"""Cheap-model probing features for the learned prompt router (v2).

The v1 router used TF-IDF over the prompt text alone. Its ROC-AUC on unseen
prompts was 0.5545 (≈ random) and Average Precision 0.2047, because the prompt
text on its own barely carries signal about *reasoning difficulty*. PROBLEMS.md
items #2, #5, #13 document this.

This module extracts a richer feature set from a SINGLE forward pass of the
cheap model and a SINGLE forward pass of the expensive model over the prompt:

  - cheap-model uncertainty aggregated over the prompt tokens (entropy, margin,
    top1 probability, surprisal);
  - cheap-vs-expensive agreement on the last position and over the prompt
    (exact top-1 match, top-k membership, symmetric KL on the union of top-k);
  - geometric features from the cheap model's hidden states (novelty,
    curvature);
  - structural density (fraction of structurally-important tokens);
  - prompt shape (token count, word count).

All functions reuse the existing pure-tensor helpers in ``gear_llm.metrics`` and
``gear_llm.teacher_calibration``. The expensive forward pass is the price of the
agreement features; if the caller only has the cheap model loaded, the
``expensive_model`` argument may be ``None`` and the agreement block is skipped
(that subset of features still carries most of the uncertainty signal).

The output is a flat ``dict[str, float]`` with stable, documented key names, so
it can be stored as columns in a CSV and consumed by the sklearn router.
"""

from __future__ import annotations

import re
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from gear_llm.metrics import (
    curvature_from_hidden,
    entropy_from_logits,
    novelty_from_hidden,
    robust_normalize,
    structural_importance_from_tokens,
    surprisal_from_logits,
)
from gear_llm.model_loader import (
    encode_prompt,
    get_cheap_tokenizer,
    get_expensive_tokenizer,
    get_model_runtime_metadata,
    tokenizers_have_compatible_vocab,
)
from gear_llm.teacher_calibration import (
    approximate_topk_kl,
    distribution_stats,
)


# Stable feature key names. These become CSV column names and are read back by
# the training script, so they are part of the on-disk contract. Do not rename
# without migrating consumers.
PROBING_FEATURE_KEYS = (
    # Cheap-model uncertainty over the prompt.
    "cheap_entropy_mean",
    "cheap_entropy_max",
    "cheap_entropy_std",
    "cheap_entropy_last",
    "cheap_margin_mean",
    "cheap_margin_min",
    "cheap_top1_prob_mean",
    "cheap_top1_prob_min",
    "cheap_surprisal_mean",
    "cheap_surprisal_max",
    # Cheap-vs-expensive agreement (last position).
    "agree_last_exact",
    "agree_last_cheap_in_exp_topk",
    "agree_last_cheap_rank_in_exp_topk",
    "agree_last_exp_in_cheap_topk",
    "agree_last_kl_exp_to_cheap",
    "agree_last_kl_cheap_to_exp",
    # Cheap-vs-expensive agreement (averaged over prompt positions).
    "agree_seq_exact_mean",
    "agree_seq_topk_mean",
    "agree_seq_kl_mean",
    # Geometric (cheap hidden states).
    "novelty_mean",
    "novelty_max",
    "curvature_mean",
    "curvature_max",
    # Structural density.
    "frac_structural",
    # Prompt shape.
    "n_tokens",
    "n_words",
)


CODE_STRUCTURAL_WORDS = {
    "def",
    "return",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "import",
    "from",
    "try",
    "except",
    "with",
    "lambda",
    "class",
    "yield",
    "raise",
}


@torch.no_grad()
def _per_position_top1(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax token id at every position. ``logits`` is [seq, vocab]."""
    return logits.argmax(dim=-1)


def _safe_stats(values: torch.Tensor) -> tuple[float, float, float, float]:
    """Return (mean, max, std, last) of a 1-D tensor as python floats.

    Empty tensors degrade to zeros rather than NaNs, so a degenerate prompt
    (length 0/1) never poisons downstream CSV rows.
    """
    if values.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    vals = values.detach().float().cpu()
    mean = float(vals.mean())
    if vals.numel() == 1:
        v = float(vals[0])
        return v, v, 0.0, v
    std = float(vals.std(unbiased=False))
    return mean, float(vals.max()), std, float(vals[-1])


def _per_position_top1_match(
    cheap_logits: torch.Tensor,
    expensive_logits: torch.Tensor,
) -> torch.Tensor:
    """Boolean tensor [seq] of cheap_top1 == expensive_top1 per position."""
    cheap_top1 = _per_position_top1(cheap_logits).detach().cpu()
    expensive_top1 = _per_position_top1(expensive_logits).detach().cpu()
    return (cheap_top1 == expensive_top1).float()


def _per_position_topk_match(
    cheap_logits: torch.Tensor,
    expensive_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Fraction overlap of the top-k sets per position, tensor [seq] in [0,1].

    Mirrors the spirit of teacher_calibration's topk_match but vectorized over
    positions: for each position we count how many of the cheap top-k ids appear
    in the expensive top-k ids, divided by k.
    """
    k = max(top_k, 1)
    cheap_topk = torch.topk(cheap_logits, k=k, dim=-1).indices.detach().cpu()
    expensive_topk = (
        torch.topk(expensive_logits, k=k, dim=-1).indices.detach().cpu()
    )
    # Expand to compute membership: for each position, for each cheap id, check
    # presence in the expensive top-k set.
    seq_len = cheap_logits.shape[0]
    matches = torch.zeros(seq_len)
    for pos in range(seq_len):
        exp_set = set(int(i) for i in expensive_topk[pos].tolist())
        cheap_ids = cheap_topk[pos].tolist()
        hits = sum(1 for i in cheap_ids if int(i) in exp_set)
        matches[pos] = hits / k
    return matches.float().cpu()


def _per_position_topk_kl(
    cheap_logits: torch.Tensor,
    expensive_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Approximate KL(expensive || cheap) over each position's top-k union.

    Only the logits in the union are copied to CPU. This keeps the helper
    device-safe for split-GPU runs without transferring full vocabularies.
    """
    k = max(top_k, 1)
    cheap_topk = torch.topk(cheap_logits, k=k, dim=-1).indices.detach().cpu()
    expensive_topk = (
        torch.topk(expensive_logits, k=k, dim=-1).indices.detach().cpu()
    )
    values = []
    for pos in range(cheap_logits.shape[0]):
        union_ids = sorted(
            set(int(value) for value in cheap_topk[pos].tolist())
            | set(int(value) for value in expensive_topk[pos].tolist())
        )
        cheap_index = torch.tensor(union_ids, device=cheap_logits.device)
        expensive_index = torch.tensor(union_ids, device=expensive_logits.device)
        cheap_union = cheap_logits[pos].index_select(0, cheap_index).float().cpu()
        expensive_union = (
            expensive_logits[pos]
            .index_select(0, expensive_index)
            .float()
            .cpu()
        )
        cheap_probs = F.softmax(cheap_union, dim=-1).clamp_min(1e-12)
        expensive_probs = F.softmax(expensive_union, dim=-1).clamp_min(1e-12)
        kl = (
            expensive_probs
            * (torch.log(expensive_probs) - torch.log(cheap_probs))
        ).sum()
        values.append(float(kl))
    return torch.tensor(values, dtype=torch.float32)


def _last_position_agreement(
    cheap_logits_last: torch.Tensor,
    expensive_logits_last: torch.Tensor,
    temperature: float,
    top_k: int,
) -> dict[str, float]:
    """Agreement features restricted to the last position (the one that would
    predict the first generated token). Uses teacher_calibration primitives so
    the definition matches the project's existing calibration work."""
    # Keeping both distributions on CPU makes the KL helper safe when the
    # models live on different CUDA devices.
    cheap_logits_last = cheap_logits_last.detach().float().cpu()
    expensive_logits_last = expensive_logits_last.detach().float().cpu()
    cheap_stats = distribution_stats(
        cheap_logits_last,
        top_k=top_k,
        temperature=temperature,
    )
    expensive_stats = distribution_stats(
        expensive_logits_last, top_k=top_k, temperature=temperature
    )

    exact = 1.0 if cheap_stats["top1_id"] == expensive_stats["top1_id"] else 0.0
    cheap_in_exp = 1.0 if cheap_stats["top1_id"] in expensive_stats["top_ids"].tolist() else 0.0
    exp_in_cheap = 1.0 if expensive_stats["top1_id"] in cheap_stats["top_ids"].tolist() else 0.0
    expensive_top_ids = expensive_stats["top_ids"].tolist()
    cheap_rank = (
        float(expensive_top_ids.index(cheap_stats["top1_id"]) + 1)
        if cheap_stats["top1_id"] in expensive_top_ids
        else 0.0
    )

    kl_exp_to_cheap = approximate_topk_kl(
        cheap_probs=cheap_stats["probs"],
        expensive_probs=expensive_stats["probs"],
        cheap_top_ids=cheap_stats["top_ids"],
        expensive_top_ids=expensive_stats["top_ids"],
    )
    # Symmetric direction: swap who is "teacher" in the helper.
    kl_cheap_to_exp = approximate_topk_kl(
        cheap_probs=expensive_stats["probs"],
        expensive_probs=cheap_stats["probs"],
        cheap_top_ids=expensive_stats["top_ids"],
        expensive_top_ids=cheap_stats["top_ids"],
    )

    return {
        "agree_last_exact": float(exact),
        "agree_last_cheap_in_exp_topk": float(cheap_in_exp),
        "agree_last_cheap_rank_in_exp_topk": cheap_rank,
        "agree_last_exp_in_cheap_topk": float(exp_in_cheap),
        "agree_last_kl_exp_to_cheap": float(kl_exp_to_cheap),
        "agree_last_kl_cheap_to_exp": float(kl_cheap_to_exp),
    }


def _decode_tokens(tokenizer, input_ids_1d: torch.Tensor) -> list[str]:
    ids = input_ids_1d.detach().cpu().tolist()
    return [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in ids
    ]


@torch.no_grad()
def compute_probing_features(
    prompt: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    prompt_format: str = "auto",
    top_k: int = 5,
    temperature: float = 1.0,
    novelty_window: int = 8,
    runtime_profiler=None,
) -> dict[str, float]:
    """Compute prompt-level probing features for one prompt.

    Runs ONE forward pass of the cheap model and (if not None) ONE forward pass
    of the expensive model over the prompt, then reduces per-token statistics to
    a flat feature dict. The dict always contains every key in
    :data:`PROBING_FEATURE_KEYS`; agreement features are 0.0 when
    ``expensive_model`` is None.

    Args:
        prompt: raw user prompt text.
        cheap_model: loaded causal LM used as the "draft" / cheap route.
        expensive_model: loaded causal LM used as the expensive route. May be
            None, in which case agreement features are skipped.
        tokenizer: the paired tokenizer handle returned by load_tokenizer_pair
            (carries both cheap/expensive tokenizers as attributes).
        device: primary device string (cheap model device).
        prompt_format: "auto" | "chat" | "raw", forwarded to encode_prompt.
        top_k: size of the top-k sets used in agreement features.
        temperature: temperature applied to logits for the agreement stats.
        novelty_window: window for novelty_from_hidden.

    Returns:
        dict with all keys in PROBING_FEATURE_KEYS (and no others).
    """
    cheap_model.eval()
    if expensive_model is not None:
        expensive_model.eval()

    cheap_tokenizer = get_cheap_tokenizer(tokenizer)

    # Each tokenizer formats the prompt independently. Agreement features are
    # computed only after vocabulary and input-id compatibility are verified.
    prompt_timer = (
        runtime_profiler.timed("prompt_format_time_seconds")
        if runtime_profiler is not None
        else nullcontext()
    )
    with prompt_timer:
        cheap_encoded, _ = encode_prompt(
            prompt,
            cheap_tokenizer,
            device,
            prompt_format=prompt_format,
        )
    cheap_input_ids = cheap_encoded["input_ids"][0]

    cheap_forward = (
        runtime_profiler.forward("cheap", device)
        if runtime_profiler is not None
        else nullcontext()
    )
    with cheap_forward:
        cheap_outputs = cheap_model(
            **cheap_encoded,
            output_hidden_states=True,
            return_dict=True,
        )
    cheap_logits = cheap_outputs.logits[0]  # [seq, vocab]
    cheap_last_hidden = cheap_outputs.hidden_states[-1][0]  # [seq, dim]

    # --- Per-token cheap uncertainty --------------------------------------
    # logits[-1] is the distribution for the first generated token, so prompt
    # uncertainty intentionally remains in causal-logit coordinates here.
    entropy = entropy_from_logits(cheap_logits)
    surprisal = surprisal_from_logits(
        logits=cheap_logits, input_ids=cheap_input_ids
    )
    novelty = robust_normalize(
        novelty_from_hidden(cheap_last_hidden, window=novelty_window)
    )
    curvature = robust_normalize(curvature_from_hidden(cheap_last_hidden))
    surprisal = robust_normalize(surprisal)

    # Margin and top1_prob are not directly provided per-position by the
    # existing helpers, so compute them from softmax once.
    probs = F.softmax(cheap_logits.float(), dim=-1)
    top2_probs_vals, _ = torch.topk(probs, k=2, dim=-1)
    top1_prob = top2_probs_vals[:, 0]
    top2_prob = top2_probs_vals[:, 1]
    margin = top1_prob - top2_prob

    ent_mean, ent_max, ent_std, ent_last = _safe_stats(entropy)
    sur_mean, sur_max, _, _ = _safe_stats(surprisal)
    margin_mean, margin_min, _, _ = _safe_stats(margin)
    top1_mean, top1_min, _, _ = _safe_stats(top1_prob)
    nov_mean, nov_max, _, _ = _safe_stats(novelty)
    curv_mean, curv_max, _, _ = _safe_stats(curvature)

    # --- Structural density -----------------------------------------------
    tokens = _decode_tokens(cheap_tokenizer, cheap_input_ids)
    structural = structural_importance_from_tokens(tokens, device="cpu")
    code_keyword_count = len(
        re.findall(
            r"\b(?:" + "|".join(sorted(CODE_STRUCTURAL_WORDS)) + r")\b",
            prompt.lower(),
        )
    )
    code_symbol_count = len(re.findall(r"[.'\"\n\t]", prompt))
    structural_count = (
        float(structural.sum()) + code_keyword_count + code_symbol_count
    )
    frac_structural = (
        min(1.0, structural_count / max(1, cheap_input_ids.shape[0]))
        if structural.numel()
        else 0.0
    )
    n_tokens = int(cheap_input_ids.shape[0])
    n_words = len(re.findall(r"\b\w+\b", prompt))

    features: dict[str, float] = {
        "cheap_entropy_mean": ent_mean,
        "cheap_entropy_max": ent_max,
        "cheap_entropy_std": ent_std,
        "cheap_entropy_last": ent_last,
        "cheap_margin_mean": margin_mean,
        "cheap_margin_min": margin_min,
        "cheap_top1_prob_mean": top1_mean,
        "cheap_top1_prob_min": top1_min,
        "cheap_surprisal_mean": sur_mean,
        "cheap_surprisal_max": sur_max,
        "novelty_mean": nov_mean,
        "novelty_max": nov_max,
        "curvature_mean": curv_mean,
        "curvature_max": curv_max,
        "frac_structural": frac_structural,
        "n_tokens": float(n_tokens),
        "n_words": float(n_words),
    }

    # --- Agreement features (need the expensive model) ---------------------
    # Zero-fill defaults so the column set is stable regardless of model pair.
    agreement_defaults = {
        "agree_last_exact": 0.0,
        "agree_last_cheap_in_exp_topk": 0.0,
        "agree_last_cheap_rank_in_exp_topk": 0.0,
        "agree_last_exp_in_cheap_topk": 0.0,
        "agree_last_kl_exp_to_cheap": 0.0,
        "agree_last_kl_cheap_to_exp": 0.0,
        "agree_seq_exact_mean": 0.0,
        "agree_seq_topk_mean": 0.0,
        "agree_seq_kl_mean": 0.0,
    }

    if expensive_model is not None:
        expensive_tokenizer = get_expensive_tokenizer(tokenizer)
        if not tokenizers_have_compatible_vocab(
            cheap_tokenizer,
            expensive_tokenizer,
        ):
            raise ValueError(
                "Probing agreement requires tokenizers with compatible vocabularies."
            )
        expensive_runtime = get_model_runtime_metadata(
            expensive_model, fallback_device=device
        )
        expensive_device = expensive_runtime["device"]

        # Encode the same prompt text with the expensive tokenizer.
        prompt_timer = (
            runtime_profiler.timed("prompt_format_time_seconds")
            if runtime_profiler is not None
            else nullcontext()
        )
        with prompt_timer:
            expensive_encoded, _ = encode_prompt(
                prompt,
                expensive_tokenizer,
                expensive_device,
                prompt_format=prompt_format,
            )
        expensive_forward = (
            runtime_profiler.forward("expensive", expensive_device)
            if runtime_profiler is not None
            else nullcontext()
        )
        with expensive_forward:
            expensive_outputs = expensive_model(
                **expensive_encoded,
                output_hidden_states=True,
                return_dict=True,
            )
        expensive_logits = expensive_outputs.logits[0]  # [seq, vocab]
        expensive_input_ids = expensive_encoded["input_ids"][0]
        if cheap_logits.shape[-1] != expensive_logits.shape[-1]:
            raise ValueError(
                "Probing agreement requires model output vocabularies with "
                "the same size."
            )

        # Agreement is meaningful only when both tokenizers produced the same
        # sequence. A matching vocabulary alone is not enough when templates
        # differ.
        same_length = (
            cheap_input_ids.shape[0] == expensive_input_ids.shape[0]
        )
        same_ids = (
            bool((cheap_input_ids == expensive_input_ids.to(cheap_input_ids.device)).all())
            if same_length
            else False
        )

        if not same_ids:
            raise ValueError(
                "Probing agreement requires identical prompt input_ids for "
                "cheap and expensive models."
            )

        last_agree = _last_position_agreement(
            cheap_logits_last=cheap_logits[-1],
            expensive_logits_last=expensive_logits[-1],
            temperature=temperature,
            top_k=top_k,
        )
        seq_exact = _per_position_top1_match(cheap_logits, expensive_logits)
        seq_topk = _per_position_topk_match(cheap_logits, expensive_logits, top_k)
        seq_kl = _per_position_topk_kl(cheap_logits, expensive_logits, top_k)
        seq_exact_mean = float(seq_exact.mean()) if seq_exact.numel() else 0.0
        seq_topk_mean = float(seq_topk.mean()) if seq_topk.numel() else 0.0
        seq_kl_mean = float(seq_kl.mean()) if seq_kl.numel() else 0.0

        features.update(last_agree)
        features["agree_seq_exact_mean"] = seq_exact_mean
        features["agree_seq_topk_mean"] = seq_topk_mean
        features["agree_seq_kl_mean"] = seq_kl_mean
    else:
        features.update(agreement_defaults)

    # Guarantee the contract: exactly PROBING_FEATURE_KEYS, in order.
    return {key: float(features.get(key, 0.0)) for key in PROBING_FEATURE_KEYS}


def probing_feature_record(prompt: str, **kwargs) -> dict[str, float]:
    """Thin convenience wrapper used by dataset builders.

    Returns compute_probing_features(...) directly; kept as a named entry point
    so callers can mock feature extraction in tests without importing the heavy
    model path.
    """
    return compute_probing_features(prompt=prompt, **kwargs)
