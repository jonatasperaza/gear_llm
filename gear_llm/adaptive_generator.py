import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gear_llm.model_loader import get_device
from gear_llm.report import save_csv


@dataclass
class AdaptiveGenerationConfig:
    cheap_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    expensive_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    max_new_tokens: int = 80
    temperature: float = 0.7
    entropy_threshold: float = 0.35
    margin_threshold: float = 0.20
    cheap_call_cost: float = 0.35
    expensive_call_cost: float = 1.00
    teacher_check_interval: int = 16
    enable_periodic_teacher_check: bool = True
    enable_repetition_guard: bool = True
    repetition_ngram_size: int = 3
    repetition_threshold: float = 0.25
    risk_gated_periodic_check: bool = True
    periodic_entropy_risk_threshold: float = 0.25
    periodic_margin_risk_threshold: float = 0.35
    periodic_repetition_risk_threshold: float = 0.05
    max_expensive_call_ratio: float = 0.40


def _load_model(model_name: str, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

    model.to(device)
    model.eval()
    return model


def load_adaptive_models(config: AdaptiveGenerationConfig):
    """
    Carrega tokenizer, modelo barato e modelo caro.

    Os dois modelos SmolLM2 usam tokenizer compatível; para manter a geração
    simples, usamos o tokenizer do modelo barato para codificar e decodificar.
    """

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.cheap_model_name)
    cheap_model = _load_model(config.cheap_model_name, device)
    expensive_model = _load_model(config.expensive_model_name, device)

    return cheap_model, expensive_model, tokenizer, device


def next_token_stats(logits: torch.Tensor, temperature: float) -> dict:
    """
    Calcula estatísticas de incerteza para o próximo token.
    """

    safe_temperature = max(temperature, 1e-6)
    scaled_logits = logits.float() / safe_temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))

    entropy = -(probs * log_probs).sum()
    normalized_entropy = entropy / math.log(probs.numel())

    top_probs, top_ids = torch.topk(probs, k=2)
    top1_prob = float(top_probs[0].detach().cpu())
    top2_prob = float(top_probs[1].detach().cpu())
    margin = top1_prob - top2_prob

    return {
        "token_id": int(top_ids[0].detach().cpu()),
        "entropy": float(normalized_entropy.detach().cpu()),
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "margin": margin,
    }


def repeated_ngram_rate_from_tokens(tokens: list[str], ngram_size: int) -> float:
    if ngram_size <= 0 or len(tokens) < ngram_size:
        return 0.0

    ngrams = [
        tuple(tokens[index : index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    ]
    counts: dict[tuple[str, ...], int] = {}

    for ngram in ngrams:
        counts[ngram] = counts.get(ngram, 0) + 1

    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def fallback_reasons(
    cheap_stats: dict,
    step: int,
    generated_tokens: list[str],
    config: AdaptiveGenerationConfig,
    expensive_model_calls_so_far: int,
) -> dict:
    reasons = []

    if cheap_stats["entropy"] > config.entropy_threshold:
        reasons.append("entropy_high")

    if cheap_stats["margin"] < config.margin_threshold:
        reasons.append("margin_low")

    repetition_rate = repeated_ngram_rate_from_tokens(
        generated_tokens,
        config.repetition_ngram_size,
    )
    repetition_triggered = (
        config.enable_repetition_guard
        and repetition_rate >= config.repetition_threshold
    )

    if repetition_triggered:
        reasons.append("repetition_guard")

    periodic_triggered = (
        config.enable_periodic_teacher_check
        and config.teacher_check_interval > 0
        and step > 0
        and step % config.teacher_check_interval == 0
    )
    periodic_risk_triggered = False
    periodic_budget_blocked = False
    budget_blocked_reason = None
    expensive_call_ratio_so_far = expensive_model_calls_so_far / max(1, step)

    if periodic_triggered:
        periodic_reason = "periodic_teacher_check"
        should_periodic_fallback = True

        if config.risk_gated_periodic_check:
            periodic_reason = "periodic_teacher_check_risk_gated"
            periodic_risk_triggered = (
                cheap_stats["entropy"] > config.periodic_entropy_risk_threshold
                or cheap_stats["margin"] < config.periodic_margin_risk_threshold
                or repetition_rate > config.periodic_repetition_risk_threshold
            )
            should_periodic_fallback = periodic_risk_triggered

        if should_periodic_fallback:
            periodic_only = not reasons
            over_budget = (
                config.max_expensive_call_ratio >= 0
                and expensive_call_ratio_so_far >= config.max_expensive_call_ratio
            )

            if periodic_only and over_budget:
                periodic_budget_blocked = True
                budget_blocked_reason = "periodic_teacher_check_budget_blocked"
            else:
                reasons.append(periodic_reason)

    return {
        "reasons": reasons,
        "budget_blocked_reason": budget_blocked_reason,
        "periodic_teacher_check_triggered": periodic_triggered,
        "periodic_risk_triggered": periodic_risk_triggered,
        "periodic_budget_blocked": periodic_budget_blocked,
        "repetition_guard_triggered": repetition_triggered,
        "expensive_call_ratio_so_far": expensive_call_ratio_so_far,
    }


@torch.no_grad()
def choose_next_token(
    input_ids: torch.Tensor,
    cheap_model,
    expensive_model,
    config: AdaptiveGenerationConfig,
    step: int,
    generated_tokens: list[str],
    expensive_model_calls_so_far: int,
) -> dict:
    cheap_outputs = cheap_model(input_ids=input_ids, return_dict=True)
    cheap_logits = cheap_outputs.logits[0, -1]
    cheap_stats = next_token_stats(
        logits=cheap_logits,
        temperature=config.temperature,
    )

    fallback = fallback_reasons(
        cheap_stats=cheap_stats,
        step=step,
        generated_tokens=generated_tokens,
        config=config,
        expensive_model_calls_so_far=expensive_model_calls_so_far,
    )
    reasons = fallback["reasons"]

    if not reasons:
        return {
            **cheap_stats,
            "route": "cheap",
            "fallback_reason": fallback["budget_blocked_reason"]
            or "cheap_confident",
            "periodic_teacher_check_triggered": fallback[
                "periodic_teacher_check_triggered"
            ],
            "repetition_guard_triggered": fallback[
                "repetition_guard_triggered"
            ],
            "expensive_call_ratio_so_far": fallback[
                "expensive_call_ratio_so_far"
            ],
            "periodic_risk_triggered": fallback["periodic_risk_triggered"],
            "periodic_budget_blocked": fallback["periodic_budget_blocked"],
        }

    expensive_outputs = expensive_model(input_ids=input_ids, return_dict=True)
    expensive_logits = expensive_outputs.logits[0, -1]
    expensive_stats = next_token_stats(
        logits=expensive_logits,
        temperature=config.temperature,
    )

    return {
        **cheap_stats,
        "token_id": expensive_stats["token_id"],
        "route": "expensive",
        "fallback_reason": "+".join(reasons),
        "periodic_teacher_check_triggered": fallback[
            "periodic_teacher_check_triggered"
        ],
        "repetition_guard_triggered": fallback["repetition_guard_triggered"],
        "expensive_call_ratio_so_far": fallback["expensive_call_ratio_so_far"],
        "periodic_risk_triggered": fallback["periodic_risk_triggered"],
        "periodic_budget_blocked": fallback["periodic_budget_blocked"],
    }


def summarize_adaptive_history(
    prompt: str,
    generated_text: str,
    full_text: str,
    history: list[dict],
    config: AdaptiveGenerationConfig,
) -> dict:
    total_generated_tokens = len(history)
    cheap_accepted_tokens = sum(1 for row in history if row["route"] == "cheap")
    expensive_model_calls = sum(
        1 for row in history if row["route"] == "expensive"
    )
    cheap_percent = (
        100 * cheap_accepted_tokens / total_generated_tokens
        if total_generated_tokens
        else 0.0
    )

    baseline_cost = total_generated_tokens * config.expensive_call_cost
    adaptive_cost = (
        total_generated_tokens * config.cheap_call_cost
        + expensive_model_calls * config.expensive_call_cost
    )
    estimated_saved_percent = (
        100 * (baseline_cost - adaptive_cost) / baseline_cost
        if baseline_cost
        else 0.0
    )

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "total_generated_tokens": total_generated_tokens,
        "cheap_accepted_tokens": cheap_accepted_tokens,
        "expensive_model_calls": expensive_model_calls,
        "cheap_percent": cheap_percent,
        "baseline_cost": baseline_cost,
        "adaptive_cost": adaptive_cost,
        "estimated_saved_percent": estimated_saved_percent,
    }


@torch.no_grad()
def adaptive_generate_with_models(
    prompt: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    config: AdaptiveGenerationConfig,
) -> tuple[str, list[dict], dict]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_length = input_ids.shape[-1]
    history = []
    generated_tokens = []

    for index in range(config.max_new_tokens):
        expensive_model_calls_so_far = sum(
            1 for row in history if row["route"] == "expensive"
        )
        decision = choose_next_token(
            input_ids=input_ids,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            config=config,
            step=index,
            generated_tokens=generated_tokens,
            expensive_model_calls_so_far=expensive_model_calls_so_far,
        )
        token_id = decision["token_id"]
        token_tensor = torch.tensor([[token_id]], device=device)
        input_ids = torch.cat([input_ids, token_tensor], dim=-1)

        token_text = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
        generated_tokens.append(token_text)

        history.append(
            {
                "index": index,
                "token": token_text,
                "route": decision["route"],
                "entropy": decision["entropy"],
                "top1_prob": decision["top1_prob"],
                "top2_prob": decision["top2_prob"],
                "margin": decision["margin"],
                "fallback_reason": decision["fallback_reason"],
                "periodic_teacher_check_triggered": decision[
                    "periodic_teacher_check_triggered"
                ],
                "repetition_guard_triggered": decision[
                    "repetition_guard_triggered"
                ],
                "expensive_call_ratio_so_far": decision[
                    "expensive_call_ratio_so_far"
                ],
                "periodic_risk_triggered": decision["periodic_risk_triggered"],
                "periodic_budget_blocked": decision[
                    "periodic_budget_blocked"
                ],
            }
        )

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    generated_ids = input_ids[0, prompt_length:]
    generated_text = tokenizer.decode(
        generated_ids,
        clean_up_tokenization_spaces=False,
        skip_special_tokens=True,
    )
    full_text = tokenizer.decode(
        input_ids[0],
        clean_up_tokenization_spaces=False,
        skip_special_tokens=True,
    )
    summary = summarize_adaptive_history(
        prompt=prompt,
        generated_text=generated_text,
        full_text=full_text,
        history=history,
        config=config,
    )

    return full_text, history, summary


def adaptive_generate(
    prompt: str,
    config: AdaptiveGenerationConfig,
) -> tuple[str, list[dict], dict]:
    cheap_model, expensive_model, tokenizer, device = load_adaptive_models(config)

    return adaptive_generate_with_models(
        prompt=prompt,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        config=config,
    )


def print_adaptive_report(summary: dict):
    print()
    print("Adaptive Dual-Model Generation")
    print("=" * 100)
    print("Texto gerado")
    print("-" * 100)
    print(summary["full_text"])
    print("-" * 100)
    print(f"total_generated_tokens : {summary['total_generated_tokens']}")
    print(f"cheap_accepted_tokens  : {summary['cheap_accepted_tokens']}")
    print(f"expensive_model_calls  : {summary['expensive_model_calls']}")
    print(f"cheap_percent          : {summary['cheap_percent']:.2f}%")
    print(f"estimated_saved_percent: {summary['estimated_saved_percent']:.2f}%")
    print("=" * 100)
    print()


def save_adaptive_history(history: list[dict], path: str | Path):
    save_csv(history, str(path))


def save_adaptive_summary_rows(summaries: list[dict], path: str | Path):
    save_csv(summaries, str(path))
