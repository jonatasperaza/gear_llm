import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from gear_llm.config import DEFAULT_CHEAP_MODEL, DEFAULT_EXPENSIVE_MODEL
from gear_llm.model_loader import (
    ensure_shared_prompt_encoding,
    get_model_runtime_metadata,
    load_causal_lm_model,
    load_tokenizer_pair,
    resolve_device,
    resolve_split_devices,
)
from gear_llm.report import save_csv
from gear_llm.runtime_profiler import RuntimeProfiler


@dataclass
class AdaptiveGenerationConfig:
    cheap_model_name: str = DEFAULT_CHEAP_MODEL
    expensive_model_name: str = DEFAULT_EXPENSIVE_MODEL
    device: str = "auto"
    cheap_device: str | None = None
    expensive_device: str | None = None
    torch_dtype: str = "auto"
    prompt_format: str = "auto"
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
    repetition_guard_requires_uncertainty: bool = True
    repetition_guard_entropy_threshold: float = 0.25
    repetition_guard_margin_threshold: float = 0.35
    repetition_guard_cooldown_tokens: int = 8
    enable_code_structural_fallback: bool = False


def load_adaptive_models(config: AdaptiveGenerationConfig):
    """
    Carrega tokenizer, modelo barato e modelo caro.

    Os dois modelos SmolLM2 usam tokenizer compatível; para manter a geração
    simples, usamos o tokenizer do modelo barato para codificar e decodificar.
    """

    cheap_device, expensive_device, primary_device = resolve_split_devices(
        device=config.device,
        cheap_device=config.cheap_device,
        expensive_device=config.expensive_device,
    )
    tokenizer, _ = load_tokenizer_pair(
        cheap_model_name=config.cheap_model_name,
        expensive_model_name=config.expensive_model_name,
    )
    cheap_model = load_causal_lm_model(
        model_name=config.cheap_model_name,
        device=cheap_device,
        torch_dtype=config.torch_dtype,
    )
    expensive_model = load_causal_lm_model(
        model_name=config.expensive_model_name,
        device=expensive_device,
        torch_dtype=config.torch_dtype,
    )

    return cheap_model, expensive_model, tokenizer, primary_device


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


def is_code_structural_token(token_text: str) -> bool:
    stripped = token_text.strip()
    lowered = stripped.lower()

    if "\n" in token_text or "\t" in token_text:
        return True
    if token_text.startswith("    ") or token_text.startswith("  "):
        return True

    structural_keywords = {
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
    }
    if lowered in structural_keywords:
        return True

    keyword_pattern = (
        r"\b(def|return|if|else|elif|for|while|import|from|try|except|with|lambda)\b"
    )
    if re.search(keyword_pattern, lowered):
        return True

    structural_symbols = (
        ":",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "=",
        "==",
        "!=",
        "<=",
        ">=",
        "+",
        "-",
        "*",
        "/",
        "%",
        ",",
        ".",
        "\"",
        "'",
        "`",
    )
    return any(symbol in token_text for symbol in structural_symbols)


def fallback_reasons(
    cheap_stats: dict,
    candidate_token_text: str,
    step: int,
    generated_tokens: list[str],
    config: AdaptiveGenerationConfig,
    expensive_model_calls_so_far: int,
    last_repetition_guard_step: int | None,
) -> dict:
    required_reasons = []
    optional_reasons = []

    if cheap_stats["entropy"] > config.entropy_threshold:
        required_reasons.append("entropy_high")

    if cheap_stats["margin"] < config.margin_threshold:
        required_reasons.append("margin_low")

    repetition_rate = repeated_ngram_rate_from_tokens(
        generated_tokens,
        config.repetition_ngram_size,
    )
    repetition_candidate = (
        config.enable_repetition_guard
        and repetition_rate >= config.repetition_threshold
    )
    repetition_uncertainty_passed = False
    repetition_cooldown_blocked = False
    repetition_triggered = False

    if repetition_candidate:
        if config.repetition_guard_requires_uncertainty:
            repetition_uncertainty_passed = (
                cheap_stats["entropy"]
                > config.repetition_guard_entropy_threshold
                or cheap_stats["margin"]
                < config.repetition_guard_margin_threshold
            )
        else:
            repetition_uncertainty_passed = True

        if (
            last_repetition_guard_step is not None
            and config.repetition_guard_cooldown_tokens > 0
        ):
            steps_since_repetition_guard = step - last_repetition_guard_step
            repetition_cooldown_blocked = (
                0 < steps_since_repetition_guard
                <= config.repetition_guard_cooldown_tokens
            )

        repetition_triggered = (
            repetition_uncertainty_passed and not repetition_cooldown_blocked
        )

    if repetition_triggered:
        optional_reasons.append("repetition_guard")

    code_structural_triggered = (
        config.enable_code_structural_fallback
        and is_code_structural_token(candidate_token_text)
    )
    if code_structural_triggered:
        optional_reasons.append("code_structural_token")

    periodic_triggered = (
        config.enable_periodic_teacher_check
        and config.teacher_check_interval > 0
        and step > 0
        and step % config.teacher_check_interval == 0
    )
    periodic_risk_triggered = False
    periodic_budget_blocked = False
    budget_blocked_reason = None
    optional_fallback_budget_blocked = False
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
            optional_reasons.append(periodic_reason)

    fallback_required = bool(required_reasons)
    fallback_optional_only = bool(optional_reasons) and not fallback_required
    over_budget = (
        config.max_expensive_call_ratio >= 0
        and expensive_call_ratio_so_far >= config.max_expensive_call_ratio
    )

    if fallback_optional_only and over_budget:
        optional_fallback_budget_blocked = True
        periodic_budget_blocked = any(
            reason.startswith("periodic_teacher_check")
            for reason in optional_reasons
        )

        if optional_reasons and all(
            reason.startswith("periodic_teacher_check")
            for reason in optional_reasons
        ):
            budget_blocked_reason = "periodic_teacher_check_budget_blocked"
        else:
            budget_blocked_reason = "optional_fallback_budget_blocked"

        reasons = []
    else:
        reasons = required_reasons + optional_reasons

    return {
        "reasons": reasons,
        "required_reasons": required_reasons,
        "optional_reasons": optional_reasons,
        "budget_blocked_reason": budget_blocked_reason,
        "periodic_teacher_check_triggered": periodic_triggered,
        "periodic_risk_triggered": periodic_risk_triggered,
        "periodic_budget_blocked": periodic_budget_blocked,
        "repetition_guard_triggered": repetition_triggered,
        "optional_fallback_budget_blocked": optional_fallback_budget_blocked,
        "repetition_guard_uncertainty_passed": repetition_uncertainty_passed,
        "repetition_guard_cooldown_blocked": repetition_cooldown_blocked,
        "code_structural_token_triggered": code_structural_triggered,
        "fallback_required": fallback_required,
        "fallback_optional_only": fallback_optional_only,
        "expensive_call_ratio_so_far": expensive_call_ratio_so_far,
    }


@torch.no_grad()
def choose_next_token(
    input_ids: torch.Tensor,
    cheap_model,
    expensive_model,
    tokenizer,
    config: AdaptiveGenerationConfig,
    step: int,
    generated_tokens: list[str],
    expensive_model_calls_so_far: int,
    last_repetition_guard_step: int | None,
    expensive_device: str,
    runtime_profiler: RuntimeProfiler | None = None,
) -> dict:
    cheap_device = str(input_ids.device)
    if runtime_profiler is None:
        cheap_outputs = cheap_model(input_ids=input_ids, return_dict=True)
    else:
        with runtime_profiler.forward("cheap", cheap_device):
            cheap_outputs = cheap_model(input_ids=input_ids, return_dict=True)

    cheap_logits = cheap_outputs.logits[0, -1]
    if runtime_profiler is None:
        cheap_stats = next_token_stats(
            logits=cheap_logits,
            temperature=config.temperature,
        )
    else:
        with runtime_profiler.timed("router_decision_time_seconds"):
            cheap_stats = next_token_stats(
                logits=cheap_logits,
                temperature=config.temperature,
            )

    if runtime_profiler is None:
        candidate_token_text = tokenizer.decode(
            [cheap_stats["token_id"]],
            clean_up_tokenization_spaces=False,
        )
    else:
        with runtime_profiler.timed("tokenizer_decode_time_seconds"):
            candidate_token_text = tokenizer.decode(
                [cheap_stats["token_id"]],
                clean_up_tokenization_spaces=False,
            )

    if runtime_profiler is None:
        fallback = fallback_reasons(
            cheap_stats=cheap_stats,
            candidate_token_text=candidate_token_text,
            step=step,
            generated_tokens=generated_tokens,
            config=config,
            expensive_model_calls_so_far=expensive_model_calls_so_far,
            last_repetition_guard_step=last_repetition_guard_step,
        )
    else:
        with runtime_profiler.timed("guard_time_seconds"):
            fallback = fallback_reasons(
                cheap_stats=cheap_stats,
                candidate_token_text=candidate_token_text,
                step=step,
                generated_tokens=generated_tokens,
                config=config,
                expensive_model_calls_so_far=expensive_model_calls_so_far,
                last_repetition_guard_step=last_repetition_guard_step,
            )
        runtime_profiler.increment("number_of_router_decisions")
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
            "optional_fallback_budget_blocked": fallback[
                "optional_fallback_budget_blocked"
            ],
            "repetition_guard_uncertainty_passed": fallback[
                "repetition_guard_uncertainty_passed"
            ],
            "repetition_guard_cooldown_blocked": fallback[
                "repetition_guard_cooldown_blocked"
            ],
            "code_structural_token_triggered": fallback[
                "code_structural_token_triggered"
            ],
            "fallback_required": fallback["fallback_required"],
            "fallback_optional_only": fallback["fallback_optional_only"],
        }

    expensive_input_ids = input_ids.to(expensive_device)
    if runtime_profiler is None:
        expensive_outputs = expensive_model(
            input_ids=expensive_input_ids,
            return_dict=True,
        )
    else:
        with runtime_profiler.forward("expensive", expensive_device):
            expensive_outputs = expensive_model(
                input_ids=expensive_input_ids,
                return_dict=True,
            )
    expensive_logits = expensive_outputs.logits[0, -1]
    if runtime_profiler is None:
        expensive_stats = next_token_stats(
            logits=expensive_logits,
            temperature=config.temperature,
        )
    else:
        with runtime_profiler.timed("router_decision_time_seconds"):
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
        "optional_fallback_budget_blocked": fallback[
            "optional_fallback_budget_blocked"
        ],
        "repetition_guard_uncertainty_passed": fallback[
            "repetition_guard_uncertainty_passed"
        ],
        "repetition_guard_cooldown_blocked": fallback[
            "repetition_guard_cooldown_blocked"
        ],
        "code_structural_token_triggered": fallback[
            "code_structural_token_triggered"
        ],
        "fallback_required": fallback["fallback_required"],
        "fallback_optional_only": fallback["fallback_optional_only"],
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
    runtime_profiler: RuntimeProfiler | None = None,
) -> tuple[str, list[dict], dict]:
    cheap_runtime = get_model_runtime_metadata(cheap_model, fallback_device=device)
    expensive_runtime = get_model_runtime_metadata(
        expensive_model,
        fallback_device=device,
    )
    cheap_device = cheap_runtime["device"]
    expensive_device = expensive_runtime["device"]

    # The adaptive loop keeps the mutable generation context on the cheap
    # model device. Expensive fallbacks move a copy only when needed.
    if runtime_profiler is None:
        encoded, effective_prompt_format_cheap, effective_prompt_format_expensive = (
            ensure_shared_prompt_encoding(
                prompt=prompt,
                tokenizer=tokenizer,
                device=device,
                prompt_format=config.prompt_format,
                cheap_device=cheap_device,
                expensive_device=expensive_device,
            )
        )
    else:
        with runtime_profiler.timed("prompt_format_time_seconds"):
            (
                encoded,
                effective_prompt_format_cheap,
                effective_prompt_format_expensive,
            ) = ensure_shared_prompt_encoding(
                prompt=prompt,
                tokenizer=tokenizer,
                device=device,
                prompt_format=config.prompt_format,
                cheap_device=cheap_device,
                expensive_device=expensive_device,
            )
    input_ids = encoded["input_ids"].to(cheap_device)
    prompt_length = input_ids.shape[-1]
    history = []
    generated_tokens = []
    expensive_model_calls_so_far = 0
    last_repetition_guard_step = None

    for index in range(config.max_new_tokens):
        decision = choose_next_token(
            input_ids=input_ids,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            tokenizer=tokenizer,
            config=config,
            step=index,
            generated_tokens=generated_tokens,
            expensive_model_calls_so_far=expensive_model_calls_so_far,
            last_repetition_guard_step=last_repetition_guard_step,
            expensive_device=expensive_device,
            runtime_profiler=runtime_profiler,
        )
        token_id = decision["token_id"]
        token_tensor = torch.tensor([[token_id]], device=cheap_device)
        input_ids = torch.cat([input_ids, token_tensor], dim=-1)

        if runtime_profiler is None:
            token_text = tokenizer.decode(
                [token_id],
                clean_up_tokenization_spaces=False,
            )
        else:
            with runtime_profiler.timed("tokenizer_decode_time_seconds"):
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
                "optional_fallback_budget_blocked": decision[
                    "optional_fallback_budget_blocked"
                ],
                "repetition_guard_uncertainty_passed": decision[
                    "repetition_guard_uncertainty_passed"
                ],
                "repetition_guard_cooldown_blocked": decision[
                    "repetition_guard_cooldown_blocked"
                ],
                "code_structural_token_triggered": decision[
                    "code_structural_token_triggered"
                ],
                "fallback_required": decision["fallback_required"],
                "fallback_optional_only": decision["fallback_optional_only"],
                "prompt_format": config.prompt_format,
                "cheap_device": cheap_device,
                "expensive_device": expensive_device,
                "effective_prompt_format_cheap": effective_prompt_format_cheap,
                "effective_prompt_format_expensive": (
                    effective_prompt_format_expensive
                ),
            }
        )

        if decision["route"] == "expensive":
            expensive_model_calls_so_far += 1

            if "repetition_guard" in decision["fallback_reason"].split("+"):
                last_repetition_guard_step = index

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    generated_ids = input_ids[0, prompt_length:]
    if runtime_profiler is None:
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
    else:
        with runtime_profiler.timed("tokenizer_decode_time_seconds"):
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
    summary.update(
        {
            "prompt_format": config.prompt_format,
            "cheap_device": cheap_device,
            "expensive_device": expensive_device,
            "effective_prompt_format_cheap": effective_prompt_format_cheap,
            "effective_prompt_format_expensive": (
                effective_prompt_format_expensive
            ),
        }
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
