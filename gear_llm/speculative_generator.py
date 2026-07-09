from dataclasses import dataclass
from pathlib import Path

import torch

from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    load_adaptive_models,
    next_token_stats,
)
from gear_llm.config import DEFAULT_CHEAP_MODEL, DEFAULT_EXPENSIVE_MODEL
from gear_llm.model_loader import ensure_shared_prompt_encoding
from gear_llm.model_loader import get_model_runtime_metadata
from gear_llm.report import save_csv
from gear_llm.runtime_profiler import RuntimeProfiler


@dataclass
class SpeculativeGenerationConfig:
    cheap_model_name: str = DEFAULT_CHEAP_MODEL
    expensive_model_name: str = DEFAULT_EXPENSIVE_MODEL
    device: str = "auto"
    cheap_device: str | None = None
    expensive_device: str | None = None
    torch_dtype: str = "auto"
    prompt_format: str = "auto"
    max_new_tokens: int = 80
    draft_length: int = 6
    verify_top_k: int = 3
    temperature: float = 0.7
    min_draft_length: int = 2
    max_draft_length: int = 8
    cheap_token_cost: float = 0.35
    expensive_block_cost: float = 1.00


def load_speculative_models(config: SpeculativeGenerationConfig):
    adaptive_config = AdaptiveGenerationConfig(
        cheap_model_name=config.cheap_model_name,
        expensive_model_name=config.expensive_model_name,
        device=config.device,
        cheap_device=config.cheap_device,
        expensive_device=config.expensive_device,
        torch_dtype=config.torch_dtype,
        prompt_format=config.prompt_format,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )
    return load_adaptive_models(adaptive_config)


@torch.no_grad()
def generate_draft(
    input_ids: torch.Tensor,
    cheap_model,
    tokenizer,
    config: SpeculativeGenerationConfig,
    draft_length: int,
    runtime_profiler: RuntimeProfiler | None = None,
) -> tuple[list[int], list[float]]:
    draft_ids = []
    entropies = []
    draft_input_ids = input_ids

    for _ in range(draft_length):
        if runtime_profiler is None:
            outputs = cheap_model(input_ids=draft_input_ids, return_dict=True)
        else:
            with runtime_profiler.forward("cheap", str(input_ids.device)):
                outputs = cheap_model(input_ids=draft_input_ids, return_dict=True)
        logits = outputs.logits[0, -1]
        if runtime_profiler is None:
            stats = next_token_stats(logits=logits, temperature=config.temperature)
        else:
            with runtime_profiler.timed("router_decision_time_seconds"):
                stats = next_token_stats(
                    logits=logits,
                    temperature=config.temperature,
                )
            runtime_profiler.increment("number_of_router_decisions")
        token_id = stats["token_id"]

        draft_ids.append(token_id)
        entropies.append(stats["entropy"])

        token_tensor = torch.tensor([[token_id]], device=input_ids.device)
        draft_input_ids = torch.cat([draft_input_ids, token_tensor], dim=-1)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    return draft_ids, entropies


@torch.no_grad()
def verify_draft(
    input_ids: torch.Tensor,
    draft_ids: list[int],
    expensive_model,
    config: SpeculativeGenerationConfig,
    expensive_device: str,
    runtime_profiler: RuntimeProfiler | None = None,
) -> dict:
    if not draft_ids:
        return {
            "accepted_ids": [],
            "correction_id": None,
            "rejected_at": -1,
            "acceptance_rate": 0.0,
        }

    expensive_input_ids = input_ids.to(expensive_device)
    draft_tensor = torch.tensor([draft_ids], device=expensive_device)
    verification_input_ids = torch.cat([expensive_input_ids, draft_tensor], dim=-1)
    context_length = input_ids.shape[-1]

    if runtime_profiler is None:
        outputs = expensive_model(input_ids=verification_input_ids, return_dict=True)
    else:
        with runtime_profiler.forward("expensive", expensive_device):
            outputs = expensive_model(
                input_ids=verification_input_ids,
                return_dict=True,
            )
    logits = outputs.logits[0].float() / max(config.temperature, 1e-6)
    accepted_ids = []
    rejected_at = -1
    correction_id = None

    for offset, draft_id in enumerate(draft_ids):
        if runtime_profiler is None:
            prediction_position = context_length + offset - 1
            token_logits = logits[prediction_position]
            topk = min(config.verify_top_k, token_logits.numel())
            top_ids = torch.topk(token_logits, k=topk).indices.detach().cpu().tolist()
        else:
            with runtime_profiler.timed("router_decision_time_seconds"):
                prediction_position = context_length + offset - 1
                token_logits = logits[prediction_position]
                topk = min(config.verify_top_k, token_logits.numel())
                top_ids = (
                    torch.topk(token_logits, k=topk)
                    .indices.detach()
                    .cpu()
                    .tolist()
                )
            runtime_profiler.increment("number_of_router_decisions")

        if draft_id in top_ids:
            accepted_ids.append(draft_id)
            continue

        rejected_at = offset
        correction_id = int(top_ids[0])
        break

    acceptance_rate = len(accepted_ids) / len(draft_ids)

    return {
        "accepted_ids": accepted_ids,
        "correction_id": correction_id,
        "rejected_at": rejected_at,
        "acceptance_rate": acceptance_rate,
    }


def adapt_draft_length(
    current_length: int,
    acceptance_rate: float,
    cheap_entropy_avg: float,
    config: SpeculativeGenerationConfig,
) -> int:
    next_length = current_length

    if acceptance_rate >= 0.95:
        next_length += 4
    elif acceptance_rate >= 0.80 and cheap_entropy_avg <= 0.55:
        next_length += 3
    elif acceptance_rate >= 0.50 and cheap_entropy_avg <= 0.55:
        next_length += 1
    elif acceptance_rate < 0.25:
        next_length -= 2

    return max(config.min_draft_length, min(config.max_draft_length, next_length))


def summarize_speculative_generation(
    prompt: str,
    generated_text: str,
    full_text: str,
    token_rows: list[dict],
    block_rows: list[dict],
    cheap_generated_tokens: int,
    config: SpeculativeGenerationConfig,
) -> dict:
    total_generated_tokens = len(token_rows)
    cheap_accepted_tokens = sum(
        1 for row in token_rows if row["source"] == "cheap_accepted"
    )
    expensive_corrected_tokens = sum(
        1 for row in token_rows if row["source"] == "expensive_corrected"
    )
    expensive_model_calls = len(block_rows)
    acceptance_rate = (
        cheap_accepted_tokens / total_generated_tokens
        if total_generated_tokens
        else 0.0
    )

    baseline_cost = total_generated_tokens * config.expensive_block_cost
    simulated_cost = (
        cheap_generated_tokens * config.cheap_token_cost
        + expensive_model_calls * config.expensive_block_cost
    )
    estimated_saved_percent = (
        100 * (baseline_cost - simulated_cost) / baseline_cost
        if baseline_cost
        else 0.0
    )

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "total_generated_tokens": total_generated_tokens,
        "cheap_generated_tokens": cheap_generated_tokens,
        "cheap_accepted_tokens": cheap_accepted_tokens,
        "expensive_corrected_tokens": expensive_corrected_tokens,
        "expensive_model_calls": expensive_model_calls,
        "acceptance_rate": acceptance_rate,
        "baseline_cost": baseline_cost,
        "simulated_cost": simulated_cost,
        "estimated_saved_percent": estimated_saved_percent,
    }


@torch.no_grad()
def speculative_generate_with_models(
    prompt: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    config: SpeculativeGenerationConfig,
    runtime_profiler: RuntimeProfiler | None = None,
) -> tuple[str, list[dict], list[dict], dict]:
    # The draft context lives on the cheap model device; verification moves
    # a copy to the expensive model device once per block.
    cheap_runtime = get_model_runtime_metadata(
        cheap_model,
        fallback_device=device,
    )
    expensive_runtime = get_model_runtime_metadata(
        expensive_model,
        fallback_device=device,
    )
    cheap_device = cheap_runtime["device"]
    expensive_device = expensive_runtime["device"]

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
    current_draft_length = max(
        config.min_draft_length,
        min(config.max_draft_length, config.draft_length),
    )
    token_rows = []
    block_rows = []
    generated_ids = []
    cheap_generated_tokens = 0
    block_index = 0

    while len(generated_ids) < config.max_new_tokens:
        remaining = config.max_new_tokens - len(generated_ids)
        block_draft_length = min(current_draft_length, remaining)
        draft_ids, entropies = generate_draft(
            input_ids=input_ids,
            cheap_model=cheap_model,
            tokenizer=tokenizer,
            config=config,
            draft_length=block_draft_length,
            runtime_profiler=runtime_profiler,
        )

        if not draft_ids:
            break

        cheap_generated_tokens += len(draft_ids)
        verification = verify_draft(
            input_ids=input_ids,
            draft_ids=draft_ids,
            expensive_model=expensive_model,
            config=config,
            expensive_device=expensive_device,
            runtime_profiler=runtime_profiler,
        )

        accepted_ids = verification["accepted_ids"]
        emitted_ids = list(accepted_ids)

        if verification["correction_id"] is not None:
            emitted_ids.append(verification["correction_id"])

        for token_id in accepted_ids:
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
            token_rows.append(
                {
                    "index": len(token_rows),
                    "token": token_text,
                    "source": "cheap_accepted",
                    "block_index": block_index,
                    "prompt_format": config.prompt_format,
                    "cheap_device": cheap_device,
                    "expensive_device": expensive_device,
                    "effective_prompt_format_cheap": (
                        effective_prompt_format_cheap
                    ),
                    "effective_prompt_format_expensive": (
                        effective_prompt_format_expensive
                    ),
                }
            )

        if verification["correction_id"] is not None:
            correction_id = verification["correction_id"]
            if runtime_profiler is None:
                correction_text = tokenizer.decode(
                    [correction_id],
                    clean_up_tokenization_spaces=False,
                )
            else:
                with runtime_profiler.timed("tokenizer_decode_time_seconds"):
                    correction_text = tokenizer.decode(
                        [correction_id],
                        clean_up_tokenization_spaces=False,
                    )
            token_rows.append(
                {
                    "index": len(token_rows),
                    "token": correction_text,
                    "source": "expensive_corrected",
                    "block_index": block_index,
                    "prompt_format": config.prompt_format,
                    "cheap_device": cheap_device,
                    "expensive_device": expensive_device,
                    "effective_prompt_format_cheap": (
                        effective_prompt_format_cheap
                    ),
                    "effective_prompt_format_expensive": (
                        effective_prompt_format_expensive
                    ),
                }
            )

        if emitted_ids:
            emitted_tensor = torch.tensor([emitted_ids], device=cheap_device)
            input_ids = torch.cat([input_ids, emitted_tensor], dim=-1)
            generated_ids.extend(emitted_ids)

        if runtime_profiler is None:
            generated_text_so_far = tokenizer.decode(
                generated_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
        else:
            with runtime_profiler.timed("tokenizer_decode_time_seconds"):
                generated_text_so_far = tokenizer.decode(
                    generated_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                )
        cheap_entropy_avg = sum(entropies) / len(entropies) if entropies else 0.0
        next_draft_length = adapt_draft_length(
            current_length=current_draft_length,
            acceptance_rate=verification["acceptance_rate"],
            cheap_entropy_avg=cheap_entropy_avg,
            config=config,
        )

        block_rows.append(
            {
                "block_index": block_index,
                "draft_length": len(draft_ids),
                "accepted_tokens": len(accepted_ids),
                "rejected_at": verification["rejected_at"],
                "acceptance_rate": verification["acceptance_rate"],
                "cheap_entropy_avg": cheap_entropy_avg,
                "expensive_calls": 1,
                "generated_text_so_far": generated_text_so_far,
                "next_draft_length": next_draft_length,
                "prompt_format": config.prompt_format,
                "cheap_device": cheap_device,
                "expensive_device": expensive_device,
                "effective_prompt_format_cheap": effective_prompt_format_cheap,
                "effective_prompt_format_expensive": (
                    effective_prompt_format_expensive
                ),
            }
        )

        current_draft_length = next_draft_length
        block_index += 1

        if (
            tokenizer.eos_token_id is not None
            and emitted_ids
            and emitted_ids[-1] == tokenizer.eos_token_id
        ):
            break

    generated_ids_tensor = input_ids[0, prompt_length:]
    if runtime_profiler is None:
        generated_text = tokenizer.decode(
            generated_ids_tensor,
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
                generated_ids_tensor,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            full_text = tokenizer.decode(
                input_ids[0],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
    summary = summarize_speculative_generation(
        prompt=prompt,
        generated_text=generated_text,
        full_text=full_text,
        token_rows=token_rows,
        block_rows=block_rows,
        cheap_generated_tokens=cheap_generated_tokens,
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

    return full_text, block_rows, token_rows, summary


def speculative_generate(
    prompt: str,
    config: SpeculativeGenerationConfig,
    runtime_profiler: RuntimeProfiler | None = None,
) -> tuple[str, list[dict], list[dict], dict]:
    cheap_model, expensive_model, tokenizer, device = load_speculative_models(config)

    return speculative_generate_with_models(
        prompt=prompt,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        config=config,
        runtime_profiler=runtime_profiler,
    )


def print_speculative_report(summary: dict):
    print()
    print("Adaptive Speculative Decoding")
    print("=" * 100)
    print("Texto gerado")
    print("-" * 100)
    print(summary["full_text"])
    print("-" * 100)
    print(f"total_generated_tokens   : {summary['total_generated_tokens']}")
    print(f"cheap_generated_tokens   : {summary['cheap_generated_tokens']}")
    print(f"cheap_accepted_tokens    : {summary['cheap_accepted_tokens']}")
    print(f"expensive_corrected_tokens: {summary['expensive_corrected_tokens']}")
    print(f"expensive_model_calls    : {summary['expensive_model_calls']}")
    print(f"acceptance_rate          : {summary['acceptance_rate']:.2%}")
    print(f"estimated_saved_percent  : {summary['estimated_saved_percent']:.2f}%")
    print("=" * 100)
    print()


def save_speculative_blocks(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))


def save_speculative_tokens(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))


def save_speculative_summary_rows(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))
