from itertools import product
from pathlib import Path

from gear_llm.quality_benchmark import (
    PROMPTS,
    generate_greedy_with_model,
    jaccard_similarity,
    repeated_ngram_rate,
    sequence_similarity,
)
from gear_llm.report import save_csv
from gear_llm.speculative_generator import (
    SpeculativeGenerationConfig,
    load_speculative_models,
    speculative_generate_with_models,
)


INITIAL_DRAFT_LENGTHS = (4, 6, 8)
VERIFY_TOP_K_VALUES = (3, 5, 10)
MIN_DRAFT_LENGTHS = (1, 2)
MAX_DRAFT_LENGTHS = (8, 12, 16)


def parse_config_filter(config_filter: str | None) -> set[str] | None:
    if not config_filter:
        return None

    names = {
        name.strip()
        for name in config_filter.split(",")
        if name.strip()
    }
    return names or None


def config_name(
    initial_draft_length: int,
    verify_top_k: int,
    min_draft_length: int,
    max_draft_length: int,
) -> str:
    return (
        f"draft_{initial_draft_length}_"
        f"topk_{verify_top_k}_"
        f"min_{min_draft_length}_"
        f"max_{max_draft_length}"
    )


def speculative_config_grid():
    for (
        initial_draft_length,
        verify_top_k,
        min_draft_length,
        max_draft_length,
    ) in product(
        INITIAL_DRAFT_LENGTHS,
        VERIFY_TOP_K_VALUES,
        MIN_DRAFT_LENGTHS,
        MAX_DRAFT_LENGTHS,
    ):
        yield {
            "config_name": config_name(
                initial_draft_length=initial_draft_length,
                verify_top_k=verify_top_k,
                min_draft_length=min_draft_length,
                max_draft_length=max_draft_length,
            ),
            "initial_draft_length": initial_draft_length,
            "verify_top_k": verify_top_k,
            "min_draft_length": min_draft_length,
            "max_draft_length": max_draft_length,
        }


def speculative_tuning_score(row: dict) -> float:
    positive_savings = max(0.0, row["estimated_saved_percent"] / 100)
    negative_savings = max(0.0, -row["estimated_saved_percent"] / 100)

    return (
        row["similarity_to_expensive"]
        + 0.25 * row["jaccard_to_expensive"]
        - 0.50 * row["repeated_3gram_rate"]
        + 0.20 * positive_savings
        - 0.75 * negative_savings
    )


def build_tuning_row(
    prompt_name: str,
    generated_text: str,
    reference_text: str,
    summary: dict,
    config_values: dict,
    config_index: int,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    row = {
        "prompt_name": prompt_name,
        "config_name": config_values["config_name"],
        "config_index": config_index,
        "initial_draft_length": config_values["initial_draft_length"],
        "verify_top_k": config_values["verify_top_k"],
        "min_draft_length": config_values["min_draft_length"],
        "max_draft_length": config_values["max_draft_length"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "generated_text": generated_text,
        "total_generated_tokens": summary["total_generated_tokens"],
        "cheap_generated_tokens": summary["cheap_generated_tokens"],
        "cheap_accepted_tokens": summary["cheap_accepted_tokens"],
        "expensive_corrected_tokens": summary["expensive_corrected_tokens"],
        "expensive_model_calls": summary["expensive_model_calls"],
        "acceptance_rate": summary["acceptance_rate"],
        "estimated_saved_percent": summary["estimated_saved_percent"],
        "similarity_to_expensive": sequence_similarity(
            generated_text,
            reference_text,
        ),
        "jaccard_to_expensive": jaccard_similarity(
            generated_text,
            reference_text,
        ),
        "repeated_3gram_rate": repeated_ngram_rate(generated_text, 3),
        "repeated_4gram_rate": repeated_ngram_rate(generated_text, 4),
    }
    row["score"] = speculative_tuning_score(row)
    return row


def summarize_speculative_tuning(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}

    for row in rows:
        grouped.setdefault(row["config_name"], []).append(row)

    summary_rows = []

    for config_name_value, group_rows in grouped.items():
        first = group_rows[0]
        total = len(group_rows)

        summary_rows.append(
            {
                "config_name": config_name_value,
                "config_index": first["config_index"],
                "initial_draft_length": first["initial_draft_length"],
                "verify_top_k": first["verify_top_k"],
                "min_draft_length": first["min_draft_length"],
                "max_draft_length": first["max_draft_length"],
                "max_new_tokens": first["max_new_tokens"],
                "temperature": first["temperature"],
                "prompt_count": total,
                "avg_total_generated_tokens": sum(
                    row["total_generated_tokens"] for row in group_rows
                )
                / total,
                "avg_cheap_accepted_tokens": sum(
                    row["cheap_accepted_tokens"] for row in group_rows
                )
                / total,
                "avg_expensive_corrected_tokens": sum(
                    row["expensive_corrected_tokens"] for row in group_rows
                )
                / total,
                "avg_expensive_model_calls": sum(
                    row["expensive_model_calls"] for row in group_rows
                )
                / total,
                "avg_acceptance_rate": sum(
                    row["acceptance_rate"] for row in group_rows
                )
                / total,
                "avg_saved_percent": sum(
                    row["estimated_saved_percent"] for row in group_rows
                )
                / total,
                "avg_similarity": sum(
                    row["similarity_to_expensive"] for row in group_rows
                )
                / total,
                "avg_jaccard": sum(
                    row["jaccard_to_expensive"] for row in group_rows
                )
                / total,
                "avg_repeated_3gram_rate": sum(
                    row["repeated_3gram_rate"] for row in group_rows
                )
                / total,
                "avg_repeated_4gram_rate": sum(
                    row["repeated_4gram_rate"] for row in group_rows
                )
                / total,
                "avg_score": sum(row["score"] for row in group_rows) / total,
            }
        )

    return sorted(
        summary_rows,
        key=lambda row: (
            row["avg_score"],
            row["avg_saved_percent"],
            row["avg_acceptance_rate"],
        ),
        reverse=True,
    )


def run_speculative_tuning(
    prompts: dict[str, str] | None = None,
    cheap_model_name: str = SpeculativeGenerationConfig.cheap_model_name,
    expensive_model_name: str = SpeculativeGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    max_configs: int | None = None,
    config_filter: str | None = None,
) -> tuple[list[dict], list[dict]]:
    if prompts is None:
        prompts = PROMPTS

    allowed_config_names = parse_config_filter(config_filter)
    all_config_names = {
        config_values["config_name"]
        for config_values in speculative_config_grid()
    }

    if allowed_config_names:
        unknown_names = sorted(allowed_config_names - all_config_names)

        if unknown_names:
            raise ValueError(
                "Configuração speculative desconhecida: "
                + ", ".join(unknown_names)
            )

    base_config = SpeculativeGenerationConfig(
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    cheap_model, expensive_model, tokenizer, device = load_speculative_models(
        base_config
    )
    expensive_references = {}

    for prompt_name, prompt in prompts.items():
        reference_text, _ = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        expensive_references[prompt_name] = reference_text

    rows = []
    matched_configs = 0

    for config_index, config_values in enumerate(speculative_config_grid()):
        if (
            allowed_config_names is not None
            and config_values["config_name"] not in allowed_config_names
        ):
            continue

        if max_configs is not None and matched_configs >= max_configs:
            break

        matched_configs += 1

        config = SpeculativeGenerationConfig(
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            max_new_tokens=max_new_tokens,
            draft_length=config_values["initial_draft_length"],
            verify_top_k=config_values["verify_top_k"],
            min_draft_length=config_values["min_draft_length"],
            max_draft_length=config_values["max_draft_length"],
            temperature=temperature,
        )

        for prompt_name, prompt in prompts.items():
            _, _, _, summary = speculative_generate_with_models(
                prompt=prompt,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                device=device,
                config=config,
            )
            rows.append(
                build_tuning_row(
                    prompt_name=prompt_name,
                    generated_text=summary["generated_text"],
                    reference_text=expensive_references[prompt_name],
                    summary=summary,
                    config_values=config_values,
                    config_index=config_index,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            )

    summary_rows = summarize_speculative_tuning(rows)

    if allowed_config_names and matched_configs == 0:
        raise ValueError(
            "Nenhuma configuração passou pelo filtro informado."
        )

    return rows, summary_rows


def print_speculative_tuning_report(summary_rows: list[dict], limit: int = 10):
    print()
    print("Speculative Tuning - Top Configurações")
    print("=" * 140)
    header = (
        f"{'rank':>4} | {'config':<28} | {'saved %':>8} | "
        f"{'sim':>7} | {'jaccard':>7} | {'rep3':>7} | "
        f"{'accept':>8} | {'calls':>7} | {'score':>7}"
    )
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(summary_rows[:limit], start=1):
        print(
            f"{rank:>4} | "
            f"{row['config_name']:<28} | "
            f"{row['avg_saved_percent']:>7.2f}% | "
            f"{row['avg_similarity']:>7.4f} | "
            f"{row['avg_jaccard']:>7.4f} | "
            f"{row['avg_repeated_3gram_rate']:>7.4f} | "
            f"{row['avg_acceptance_rate']:>7.2%} | "
            f"{row['avg_expensive_model_calls']:>7.2f} | "
            f"{row['avg_score']:>7.4f}"
        )

    print("=" * 140)
    print()


def save_speculative_tuning(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))
