from itertools import product
from pathlib import Path

from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    adaptive_generate_with_models,
    load_adaptive_models,
)
from gear_llm.quality_benchmark import (
    PROMPTS,
    build_quality_row,
    generate_greedy_with_model,
)
from gear_llm.report import save_csv


TEACHER_CHECK_INTERVALS = (8, 12, 16, 24)
REPETITION_THRESHOLDS = (0.15, 0.20, 0.25, 0.30)
ENABLE_OPTIONS = (True, False)


def guard_score(row: dict) -> float:
    """
    Combina qualidade, repetição e penalidade por economia negativa.
    """

    negative_savings = max(0.0, -row["estimated_saved_percent"] / 100)

    return (
        row["similarity_to_expensive"]
        + 0.25 * row["jaccard_to_expensive"]
        - 0.50 * row["repeated_3gram_rate"]
        - 0.75 * negative_savings
    )


def config_name(
    teacher_check_interval: int,
    repetition_threshold: float,
    enable_periodic_teacher_check: bool,
    enable_repetition_guard: bool,
) -> str:
    periodic = "periodic_on" if enable_periodic_teacher_check else "periodic_off"
    repetition = "repetition_on" if enable_repetition_guard else "repetition_off"

    return (
        f"tci_{teacher_check_interval}_"
        f"rt_{repetition_threshold:.2f}_"
        f"{periodic}_{repetition}"
    )


def guard_config_grid():
    for (
        teacher_check_interval,
        repetition_threshold,
        enable_periodic_teacher_check,
        enable_repetition_guard,
    ) in product(
        TEACHER_CHECK_INTERVALS,
        REPETITION_THRESHOLDS,
        ENABLE_OPTIONS,
        ENABLE_OPTIONS,
    ):
        yield {
            "config_name": config_name(
                teacher_check_interval=teacher_check_interval,
                repetition_threshold=repetition_threshold,
                enable_periodic_teacher_check=enable_periodic_teacher_check,
                enable_repetition_guard=enable_repetition_guard,
            ),
            "teacher_check_interval": teacher_check_interval,
            "repetition_threshold": repetition_threshold,
            "enable_periodic_teacher_check": enable_periodic_teacher_check,
            "enable_repetition_guard": enable_repetition_guard,
        }


def run_guard_tuning(
    prompts: dict[str, str] | None = None,
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    max_configs: int | None = None,
) -> tuple[list[dict], list[dict]]:
    if prompts is None:
        prompts = PROMPTS

    base_config = AdaptiveGenerationConfig(
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    cheap_model, expensive_model, tokenizer, device = load_adaptive_models(base_config)

    expensive_references: dict[str, str] = {}

    for prompt_name, prompt in prompts.items():
        expensive_text, _ = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        expensive_references[prompt_name] = expensive_text

    detailed_rows = []

    for config_index, config_values in enumerate(guard_config_grid()):
        if max_configs is not None and config_index >= max_configs:
            break

        adaptive_config = AdaptiveGenerationConfig(
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            entropy_threshold=0.35,
            margin_threshold=0.20,
            teacher_check_interval=config_values["teacher_check_interval"],
            enable_periodic_teacher_check=config_values[
                "enable_periodic_teacher_check"
            ],
            enable_repetition_guard=config_values["enable_repetition_guard"],
            repetition_ngram_size=3,
            repetition_threshold=config_values["repetition_threshold"],
        )

        for prompt_name, prompt in prompts.items():
            _, _, adaptive_summary = adaptive_generate_with_models(
                prompt=prompt,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                device=device,
                config=adaptive_config,
            )
            row = build_quality_row(
                prompt_name=prompt_name,
                mode="adaptive_guarded_tuned",
                generated_text=adaptive_summary["generated_text"],
                reference_text=expensive_references[prompt_name],
                total_generated_tokens=adaptive_summary["total_generated_tokens"],
                cheap_accepted_tokens=adaptive_summary["cheap_accepted_tokens"],
                expensive_model_calls=adaptive_summary["expensive_model_calls"],
                saved_percent=adaptive_summary["estimated_saved_percent"],
            )
            row.update(config_values)
            row.update(
                {
                    "config_index": config_index,
                    "entropy_threshold": adaptive_config.entropy_threshold,
                    "margin_threshold": adaptive_config.margin_threshold,
                    "repetition_ngram_size": adaptive_config.repetition_ngram_size,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
            )
            row["score"] = guard_score(row)
            detailed_rows.append(row)

    summary_rows = summarize_guard_tuning(detailed_rows)
    return detailed_rows, summary_rows


def summarize_guard_tuning(rows: list[dict]) -> list[dict]:
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
                "teacher_check_interval": first["teacher_check_interval"],
                "repetition_threshold": first["repetition_threshold"],
                "enable_periodic_teacher_check": first[
                    "enable_periodic_teacher_check"
                ],
                "enable_repetition_guard": first["enable_repetition_guard"],
                "entropy_threshold": first["entropy_threshold"],
                "margin_threshold": first["margin_threshold"],
                "repetition_ngram_size": first["repetition_ngram_size"],
                "max_new_tokens": first["max_new_tokens"],
                "temperature": first["temperature"],
                "prompt_count": total,
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
                "avg_score": sum(row["score"] for row in group_rows) / total,
                "avg_expensive_model_calls": sum(
                    row["expensive_model_calls"] for row in group_rows
                )
                / total,
            }
        )

    return sorted(summary_rows, key=lambda row: row["avg_score"], reverse=True)


def print_guard_tuning_report(summary_rows: list[dict], limit: int = 10):
    print()
    print("Guard Tuning - Top Configurações")
    print("=" * 140)
    header = (
        f"{'rank':>4} | {'config':<48} | {'saved %':>8} | "
        f"{'sim':>7} | {'jaccard':>7} | {'rep3':>7} | "
        f"{'score':>7} | {'calls':>7}"
    )
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(summary_rows[:limit], start=1):
        print(
            f"{rank:>4} | "
            f"{row['config_name']:<48} | "
            f"{row['avg_saved_percent']:>7.2f}% | "
            f"{row['avg_similarity']:>7.4f} | "
            f"{row['avg_jaccard']:>7.4f} | "
            f"{row['avg_repeated_3gram_rate']:>7.4f} | "
            f"{row['avg_score']:>7.4f} | "
            f"{row['avg_expensive_model_calls']:>7.2f}"
        )

    print("=" * 140)
    print()


def save_guard_tuning(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))
