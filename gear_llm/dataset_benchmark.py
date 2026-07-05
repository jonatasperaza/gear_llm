from collections import defaultdict
from pathlib import Path

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.dataset_loader import filter_prompts, load_prompts
from gear_llm.hybrid_router import (
    choose_mode,
    classify_prompt,
    generate_with_mode,
    load_hybrid_models,
)
from gear_llm.quality_benchmark import (
    generate_greedy_with_model,
    jaccard_similarity,
    repeated_ngram_rate,
    sequence_similarity,
)
from gear_llm.report import save_csv


DATASET_MODES = (
    "adaptive_calibrated",
    "adaptive_guarded_v3",
    "speculative_adaptive",
)


def parse_categories(value: str | None) -> list[str] | None:
    if not value:
        return None

    categories = [item.strip() for item in value.split(",") if item.strip()]
    return categories or None


def build_dataset_row(
    prompt_item: dict,
    mode: str,
    selected_mode: str,
    prompt_type: str,
    summary: dict,
    reference_text: str,
) -> dict:
    generated_text = summary["generated_text"]

    return {
        "prompt_id": prompt_item["id"],
        "category": prompt_item["category"],
        "expected_category": prompt_item["category"],
        "prompt_type": prompt_type,
        "mode": mode,
        "selected_mode": selected_mode,
        "generated_text": generated_text,
        "estimated_saved_percent": summary["estimated_saved_percent"],
        "expensive_model_calls": summary["expensive_model_calls"],
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


def summarize_dataset_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in rows:
        groups[(row["category"], row["mode"])].append(row)

    summary_rows = []
    for (category, mode), group_rows in sorted(groups.items()):
        count = len(group_rows)
        summary_rows.append(
            {
                "category": category,
                "mode": mode,
                "prompt_count": count,
                "avg_saved_percent": _average(
                    row["estimated_saved_percent"] for row in group_rows
                ),
                "avg_similarity": _average(
                    row["similarity_to_expensive"] for row in group_rows
                ),
                "avg_jaccard": _average(
                    row["jaccard_to_expensive"] for row in group_rows
                ),
                "avg_repeated_3gram_rate": _average(
                    row["repeated_3gram_rate"] for row in group_rows
                ),
                "avg_repeated_4gram_rate": _average(
                    row["repeated_4gram_rate"] for row in group_rows
                ),
                "avg_expensive_model_calls": _average(
                    row["expensive_model_calls"] for row in group_rows
                ),
            }
        )

    return summary_rows


def build_hybrid_mode_matrix(rows: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        if row["mode"] != "hybrid":
            continue
        counts[(row["expected_category"], row["selected_mode"])] += 1

    return [
        {
            "expected_category": expected_category,
            "selected_mode": selected_mode,
            "count": count,
        }
        for (expected_category, selected_mode), count in sorted(counts.items())
    ]


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_dataset_benchmark(
    dataset_path: str = "data/prompts.jsonl",
    categories: list[str] | None = None,
    limit: int | None = None,
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    models=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    prompts = filter_prompts(
        prompts=load_prompts(dataset_path),
        categories=categories,
        limit=limit,
    )

    if models is None:
        cheap_model, expensive_model, tokenizer, device = load_hybrid_models(
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        cheap_model, expensive_model, tokenizer, device = models

    rows = []

    for prompt_item in prompts:
        prompt = prompt_item["prompt"]
        prompt_type = classify_prompt(prompt)
        selected_mode = choose_mode(prompt_type, prompt)
        reference_text, _ = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        mode_summaries = {}
        for mode in DATASET_MODES:
            summary = generate_with_mode(
                prompt=prompt,
                mode=mode,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                device=device,
                cheap_model_name=cheap_model_name,
                expensive_model_name=expensive_model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            mode_summaries[mode] = summary
            rows.append(
                build_dataset_row(
                    prompt_item=prompt_item,
                    mode=mode,
                    selected_mode="",
                    prompt_type=prompt_type,
                    summary=summary,
                    reference_text=reference_text,
                )
            )

        rows.append(
            build_dataset_row(
                prompt_item=prompt_item,
                mode="hybrid",
                selected_mode=selected_mode,
                prompt_type=prompt_type,
                summary=mode_summaries[selected_mode],
                reference_text=reference_text,
            )
        )

    summary_rows = summarize_dataset_rows(rows)
    matrix_rows = build_hybrid_mode_matrix(rows)

    return rows, summary_rows, matrix_rows


def print_dataset_benchmark_report(
    summary_rows: list[dict],
    matrix_rows: list[dict],
):
    print()
    print("Dataset Benchmark Summary")
    print("=" * 124)
    header = (
        f"{'category':<14} | {'mode':<22} | {'n':>3} | "
        f"{'saved %':>8} | {'sim':>7} | {'jacc':>7} | "
        f"{'rep3':>7} | {'rep4':>7} | {'calls':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['category']:<14} | "
            f"{row['mode']:<22} | "
            f"{row['prompt_count']:>3} | "
            f"{row['avg_saved_percent']:>7.2f}% | "
            f"{row['avg_similarity']:>7.4f} | "
            f"{row['avg_jaccard']:>7.4f} | "
            f"{row['avg_repeated_3gram_rate']:>7.4f} | "
            f"{row['avg_repeated_4gram_rate']:>7.4f} | "
            f"{row['avg_expensive_model_calls']:>7.2f}"
        )

    print("=" * 124)
    print()
    print("Hybrid Mode Matrix")
    print("=" * 72)
    matrix_header = (
        f"{'expected_category':<20} | {'selected_mode':<24} | {'count':>5}"
    )
    print(matrix_header)
    print("-" * len(matrix_header))
    for row in matrix_rows:
        print(
            f"{row['expected_category']:<20} | "
            f"{row['selected_mode']:<24} | "
            f"{row['count']:>5}"
        )
    print("=" * 72)
    print()


def save_dataset_benchmark_outputs(
    rows: list[dict],
    summary_rows: list[dict],
    matrix_rows: list[dict],
    output_dir: str | Path,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detailed_csv = output_path / "dataset_benchmark.csv"
    summary_csv = output_path / "dataset_benchmark_summary.csv"
    matrix_csv = output_path / "hybrid_mode_matrix.csv"

    save_csv(rows, str(detailed_csv))
    save_csv(summary_rows, str(summary_csv))
    save_csv(matrix_rows, str(matrix_csv))

    return detailed_csv, summary_csv, matrix_csv
