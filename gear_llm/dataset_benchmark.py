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
from gear_llm.model_loader import get_model_runtime_metadata
from gear_llm.model_loader import (
    get_expensive_tokenizer,
    prompt_format_metadata,
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
    model_metadata: dict,
) -> dict:
    generated_text = summary["generated_text"]

    return {
        "prompt_id": prompt_item["id"],
        "category": prompt_item["category"],
        "expected_category": prompt_item["category"],
        "prompt_type": prompt_type,
        "mode": mode,
        "selected_mode": selected_mode,
        "cheap_model_name": model_metadata["cheap_model_name"],
        "expensive_model_name": model_metadata["expensive_model_name"],
        "device": model_metadata["device"],
        "torch_dtype": model_metadata["torch_dtype"],
        "prompt_format": model_metadata["prompt_format"],
        "effective_prompt_format_cheap": model_metadata[
            "effective_prompt_format_cheap"
        ],
        "effective_prompt_format_expensive": model_metadata[
            "effective_prompt_format_expensive"
        ],
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
        first_row = group_rows[0]
        summary_rows.append(
            {
                "category": category,
                "mode": mode,
                "cheap_model_name": first_row.get("cheap_model_name", ""),
                "expensive_model_name": first_row.get(
                    "expensive_model_name",
                    "",
                ),
                "device": first_row.get("device", ""),
                "torch_dtype": first_row.get("torch_dtype", ""),
                "prompt_format": first_row.get("prompt_format", ""),
                "effective_prompt_format_cheap": first_row.get(
                    "effective_prompt_format_cheap",
                    "",
                ),
                "effective_prompt_format_expensive": first_row.get(
                    "effective_prompt_format_expensive",
                    "",
                ),
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


def build_hybrid_mode_matrix(
    rows: list[dict],
    model_metadata: dict,
) -> list[dict]:
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        if row["mode"] != "hybrid":
            continue
        counts[(row["expected_category"], row["selected_mode"])] += 1

    return [
        {
            "expected_category": expected_category,
            "selected_mode": selected_mode,
            "cheap_model_name": model_metadata["cheap_model_name"],
            "expensive_model_name": model_metadata["expensive_model_name"],
            "device": model_metadata["device"],
            "torch_dtype": model_metadata["torch_dtype"],
            "prompt_format": model_metadata["prompt_format"],
            "effective_prompt_format_cheap": model_metadata[
                "effective_prompt_format_cheap"
            ],
            "effective_prompt_format_expensive": model_metadata[
                "effective_prompt_format_expensive"
            ],
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
    device: str = "auto",
    torch_dtype: str = "auto",
    prompt_format: str = "auto",
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
            device=device,
            torch_dtype=torch_dtype,
            prompt_format=prompt_format,
        )
    else:
        cheap_model, expensive_model, tokenizer, device = models

    runtime_metadata = get_model_runtime_metadata(
        cheap_model,
        fallback_device=device,
    )
    expensive_runtime_metadata = get_model_runtime_metadata(
        expensive_model,
        fallback_device=device,
    )
    if runtime_metadata != expensive_runtime_metadata:
        raise ValueError(
            "cheap_model e expensive_model precisam usar o mesmo device/dtype. "
            f"cheap={runtime_metadata}, expensive={expensive_runtime_metadata}"
        )

    model_metadata = {
        "cheap_model_name": cheap_model_name,
        "expensive_model_name": expensive_model_name,
        "device": runtime_metadata["device"],
        "torch_dtype": runtime_metadata["torch_dtype"],
        **prompt_format_metadata(tokenizer, prompt_format),
    }
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)
    rows = []

    for prompt_item in prompts:
        prompt = prompt_item["prompt"]
        prompt_type = classify_prompt(prompt)
        selected_mode = choose_mode(prompt_type, prompt)
        reference_text, _ = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=expensive_tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
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
                prompt_format=prompt_format,
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
                    model_metadata=model_metadata,
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
                model_metadata=model_metadata,
            )
        )

    summary_rows = summarize_dataset_rows(rows)
    matrix_rows = build_hybrid_mode_matrix(rows, model_metadata)

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
