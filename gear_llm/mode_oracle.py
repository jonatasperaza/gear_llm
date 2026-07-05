import csv
from collections import defaultdict
from pathlib import Path

from gear_llm.report import save_csv


BASE_MODES = (
    "adaptive_calibrated",
    "adaptive_guarded_v3",
    "speculative_adaptive",
)


def row_score(row: dict) -> float:
    similarity = _float(row["similarity_to_expensive"])
    jaccard = _float(row["jaccard_to_expensive"])
    repeated_3gram = _float(row["repeated_3gram_rate"])
    saved_percent = _float(row["estimated_saved_percent"])

    return (
        similarity
        + 0.25 * jaccard
        - 0.50 * repeated_3gram
        + 0.20 * max(0.0, saved_percent / 100)
        - 0.75 * max(0.0, -saved_percent / 100)
    )


def load_dataset_benchmark(path: str | Path) -> list[dict]:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            "Arquivo results/dataset_benchmark.csv nao encontrado. "
            "Rode primeiro: python run_dataset_benchmark.py --max-new-tokens 32"
        )

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_mode_oracle(
    dataset_csv: str | Path = "results/dataset_benchmark.csv",
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    rows = load_dataset_benchmark(dataset_csv)
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    hybrid_rows: dict[str, dict] = {}

    for row in rows:
        prompt_id = row["prompt_id"]
        mode = row["mode"]

        if mode in BASE_MODES:
            by_prompt[prompt_id][mode] = row
        elif mode == "hybrid":
            hybrid_rows[prompt_id] = row

    oracle_rows = []
    comparison_rows = []

    for prompt_id in sorted(by_prompt):
        mode_rows = by_prompt[prompt_id]
        missing = [mode for mode in BASE_MODES if mode not in mode_rows]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Prompt {prompt_id} sem modos: {missing_text}")

        scores = {mode: row_score(mode_rows[mode]) for mode in BASE_MODES}
        ranked_modes = sorted(
            BASE_MODES,
            key=lambda mode: scores[mode],
            reverse=True,
        )
        best_mode = ranked_modes[0]
        second_best_mode = ranked_modes[1]
        best_score = scores[best_mode]
        second_best_score = scores[second_best_mode]
        score_margin = best_score - second_best_score
        oracle_confidence = classify_oracle_confidence(
            best_score=best_score,
            score_margin=score_margin,
        )
        best_mode_confident = (
            best_mode if oracle_confidence == "high" else "inconclusive"
        )
        best_row = mode_rows[best_mode]

        oracle_rows.append(
            {
                "prompt_id": prompt_id,
                "category": best_row["category"],
                "best_mode": best_mode,
                "second_best_mode": second_best_mode,
                "best_score": best_score,
                "second_best_score": second_best_score,
                "score_margin": score_margin,
                "oracle_confidence": oracle_confidence,
                "best_mode_confident": best_mode_confident,
                "best_saved_percent": _float(
                    best_row["estimated_saved_percent"]
                ),
                "best_similarity": _float(
                    best_row["similarity_to_expensive"]
                ),
                "best_jaccard": _float(best_row["jaccard_to_expensive"]),
                "best_repeated_3gram_rate": _float(
                    best_row["repeated_3gram_rate"]
                ),
                "calibrated_score": scores["adaptive_calibrated"],
                "guarded_score": scores["adaptive_guarded_v3"],
                "speculative_score": scores["speculative_adaptive"],
            }
        )

        hybrid_row = hybrid_rows.get(prompt_id)
        hybrid_selected_mode = (
            hybrid_row["selected_mode"] if hybrid_row is not None else ""
        )
        comparison_rows.append(
            {
                "prompt_id": prompt_id,
                "category": best_row["category"],
                "hybrid_selected_mode": hybrid_selected_mode,
                "oracle_best_mode": best_mode,
                "oracle_best_mode_confident": best_mode_confident,
                "hybrid_matches_oracle_raw": (
                    "true" if hybrid_selected_mode == best_mode else "false"
                ),
                "hybrid_matches_oracle_confident": (
                    _confident_match_text(
                        hybrid_selected_mode=hybrid_selected_mode,
                        best_mode_confident=best_mode_confident,
                        oracle_confidence=oracle_confidence,
                    )
                ),
                "oracle_confidence": oracle_confidence,
                "score_margin": score_margin,
            }
        )

    summary_rows = summarize_oracle(oracle_rows)
    confidence_summary_rows = summarize_oracle_confidence(oracle_rows)
    metrics = hybrid_accuracy_metrics(comparison_rows)

    return (
        oracle_rows,
        summary_rows,
        confidence_summary_rows,
        comparison_rows,
        metrics,
    )


def classify_oracle_confidence(best_score: float, score_margin: float) -> str:
    if best_score < 0.10:
        return "low"
    if score_margin < 0.02:
        return "tie"
    return "high"


def summarize_oracle(oracle_rows: list[dict]) -> list[dict]:
    category_totals: dict[str, int] = defaultdict(int)
    mode_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in oracle_rows:
        category = row["category"]
        best_mode = row["best_mode"]
        category_totals[category] += 1
        mode_counts[(category, best_mode)] += 1

    summary_rows = []
    for (category, best_mode), count in sorted(mode_counts.items()):
        total = category_totals[category]
        summary_rows.append(
            {
                "category": category,
                "best_mode": best_mode,
                "count": count,
                "percentage": 100 * count / total if total else 0.0,
            }
        )

    return summary_rows


def summarize_oracle_confidence(oracle_rows: list[dict]) -> list[dict]:
    category_totals: dict[str, int] = defaultdict(int)
    confidence_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in oracle_rows:
        category = row["category"]
        confidence = row["oracle_confidence"]
        category_totals[category] += 1
        confidence_counts[(category, confidence)] += 1

    summary_rows = []
    for (category, confidence), count in sorted(confidence_counts.items()):
        total = category_totals[category]
        summary_rows.append(
            {
                "category": category,
                "oracle_confidence": confidence,
                "count": count,
                "percentage": 100 * count / total if total else 0.0,
            }
        )

    return summary_rows


def hybrid_accuracy_metrics(comparison_rows: list[dict]) -> dict:
    total = len(comparison_rows)
    if not total:
        return {
            "raw_accuracy": 0.0,
            "confident_accuracy": 0.0,
            "total_cases": 0,
            "high_cases": 0,
            "low_cases": 0,
            "tie_cases": 0,
            "inconclusive_cases": 0,
        }

    matches = sum(
        1
        for row in comparison_rows
        if row["hybrid_matches_oracle_raw"] == "true"
    )
    high_rows = [
        row for row in comparison_rows if row["oracle_confidence"] == "high"
    ]
    high_matches = sum(
        1
        for row in high_rows
        if row["hybrid_matches_oracle_confident"] == "true"
    )
    low_count = sum(
        1 for row in comparison_rows if row["oracle_confidence"] == "low"
    )
    tie_count = sum(
        1 for row in comparison_rows if row["oracle_confidence"] == "tie"
    )

    return {
        "raw_accuracy": 100 * matches / total,
        "confident_accuracy": (
            100 * high_matches / len(high_rows) if high_rows else 0.0
        ),
        "total_cases": total,
        "high_cases": len(high_rows),
        "low_cases": low_count,
        "tie_cases": tie_count,
        "inconclusive_cases": low_count + tie_count,
    }


def print_mode_oracle_report(
    summary_rows: list[dict],
    confidence_summary_rows: list[dict],
    metrics: dict,
):
    print()
    print("Mode Oracle Summary")
    print("=" * 86)
    header = (
        f"{'category':<16} | {'best_mode':<22} | {'count':>5} | "
        f"{'percentage':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['category']:<16} | "
            f"{row['best_mode']:<22} | "
            f"{row['count']:>5} | "
            f"{row['percentage']:>9.2f}%"
        )

    print("=" * 86)
    print()
    print("Oracle Confidence Summary")
    print("=" * 86)
    confidence_header = (
        f"{'category':<16} | {'confidence':<12} | {'count':>5} | "
        f"{'percentage':>10}"
    )
    print(confidence_header)
    print("-" * len(confidence_header))

    for row in confidence_summary_rows:
        print(
            f"{row['category']:<16} | "
            f"{row['oracle_confidence']:<12} | "
            f"{row['count']:>5} | "
            f"{row['percentage']:>9.2f}%"
        )

    print("=" * 86)
    print(f"Hybrid raw accuracy vs oracle      : {metrics['raw_accuracy']:.2f}%")
    print(
        "Hybrid confident accuracy vs oracle: "
        f"{metrics['confident_accuracy']:.2f}% "
        f"({metrics['high_cases']} high-confidence cases)"
    )
    print(
        "Confidence counts                  : "
        f"high={metrics['high_cases']}, "
        f"low={metrics['low_cases']}, "
        f"tie={metrics['tie_cases']}, "
        f"inconclusive={metrics['inconclusive_cases']}"
    )
    print()


def save_mode_oracle_outputs(
    oracle_rows: list[dict],
    summary_rows: list[dict],
    confidence_summary_rows: list[dict],
    comparison_rows: list[dict],
    output_dir: str | Path = "results",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    oracle_csv = output_path / "mode_oracle.csv"
    summary_csv = output_path / "mode_oracle_summary.csv"
    confidence_summary_csv = output_path / "mode_oracle_confidence_summary.csv"
    comparison_csv = output_path / "hybrid_vs_oracle.csv"

    save_csv(oracle_rows, str(oracle_csv))
    save_csv(summary_rows, str(summary_csv))
    save_csv(confidence_summary_rows, str(confidence_summary_csv))
    save_csv(comparison_rows, str(comparison_csv))

    return oracle_csv, summary_csv, confidence_summary_csv, comparison_csv


def _confident_match_text(
    hybrid_selected_mode: str,
    best_mode_confident: str,
    oracle_confidence: str,
) -> str:
    if oracle_confidence != "high":
        return "inconclusive"
    return "true" if hybrid_selected_mode == best_mode_confident else "false"


def _float(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
