import csv
from collections import defaultdict
from pathlib import Path

from gear_llm.report import save_csv


def load_csv(path: str | Path) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_quality_latency_report(
    latency_winners_csv: str | Path = "results/latency_winners.csv",
    latency_summary_csv: str | Path = "results/latency_benchmark_summary.csv",
    quality_csv: str | Path = "results/quality_benchmark.csv",
) -> tuple[list[dict], list[dict]]:
    latency_winners = load_csv(latency_winners_csv)
    latency_summary = load_csv(latency_summary_csv)
    quality_rows = load_csv(quality_csv)

    summary_lookup = {
        (row["prompt_name"], row["mode"]): row for row in latency_summary
    }
    quality_lookup = {
        (row["prompt_name"], row["mode"]): row for row in quality_rows
    }

    report_rows = []
    for winner in latency_winners:
        prompt_name = winner["prompt_name"]
        latency_winner = winner["fastest_mode_excluding_cheap"]
        latency_summary_row = summary_lookup.get((prompt_name, latency_winner))
        quality_row = quality_lookup.get((prompt_name, latency_winner))
        cheap_row = quality_lookup.get((prompt_name, "cheap_only"))
        expensive_row = quality_lookup.get((prompt_name, "expensive_only"))

        if latency_summary_row is None:
            raise ValueError(
                "Modo vencedor de latencia nao encontrado no summary: "
                f"prompt={prompt_name}, mode={latency_winner}"
            )
        if quality_row is None:
            raise ValueError(
                "Modo vencedor de latencia nao encontrado no quality benchmark: "
                f"prompt={prompt_name}, mode={latency_winner}"
            )
        if cheap_row is None or expensive_row is None:
            raise ValueError(
                "Quality benchmark precisa conter cheap_only e expensive_only "
                f"para prompt={prompt_name}."
            )

        real_speedup = _float(
            winner,
            "best_real_speedup_excluding_cheap_percent",
        )
        similarity = _float(quality_row, "similarity_to_expensive")
        jaccard = _float(quality_row, "jaccard_to_expensive")
        repeated_3gram_rate = _float(quality_row, "repeated_3gram_rate")
        quality_latency_score = (
            similarity
            + 0.25 * jaccard
            - 0.50 * repeated_3gram_rate
            + 0.25 * max(0.0, real_speedup / 100)
        )

        report_rows.append(
            {
                "prompt_name": prompt_name,
                "cheap_model_name": _first_value(
                    winner,
                    quality_row,
                    "cheap_model_name",
                ),
                "expensive_model_name": _first_value(
                    winner,
                    quality_row,
                    "expensive_model_name",
                ),
                "device": _first_value(winner, quality_row, "device"),
                "torch_dtype": _first_value(
                    winner,
                    quality_row,
                    "torch_dtype",
                ),
                "prompt_format": _first_value(
                    winner,
                    quality_row,
                    "prompt_format",
                ),
                "latency_winner_excluding_cheap": latency_winner,
                "fastest_seconds_excluding_cheap": _float(
                    winner,
                    "fastest_seconds_excluding_cheap",
                ),
                "real_speedup_vs_expensive_percent": real_speedup,
                "quality_mode": quality_row["mode"],
                "estimated_saved_percent": _float(
                    quality_row,
                    "estimated_saved_percent",
                ),
                "similarity_to_expensive": similarity,
                "jaccard_to_expensive": jaccard,
                "repeated_3gram_rate": repeated_3gram_rate,
                "repeated_4gram_rate": _float(
                    quality_row,
                    "repeated_4gram_rate",
                ),
                "expensive_model_calls": _float(
                    quality_row,
                    "expensive_model_calls",
                ),
                "selected_mode": quality_row.get("selected_mode", ""),
                "cheap_only_similarity": _float(
                    cheap_row,
                    "similarity_to_expensive",
                ),
                "cheap_only_jaccard": _float(
                    cheap_row,
                    "jaccard_to_expensive",
                ),
                "expensive_only_generated_text": expensive_row.get(
                    "generated_text",
                    "",
                ),
                "latency_winner_generated_text": quality_row.get(
                    "generated_text",
                    "",
                ),
                "quality_latency_score": quality_latency_score,
            }
        )

    return report_rows, build_quality_latency_summary(report_rows)


def build_quality_latency_summary(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped["ALL"] = rows
    for row in rows:
        grouped[f"mode:{row['latency_winner_excluding_cheap']}"].append(row)

    summary_rows = []
    for group_name, group_rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "summary_group": group_name,
                "prompt_count": len(group_rows),
                "avg_quality_latency_score": _average(
                    row["quality_latency_score"] for row in group_rows
                ),
                "avg_real_speedup_vs_expensive_percent": _average(
                    row["real_speedup_vs_expensive_percent"]
                    for row in group_rows
                ),
                "avg_estimated_saved_percent": _average(
                    row["estimated_saved_percent"] for row in group_rows
                ),
                "avg_similarity_to_expensive": _average(
                    row["similarity_to_expensive"] for row in group_rows
                ),
                "avg_jaccard_to_expensive": _average(
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
                "prompts": ",".join(row["prompt_name"] for row in group_rows),
            }
        )

    return summary_rows


def save_quality_latency_outputs(
    report_rows: list[dict],
    summary_rows: list[dict],
    output_dir: str | Path = "results",
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_csv = output_path / "quality_latency_report.csv"
    summary_csv = output_path / "quality_latency_summary.csv"

    save_csv(report_rows, str(report_csv))
    save_csv(summary_rows, str(summary_csv))

    return report_csv, summary_csv


def print_quality_latency_report(rows: list[dict]):
    print()
    print("Quality-Latency Report")
    print("=" * 132)
    header = (
        f"{'prompt':<16} | {'winner':<22} | {'speedup':>8} | "
        f"{'saved':>8} | {'sim':>7} | {'jacc':>7} | "
        f"{'rep3':>7} | {'calls':>7} | {'score':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['prompt_name']:<16} | "
            f"{row['latency_winner_excluding_cheap']:<22} | "
            f"{row['real_speedup_vs_expensive_percent']:>7.2f}% | "
            f"{row['estimated_saved_percent']:>7.2f}% | "
            f"{row['similarity_to_expensive']:>7.4f} | "
            f"{row['jaccard_to_expensive']:>7.4f} | "
            f"{row['repeated_3gram_rate']:>7.4f} | "
            f"{row['expensive_model_calls']:>7.2f} | "
            f"{row['quality_latency_score']:>7.4f}"
        )

    print("=" * 132)
    print()


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _first_value(*rows_and_key):
    *rows, key = rows_and_key
    for row in rows:
        value = row.get(key, "")
        if value != "":
            return value
    return ""
