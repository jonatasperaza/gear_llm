import argparse
import csv
import json
from pathlib import Path


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_tasks(path: str | Path) -> dict[str, dict]:
    dataset_path = Path(path)
    tasks = {}
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            task_id = item.get("id")
            if not task_id:
                raise ValueError(
                    f"Missing id in {dataset_path} line {line_number}."
                )
            tasks[task_id] = item
    return tasks


def load_task_results(path: str | Path) -> dict[str, dict[str, dict]]:
    results_path = Path(path)
    grouped: dict[str, dict[str, dict]] = {}
    with results_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"task_id", "mode", "score", "passed"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"{results_path} missing columns: {fields}")

        for row in reader:
            mode = row.get("mode", "")
            if mode not in {"cheap_only", "expensive_only"}:
                continue
            grouped.setdefault(row["task_id"], {})[mode] = row
    return grouped


def build_router_rows(tasks: dict[str, dict], results: dict[str, dict[str, dict]]):
    rows = []
    skipped = []

    for task_id, task in tasks.items():
        task_results = results.get(task_id, {})
        cheap_row = task_results.get("cheap_only")
        expensive_row = task_results.get("expensive_only")
        if cheap_row is None or expensive_row is None:
            skipped.append(task_id)
            continue

        cheap_score = _as_float(cheap_row.get("score"))
        expensive_score = _as_float(expensive_row.get("score"))
        cheap_pass = _as_bool(cheap_row.get("passed"))
        expensive_pass = _as_bool(expensive_row.get("passed"))
        oracle_strict_label = (
            "expensive_only"
            if cheap_score != 1.0 and expensive_score == 1.0
            else "cheap_only"
        )
        oracle_score_label = (
            "expensive_only" if expensive_score > cheap_score else "cheap_only"
        )

        rows.append(
            {
                "task_id": task_id,
                "category": task.get("category", cheap_row.get("category", "")),
                "difficulty": task.get(
                    "difficulty",
                    cheap_row.get("difficulty", ""),
                ),
                "prompt": task.get("prompt", ""),
                "function_name": task.get("function_name", ""),
                "cheap_score": cheap_score,
                "expensive_score": expensive_score,
                "oracle_strict_label": oracle_strict_label,
                "oracle_score_label": oracle_score_label,
                "oracle_best_score": max(cheap_score, expensive_score),
                "cheap_pass": cheap_pass,
                "expensive_pass": expensive_pass,
            }
        )

    return rows, skipped


def save_rows(rows: list[dict], path: str | Path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "category",
        "difficulty",
        "prompt",
        "function_name",
        "cheap_score",
        "expensive_score",
        "oracle_strict_label",
        "oracle_score_label",
        "oracle_best_score",
        "cheap_pass",
        "expensive_pass",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], skipped: list[str]):
    total = len(rows)
    strict_expensive = sum(
        1 for row in rows if row["oracle_strict_label"] == "expensive_only"
    )
    score_expensive = sum(
        1 for row in rows if row["oracle_score_label"] == "expensive_only"
    )
    print("Prompt Router Dataset")
    print("=" * 72)
    print(f"rows                         : {total}")
    print(f"skipped_missing_pair          : {len(skipped)}")
    if total:
        print(
            "oracle_strict_expensive_rate : "
            f"{100 * strict_expensive / total:.2f}%"
        )
        print(
            "oracle_score_expensive_rate  : "
            f"{100 * score_expensive / total:.2f}%"
        )
    if skipped:
        preview = ", ".join(skipped[:10])
        suffix = " ..." if len(skipped) > 10 else ""
        print(f"skipped_task_ids              : {preview}{suffix}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Build a prompt-router training dataset from task results."
    )
    parser.add_argument(
        "--task-results",
        default="results/task_evaluation.csv",
        help="CSV containing cheap_only and expensive_only task evaluation rows.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Original JSONL task dataset with prompts.",
    )
    parser.add_argument(
        "--output",
        default="results/prompt_router_dataset.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.dataset)
    results = load_task_results(args.task_results)
    rows, skipped = build_router_rows(tasks, results)
    if not rows:
        raise RuntimeError(
            "No complete cheap_only/expensive_only task pairs found. "
            "Run task evaluation with --modes cheap_only,expensive_only first."
        )
    save_rows(rows, args.output)
    print_summary(rows, skipped)
    print(f"output -> {args.output}")


if __name__ == "__main__":
    main()
