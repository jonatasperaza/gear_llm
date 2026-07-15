"""Freeze one prompt-router v2 policy using validation data only.

This script never reads the test split. It sweeps each trained candidate's
validation scores, enforces pass-rate and average-score preservation floors
relative to expensive_only, and selects the feasible policy with the fewest
expensive routes.
"""

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CANDIDATES = (
    "classifier_probing",
    "classifier_tfidf",
    "l2d_probing",
    "l2d_tfidf",
)


def as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def evaluate_threshold(
    validation_rows: dict[str, dict],
    scored_tasks: list[tuple[str, float]],
    mode: str,
    threshold: float,
) -> dict:
    selected_routes = {}
    for task_id, score in scored_tasks:
        is_expensive = (
            score >= threshold if mode == "classifier" else score > threshold
        )
        selected_routes[task_id] = (
            "expensive" if is_expensive else "cheap"
        )

    passes = []
    scores = []
    realized_costs = []
    true_expensive = []
    predicted_expensive = []
    for task_id, route in selected_routes.items():
        row = validation_rows[task_id]
        passes.append(as_bool(row, f"{route}_pass"))
        scores.append(as_float(row, f"{route}_score"))
        realized_costs.append(as_float(row, f"route_{route}_cost"))
        true_expensive.append(row["oracle_score_label"] == "expensive_only")
        predicted_expensive.append(route == "expensive")

    count = len(selected_routes)
    expensive_routes = sum(predicted_expensive)
    true_positive = sum(
        truth and prediction
        for truth, prediction in zip(true_expensive, predicted_expensive)
    )
    false_positive = sum(
        not truth and prediction
        for truth, prediction in zip(true_expensive, predicted_expensive)
    )
    false_negative = sum(
        truth and not prediction
        for truth, prediction in zip(true_expensive, predicted_expensive)
    )
    return {
        "threshold": threshold,
        "pass_rate": sum(passes) / count,
        "avg_score": sum(scores) / count,
        "expensive_routes": expensive_routes,
        "expensive_route_percent": 100.0 * expensive_routes / count,
        "estimated_saved_percent": 65.0 * (1.0 - expensive_routes / count),
        "realized_cost": sum(realized_costs),
        "precision_expensive": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "recall_expensive": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
    }


def best_candidate_threshold(
    validation_rows: dict[str, dict],
    candidate_dir: Path,
    pass_floor: float,
    score_floor: float,
) -> dict:
    with (candidate_dir / "policy_meta.json").open(encoding="utf-8") as file:
        policy_meta = json.load(file)
    predictions = load_csv(candidate_dir / "val_predictions.csv")
    mode = policy_meta["mode"]
    score_key = (
        "expensive_prob" if mode == "classifier" else "delta_score_pred"
    )
    scored_tasks = [
        (row["task_id"], float(row[score_key])) for row in predictions
    ]
    unique_scores = sorted({score for _, score in scored_tasks})
    epsilon = 1e-12
    thresholds = (
        [unique_scores[0] - epsilon]
        + unique_scores
        + [unique_scores[-1] + epsilon]
    )
    points = [
        evaluate_threshold(
            validation_rows,
            scored_tasks,
            mode,
            threshold,
        )
        for threshold in thresholds
    ]
    feasible = [
        point
        for point in points
        if point["pass_rate"] + epsilon >= pass_floor
        and point["avg_score"] + epsilon >= score_floor
    ]
    pool = feasible or points
    if feasible:
        best = min(
            pool,
            key=lambda point: (
                point["expensive_routes"],
                -point["pass_rate"],
                -point["avg_score"],
                -point["recall_expensive"],
            ),
        )
    else:
        best = max(
            pool,
            key=lambda point: (
                point["pass_rate"],
                point["avg_score"],
                -point["expensive_routes"],
            ),
        )
    return {
        "candidate_name": candidate_dir.name,
        "candidate_dir": str(candidate_dir),
        "mode": mode,
        "use_probing": bool(policy_meta.get("use_probing", True)),
        "feasible": bool(feasible),
        **best,
    }


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Select and freeze prompt-router v2 using validation only."
    )
    parser.add_argument(
        "--validation-csv",
        default="results/router_dataset_v2/val_features.csv",
    )
    parser.add_argument(
        "--candidates-root",
        default="results/router_v2",
    )
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
    )
    parser.add_argument("--pass-preservation", type=float, default=0.95)
    parser.add_argument("--score-preservation", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        default="results/router_v2/frozen_validation_policy",
    )
    args = parser.parse_args()

    validation_list = load_csv(Path(args.validation_csv))
    validation_rows = {row["task_id"]: row for row in validation_list}
    count = len(validation_list)
    expensive_pass_rate = sum(
        as_bool(row, "expensive_pass") for row in validation_list
    ) / count
    expensive_avg_score = sum(
        as_float(row, "expensive_score") for row in validation_list
    ) / count
    pass_floor = args.pass_preservation * expensive_pass_rate
    score_floor = args.score_preservation * expensive_avg_score

    root = Path(args.candidates_root)
    names = [name.strip() for name in args.candidates.split(",") if name.strip()]
    results = [
        best_candidate_threshold(
            validation_rows,
            root / name,
            pass_floor,
            score_floor,
        )
        for name in names
    ]
    feasible = [row for row in results if row["feasible"]]
    if not feasible:
        raise RuntimeError("No candidate satisfied the validation quality floors.")
    selected = min(
        feasible,
        key=lambda row: (
            row["expensive_routes"],
            -row["pass_rate"],
            -row["avg_score"],
            -row["recall_expensive"],
        ),
    )
    for row in results:
        row["selected"] = row is selected
        row["expensive_only_pass_rate"] = expensive_pass_rate
        row["expensive_only_avg_score"] = expensive_avg_score
        row["pass_floor"] = pass_floor
        row["score_floor"] = score_floor

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, output_dir / "validation_policy_selection.csv")

    source_dir = Path(selected["candidate_dir"])
    shutil.copy2(source_dir / "model.joblib", output_dir / "model.joblib")
    with (source_dir / "policy_meta.json").open(encoding="utf-8") as file:
        frozen_meta = json.load(file)
    frozen_meta.update(
        {
            "threshold": selected["threshold"],
            "frozen_from_candidate": selected["candidate_name"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_split": "validation",
            "selection_rule": (
                "minimize expensive routes subject to pass-rate and avg-score "
                "preservation floors"
            ),
            "pass_preservation_floor": args.pass_preservation,
            "score_preservation_floor": args.score_preservation,
            "validation_pass_rate": selected["pass_rate"],
            "validation_avg_score": selected["avg_score"],
            "validation_expensive_routes": selected["expensive_routes"],
            "validation_estimated_saved_percent": selected[
                "estimated_saved_percent"
            ],
        }
    )
    with (output_dir / "policy_meta.json").open("w", encoding="utf-8") as file:
        json.dump(frozen_meta, file, indent=2, ensure_ascii=False)

    print("Prompt Router v2 Validation Selection")
    print("=" * 88)
    print(
        f"expensive baseline: pass={expensive_pass_rate:.4f} "
        f"score={expensive_avg_score:.4f}"
    )
    print(f"quality floors    : pass={pass_floor:.4f} score={score_floor:.4f}")
    print("-" * 88)
    for row in results:
        marker = "*" if row["selected"] else " "
        print(
            f"{marker} {row['candidate_name']:<22} "
            f"pass={row['pass_rate']:.4f} score={row['avg_score']:.4f} "
            f"exp={row['expensive_routes']:>2}/{count} "
            f"saved={row['estimated_saved_percent']:.2f}% "
            f"threshold={row['threshold']:.6f}"
        )
    print("=" * 88)
    print(f"frozen model -> {output_dir / 'model.joblib'}")
    print(f"policy meta  -> {output_dir / 'policy_meta.json'}")


if __name__ == "__main__":
    main()
