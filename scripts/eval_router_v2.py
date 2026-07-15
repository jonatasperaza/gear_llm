"""Evaluate the frozen prompt router v2 ONCE on the held-out test split.

This script is the single point of contact with the test set. The protocol
(PROBLEMS.md #13) requires:

  1. fit on train (train_prompt_router_v2.py),
  2. select threshold/features on val (train_prompt_router_v2.py),
  3. FREEZE the policy, then run this script exactly once on test.

Running it more than once with different knobs turns the test set into another
validation set and invalidates the generalization claim.

It compares, on the same test tasks:
  - cheap_only baseline
  - expensive_only baseline
  - prompt_router_ml_v1 baseline (TF-IDF only) -- optional, if model provided
  - prompt_router_ml_v2 (this work)

Reported metrics:
  - pass rate (fraction of tasks passed under each policy)
  - average task score
  - PR-AUC / ROC-AUC for the expensive class (v2 only)
  - expensive-route recall and precision (v2 only)
  - realized routing cost vs expensive_only / cheap_only
  - fraction routed to expensive

Outputs:
  results/router_v2/<run>/test_report.md
  results/router_v2/<run>/test_metrics.csv
  results/router_v2/<run>/test_predictions.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gear_llm.probing_features import PROBING_FEATURE_KEYS  # noqa: E402


LABELS = ("cheap_only", "expensive_only")


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_test_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("prompt", "").strip()]
    if not rows:
        raise RuntimeError(f"No usable rows in {path}")
    return rows


def to_dataframe(rows):
    import pandas as pd

    data = {"prompt": [r["prompt"] for r in rows]}
    for key in PROBING_FEATURE_KEYS:
        data[key] = [_as_float(r.get(key)) for r in rows]
    return pd.DataFrame(data)


def baseline_metrics(rows, route_column, score_column, pass_column, label):
    """Compute pass-rate/avg-score for a fixed route (cheap_only/expensive_only)."""
    scores = [_as_float(r.get(score_column)) for r in rows]
    passes = [_as_bool(r.get(pass_column)) for r in rows]
    n = len(rows)
    pass_rate = sum(1 for p in passes if p) / n if n else 0.0
    avg_score = sum(scores) / n if n else 0.0
    return {
        "policy": label,
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "n_total": n,
    }


def evaluate_v1_on_test(rows, model):
    """Evaluate the frozen TF-IDF v1 router on the same task outcomes."""
    prompts = [row["prompt"] for row in rows]
    predictions = [str(value) for value in model.predict(prompts)]
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if "expensive_only" in classes:
            index = classes.index("expensive_only")
            rank_scores = model.predict_proba(prompts)[:, index].tolist()
        else:
            rank_scores = [0.0] * len(rows)
    else:
        rank_scores = [
            1.0 if value == "expensive_only" else 0.0
            for value in predictions
        ]

    realized_scores = []
    realized_passes = []
    realized_cost = 0.0
    n_expensive = 0
    for row, prediction in zip(rows, predictions):
        if prediction == "expensive_only":
            realized_scores.append(_as_float(row.get("expensive_score")))
            realized_passes.append(_as_bool(row.get("expensive_pass")))
            realized_cost += _as_float(row.get("route_expensive_cost"))
            n_expensive += 1
        else:
            realized_scores.append(_as_float(row.get("cheap_score")))
            realized_passes.append(_as_bool(row.get("cheap_pass")))
            realized_cost += _as_float(row.get("route_cheap_cost"))

    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = [
        1 if row.get("oracle_score_label") == "expensive_only" else 0
        for row in rows
    ]
    predicted_binary = [
        1 if value == "expensive_only" else 0 for value in predictions
    ]
    try:
        roc = float(roc_auc_score(y_true, rank_scores))
    except ValueError:
        roc = float("nan")
    try:
        ap = float(average_precision_score(y_true, rank_scores))
    except ValueError:
        ap = float("nan")
    tp = sum(1 for y, p in zip(y_true, predicted_binary) if y == p == 1)
    fp = sum(1 for y, p in zip(y_true, predicted_binary) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, predicted_binary) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    n = len(rows)
    return {
        "policy": "prompt_router_ml_v1",
        "mode": "classifier",
        "pass_rate": sum(realized_passes) / n if n else 0.0,
        "avg_score": sum(realized_scores) / n if n else 0.0,
        "roc_auc": roc,
        "pr_auc_average_precision": ap,
        "precision_expensive": precision,
        "recall_expensive": recall,
        "n_expensive_routes": n_expensive,
        "n_cheap_routes": n - n_expensive,
        "frac_expensive": n_expensive / n if n else 0.0,
        "realized_cost": realized_cost,
        "n_total": n,
    }


def evaluate_v2_on_test(rows, model, policy_meta):
    """Apply the frozen v2 policy to the test rows and compute routing metrics."""
    import numpy as np

    X = to_dataframe(rows)
    mode = policy_meta.get("mode", "l2d")
    threshold = float(policy_meta.get("threshold", 0.5))

    if mode == "l2d":
        # Regressor predicts delta_score; defer to expensive if > threshold.
        delta_pred = model.predict(X)
        preds = ["expensive_only" if d > threshold else "cheap_only" for d in delta_pred]
        scores_for_ranking = [float(d) for d in delta_pred]
    else:
        # Classifier: predict probability of expensive class; threshold it.
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            probs = model.predict_proba(X)
            if "expensive_only" in classes:
                idx = classes.index("expensive_only")
                scores_for_ranking = probs[:, idx].tolist()
            else:
                scores_for_ranking = [0.0] * len(rows)
        else:
            scores_for_ranking = [0.0] * len(rows)
        preds = ["expensive_only" if s >= threshold else "cheap_only" for s in scores_for_ranking]

    # Realized outcomes: for each task, use the score of the route the policy chose.
    realized_scores = []
    realized_passes = []
    n_expensive = 0
    for row, pred in zip(rows, preds):
        if pred == "expensive_only":
            realized_scores.append(_as_float(row.get("expensive_score")))
            realized_passes.append(_as_bool(row.get("expensive_pass")))
            n_expensive += 1
        else:
            realized_scores.append(_as_float(row.get("cheap_score")))
            realized_passes.append(_as_bool(row.get("cheap_pass")))

    n = len(rows)
    pass_rate = sum(1 for p in realized_passes if p) / n if n else 0.0
    avg_score = sum(realized_scores) / n if n else 0.0

    # Oracle upper bound: best of (cheap, expensive) per task.
    oracle_scores = [
        max(_as_float(r.get("cheap_score")), _as_float(r.get("expensive_score")))
        for r in rows
    ]
    oracle_passes = [
        _as_bool(r.get("cheap_pass")) or _as_bool(r.get("expensive_pass"))
        for r in rows
    ]
    oracle_pass_rate = sum(1 for p in oracle_passes if p) / n if n else 0.0
    oracle_avg_score = sum(oracle_scores) / n if n else 0.0

    # Ranking metrics (PR-AUC, ROC-AUC) against oracle_score_label.
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = [1 if r.get("oracle_score_label") == "expensive_only" else 0 for r in rows]
    roc = float("nan")
    ap = float("nan")
    try:
        roc = float(roc_auc_score(y_true, scores_for_ranking))
    except ValueError:
        pass
    try:
        ap = float(average_precision_score(y_true, scores_for_ranking))
    except ValueError:
        pass

    # Confusion matrix against oracle_score_label.
    tp = sum(1 for y, p in zip(y_true, [1 if x == "expensive_only" else 0 for x in preds]) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, [1 if x == "expensive_only" else 0 for x in preds]) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, [1 if x == "expensive_only" else 0 for x in preds]) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, [1 if x == "expensive_only" else 0 for x in preds]) if y == 0 and p == 0)
    prec_e = tp / (tp + fp) if (tp + fp) else 0.0
    rec_e = tp / (tp + fn) if (tp + fn) else 0.0

    # Cost: approximate realized routing cost using the cost columns.
    realized_cost = 0.0
    for row, pred in zip(rows, preds):
        if pred == "expensive_only":
            realized_cost += _as_float(row.get("route_expensive_cost"))
        else:
            realized_cost += _as_float(row.get("route_cheap_cost"))
    cheap_only_cost = sum(_as_float(r.get("route_cheap_cost")) for r in rows)
    expensive_only_cost = sum(_as_float(r.get("route_expensive_cost")) for r in rows)

    metrics = {
        "policy": "prompt_router_ml_v2",
        "mode": mode,
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "roc_auc": roc,
        "pr_auc_average_precision": ap,
        "precision_expensive": prec_e,
        "recall_expensive": rec_e,
        "n_expensive_routes": n_expensive,
        "n_cheap_routes": n - n_expensive,
        "frac_expensive": n_expensive / n if n else 0.0,
        "realized_cost": realized_cost,
        "cost_vs_cheap_only_baseline": realized_cost - cheap_only_cost,
        "cost_vs_expensive_only_baseline": realized_cost - expensive_only_cost,
        "oracle_pass_rate": oracle_pass_rate,
        "oracle_avg_score": oracle_avg_score,
        "n_total": n,
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    predictions = []
    for row, pred, score_rank in zip(rows, preds, scores_for_ranking):
        predictions.append(
            {
                "task_id": row.get("task_id", ""),
                "mbpp_task_id": row.get("mbpp_task_id", ""),
                "prompt": row.get("prompt", "")[:120],
                "pred_route": pred,
                "true_label": row.get("oracle_score_label", ""),
                "rank_score": float(score_rank),
                "cheap_score": row.get("cheap_score", ""),
                "expensive_score": row.get("expensive_score", ""),
                "realized_score": _as_float(
                    row.get("cheap_score") if pred == "cheap_only" else row.get("expensive_score")
                ),
            }
        )
    return metrics, predictions


def render_markdown_report(
    all_metrics, v2_metrics, v1_metrics, policy_meta, test_csv, n_rows
):
    lines = []
    lines.append("# Prompt Router v2 — Test Report")
    lines.append("")
    lines.append("**One-shot evaluation on the held-out test split.**")
    lines.append(
        "Per PROBLEMS.md #13, this script must be run at most once with the frozen "
        "policy; re-running with different knobs invalidates the generalization claim."
    )
    lines.append("")
    lines.append(f"- test csv: `{test_csv}`")
    lines.append(f"- test rows: {n_rows}")
    lines.append(f"- policy mode: `{policy_meta.get('mode')}`")
    lines.append(f"- threshold: `{policy_meta.get('threshold')}`")
    lines.append(f"- probing features: `{policy_meta.get('use_probing', True)}`")
    if v1_metrics is not None:
        lines.append(
            "- v1 is a historical TF-IDF reference; unless retrained on the "
            "fixed train split, it is not a protocol-matched baseline."
        )
    lines.append("")

    lines.append("## Comparison table")
    lines.append("")
    header = (
        "| Policy | Pass rate | Avg score | Expensive routes | "
        "Cost | PR-AUC | ROC-AUC | Exp recall |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)

    def fmt(v, nd=4, default="—"):
        if v is None:
            return default
        try:
            if isinstance(v, float) and (v != v):  # NaN
                return default
            return f"{v:.{nd}f}"
        except (TypeError, ValueError):
            return str(v)

    for m in all_metrics:
        name = m["policy"]
        exp_frac = m.get("frac_expensive")
        exp_routes = (
            f"{m.get('n_expensive_routes','—')}/{m.get('n_total','—')}"
            if "n_expensive_routes" in m
            else "—"
        )
        cost = m.get("realized_cost")
        pr = m.get("pr_auc_average_precision")
        roc = m.get("roc_auc")
        rec = m.get("recall_expensive")
        lines.append(
            f"| {name} | {fmt(m.get('pass_rate'), 4)} | {fmt(m.get('avg_score'), 4)} | "
            f"{exp_routes} | {fmt(cost, 2)} | {fmt(pr)} | {fmt(roc)} | {fmt(rec)} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "1. **Quality**: compare v2 with both single-model baselines and the "
        "oracle ceiling; a small difference on 85 tasks remains preliminary."
    )
    lines.append(
        "2. **Cost**: route-mix savings and token-count-derived realized cost "
        "are proxies, not wall-clock speedup."
    )
    lines.append(
        "3. **Generalization**: PR-AUC, precision and expensive recall on this "
        "held-out split describe routing quality, but require external replication."
    )
    lines.append(
        "4. **Probing**: if validation selects a no-probing candidate, report "
        "the current probing representation as a negative result."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen prompt router v2 on the held-out test split."
    )
    parser.add_argument(
        "--test-csv",
        default="results/router_dataset_v2/test_features.csv",
    )
    parser.add_argument(
        "--model",
        default="results/router_v2/model.joblib",
        help="Frozen v2 model.joblib from train_prompt_router_v2.py.",
    )
    parser.add_argument(
        "--policy-meta",
        default="results/router_v2/policy_meta.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/router_v2",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite an existing held-out test report. Avoid this for the "
            "official frozen-policy evaluation."
        ),
    )
    parser.add_argument(
        "--v1-model",
        default=(
            "results/kaggle/prompt_router_ml_v1/"
            "seed123_train/model.joblib"
        ),
        help="Optional frozen v1 model used as a reference baseline.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    protected_outputs = (
        output_dir / "test_metrics.csv",
        output_dir / "test_predictions.csv",
        output_dir / "test_report.md",
    )
    existing_outputs = [path for path in protected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Held-out test outputs already exist. Refusing to re-run the "
            f"one-shot evaluation: {existing}. Use --overwrite only for a "
            "deliberate non-canonical rerun."
        )

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "eval_router_v2 requires joblib. pip install -r requirements.txt"
        ) from exc

    rows = load_test_csv(Path(args.test_csv))
    model = joblib.load(args.model)
    with Path(args.policy_meta).open("r", encoding="utf-8") as file:
        policy_meta = json.load(file)

    # Baselines computed directly from the CSV columns (no model needed).
    cheap_baseline = baseline_metrics(
        rows, "cheap", "cheap_score", "cheap_pass", "cheap_only"
    )
    expensive_baseline = baseline_metrics(
        rows, "expensive", "expensive_score", "expensive_pass", "expensive_only"
    )
    cheap_baseline["realized_cost"] = sum(_as_float(r.get("route_cheap_cost")) for r in rows)
    expensive_baseline["realized_cost"] = sum(_as_float(r.get("route_expensive_cost")) for r in rows)

    v2_metrics, predictions = evaluate_v2_on_test(rows, model, policy_meta)

    v1_metrics = None
    v1_path = Path(args.v1_model) if args.v1_model else None
    if v1_path and v1_path.exists():
        try:
            v1_metrics = evaluate_v1_on_test(rows, joblib.load(v1_path))
        except Exception as exc:  # noqa: BLE001 - optional reference only
            print(f"WARNING: v1 baseline could not be loaded: {exc}")
    elif v1_path:
        print(f"WARNING: v1 baseline not found: {v1_path}")

    all_metrics = [cheap_baseline, expensive_baseline]
    if v1_metrics is not None:
        all_metrics.append(v1_metrics)
    all_metrics.append(v2_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV of all metrics.
    metrics_csv = output_dir / "test_metrics.csv"
    fieldnames = sorted({k for m in all_metrics for k in m})
    with metrics_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

    # Predictions.
    pred_csv = output_dir / "test_predictions.csv"
    pred_fields = sorted({k for p in predictions for k in p})
    with pred_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(predictions)

    # Markdown report.
    report_md = output_dir / "test_report.md"
    report_text = render_markdown_report(
        all_metrics=all_metrics,
        v2_metrics=v2_metrics,
        v1_metrics=v1_metrics,
        policy_meta=policy_meta,
        test_csv=args.test_csv,
        n_rows=len(rows),
    )
    with report_md.open("w", encoding="utf-8") as file:
        file.write(report_text)

    # Console summary.
    print()
    print("Prompt Router v2 — Test Evaluation (one-shot)")
    print("=" * 72)
    print(f"test rows : {len(rows)}")
    print(f"mode      : {policy_meta.get('mode')}")
    print(f"threshold : {policy_meta.get('threshold')}")
    print("-" * 72)
    print(f"{'policy':<22} {'pass':>7} {'score':>7} {'exp%':>6} {'PR-AUC':>7} {'ROC':>7}")
    for m in all_metrics:
        print(
            f"{m['policy']:<22} "
            f"{m.get('pass_rate',0):>7.4f} "
            f"{m.get('avg_score',0):>7.4f} "
            f"{m.get('frac_expensive',0):>6.2f} "
            f"{m.get('pr_auc_average_precision', float('nan')):>7.4f} "
            f"{m.get('roc_auc', float('nan')):>7.4f}"
        )
    print("-" * 72)
    print(f"oracle pass rate     : {v2_metrics['oracle_pass_rate']:.4f}")
    print(f"oracle avg score     : {v2_metrics['oracle_avg_score']:.4f}")
    print(f"expensive recall (v2): {v2_metrics['recall_expensive']:.4f}")
    print(f"expensive precision  : {v2_metrics['precision_expensive']:.4f}")
    print("=" * 72)
    print(f"report    -> {report_md}")
    print(f"metrics   -> {metrics_csv}")
    print(f"predicts  -> {pred_csv}")


if __name__ == "__main__":
    main()
