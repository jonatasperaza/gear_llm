"""Train the learned prompt router v2 (TF-IDF + probing features, classifier or L2D).

This replaces train_prompt_router_ml.py for the v2 pipeline. Differences from v1:

  - Feature pipeline is a ColumnTransformer combining TF-IDF over the prompt
    text with the precomputed cheap-model probing features (see
    gear_llm.probing_features). v1 used TF-IDF alone and reached ROC-AUC 0.5545
    / AP 0.2047 on unseen prompts (≈ random).
  - Two selectable objectives via --loss:
      * classifier : predict oracle_score_label directly (comparable to v1).
      * l2d        : learning-to-defer. Regress delta_score (how much the
        expensive route beats the cheap route), then calibrate a deferral
        threshold on the VALIDATION split only, minimizing the expected routing
        cost route_cheap_cost / route_expensive_cost. This optimizes the
        quality-cost frontier instead of accuracy.

Protocol (enforced):
  - Fit on TRAIN only.
  - Select hyperparameters, class_weight, and threshold on VALIDATION only.
  - Touch TEST at most once, with the frozen policy, via eval_router_v2.py.

Outputs:
  results/router_v2/<run>/model.joblib           -- sklearn Pipeline
  results/router_v2/<run>/policy_meta.json       -- threshold, features, costs
  results/router_v2/<run>/val_metrics.csv        -- validation diagnostics
  results/router_v2/<run>/val_predictions.csv    -- per-prompt val predictions
"""

import argparse
import json
import sys
from pathlib import Path

# scripts/ has no __init__.py; import sibling modules by path.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gear_llm.probing_features import PROBING_FEATURE_KEYS  # noqa: E402

import csv  # noqa: E402

LABELS = ("cheap_only", "expensive_only")


# ---------------------------------------------------------------------------
# Cost model -- must match build_router_dataset_v2.derive_labels_and_costs.
# These are used by the L2D threshold search. C_fail is the penalty for routing
# to a model that fails the task; it dominates the call-cost difference, which
# encodes "a wrong answer is far more expensive than a slow one".
# ---------------------------------------------------------------------------
CHEAP_CALL_COST = 0.35
EXPENSIVE_CALL_COST = 1.00
DEFAULT_C_FAIL = 50.0


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row.get("prompt", "").strip()]
    if not rows:
        raise RuntimeError(f"No usable rows in {path}")
    return rows


def xy_from_rows(rows: list[dict]):
    """Return (prompts, feature_matrix, labels_score, labels_strict, costs).

    feature_matrix is a list of dict rows (one per prompt) keyed by probing
    feature name, suitable for DataFrame construction or dict-vectorizing. We
    keep it as a list of dicts and let the ColumnTransformer slice columns.
    """
    prompts = [row["prompt"] for row in rows]
    # Probing features as a list of plain dicts with float values.
    feat_rows = []
    for row in rows:
        feat_rows.append({key: _as_float(row.get(key)) for key in PROBING_FEATURE_KEYS})
    labels_score = [row.get("oracle_score_label", "") for row in rows]
    labels_strict = [row.get("oracle_strict_label", "") for row in rows]
    delta_score = [_as_float(row.get("delta_score")) for row in rows]
    route_cheap_cost = [_as_float(row.get("route_cheap_cost")) for row in rows]
    route_expensive_cost = [_as_float(row.get("route_expensive_cost")) for row in rows]
    delta_cost = [_as_float(row.get("delta_cost")) for row in rows]
    return {
        "prompts": prompts,
        "feat_rows": feat_rows,
        "labels_score": labels_score,
        "labels_strict": labels_strict,
        "delta_score": delta_score,
        "route_cheap_cost": route_cheap_cost,
        "route_expensive_cost": route_expensive_cost,
        "delta_cost": delta_cost,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# sklearn pipeline construction
# ---------------------------------------------------------------------------

def build_feature_pipeline(
    use_probing: bool,
    model_kind: str,
    class_weight=None,
    estimator_param: float | int | None = None,
):
    """Build the sklearn Pipeline.

    The pipeline accepts a DataFrame with a ``prompt`` text column and the
    probing feature columns. A ColumnTransformer routes text -> TF-IDF and
    numeric probing features -> passthrough/StandardScaler.

    Returns a Pipeline ending in either a classifier or a regressor.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    text_col = "prompt"
    text_transformer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    if use_probing:
        numeric_cols = list(PROBING_FEATURE_KEYS)
        # Coerce numeric inputs to float arrays for the scaler.
        numeric_transformer = StandardScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ("tfidf", text_transformer, text_col),
                ("probing", numeric_transformer, numeric_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0 if model_kind == "gbdt" else 0.3,
        )
    else:
        # TF-IDF only (ablation comparable to v1).
        preprocessor = ColumnTransformer(
            transformers=[("tfidf", text_transformer, text_col)],
            remainder="drop",
            sparse_threshold=0.0 if model_kind == "gbdt" else 0.3,
        )

    if model_kind == "classifier":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            C=float(estimator_param or 1.0),
            solver="liblinear",
            random_state=0,
        )
    elif model_kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        # class_weight supported in sklearn >= 1.2.
        try:
            estimator = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=int(estimator_param or 31),
                l2_regularization=1.0,
                random_state=0,
                class_weight=class_weight,
            )
        except TypeError:
            estimator = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                random_state=0,
            )
    elif model_kind == "regressor":
        from sklearn.linear_model import Ridge

        # Ridge with lsqr accepts the sparse TF-IDF output directly. The
        # previous HistGradientBoostingRegressor path required dense input and
        # failed for realistic vocabularies.
        estimator = Ridge(
            alpha=float(estimator_param or 1.0),
            solver="lsqr",
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    return Pipeline([("pre", preprocessor), ("model", estimator)])


def to_dataframe(prompts, feat_rows):
    """Build a DataFrame with the prompt column + probing feature columns."""
    import pandas as pd

    data = {"prompt": prompts}
    for key in PROBING_FEATURE_KEYS:
        data[key] = [fr.get(key, 0.0) for fr in feat_rows]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Threshold calibration (L2D) and validation metrics
# ---------------------------------------------------------------------------

def expected_cost_of_routes(delta_pred, threshold, route_cheap_cost, route_expensive_cost):
    """Compute the realized expected cost on the validation set given a threshold.

    delta_pred is the predicted delta_score (expensive - cheap). We defer to
    expensive when delta_pred > threshold. The realized cost for each prompt is
    the actual route_cheap_cost or route_expensive_cost depending on the route
    taken. Lower is better.

    Returns (total_cost, n_expensive, n_cheap).
    """
    total = 0.0
    n_expensive = 0
    n_cheap = 0
    for pred, cc, ec in zip(delta_pred, route_cheap_cost, route_expensive_cost):
        if pred > threshold:
            total += ec
            n_expensive += 1
        else:
            total += cc
            n_cheap += 1
    return total, n_expensive, n_cheap


def calibrate_l2d_threshold(
    delta_pred_val,
    route_cheap_cost_val,
    route_expensive_cost_val,
    n_points=101,
):
    """Sweep thresholds over the validation set and pick the cost-minimizing one.

    The candidate thresholds are quantiles of the predicted delta distribution,
    which is more robust than a uniform grid when the scores are skewed.
    """
    import numpy as np

    arr = np.asarray(delta_pred_val, dtype=float)
    if arr.size == 0:
        return 0.0, None

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        # No spread: defer to cheap (threshold above max -> never expensive).
        return hi + 1e-6, None

    candidates = np.linspace(lo - 1e-6, hi + 1e-6, n_points)
    best_threshold = candidates[0]
    best_cost = float("inf")
    best_counts = None
    for threshold in candidates:
        cost, n_exp, n_cheap = expected_cost_of_routes(
            arr, threshold, route_cheap_cost_val, route_expensive_cost_val
        )
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
            best_counts = (n_exp, n_cheap)
    return best_threshold, best_counts


def pareto_frontier(
    delta_pred_val,
    route_cheap_cost_val,
    route_expensive_cost_val,
    n_points=41,
):
    """Sweep thresholds and return the quality-cost frontier for reporting.

    Each point reports (threshold, total_cost, n_expensive_frac, avg_oracle_score).
    The avg_oracle_score is approximated by realized route scores inferred from
    cost: when cheap route is taken its contribution is route_cheap_cost minus
    its call cost; we don't have per-prompt scores here so we return cost-only
    frontier plus expensive fraction.
    """
    import numpy as np

    arr = np.asarray(delta_pred_val, dtype=float)
    if arr.size == 0:
        return []
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return [
            {"threshold": hi + 1e-6, "total_cost": float(sum(route_cheap_cost_val)), "n_expensive_frac": 0.0, "n_expensive": 0}
        ]
    candidates = np.linspace(lo - 1e-6, hi + 1e-6, n_points)
    frontier = []
    n = len(arr)
    for threshold in candidates:
        cost, n_exp, n_cheap = expected_cost_of_routes(
            arr, threshold, route_cheap_cost_val, route_expensive_cost_val
        )
        frontier.append(
            {
                "threshold": float(threshold),
                "total_cost": float(cost),
                "n_expensive_frac": float(n_exp / n) if n else 0.0,
                "n_expensive": int(n_exp),
            }
        )
    return frontier


def classification_metrics(y_true, y_pred, y_score_expensive):
    """Compute metrics for the classifier mode.

    y_score_expensive is the predicted probability of the expensive class.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        precision_recall_fscore_support,
    )

    # Binary encode: expensive_only = 1.
    y_true_bin = [1 if y == "expensive_only" else 0 for y in y_true]
    y_pred_bin = [1 if y == "expensive_only" else 0 for y in y_pred]

    roc = float("nan")
    ap = float("nan")
    try:
        roc = float(roc_auc_score(y_true_bin, y_score_expensive))
    except ValueError:
        pass
    try:
        ap = float(average_precision_score(y_true_bin, y_score_expensive))
    except ValueError:
        pass

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "roc_auc": roc,
        "pr_auc_average_precision": ap,
        "precision_cheap": float(precision[0]),
        "recall_cheap": float(recall[0]),
        "f1_cheap": float(f1[0]),
        "support_cheap": int(support[0]),
        "precision_expensive": float(precision[1]),
        "recall_expensive": float(recall[1]),
        "f1_expensive": float(f1[1]),
        "support_expensive": int(support[1]),
        "n_total": len(y_true_bin),
    }


def classification_threshold_sweep(y_true, y_score_expensive, n_points=41):
    """Find the probability threshold maximizing macro-F1 on validation."""
    import numpy as np

    y_true_bin = [1 if y == "expensive_only" else 0 for y in y_true]
    arr = np.asarray(y_score_expensive, dtype=float)
    if arr.size == 0:
        return 0.5
    candidates = np.linspace(0.01, 0.99, n_points)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        y_pred_bin = [1 if s >= threshold else 0 for s in arr]
        tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0)
        # macro F1
        prec_e = tp / (tp + fp) if (tp + fp) else 0.0
        rec_e = tp / (tp + fn) if (tp + fn) else 0.0
        f1_e = (
            2 * prec_e * rec_e / (prec_e + rec_e) if (prec_e + rec_e) else 0.0
        )
        prec_c = tn / (tn + fn) if (tn + fn) else 0.0
        rec_c = tn / (tn + fp) if (tn + fp) else 0.0
        f1_c = (
            2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) else 0.0
        )
        macro_f1 = 0.5 * (f1_e + f1_c)
        if macro_f1 > best_score:
            best_score = macro_f1
            best_threshold = float(threshold)
    return best_threshold


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def train_classifier(
    train_data,
    val_data,
    model_kind,
    use_probing,
    class_weights,
    estimator_params,
):
    """Select estimator settings and threshold using validation only."""
    X_train = to_dataframe(train_data["prompts"], train_data["feat_rows"])
    y_train = train_data["labels_score"]
    X_val = to_dataframe(val_data["prompts"], val_data["feat_rows"])
    y_val = val_data["labels_score"]

    # Handle degenerate single-class training sets.
    if len(set(y_train)) < 2:
        from sklearn.dummy import DummyClassifier

        text_col = "prompt"
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        pre = ColumnTransformer(
            transformers=[("tfidf", TfidfVectorizer(), text_col)], remainder="drop"
        )
        pipe = Pipeline(
            [
                ("pre", pre),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        )
        pipe.fit(X_train, y_train)
        candidates = [(pipe, None, None)]
    else:
        candidates = []
        for class_weight in class_weights:
            for estimator_param in estimator_params:
                candidate = build_feature_pipeline(
                    use_probing=use_probing,
                    model_kind=model_kind,
                    class_weight=class_weight,
                    estimator_param=estimator_param,
                )
                candidate.fit(X_train, y_train)
                candidates.append(
                    (candidate, class_weight, estimator_param)
                )

    best = None
    for candidate, class_weight, estimator_param in candidates:
        classes = list(candidate.classes_)
        exp_index = (
            classes.index("expensive_only")
            if "expensive_only" in classes
            else -1
        )
        if exp_index >= 0 and hasattr(candidate, "predict_proba"):
            probs = candidate.predict_proba(X_val)
            y_score_expensive = probs[:, exp_index].tolist()
        else:
            hard = candidate.predict(X_val)
            y_score_expensive = [
                1.0 if value == "expensive_only" else 0.0
                for value in hard
            ]

        threshold = classification_threshold_sweep(y_val, y_score_expensive)
        predictions = [
            "expensive_only" if score >= threshold else "cheap_only"
            for score in y_score_expensive
        ]
        metrics = classification_metrics(
            y_val,
            predictions,
            y_score_expensive,
        )
        metrics["threshold"] = threshold
        macro_f1 = 0.5 * (metrics["f1_cheap"] + metrics["f1_expensive"])
        selection_key = (
            macro_f1,
            metrics["pr_auc_average_precision"],
            metrics["recall_expensive"],
        )
        if best is None or selection_key > best[0]:
            best = (
                selection_key,
                candidate,
                classes,
                y_score_expensive,
                predictions,
                metrics,
                class_weight,
                estimator_param,
            )

    (
        _,
        pipe,
        classes,
        y_score_expensive,
        y_pred_thresholded,
        metrics,
        selected_class_weight,
        selected_estimator_param,
    ) = best
    metrics["selected_class_weight"] = str(selected_class_weight)
    metrics["selected_estimator_param"] = selected_estimator_param

    predictions = []
    for row, score, pred in zip(
        val_data["rows"], y_score_expensive, y_pred_thresholded
    ):
        predictions.append(
            {
                "task_id": row.get("task_id", ""),
                "mbpp_task_id": row.get("mbpp_task_id", ""),
                "prompt": row.get("prompt", "")[:120],
                "true_label": row.get("oracle_score_label", ""),
                "pred_label": pred,
                "expensive_prob": float(score),
                "cheap_score": row.get("cheap_score", ""),
                "expensive_score": row.get("expensive_score", ""),
            }
        )
    return pipe, metrics, predictions, {
        "threshold": metrics["threshold"],
        "classes": classes,
        "selected_class_weight": selected_class_weight,
        "selected_estimator_param": selected_estimator_param,
    }


def train_l2d(train_data, val_data, use_probing, alpha_grid):
    """Select a delta-score regressor and deferral threshold on validation."""
    import numpy as np

    X_train = to_dataframe(train_data["prompts"], train_data["feat_rows"])
    y_train = np.asarray(train_data["delta_score"], dtype=float)
    X_val = to_dataframe(val_data["prompts"], val_data["feat_rows"])

    best = None
    for alpha in alpha_grid:
        candidate = build_feature_pipeline(
            use_probing=use_probing,
            model_kind="regressor",
            class_weight=None,
            estimator_param=alpha,
        )
        candidate.fit(X_train, y_train)
        candidate_predictions = candidate.predict(X_val).tolist()
        candidate_threshold, _ = calibrate_l2d_threshold(
            candidate_predictions,
            val_data["route_cheap_cost"],
            val_data["route_expensive_cost"],
        )
        candidate_cost, candidate_n_exp, _ = expected_cost_of_routes(
            candidate_predictions,
            candidate_threshold,
            val_data["route_cheap_cost"],
            val_data["route_expensive_cost"],
        )
        selection_key = (candidate_cost, candidate_n_exp, float(alpha))
        if best is None or selection_key < best[0]:
            best = (
                selection_key,
                candidate,
                candidate_predictions,
                candidate_threshold,
                float(alpha),
            )

    _, pipe, delta_pred_val, threshold, selected_alpha = best
    frontier = pareto_frontier(
        delta_pred_val,
        val_data["route_cheap_cost"],
        val_data["route_expensive_cost"],
    )

    # Build derived classification metrics so the two modes are comparable.
    # Route=expensive iff delta_pred > threshold. Compare against oracle_score_label.
    y_pred = [
        "expensive_only" if d > threshold else "cheap_only"
        for d in delta_pred_val
    ]
    # Score = predicted delta (higher => more likely expensive).
    metrics = classification_metrics(
        val_data["labels_score"], y_pred, delta_pred_val
    )
    total_cost, n_exp, n_cheap = expected_cost_of_routes(
        delta_pred_val,
        threshold,
        val_data["route_cheap_cost"],
        val_data["route_expensive_cost"],
    )
    metrics["threshold"] = float(threshold)
    metrics["val_total_cost"] = float(total_cost)
    metrics["val_n_expensive"] = int(n_exp)
    metrics["val_n_cheap"] = int(n_cheap)
    metrics["val_cost_baseline_cheap_only"] = float(sum(val_data["route_cheap_cost"]))
    metrics["val_cost_baseline_expensive_only"] = float(
        sum(val_data["route_expensive_cost"])
    )
    metrics["selected_regressor_alpha"] = selected_alpha

    predictions = []
    for row, dval, pred in zip(val_data["rows"], delta_pred_val, y_pred):
        predictions.append(
            {
                "task_id": row.get("task_id", ""),
                "mbpp_task_id": row.get("mbpp_task_id", ""),
                "prompt": row.get("prompt", "")[:120],
                "true_label": row.get("oracle_score_label", ""),
                "pred_label": pred,
                "delta_score_pred": float(dval),
                "delta_score_true": row.get("delta_score", ""),
                "cheap_score": row.get("cheap_score", ""),
                "expensive_score": row.get("expensive_score", ""),
                "route_cheap_cost": row.get("route_cheap_cost", ""),
                "route_expensive_cost": row.get("route_expensive_cost", ""),
            }
        )

    policy_meta = {
        "threshold": float(threshold),
        "mode": "l2d",
        "cheap_call_cost": CHEAP_CALL_COST,
        "expensive_call_cost": EXPENSIVE_CALL_COST,
        "c_fail": DEFAULT_C_FAIL,
        "use_probing": use_probing,
        "selected_regressor_alpha": selected_alpha,
    }
    return pipe, metrics, predictions, policy_meta, frontier


def save_val_outputs(output_dir, metrics, predictions, frontier=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "val_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])

    pred_path = output_dir / "val_predictions.csv"
    fieldnames = sorted({k for p in predictions for k in p})
    with pred_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    if frontier is not None:
        frontier_path = output_dir / "val_pareto_frontier.csv"
        fieldnames = ["threshold", "total_cost", "n_expensive_frac", "n_expensive"]
        with frontier_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frontier)


def print_summary(loss, model_kind, use_probing, train_n, val_n, metrics):
    print()
    print("Prompt Router v2 Training")
    print("=" * 72)
    print(f"loss              : {loss}")
    print(f"model_kind        : {model_kind}")
    print(f"use_probing       : {use_probing}")
    print(f"train rows        : {train_n}")
    print(f"val rows          : {val_n}")
    print(f"probing features  : {len(PROBING_FEATURE_KEYS)}")
    print("-" * 72)
    print("VALIDATION metrics (policy frozen here, test reported separately):")
    for key in [
        "roc_auc",
        "pr_auc_average_precision",
        "precision_expensive",
        "recall_expensive",
        "f1_expensive",
        "support_expensive",
        "precision_cheap",
        "recall_cheap",
        "threshold",
    ]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key:<28} : {value:.4f}")
            else:
                print(f"  {key:<28} : {value}")
    if "val_total_cost" in metrics:
        print("-" * 72)
        print("L2D expected-cost diagnostics (validation):")
        for key in [
            "val_total_cost",
            "val_cost_baseline_cheap_only",
            "val_cost_baseline_expensive_only",
            "val_n_expensive",
            "val_n_cheap",
        ]:
            if key in metrics:
                print(f"  {key:<28} : {metrics[key]:.4f}")
    print("=" * 72)


def parse_float_grid(value: str, name: str) -> list[float]:
    try:
        values = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers.") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive.")
    return values


def resolve_class_weights(value: str) -> list[str | None]:
    normalized = value.strip().lower()
    if normalized == "auto":
        return [None, "balanced"]
    if normalized in {"none", "null"}:
        return [None]
    if normalized == "balanced":
        return ["balanced"]
    raise ValueError("--class-weight must be auto, balanced, or none.")


def main():
    parser = argparse.ArgumentParser(
        description="Train the learned prompt router v2 (TF-IDF + probing features)."
    )
    parser.add_argument(
        "--loss",
        choices=("classifier", "l2d"),
        default="l2d",
        help="Objective: classifier (predict oracle label) or l2d (regret/cost).",
    )
    parser.add_argument(
        "--model-kind",
        choices=("classifier", "gbdt"),
        default="classifier",
        help="Classifier estimator (ignored for --loss l2d, which uses a regressor).",
    )
    parser.add_argument(
        "--train-csv",
        default="results/router_dataset_v2/train_features.csv",
    )
    parser.add_argument(
        "--val-csv",
        default="results/router_dataset_v2/val_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/router_v2",
    )
    parser.add_argument(
        "--no-probing",
        action="store_true",
        help="Ablation: use TF-IDF only (drops probing features).",
    )
    parser.add_argument(
        "--class-weight",
        default="auto",
        help="Classifier class weight: auto, balanced, or none.",
    )
    parser.add_argument(
        "--classifier-param-grid",
        default="0.1,1.0,10.0",
        help="Logistic C values (or GBDT leaf counts) selected on validation.",
    )
    parser.add_argument(
        "--l2d-alpha-grid",
        default="0.1,1.0,10.0",
        help="Ridge alpha values selected on validation for L2D.",
    )
    args = parser.parse_args()

    try:
        import joblib
        import pandas as pd  # noqa: F401 -- imported for side-effect availability
    except ImportError as exc:
        raise ImportError(
            "train_prompt_router_v2 requires joblib, pandas, scikit-learn. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    train_rows = load_split_csv(Path(args.train_csv))
    val_rows = load_split_csv(Path(args.val_csv))
    train_data = xy_from_rows(train_rows)
    val_data = xy_from_rows(val_rows)

    use_probing = not args.no_probing
    classifier_params = parse_float_grid(
        args.classifier_param_grid,
        "--classifier-param-grid",
    )
    if args.model_kind == "gbdt":
        classifier_params = [max(2, int(value)) for value in classifier_params]
    l2d_alpha_grid = parse_float_grid(
        args.l2d_alpha_grid,
        "--l2d-alpha-grid",
    )
    class_weights = resolve_class_weights(args.class_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.loss == "l2d":
        pipe, metrics, predictions, policy_meta, frontier = train_l2d(
            train_data,
            val_data,
            use_probing,
            l2d_alpha_grid,
        )
        save_val_outputs(output_dir, metrics, predictions, frontier)
    else:
        pipe, metrics, predictions, policy_meta_extra = train_classifier(
            train_data,
            val_data,
            args.model_kind,
            use_probing,
            class_weights,
            classifier_params,
        )
        policy_meta = {
            "mode": "classifier",
            "model_kind": args.model_kind,
            "use_probing": use_probing,
            **policy_meta_extra,
        }
        save_val_outputs(output_dir, metrics, predictions)

    # Persist artifacts.
    model_path = output_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    meta_path = output_dir / "policy_meta.json"
    full_meta = {
        **policy_meta,
        "feature_keys": list(PROBING_FEATURE_KEYS),
        "labels": list(LABELS),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    import sklearn

    full_meta["scikit_learn_version"] = sklearn.__version__
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(full_meta, file, indent=2, ensure_ascii=False)

    model_kind = (
        "regressor" if args.loss == "l2d" else args.model_kind
    )
    print_summary(
        loss=args.loss,
        model_kind=model_kind,
        use_probing=use_probing,
        train_n=len(train_rows),
        val_n=len(val_rows),
        metrics=metrics,
    )
    print(f"model       -> {model_path}")
    print(f"policy_meta -> {meta_path}")
    print(f"val metrics -> {output_dir / 'val_metrics.csv'}")


if __name__ == "__main__":
    main()
