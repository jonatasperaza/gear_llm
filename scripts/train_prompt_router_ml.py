import argparse
import csv
from pathlib import Path


LABELS = ("cheap_only", "expensive_only")


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: str | Path, label_column: str) -> list[dict]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"task_id", "prompt", label_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"{dataset_path} missing columns: {fields}")

        rows = []
        for row in reader:
            label = row[label_column]
            if label not in LABELS:
                raise ValueError(
                    f"Unsupported label {label!r} in task {row.get('task_id')}. "
                    f"Expected one of: {', '.join(LABELS)}"
                )
            if not row.get("prompt", "").strip():
                continue
            rows.append(row)

    if len(rows) < 2:
        raise RuntimeError("Need at least two rows to train prompt_router_ml_v1.")

    return rows


def can_stratify(labels: list[str], test_size: float) -> bool:
    counts = {label: labels.count(label) for label in set(labels)}
    if len(counts) < 2 or min(counts.values()) < 2:
        return False

    test_count = max(1, round(len(labels) * test_size))
    train_count = len(labels) - test_count
    return test_count >= len(counts) and train_count >= len(counts)


def make_pipeline(use_dummy: bool = False):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError(
            "Training prompt_router_ml_v1 requires scikit-learn. "
            "Install project requirements first: pip install -r requirements.txt"
        ) from exc

    if use_dummy:
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=0,
        )

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def save_csv(rows: list[dict], path: str | Path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_rows(
    label_column: str,
    seed: int,
    test_size: float,
    rows: list[dict],
    y_train: list[str],
    y_test: list[str],
    y_pred: list[str],
) -> list[dict]:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=list(LABELS),
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=list(LABELS),
        average="weighted",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=list(LABELS),
        zero_division=0,
    )
    true_expensive = sum(1 for label in y_test if label == "expensive_only")
    pred_expensive = sum(1 for label in y_pred if label == "expensive_only")

    rows_out = [
        {
            "metric": "overall",
            "label_column": label_column,
            "seed": seed,
            "test_size": test_size,
            "total_count": len(rows),
            "train_count": len(y_train),
            "test_count": len(y_test),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "true_expensive_percent": (
                100 * true_expensive / len(y_test) if y_test else 0.0
            ),
            "predicted_expensive_percent": (
                100 * pred_expensive / len(y_pred) if y_pred else 0.0
            ),
            "trained_output_model_on_full_dataset": True,
        }
    ]

    for index, label in enumerate(LABELS):
        rows_out.append(
            {
                "metric": f"label:{label}",
                "label_column": label_column,
                "seed": seed,
                "test_size": test_size,
                "total_count": len(rows),
                "train_count": len(y_train),
                "test_count": len(y_test),
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
                "trained_output_model_on_full_dataset": True,
            }
        )

    return rows_out


def prediction_rows(split_rows, split_name: str, y_true: list[str], y_pred: list[str]):
    rows = []
    for row, true_label, predicted_label in zip(split_rows, y_true, y_pred):
        rows.append(
            {
                "split": split_name,
                "task_id": row.get("task_id", ""),
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "prompt": row.get("prompt", ""),
                "function_name": row.get("function_name", ""),
                "cheap_score": row.get("cheap_score", ""),
                "expensive_score": row.get("expensive_score", ""),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
                "oracle_best_score": row.get("oracle_best_score", ""),
            }
        )
    return rows


def output_paths(model_path: str | Path) -> tuple[Path, Path]:
    path = Path(model_path)
    return (
        path.with_name(f"{path.stem}_metrics.csv"),
        path.with_name(f"{path.stem}_predictions.csv"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train prompt_router_ml_v1 from prompt text."
    )
    parser.add_argument(
        "--router-dataset",
        default="results/prompt_router_dataset.csv",
        help="CSV from scripts/build_prompt_router_dataset.py.",
    )
    parser.add_argument(
        "--label-column",
        choices=("oracle_score_label", "oracle_strict_label"),
        default="oracle_score_label",
        help="Oracle label to learn.",
    )
    parser.add_argument(
        "--output-model",
        default="results/prompt_router_ml_v1.joblib",
        help="Output joblib model path.",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "Training prompt_router_ml_v1 requires joblib and scikit-learn. "
            "Install project requirements first: pip install -r requirements.txt"
        ) from exc

    rows = load_rows(args.router_dataset, args.label_column)
    prompts = [row["prompt"] for row in rows]
    labels = [row[args.label_column] for row in rows]
    stratify = labels if can_stratify(labels, args.test_size) else None
    train_rows, test_rows, y_train, y_test = train_test_split(
        rows,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    split_model = make_pipeline(use_dummy=len(set(y_train)) < 2)
    split_model.fit([row["prompt"] for row in train_rows], y_train)
    train_predictions = list(split_model.predict([row["prompt"] for row in train_rows]))
    test_predictions = list(split_model.predict([row["prompt"] for row in test_rows]))

    final_model = make_pipeline(use_dummy=len(set(labels)) < 2)
    final_model.fit(prompts, labels)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_model)
    metrics_path, predictions_path = output_paths(output_model)

    metrics = build_metric_rows(
        label_column=args.label_column,
        seed=args.seed,
        test_size=args.test_size,
        rows=rows,
        y_train=y_train,
        y_test=y_test,
        y_pred=test_predictions,
    )
    predictions = prediction_rows(
        train_rows,
        "train",
        y_train,
        train_predictions,
    ) + prediction_rows(
        test_rows,
        "test",
        y_test,
        test_predictions,
    )
    save_csv(metrics, metrics_path)
    save_csv(predictions, predictions_path)

    overall = metrics[0]
    print("Prompt Router ML Training")
    print("=" * 72)
    print(f"rows              : {len(rows)}")
    print(f"label_column      : {args.label_column}")
    print(f"train_count       : {len(y_train)}")
    print(f"test_count        : {len(y_test)}")
    print(f"accuracy          : {overall['accuracy']:.4f}")
    print(f"macro_f1          : {overall['macro_f1']:.4f}")
    print(f"model             : {output_model}")
    print(f"metrics           : {metrics_path}")
    print(f"predictions       : {predictions_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
