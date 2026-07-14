"""Build the enriched router dataset (v2): oracle labels + probing features.

This is the offline pipeline that produces the training data for the learned
prompt router v2. For each MBPP task in the fixed split (see
build_mbpp_split.py) it:

  1. generates a solution with cheap_only and expensive_only (greedy),
  2. evaluates each with the project's task evaluator (code tests / math /
     logic labels),
  3. extracts cheap-model + cheap-vs-expensive probing features in a single
     forward pass per model,
  4. derives oracle labels (oracle_score_label, oracle_strict_label) and the
     learning-to-defer cost fields (delta_score, route_*_cost).

Output (per split + an _all CSV):
    results/router_dataset_v2/<split>_features.csv

The pipeline checkpoints incrementally: completed task ids are appended to a
JSONL sidecar so a crashed Kaggle run can be resumed without recomputing.

Usage (Kaggle/Colab with GPU):
    python scripts/build_router_dataset_v2.py \\
        --cheap-model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
        --expensive-model Qwen/Qwen2.5-Coder-3B-Instruct \\
        --device cuda --torch-dtype float16 \\
        --max-new-tokens 256 \\
        --train-data data/mbpp_train_257.jsonl \\
        --val-data data/mbpp_val_85.jsonl \\
        --test-data data/mbpp_test_85.jsonl \\
        --output-dir results/router_dataset_v2

Smoke (CPU, tiny):
    python scripts/build_router_dataset_v2.py \\
        --max-new-tokens 16 --limit 4 \\
        --train-data data/mbpp_train_257.jsonl \\
        --output-dir results/router_dataset_v2_smoke
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

# Make the repo root importable when run as `python scripts/<name>.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_llm.adaptive_generator import (  # noqa: E402
    AdaptiveGenerationConfig,
    load_adaptive_models,
)
from gear_llm.model_loader import (  # noqa: E402
    get_cheap_tokenizer,
    get_expensive_tokenizer,
    get_model_runtime_metadata,
)
from gear_llm.probing_features import PROBING_FEATURE_KEYS, compute_probing_features  # noqa: E402
from gear_llm.quality_benchmark import estimated_saved_percent, generate_greedy_with_model  # noqa: E402
from gear_llm.task_evaluation import evaluate_task  # noqa: E402


# Default cost model used by the adaptive generator (see AdaptiveGenerationConfig).
# Kept explicit here because the L2D threshold depends on it.
CHEAP_CALL_COST = 0.35
EXPENSIVE_CALL_COST = 1.00
# Penalty for a failing answer. C_fail is the *quality* cost of a wrong answer,
# expressed in the same units as the per-token compute cost. The router should
# avoid sending a prompt to the cheap model when it will fail AND the expensive
# model would have succeeded. C_FAIL is large relative to per-token cost so the
# expected-cost optimum pushes toward the expensive model only when the cheap
# model genuinely fails. This is reported and tunable.
DEFAULT_C_FAIL = 50.0


def load_tasks_jsonl(path: Path) -> list[dict]:
    tasks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                tasks.append(json.loads(stripped))
    return tasks


def load_split_manifest(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    assignments = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(row["mbpp_task_id"])
            if task_id in assignments:
                raise ValueError(f"Duplicate mbpp_task_id in manifest: {task_id}")
            assignments[task_id] = str(row["split"])
    return assignments


def validate_split_tasks(
    split_name: str,
    tasks: list[dict],
    assignments: dict[int, str],
):
    ids = [int(task["mbpp_task_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate mbpp_task_id values in {split_name} data.")
    mismatches = [
        task_id
        for task_id in ids
        if assignments.get(task_id) != split_name
    ]
    if mismatches:
        raise ValueError(
            f"{split_name} data disagrees with split manifest for task ids: "
            + ", ".join(str(value) for value in mismatches[:10])
        )


def load_checkpoint_rows(checkpoint_path: Path) -> list[dict]:
    """Load the latest row for each completed task id."""
    if not checkpoint_path.exists():
        return []
    rows_by_id: dict[str, dict] = {}
    with checkpoint_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if task_id is not None:
                rows_by_id[str(task_id)] = record
    return list(rows_by_id.values())


def write_checkpoint_rows(checkpoint_path: Path, rows: list[dict]):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_checkpoint(checkpoint_path: Path, row: dict):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def synchronize_devices(*devices: str):
    if not torch.cuda.is_available():
        return
    seen = set()
    for device in devices:
        value = str(device)
        if value.startswith("cuda") and value not in seen:
            torch.cuda.synchronize(value)
            seen.add(value)


def write_features_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable column order: identity, labels, cost fields, then the probing
    # features, then generation telemetry. This order is the on-disk contract.
    base_fields = [
        "task_id",
        "mbpp_task_id",
        "split",
        "category",
        "difficulty",
        "prompt",
        "function_name",
        "cheap_score",
        "expensive_score",
        "cheap_pass",
        "expensive_pass",
        "oracle_score_label",
        "oracle_strict_label",
        "oracle_best_score",
        "delta_score",
        "route_cheap_cost",
        "route_expensive_cost",
        "delta_cost",
        "n_failed_by_cheap_only",
        "cheap_model_name",
        "expensive_model_name",
        "device",
        "cheap_device",
        "expensive_device",
        "torch_dtype",
        "prompt_format",
    ]
    feature_fields = list(PROBING_FEATURE_KEYS)
    telemetry_fields = [
        "cheap_generated_tokens",
        "expensive_generated_tokens",
        "cheap_generation_time_seconds",
        "expensive_generation_time_seconds",
        "cheap_failed_tests",
        "expensive_failed_tests",
        "cheap_test_count",
        "expensive_test_count",
        "feature_time_seconds",
        "generation_time_seconds",
    ]
    fieldnames = base_fields + feature_fields + telemetry_fields
    # Include any extra keys that slipped in, defensively.
    extra = set()
    for row in rows:
        extra.update(row.keys())
    extra -= set(fieldnames)
    fieldnames += sorted(extra)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def derive_labels_and_costs(cheap_score, expensive_score, cheap_pass, expensive_pass):
    """Mirror build_prompt_router_dataset.py labels and add L2D cost fields."""
    oracle_strict_label = (
        "expensive_only"
        if cheap_score != 1.0 and expensive_score == 1.0
        else "cheap_only"
    )
    oracle_score_label = (
        "expensive_only" if expensive_score > cheap_score else "cheap_only"
    )
    oracle_best_score = max(cheap_score, expensive_score)
    delta_score = expensive_score - cheap_score

    # Expected cost of each route. C_fail is charged when the answer fails.
    # A failure is binary here (score < 1.0), matching oracle_strict_label.
    cheap_fail = 1.0 if not cheap_pass else 0.0
    expensive_fail = 1.0 if not expensive_pass else 0.0
    route_cheap_cost = CHEAP_CALL_COST + cheap_fail * DEFAULT_C_FAIL
    route_expensive_cost = EXPENSIVE_CALL_COST + expensive_fail * DEFAULT_C_FAIL
    delta_cost = route_cheap_cost - route_expensive_cost  # >0 => expensive cheaper

    # How many of the cheap-only failures the expensive route would have fixed.
    # This is 0/1 at the task level for the strict oracle, but kept as a column
    # so an aggregate "fraction recoverable" can be computed per split.
    n_failed_by_cheap_only = 1 if (not cheap_pass and expensive_pass) else 0

    return {
        "oracle_strict_label": oracle_strict_label,
        "oracle_score_label": oracle_score_label,
        "oracle_best_score": oracle_best_score,
        "delta_score": delta_score,
        "route_cheap_cost": route_cheap_cost,
        "route_expensive_cost": route_expensive_cost,
        "delta_cost": delta_cost,
        "n_failed_by_cheap_only": n_failed_by_cheap_only,
    }


def process_one_task(
    task: dict,
    split: str,
    cheap_model,
    expensive_model,
    tokenizer,
    cheap_device: str,
    expensive_device: str,
    cheap_tokenizer,
    expensive_tokenizer,
    max_new_tokens: int,
    temperature: float,
    prompt_format: str,
    top_k: int,
    model_metadata: dict,
) -> dict:
    """Generate + evaluate + extract features for one task. Returns a full row."""
    prompt = task["prompt"]
    task_id = task["id"]

    # --- Generation: cheap_only and expensive_only -------------------------
    synchronize_devices(cheap_device)
    cheap_gen_start = time.perf_counter()
    cheap_text, cheap_tokens = generate_greedy_with_model(
        prompt=prompt,
        model=cheap_model,
        tokenizer=cheap_tokenizer,
        device=cheap_device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_format=prompt_format,
        model_role="cheap",
    )
    synchronize_devices(cheap_device)
    cheap_generation_time = time.perf_counter() - cheap_gen_start

    synchronize_devices(expensive_device)
    expensive_gen_start = time.perf_counter()
    expensive_text, expensive_tokens = generate_greedy_with_model(
        prompt=prompt,
        model=expensive_model,
        tokenizer=expensive_tokenizer,
        device=expensive_device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_format=prompt_format,
        model_role="expensive",
    )
    synchronize_devices(expensive_device)
    expensive_generation_time = time.perf_counter() - expensive_gen_start
    generation_time = cheap_generation_time + expensive_generation_time

    # --- Evaluation --------------------------------------------------------
    cheap_eval = evaluate_task(task, cheap_text)
    expensive_eval = evaluate_task(task, expensive_text)

    cheap_score = float(cheap_eval.get("score", 0.0))
    expensive_score = float(expensive_eval.get("score", 0.0))
    cheap_pass = bool(cheap_eval.get("passed", False))
    expensive_pass = bool(expensive_eval.get("passed", False))

    labels = derive_labels_and_costs(
        cheap_score=cheap_score,
        expensive_score=expensive_score,
        cheap_pass=cheap_pass,
        expensive_pass=expensive_pass,
    )

    # --- Probing features --------------------------------------------------
    synchronize_devices(cheap_device, expensive_device)
    feat_start = time.perf_counter()
    features = compute_probing_features(
        prompt=prompt,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=cheap_device,
        prompt_format=prompt_format,
        top_k=top_k,
    )
    synchronize_devices(cheap_device, expensive_device)
    feature_time = time.perf_counter() - feat_start

    row = {
        "task_id": task_id,
        "mbpp_task_id": task.get("mbpp_task_id", ""),
        "split": split,
        "category": task.get("category", ""),
        "difficulty": task.get("difficulty", ""),
        "prompt": prompt,
        "function_name": task.get("function_name", ""),
        **model_metadata,
        "cheap_score": cheap_score,
        "expensive_score": expensive_score,
        "cheap_pass": cheap_pass,
        "expensive_pass": expensive_pass,
        **labels,
        "cheap_generated_tokens": cheap_tokens,
        "expensive_generated_tokens": expensive_tokens,
        "cheap_generation_time_seconds": cheap_generation_time,
        "expensive_generation_time_seconds": expensive_generation_time,
        "cheap_failed_tests": cheap_eval.get("test_count", 0)
        - cheap_eval.get("passed_tests", 0),
        "expensive_failed_tests": expensive_eval.get("test_count", 0)
        - expensive_eval.get("passed_tests", 0),
        "cheap_test_count": cheap_eval.get("test_count", 0),
        "expensive_test_count": expensive_eval.get("test_count", 0),
        "feature_time_seconds": feature_time,
        "generation_time_seconds": generation_time,
        **features,
    }
    return row


def run_split(
    split_name: str,
    tasks: list[dict],
    output_dir: Path,
    resume: bool,
    models_bundle,
    max_new_tokens: int,
    temperature: float,
    prompt_format: str,
    top_k: int,
    checkpoint_every: int,
    limit: int | None,
):
    (
        cheap_model,
        expensive_model,
        tokenizer,
        _device,
        cheap_device,
        expensive_device,
        cheap_tokenizer,
        expensive_tokenizer,
        model_metadata,
    ) = models_bundle

    checkpoint_path = output_dir / f"{split_name}_checkpoint.jsonl"
    if not resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    if limit is not None:
        tasks = tasks[:limit]
    allowed_ids = {str(task["id"]) for task in tasks}

    rows = (
        [
            row
            for row in load_checkpoint_rows(checkpoint_path)
            if str(row.get("task_id")) in allowed_ids
        ]
        if resume
        else []
    )
    done_ids = {str(row["task_id"]) for row in rows}
    if done_ids:
        print(f"[{split_name}] resuming: {len(done_ids)} tasks already done")
    if resume and checkpoint_path.exists():
        write_checkpoint_rows(checkpoint_path, rows)

    total = len(tasks)
    processed_this_run = 0
    skipped_done = 0
    t0 = time.perf_counter()

    for index, task in enumerate(tasks, start=1):
        task_id = task["id"]
        if str(task_id) in done_ids:
            skipped_done += 1
            continue

        try:
            row = process_one_task(
                task=task,
                split=split_name,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                cheap_device=cheap_device,
                expensive_device=expensive_device,
                cheap_tokenizer=cheap_tokenizer,
                expensive_tokenizer=expensive_tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_format=prompt_format,
                top_k=top_k,
                model_metadata=model_metadata,
            )
        except Exception as exc:  # noqa: BLE001 - keep the run alive, record failure
            print(f"[{split_name}] [{index}/{total}] {task_id} FAILED: {exc}")
            continue

        rows.append(row)
        append_checkpoint(checkpoint_path, row)
        done_ids.add(str(task_id))
        processed_this_run += 1

        if index % max(1, checkpoint_every) == 0 or index == total:
            elapsed = time.perf_counter() - t0
            rate = processed_this_run / elapsed if elapsed > 0 else 0.0
            print(
                f"[{split_name}] [{index}/{total}] {task_id} | "
                f"cheap={row['cheap_score']:.2f} exp={row['expensive_score']:.2f} "
                f"label={row['oracle_score_label']} | "
                f"{rate:.2f} tasks/s"
            )

    # Final CSV for this split.
    csv_path = output_dir / f"{split_name}_features.csv"
    write_features_csv(rows, csv_path)
    if checkpoint_path.exists():
        write_checkpoint_rows(checkpoint_path, rows)
    return rows, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Build the enriched router dataset v2 (labels + probing features)."
    )
    parser.add_argument("--cheap-model", default=None)
    parser.add_argument("--expensive-model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cheap-device", default=None)
    parser.add_argument("--expensive-device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--prompt-format", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--train-data", default="data/mbpp_train_257.jsonl")
    parser.add_argument("--val-data", default="data/mbpp_val_85.jsonl")
    parser.add_argument("--test-data", default="data/mbpp_test_85.jsonl")
    parser.add_argument(
        "--split-manifest",
        default="data/mbpp_split_manifest.jsonl",
        help="Fixed split manifest used to reject accidental overlap/mismatch.",
    )
    parser.add_argument("--output-dir", default="results/router_dataset_v2")
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated subset of train,val,test to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tasks per split (smoke testing).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="Print progress every N tasks (checkpoint is per-task).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint files and start fresh.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AdaptiveGenerationConfig(
        cheap_model_name=args.cheap_model or AdaptiveGenerationConfig.cheap_model_name,
        expensive_model_name=args.expensive_model or AdaptiveGenerationConfig.expensive_model_name,
        device=args.device,
        cheap_device=args.cheap_device,
        expensive_device=args.expensive_device,
        torch_dtype=args.torch_dtype,
        prompt_format=args.prompt_format,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    cheap_model, expensive_model, tokenizer, device = load_adaptive_models(config)
    cheap_runtime = get_model_runtime_metadata(cheap_model, fallback_device=device)
    expensive_runtime = get_model_runtime_metadata(expensive_model, fallback_device=device)
    cheap_tokenizer = get_cheap_tokenizer(tokenizer)
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)
    model_metadata = {
        "cheap_model_name": config.cheap_model_name,
        "expensive_model_name": config.expensive_model_name,
        "device": (
            cheap_runtime["device"]
            if cheap_runtime["device"] == expensive_runtime["device"]
            else "split"
        ),
        "cheap_device": cheap_runtime["device"],
        "expensive_device": expensive_runtime["device"],
        "torch_dtype": cheap_runtime["torch_dtype"],
        "prompt_format": args.prompt_format,
    }
    models_bundle = (
        cheap_model,
        expensive_model,
        tokenizer,
        device,
        cheap_runtime["device"],
        expensive_runtime["device"],
        cheap_tokenizer,
        expensive_tokenizer,
        model_metadata,
    )

    split_data_paths = {
        "train": Path(args.train_data),
        "val": Path(args.val_data),
        "test": Path(args.test_data),
    }
    split_assignments = load_split_manifest(Path(args.split_manifest))
    requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    invalid_splits = sorted(set(requested) - set(split_data_paths))
    if invalid_splits:
        raise ValueError(
            "Unknown splits: " + ", ".join(invalid_splits)
        )

    print("Router Dataset v2 builder")
    print("=" * 72)
    print(f"cheap     : {config.cheap_model_name} @ {cheap_runtime['device']}")
    print(f"expensive : {config.expensive_model_name} @ {expensive_runtime['device']}")
    print(f"dtype     : {cheap_runtime['torch_dtype']}")
    print(f"max_new_tokens : {args.max_new_tokens}")
    print(f"splits    : {', '.join(requested)}")
    print(f"output    : {output_dir}")
    print("=" * 72)

    all_rows: list[dict] = []
    for split in requested:
        data_path = split_data_paths[split]
        if not data_path.exists():
            print(f"[{split}] skipping, file not found: {data_path}")
            continue
        tasks = load_tasks_jsonl(data_path)
        validate_split_tasks(split, tasks, split_assignments)
        print(f"[{split}] {len(tasks)} tasks from {data_path}")
        rows, csv_path = run_split(
            split_name=split,
            tasks=tasks,
            output_dir=output_dir,
            resume=not args.no_resume,
            models_bundle=models_bundle,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            prompt_format=args.prompt_format,
            top_k=args.top_k,
            checkpoint_every=args.checkpoint_every,
            limit=args.limit,
        )
        all_rows.extend(rows)
        print(f"[{split}] wrote {len(rows)} rows -> {csv_path}")

    if all_rows:
        all_csv = output_dir / "all_features.csv"
        write_features_csv(all_rows, all_csv)
        print(f"[all] wrote {len(all_rows)} rows -> {all_csv}")

    # Quick label balance summary.
    if all_rows:
        from collections import Counter
        by_split_label = Counter(
            (r["split"], r["oracle_score_label"]) for r in all_rows
        )
        print("-" * 72)
        print("oracle_score_label balance by split:")
        for (split, label), count in sorted(by_split_label.items()):
            print(f"  {split:<5} {label:<14} : {count}")


if __name__ == "__main__":
    main()
