"""Build a fixed, non-overlapping train/validation/test split over all MBPP tasks.

This is the methodological foundation for the learned prompt router (v2). Random
seeds do not guarantee independent datasets: the seed123/seed999 comparison in
PROBLEMS.md showed 20 overlapping prompts. This script persists one deterministic
split keyed on the *stable original MBPP task_id* (not the list position), so the
same split is reproducible across machines and runs.

Split sizes over all 427 MBPP tasks (stratified by difficulty):
    train: 257
    val  : 85
    test : 85

Outputs:
    data/mbpp_split_manifest.jsonl     -- {mbpp_task_id, split, difficulty, index}
    data/mbpp_train_257.jsonl          -- eval-ready tasks (convert_mbpp schema)
    data/mbpp_val_85.jsonl
    data/mbpp_test_85.jsonl

Sanity checks (asserted before writing): zero pairwise overlap, sizes sum to 427.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Reuse the canonical MBPP normalization from the existing builder instead of
# duplicating the logic. These helpers are pure functions over a raw record.
# scripts/ has no __init__.py, so add its directory to sys.path and import the
# module directly. This keeps the script runnable both as
# `python scripts/build_mbpp_split.py` and `python -m scripts.build_mbpp_split`.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from build_external_eval_tasks import (  # noqa: E402
    convert_mbpp,
    mbpp_difficulty,
    normalize_mbpp_tests,
    infer_function_name,
)


SOURCE_PATH = Path("data/external_sources/mbpp/sanitized-mbpp.json")
MANIFEST_PATH = Path("data/mbpp_split_manifest.jsonl")
SPLIT_PATHS = {
    "train": Path("data/mbpp_train_257.jsonl"),
    "val": Path("data/mbpp_val_85.jsonl"),
    "test": Path("data/mbpp_test_85.jsonl"),
}
# Fixed, announced in advance. Do not change: it would invalidate every
# downstream artifact and every reported number.
DEFAULT_SEED = 20260714
TARGET_SIZES = {"train": 257, "val": 85, "test": 85}
TOTAL_TARGET = sum(TARGET_SIZES.values())  # 427


def load_raw_mbpp(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} did not contain a JSON list.")
    return data


def stable_task_id(record: dict, list_index: int) -> int:
    """Return the original MBPP task_id, falling back to list index if absent.

    The raw sanitized-mbpp.json carries an integer `task_id` on every record
    (e.g. the first task has task_id=2). That field is the stable identifier
    across re-downloads; the list position is not.
    """
    raw = record.get("task_id")
    if raw is None:
        return list_index
    try:
        return int(raw)
    except (TypeError, ValueError):
        return list_index


def difficulty_for(record: dict) -> str:
    """Compute difficulty for a raw MBPP record, mirroring convert_mbpp."""
    text = (
        record.get("text")
        or record.get("prompt")
        or record.get("task")
        or record.get("description")
    )
    raw_tests = record.get("test_list") or record.get("tests") or []
    tests = normalize_mbpp_tests(raw_tests)
    return mbpp_difficulty(str(text or ""), tests)


def stratified_split(
    items: list[dict],
    target_sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict]]:
    """Split items into named buckets, stratified by their `difficulty` key.

    Each item must carry `mbpp_task_id` and `difficulty`. Within each difficulty
    stratum we shuffle once (fixed seed) and assign to splits in proportion to
    the global target sizes. This keeps the difficulty distribution balanced
    across train/val/test, which matters because MBPP difficulty is correlated
    with the cheap/expensive gap.
    """
    rng = random.Random(seed)

    # Group indices by difficulty.
    by_difficulty: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        by_difficulty.setdefault(item["difficulty"], []).append(index)

    total = len(items)
    fractions = {split: size / total for split, size in target_sizes.items()}

    assignments: dict[str, list[int]] = {split: [] for split in target_sizes}
    for difficulty, indices in sorted(by_difficulty.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n = len(shuffled)
        # Target counts for this stratum, proportional to the global split.
        stratum_targets = {
            split: int(round(fractions[split] * n)) for split in target_sizes
        }
        # Correct rounding drift onto the largest split (train).
        drift = n - sum(stratum_targets.values())
        if drift:
            stratum_targets["train"] += drift

        cursor = 0
        # Assign val and test first (they are the smaller, more sensitive
        # buckets), then give the remainder to train.
        for split in ("val", "test"):
            count = stratum_targets[split]
            assignments[split].extend(shuffled[cursor : cursor + count])
            cursor += count
        assignments["train"].extend(shuffled[cursor:])

    return {
        split: [items[i] for i in assignments[split]] for split in target_sizes
    }


def assert_no_overlap(splits: dict[str, list[dict]]):
    ids = {
        split: {item["mbpp_task_id"] for item in items}
        for split, items in splits.items()
    }
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = ids[a] & ids[b]
            assert not overlap, (
                f"Overlap between {a} and {b}: {sorted(overlap)[:5]} ..."
            )
    total = sum(len(s) for s in ids.values())
    assert total == TOTAL_TARGET, (
        f"Split sizes sum to {total}, expected {TOTAL_TARGET}. "
        f"Counts: {', '.join(f'{k}={len(v)}' for k, v in ids.items())}"
    )
    # Also assert each split hit its target exactly.
    for split, target in TARGET_SIZES.items():
        assert len(ids[split]) == target, (
            f"{split} has {len(ids[split])} tasks, expected {target}"
        )


def write_manifest(splits: dict[str, list[dict]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, items in splits.items():
        for position, item in enumerate(items):
            rows.append(
                {
                    "mbpp_task_id": item["mbpp_task_id"],
                    "split": split,
                    "difficulty": item["difficulty"],
                    # Position within the split: useful for stable sub-sampling.
                    "index": position,
                }
            )
    # Sort by task_id for deterministic file layout.
    rows.sort(key=lambda r: r["mbpp_task_id"])
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_split_jsonl(items: list[dict], path: Path):
    """Write eval-ready tasks. The id embeds the STABLE original task_id, so the
    same task keeps the same id across re-runs (unlike convert_mbpp's list-index
    id). Downstream (build_router_dataset_v2.py, task_evaluation.py) joins on id.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            record = item["record"]
            task_id = item["mbpp_task_id"]
            converted = convert_mbpp(record, task_id)
            if converted is None:
                # convert_mbpp drops records without text/tests/function_name.
                # Keep them visible so the count is auditable.
                raise RuntimeError(
                    f"convert_mbpp returned None for task_id={task_id}; "
                    "the raw record is missing text/tests/function_name."
                )
            # Override the id to guarantee it is keyed on the stable task_id
            # (convert_mbpp already received task_id as `index`, but this makes
            # the intent explicit and robust to future changes).
            converted["id"] = f"mbpp_task{task_id}"
            converted["mbpp_task_id"] = task_id
            file.write(json.dumps(converted, ensure_ascii=False) + "\n")


def print_summary(
    splits: dict[str, list[dict]], raw_count: int, skipped: int
):
    print("MBPP Fixed Split")
    print("=" * 72)
    print(f"raw records           : {raw_count}")
    print(f"skipped (no task_id)  : {skipped}")
    total = sum(len(v) for v in splits.values())
    print(f"split total           : {total}")
    for split in ("train", "val", "test"):
        items = splits[split]
        diff_counts = {}
        for item in items:
            diff_counts[item["difficulty"]] = diff_counts.get(item["difficulty"], 0) + 1
        diff_str = ", ".join(
            f"{d}={c}" for d, c in sorted(diff_counts.items())
        )
        print(f"  {split:<5} ({len(items):>3}): {diff_str}")
    print(f"seed                  : {DEFAULT_SEED}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Build a fixed non-overlapping MBPP train/val/test split."
    )
    parser.add_argument(
        "--source",
        default=str(SOURCE_PATH),
        help="Path to sanitized-mbpp.json.",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Output manifest JSONL path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed. Changing this invalidates downstream artifacts.",
    )
    args = parser.parse_args()

    raw = load_raw_mbpp(Path(args.source))

    # Build split items keyed on the stable task_id.
    seen_ids = set()
    items: list[dict] = []
    skipped = 0
    for list_index, record in enumerate(raw):
        task_id = stable_task_id(record, list_index)
        if task_id in seen_ids:
            # Defensive: dedupe on task_id in case the source has collisions.
            skipped += 1
            continue
        seen_ids.add(task_id)
        difficulty = difficulty_for(record)
        items.append(
            {
                "mbpp_task_id": task_id,
                "difficulty": difficulty,
                "record": record,
            }
        )

    if len(items) != TOTAL_TARGET:
        print(
            f"WARNING: found {len(items)} usable tasks, expected {TOTAL_TARGET}. "
            "Split targets are fixed; proceeding with the available pool."
        )
        # Adjust targets proportionally if the pool differs, but keep val/test
        # as large as possible (they bound the statistical signal).
        if len(items) < TOTAL_TARGET:
            raise SystemExit(
                f"Only {len(items)} MBPP tasks available; cannot form the "
                f"announced {TOTAL_TARGET}-task split. Check the source file."
            )

    splits = stratified_split(items, TARGET_SIZES, seed=args.seed)
    assert_no_overlap(splits)

    manifest_path = Path(args.manifest)
    write_manifest(splits, manifest_path)
    for split, path in SPLIT_PATHS.items():
        write_split_jsonl(splits[split], path)

    print_summary(splits, raw_count=len(raw), skipped=skipped)
    print(f"manifest -> {manifest_path}")
    for split, path in SPLIT_PATHS.items():
        print(f"{split:<5}     -> {path}")


if __name__ == "__main__":
    main()
