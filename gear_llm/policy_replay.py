import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from gear_llm.report import save_csv


@dataclass(frozen=True)
class ReplayPolicy:
    name: str
    entropy_threshold: float
    margin_threshold: float


POLICIES = (
    ReplayPolicy("old_0.45_0.20", entropy_threshold=0.45, margin_threshold=0.20),
    ReplayPolicy(
        "calibrated_0.35_0.20",
        entropy_threshold=0.35,
        margin_threshold=0.20,
    ),
    ReplayPolicy("strict_0.30_0.20", entropy_threshold=0.30, margin_threshold=0.20),
    ReplayPolicy("loose_0.50_0.15", entropy_threshold=0.50, margin_threshold=0.15),
)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "sim"}


def load_teacher_rows(path: str | Path) -> list[dict]:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path}. Rode antes: "
            "python benchmark.py --teacher-calibration"
        )

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []

        for row in reader:
            rows.append(
                {
                    "prompt_name": row.get("prompt_name") or "unknown",
                    "cheap_entropy": float(row["cheap_entropy"]),
                    "cheap_margin": float(row["cheap_margin"]),
                    "exact_match": parse_bool(row["exact_match"]),
                    "topk_match": parse_bool(row["topk_match"]),
                }
            )

    return rows


def accepted_by_policy(row: dict, policy: ReplayPolicy) -> bool:
    return (
        row["cheap_entropy"] <= policy.entropy_threshold
        and row["cheap_margin"] >= policy.margin_threshold
    )


def replay_policy(
    rows: list[dict],
    policy: ReplayPolicy,
    prompt_name: str,
    cheap_cost: float = 0.35,
    expensive_cost: float = 1.00,
) -> dict:
    total_steps = len(rows)
    accepted_rows = [row for row in rows if accepted_by_policy(row, policy)]
    accepted_steps = len(accepted_rows)
    fallback_steps = total_steps - accepted_steps

    exact_matches = sum(1 for row in accepted_rows if row["exact_match"])
    topk_matches = sum(1 for row in accepted_rows if row["topk_match"])
    false_accepts = accepted_steps - topk_matches

    baseline_cost = total_steps * expensive_cost
    simulated_cost = total_steps * cheap_cost + fallback_steps * expensive_cost
    estimated_saved_percent = (
        (baseline_cost - simulated_cost) / baseline_cost * 100
        if baseline_cost
        else 0.0
    )

    return {
        "policy_name": policy.name,
        "prompt_name": prompt_name,
        "entropy_threshold": policy.entropy_threshold,
        "margin_threshold": policy.margin_threshold,
        "total_steps": total_steps,
        "accepted_steps": accepted_steps,
        "fallback_steps": fallback_steps,
        "accept_rate": accepted_steps / total_steps if total_steps else 0.0,
        "exact_precision_accept": (
            exact_matches / accepted_steps if accepted_steps else None
        ),
        "topk_precision_accept": (
            topk_matches / accepted_steps if accepted_steps else None
        ),
        "false_accept_rate": (
            false_accepts / accepted_steps if accepted_steps else None
        ),
        "estimated_saved_percent": estimated_saved_percent,
    }


def run_policy_replay_from_rows(
    rows: list[dict],
    policies: tuple[ReplayPolicy, ...] = POLICIES,
    cheap_cost: float = 0.35,
    expensive_cost: float = 1.00,
) -> list[dict]:
    grouped_rows: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped_rows[row["prompt_name"]].append(row)

    grouped_rows["ALL"] = list(rows)

    results = []

    for prompt_name in sorted(grouped_rows):
        for policy in policies:
            results.append(
                replay_policy(
                    rows=grouped_rows[prompt_name],
                    policy=policy,
                    prompt_name=prompt_name,
                    cheap_cost=cheap_cost,
                    expensive_cost=expensive_cost,
                )
            )

    return results


def run_policy_replay(
    teacher_csv: str | Path = "results/teacher_calibration.csv",
    cheap_cost: float = 0.35,
    expensive_cost: float = 1.00,
) -> list[dict]:
    rows = load_teacher_rows(teacher_csv)

    return run_policy_replay_from_rows(
        rows=rows,
        cheap_cost=cheap_cost,
        expensive_cost=expensive_cost,
    )


def save_policy_replay(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))


def _format_percent(value) -> str:
    if value is None:
        return "N/A"

    return f"{100 * value:>6.2f}%"


def print_policy_replay_report(rows: list[dict]):
    print()
    print("Policy Replay")
    print("=" * 120)

    header = (
        f"{'policy':<23} | {'prompt':<14} | {'accept':>8} | "
        f"{'exact':>8} | {'topk':>8} | {'false':>8} | {'saved %':>8}"
    )
    print(header)
    print("-" * len(header))

    prompt_order = ["ALL", "easy", "math", "logic_negation", "code", "long_simple"]

    def sort_key(row: dict):
        prompt = row["prompt_name"]
        prompt_index = (
            prompt_order.index(prompt)
            if prompt in prompt_order
            else len(prompt_order)
        )
        policy_index = next(
            index
            for index, policy in enumerate(POLICIES)
            if policy.name == row["policy_name"]
        )
        return prompt_index, policy_index

    for row in sorted(rows, key=sort_key):
        print(
            f"{row['policy_name']:<23} | "
            f"{row['prompt_name']:<14} | "
            f"{_format_percent(row['accept_rate']):>8} | "
            f"{_format_percent(row['exact_precision_accept']):>8} | "
            f"{_format_percent(row['topk_precision_accept']):>8} | "
            f"{_format_percent(row['false_accept_rate']):>8} | "
            f"{row['estimated_saved_percent']:>7.2f}%"
        )

    print("=" * 120)
    print()
