import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from gear_llm.adaptive_generator import load_adaptive_models
from gear_llm.report import save_csv


ENTROPY_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
MARGIN_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass
class TeacherCalibrationConfig:
    cheap_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    expensive_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    max_steps: int = 40
    top_k: int = 5
    temperature: float = 0.7
    cheap_cost: float = 0.35
    expensive_cost: float = 1.00


def _clean_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t")


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _format_optional(value, decimals: int = 4) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return "sim" if value else "não"

    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"

    return str(value)


def load_teacher_models(config: TeacherCalibrationConfig):
    return load_adaptive_models(config)


def distribution_stats(
    logits: torch.Tensor,
    top_k: int,
    temperature: float,
) -> dict:
    safe_temperature = max(temperature, 1e-6)
    scaled_logits = logits.float() / safe_temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))

    entropy = -(probs * log_probs).sum()
    normalized_entropy = entropy / math.log(probs.numel())

    top_probs, top_ids = torch.topk(probs, k=max(top_k, 2))

    top1_prob = float(top_probs[0].detach().cpu())
    top2_prob = float(top_probs[1].detach().cpu())
    margin = top1_prob - top2_prob

    return {
        "probs": probs,
        "top_ids": top_ids,
        "top_probs": top_probs,
        "top1_id": int(top_ids[0].detach().cpu()),
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "margin": margin,
        "entropy": float(normalized_entropy.detach().cpu()),
    }


def approximate_topk_kl(
    cheap_probs: torch.Tensor,
    expensive_probs: torch.Tensor,
    cheap_top_ids: torch.Tensor,
    expensive_top_ids: torch.Tensor,
) -> float:
    """
    KL aproximada no conjunto união dos top-k tokens dos dois modelos.

    Calculamos KL(expensive || cheap) após renormalizar as massas nesse
    pequeno conjunto. É uma aproximação didática, não uma KL full-vocab.
    """

    union_ids = sorted(
        set(int(token_id.detach().cpu()) for token_id in cheap_top_ids)
        | set(int(token_id.detach().cpu()) for token_id in expensive_top_ids)
    )

    if not union_ids:
        return 0.0

    device = expensive_probs.device
    index = torch.tensor(union_ids, device=device)
    teacher = expensive_probs[index].clamp_min(1e-12)
    student = cheap_probs[index].clamp_min(1e-12)

    teacher = teacher / teacher.sum()
    student = student / student.sum()

    kl = (teacher * (torch.log(teacher) - torch.log(student))).sum()
    return float(kl.detach().cpu())


@torch.no_grad()
def collect_teacher_calibration_rows(
    prompt: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    config: TeacherCalibrationConfig,
    prompt_name: str = "",
) -> list[dict]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    rows = []

    for step in range(config.max_steps):
        cheap_outputs = cheap_model(input_ids=input_ids, return_dict=True)
        expensive_outputs = expensive_model(input_ids=input_ids, return_dict=True)

        cheap_stats = distribution_stats(
            logits=cheap_outputs.logits[0, -1],
            top_k=config.top_k,
            temperature=config.temperature,
        )
        expensive_stats = distribution_stats(
            logits=expensive_outputs.logits[0, -1],
            top_k=config.top_k,
            temperature=config.temperature,
        )

        cheap_top1_id = cheap_stats["top1_id"]
        expensive_top1_id = expensive_stats["top1_id"]
        expensive_topk_ids = [
            int(token_id.detach().cpu())
            for token_id in expensive_stats["top_ids"][: config.top_k]
        ]
        exact_match = cheap_top1_id == expensive_top1_id
        topk_match = cheap_top1_id in expensive_topk_ids
        cheap_rank = (
            expensive_topk_ids.index(cheap_top1_id) + 1
            if topk_match
            else None
        )
        topk_kl = approximate_topk_kl(
            cheap_probs=cheap_stats["probs"],
            expensive_probs=expensive_stats["probs"],
            cheap_top_ids=cheap_stats["top_ids"][: config.top_k],
            expensive_top_ids=expensive_stats["top_ids"][: config.top_k],
        )

        cheap_token = tokenizer.decode(
            [cheap_top1_id],
            clean_up_tokenization_spaces=False,
        )
        expensive_token = tokenizer.decode(
            [expensive_top1_id],
            clean_up_tokenization_spaces=False,
        )
        expensive_topk_tokens = [
            _clean_token(
                tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                )
            )
            for token_id in expensive_topk_ids
        ]

        rows.append(
            {
                "prompt_name": prompt_name,
                "step": step,
                "cheap_entropy": cheap_stats["entropy"],
                "cheap_top1_prob": cheap_stats["top1_prob"],
                "cheap_top2_prob": cheap_stats["top2_prob"],
                "cheap_margin": cheap_stats["margin"],
                "cheap_token_id": cheap_top1_id,
                "cheap_token": _clean_token(cheap_token),
                "expensive_top1_token_id": expensive_top1_id,
                "expensive_top1_token": _clean_token(expensive_token),
                "expensive_topk_token_ids": " ".join(
                    str(token_id) for token_id in expensive_topk_ids
                ),
                "expensive_topk_tokens": " | ".join(expensive_topk_tokens),
                "exact_match": exact_match,
                "topk_match": topk_match,
                "cheap_rank_in_expensive": cheap_rank,
                "topk_kl_expensive_to_cheap": topk_kl,
            }
        )

        next_token = torch.tensor([[expensive_top1_id]], device=device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if (
            tokenizer.eos_token_id is not None
            and expensive_top1_id == tokenizer.eos_token_id
        ):
            break

    return rows


def summarize_teacher_rows(rows: list[dict]) -> dict:
    total_steps = len(rows)
    exact_rows = [row for row in rows if row["exact_match"]]
    mismatch_rows = [row for row in rows if not row["exact_match"]]
    topk_rows = [row for row in rows if row["topk_match"]]

    return {
        "total_steps": total_steps,
        "exact_match_rate": (
            len(exact_rows) / total_steps if total_steps else 0.0
        ),
        "topk_match_rate": (
            len(topk_rows) / total_steps if total_steps else 0.0
        ),
        "mean_entropy_exact_match": _safe_mean(
            [row["cheap_entropy"] for row in exact_rows]
        ),
        "mean_entropy_not_exact_match": _safe_mean(
            [row["cheap_entropy"] for row in mismatch_rows]
        ),
        "mean_margin_exact_match": _safe_mean(
            [row["cheap_margin"] for row in exact_rows]
        ),
        "mean_margin_not_exact_match": _safe_mean(
            [row["cheap_margin"] for row in mismatch_rows]
        ),
    }


def threshold_grid_search(
    rows: list[dict],
    config: TeacherCalibrationConfig,
    prompt_name: str = "",
) -> list[dict]:
    total_steps = len(rows)
    grid_rows = []

    for entropy_threshold in ENTROPY_THRESHOLDS:
        for margin_threshold in MARGIN_THRESHOLDS:
            accepted = [
                row
                for row in rows
                if row["cheap_entropy"] <= entropy_threshold
                and row["cheap_margin"] >= margin_threshold
            ]
            accepted_count = len(accepted)
            matched_accepts = [
                row
                for row in accepted
                if row["exact_match"] or row["topk_match"]
            ]
            false_accepts = accepted_count - len(matched_accepts)
            expensive_calls = total_steps - accepted_count

            baseline_cost = total_steps * config.expensive_cost
            adaptive_cost = (
                total_steps * config.cheap_cost
                + expensive_calls * config.expensive_cost
            )
            estimated_saved_percent = (
                100 * (baseline_cost - adaptive_cost) / baseline_cost
                if baseline_cost
                else 0.0
            )

            grid_rows.append(
                {
                    "prompt_name": prompt_name,
                    "entropy_threshold": entropy_threshold,
                    "margin_threshold": margin_threshold,
                    "total_steps": total_steps,
                    "accepted_count": accepted_count,
                    "accept_rate": (
                        accepted_count / total_steps if total_steps else 0.0
                    ),
                    "precision_accept": (
                        len(matched_accepts) / accepted_count
                        if accepted_count
                        else None
                    ),
                    "false_accept_count": false_accepts,
                    "false_accept_rate": (
                        false_accepts / accepted_count
                        if accepted_count
                        else None
                    ),
                    "expensive_calls": expensive_calls,
                    "estimated_saved_percent": estimated_saved_percent,
                }
            )

    return grid_rows


def run_teacher_calibration_with_models(
    prompt: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    config: TeacherCalibrationConfig,
    prompt_name: str = "",
) -> tuple[list[dict], dict, list[dict]]:
    rows = collect_teacher_calibration_rows(
        prompt=prompt,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        config=config,
        prompt_name=prompt_name,
    )
    summary = summarize_teacher_rows(rows)
    summary["prompt_name"] = prompt_name
    grid_rows = threshold_grid_search(
        rows=rows,
        config=config,
        prompt_name=prompt_name,
    )

    return rows, summary, grid_rows


def run_teacher_calibration(
    prompt: str,
    config: TeacherCalibrationConfig,
    prompt_name: str = "",
) -> tuple[list[dict], dict, list[dict]]:
    cheap_model, expensive_model, tokenizer, device = load_teacher_models(config)

    return run_teacher_calibration_with_models(
        prompt=prompt,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        config=config,
        prompt_name=prompt_name,
    )


def print_teacher_summary(summary: dict, grid_rows: list[dict]):
    print()
    print("Teacher Calibration")
    print("=" * 100)

    if summary.get("prompt_name"):
        print(f"prompt_name                 : {summary['prompt_name']}")

    print(f"total_steps                 : {summary['total_steps']}")
    print(f"exact_match_rate            : {summary['exact_match_rate']:.2%}")
    print(f"topk_match_rate             : {summary['topk_match_rate']:.2%}")
    print(
        "mean_entropy exact          : "
        f"{_format_optional(summary['mean_entropy_exact_match'])}"
    )
    print(
        "mean_entropy not exact      : "
        f"{_format_optional(summary['mean_entropy_not_exact_match'])}"
    )
    print(
        "mean_margin exact           : "
        f"{_format_optional(summary['mean_margin_exact_match'])}"
    )
    print(
        "mean_margin not exact       : "
        f"{_format_optional(summary['mean_margin_not_exact_match'])}"
    )
    print()

    candidates = [
        row
        for row in grid_rows
        if row["precision_accept"] is not None
        and row["estimated_saved_percent"] > 0
    ]
    candidates = sorted(
        candidates,
        key=lambda row: (
            row["precision_accept"],
            row["estimated_saved_percent"],
            -row["false_accept_rate"],
        ),
        reverse=True,
    )

    if candidates:
        print("Melhores thresholds com economia positiva")
        print("-" * 100)
        header = (
            f"{'entropy':>8} | {'margin':>6} | {'accept':>7} | "
            f"{'precision':>9} | {'false':>7} | {'saved %':>8}"
        )
        print(header)
        print("-" * len(header))

        for row in candidates[:5]:
            print(
                f"{row['entropy_threshold']:>8.2f} | "
                f"{row['margin_threshold']:>6.2f} | "
                f"{row['accept_rate']:>7.2%} | "
                f"{row['precision_accept']:>9.2%} | "
                f"{row['false_accept_rate']:>7.2%} | "
                f"{row['estimated_saved_percent']:>7.2f}%"
            )

    print("=" * 100)
    print()


def save_teacher_rows(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))


def save_teacher_grid(grid_rows: list[dict], path: str | Path):
    save_csv(grid_rows, str(path))
