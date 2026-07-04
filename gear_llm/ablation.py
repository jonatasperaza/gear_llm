import math
import random
import statistics
from pathlib import Path

import torch

from gear_llm.analyzer import analyze_prompt_with_model
from gear_llm.config import ModelConfig, RouterConfig
from gear_llm.model_loader import load_model_and_tokenizer
from gear_llm.report import save_csv


ROUTES = ("cheap", "medium", "expensive")


def safe_perplexity(loss: float) -> float:
    """
    Converte loss em perplexity sem explodir em casos extremos.
    """

    return math.exp(min(loss, 50.0))


def neutral_token_id(tokenizer, neutral_text: str = " ") -> int:
    """
    Escolhe um token neutro para substituir tokens removidos.

    Por padrão tentamos um espaço. Se o tokenizer não produzir ID para isso,
    usamos pad/eos/unk como fallback.
    """

    token_ids = tokenizer.encode(neutral_text, add_special_tokens=False)

    if token_ids:
        return token_ids[0]

    for candidate in (
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
    ):
        if candidate is not None:
            return candidate

    raise ValueError("Não foi possível encontrar um token neutro para ablation.")


@torch.no_grad()
def causal_loss_from_ids(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    """
    Calcula loss/perplexity causal.

    input_ids podem estar ablated, mas labels podem continuar sendo os IDs
    originais. Isso mede quanto a corrupção do contexto degrada o prompt real.
    """

    if input_ids.numel() < 2:
        return 0.0, 1.0

    input_ids = input_ids.to(device).unsqueeze(0)
    labels = labels.to(device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )

    loss = float(outputs.loss.detach().cpu())
    return loss, safe_perplexity(loss)


def ablate_ids_by_route(
    input_ids: torch.Tensor,
    rows: list[dict],
    route: str,
    replacement_id: int,
) -> tuple[torch.Tensor, list[int]]:
    """
    Substitui por token neutro todos os tokens classificados como route.
    """

    ablated_ids = input_ids.clone()
    selected_indices: list[int] = []

    for row in rows:
        if row["route"] != route:
            continue

        index = int(row["index"])

        if index >= ablated_ids.numel():
            continue

        ablated_ids[index] = replacement_id
        selected_indices.append(index)

    return ablated_ids, selected_indices


def ablate_ids_by_indices(
    input_ids: torch.Tensor,
    indices: list[int],
    replacement_id: int,
) -> torch.Tensor:
    """
    Substitui índices específicos por um token neutro.
    """

    ablated_ids = input_ids.clone()

    for index in indices:
        if 0 <= index < ablated_ids.numel():
            ablated_ids[index] = replacement_id

    return ablated_ids


def summarize_selected_rows(rows: list[dict], selected_indices: list[int]) -> dict:
    selected = [rows[index] for index in selected_indices if index < len(rows)]

    if not selected:
        return {
            "token_count": 0,
            "mean_rho": 0.0,
            "tokens_preview": "",
        }

    mean_rho = sum(row["rho"] for row in selected) / len(selected)
    tokens_preview = "".join(row["token"] for row in selected[:12])
    tokens_preview = tokens_preview.replace("\n", "\\n").replace("\t", "\\t")

    return {
        "token_count": len(selected),
        "mean_rho": mean_rho,
        "tokens_preview": tokens_preview,
    }


def _select_indices(rows: list[dict], route: str, k: int, reverse: bool) -> list[int]:
    candidates = [row for row in rows if row["route"] == route]

    if len(candidates) < k:
        return []

    selected = sorted(candidates, key=lambda row: row["rho"], reverse=reverse)[:k]
    return [int(row["index"]) for row in selected]


def _index_preview(rows: list[dict], indices: list[int]) -> str:
    index_set = set(indices)
    tokens = [
        row["token"]
        for row in rows
        if int(row["index"]) in index_set
    ]
    preview = "".join(tokens[:12])
    return preview.replace("\n", "\\n").replace("\t", "\\t")


def _evaluate_index_group(
    *,
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    indices: list[int],
    replacement_id: int,
    original_loss: float,
    device: str,
) -> tuple[float, float, float, float]:
    ablated_ids = ablate_ids_by_indices(
        input_ids=input_ids,
        indices=indices,
        replacement_id=replacement_id,
    )
    loss, perplexity = causal_loss_from_ids(
        model=model,
        input_ids=ablated_ids,
        labels=labels,
        device=device,
    )
    delta_loss = loss - original_loss
    delta_per_token = delta_loss / len(indices) if indices else 0.0

    return loss, perplexity, delta_loss, delta_per_token


def run_ablation_with_model(
    prompt: str,
    model,
    tokenizer,
    device: str,
    router_config: RouterConfig,
    neutral_text: str = " ",
) -> dict:
    """
    Roda ablation cheap/medium/expensive para um prompt.

    A sequência ablated preserva o número de tokens: os tokens da classe-alvo
    são substituídos por um token neutro, mas a loss é medida contra os labels
    originais do prompt.
    """

    rows = analyze_prompt_with_model(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        router_config=router_config,
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"][0].to(device)
    original_labels = input_ids.clone()

    original_loss, original_perplexity = causal_loss_from_ids(
        model=model,
        input_ids=input_ids,
        labels=original_labels,
        device=device,
    )

    replacement_id = neutral_token_id(tokenizer, neutral_text=neutral_text)
    replacement_text = tokenizer.decode(
        [replacement_id],
        clean_up_tokenization_spaces=False,
    )

    summary = {
        "prompt": prompt,
        "neutral_text": neutral_text,
        "replacement_token": replacement_text,
        "replacement_token_id": replacement_id,
        "original_loss": original_loss,
        "original_perplexity": original_perplexity,
    }

    for route in ROUTES:
        ablated_ids, selected_indices = ablate_ids_by_route(
            input_ids=input_ids,
            rows=rows,
            route=route,
            replacement_id=replacement_id,
        )
        ablated_loss, ablated_perplexity = causal_loss_from_ids(
            model=model,
            input_ids=ablated_ids,
            labels=original_labels,
            device=device,
        )
        selected_stats = summarize_selected_rows(rows, selected_indices)
        token_count = selected_stats["token_count"]
        delta_loss = ablated_loss - original_loss

        summary[f"{route}_token_count"] = token_count
        summary[f"{route}_mean_rho"] = selected_stats["mean_rho"]
        summary[f"{route}_tokens_preview"] = selected_stats["tokens_preview"]
        summary[f"{route}_removed_loss"] = ablated_loss
        summary[f"{route}_removed_perplexity"] = ablated_perplexity
        summary[f"{route}_delta_loss"] = delta_loss
        summary[f"{route}_delta_loss_per_token"] = (
            delta_loss / token_count if token_count else 0.0
        )

    criterion_applicable = (
        summary["cheap_token_count"] > 0
        and summary["expensive_token_count"] > 0
    )
    criterion_passed_raw = (
        summary["expensive_delta_loss"] > summary["cheap_delta_loss"]
        if criterion_applicable
        else False
    )
    criterion_passed_per_token = (
        summary["expensive_delta_loss_per_token"]
        > summary["cheap_delta_loss_per_token"]
        if criterion_applicable
        else False
    )

    if not criterion_applicable:
        criterion_status = "not_applicable"
    elif criterion_passed_raw:
        criterion_status = "passed"
    else:
        criterion_status = "failed"

    summary["criterion_applicable"] = criterion_applicable
    summary["criterion_passed"] = criterion_passed_raw
    summary["criterion_passed_raw"] = criterion_passed_raw
    summary["criterion_passed_per_token"] = criterion_passed_per_token
    summary["criterion_status"] = criterion_status

    return summary


def run_balanced_ablation_with_model(
    prompt: str,
    model,
    tokenizer,
    device: str,
    router_config: RouterConfig,
    neutral_text: str = " ",
    random_trials: int = 20,
    seed: int = 42,
) -> dict:
    """
    Roda ablation balanceada.

    O tamanho de todos os grupos é k, definido pelo número de tokens expensive.
    Isso evita comparar delta bruto de grupos com quantidades muito diferentes
    de tokens.
    """

    rows = analyze_prompt_with_model(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        router_config=router_config,
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"][0].to(device)
    labels = input_ids.clone()

    original_loss, original_perplexity = causal_loss_from_ids(
        model=model,
        input_ids=input_ids,
        labels=labels,
        device=device,
    )

    route_counts = {
        route: sum(1 for row in rows if row["route"] == route)
        for route in ROUTES
    }
    k = route_counts["expensive"]
    replacement_id = neutral_token_id(tokenizer, neutral_text=neutral_text)
    replacement_text = tokenizer.decode(
        [replacement_id],
        clean_up_tokenization_spaces=False,
    )

    summary = {
        "prompt": prompt,
        "neutral_text": neutral_text,
        "replacement_token": replacement_text,
        "replacement_token_id": replacement_id,
        "original_loss": original_loss,
        "original_perplexity": original_perplexity,
        "k": k,
        "random_trials": random_trials,
        "random_seed": seed,
        "cheap_token_count": route_counts["cheap"],
        "medium_token_count": route_counts["medium"],
        "expensive_token_count": route_counts["expensive"],
        "balanced_status": "ok" if k > 0 else "not_applicable",
    }

    group_specs = {
        "expensive": _select_indices(rows, "expensive", k, reverse=True),
        "cheap": _select_indices(rows, "cheap", k, reverse=False),
        "medium": _select_indices(rows, "medium", k, reverse=True),
    }

    for group_name, indices in group_specs.items():
        available = k > 0 and len(indices) == k
        summary[f"{group_name}_available"] = available
        summary[f"{group_name}_indices"] = " ".join(str(index) for index in indices)
        summary[f"{group_name}_tokens_preview"] = _index_preview(rows, indices)

        if not available:
            summary[f"{group_name}_loss"] = None
            summary[f"{group_name}_perplexity"] = None
            summary[f"{group_name}_delta_loss"] = None
            summary[f"{group_name}_delta_per_token"] = None
            continue

        loss, perplexity, delta_loss, delta_per_token = _evaluate_index_group(
            model=model,
            input_ids=input_ids,
            labels=labels,
            indices=indices,
            replacement_id=replacement_id,
            original_loss=original_loss,
            device=device,
        )
        summary[f"{group_name}_loss"] = loss
        summary[f"{group_name}_perplexity"] = perplexity
        summary[f"{group_name}_delta_loss"] = delta_loss
        summary[f"{group_name}_delta_per_token"] = delta_per_token

    valid_indices = [int(row["index"]) for row in rows]
    random_delta_per_token_values: list[float] = []
    rng = random.Random(seed)

    if k > 0 and len(valid_indices) >= k:
        for _ in range(random_trials):
            indices = rng.sample(valid_indices, k)
            _, _, _, delta_per_token = _evaluate_index_group(
                model=model,
                input_ids=input_ids,
                labels=labels,
                indices=indices,
                replacement_id=replacement_id,
                original_loss=original_loss,
                device=device,
            )
            random_delta_per_token_values.append(delta_per_token)

    random_mean = (
        statistics.mean(random_delta_per_token_values)
        if random_delta_per_token_values
        else None
    )
    random_std = (
        statistics.pstdev(random_delta_per_token_values)
        if len(random_delta_per_token_values) > 1
        else 0.0 if random_delta_per_token_values else None
    )

    summary["random_trials_completed"] = len(random_delta_per_token_values)
    summary["random_mean_delta_per_token"] = random_mean
    summary["random_std_delta_per_token"] = random_std

    expensive_delta = summary["expensive_delta_per_token"]
    cheap_delta = summary["cheap_delta_per_token"]
    medium_delta = summary["medium_delta_per_token"]

    summary["expensive_gt_cheap"] = (
        expensive_delta is not None
        and cheap_delta is not None
        and expensive_delta > cheap_delta
    )
    summary["expensive_gt_medium"] = (
        expensive_delta is not None
        and medium_delta is not None
        and expensive_delta > medium_delta
    )
    summary["expensive_gt_random_mean"] = (
        expensive_delta is not None
        and random_mean is not None
        and expensive_delta > random_mean
    )

    if k == 0:
        summary["balanced_status"] = "not_applicable"
    elif not summary["cheap_available"]:
        summary["balanced_status"] = "not_enough_cheap"
    elif random_mean is None:
        summary["balanced_status"] = "not_enough_random"
    elif summary["expensive_gt_cheap"] and summary["expensive_gt_random_mean"]:
        summary["balanced_status"] = "passed"
    else:
        summary["balanced_status"] = "failed"

    return summary


def run_ablation(
    prompt: str,
    model_config: ModelConfig,
    router_config: RouterConfig,
    neutral_text: str = " ",
) -> dict:
    model, tokenizer, device = load_model_and_tokenizer(model_config.model_name)

    return run_ablation_with_model(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        router_config=router_config,
        neutral_text=neutral_text,
    )


def run_balanced_ablation(
    prompt: str,
    model_config: ModelConfig,
    router_config: RouterConfig,
    neutral_text: str = " ",
    random_trials: int = 20,
    seed: int = 42,
) -> dict:
    model, tokenizer, device = load_model_and_tokenizer(model_config.model_name)

    return run_balanced_ablation_with_model(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        router_config=router_config,
        neutral_text=neutral_text,
        random_trials=random_trials,
        seed=seed,
    )


def save_ablation_csv(summary: dict, path: str | Path):
    save_csv([summary], str(path))


def save_ablation_rows(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))


def print_ablation_report(summary: dict):
    print()
    print("Validação por ablation")
    print("=" * 100)
    print(f"original_loss       : {summary['original_loss']:.4f}")
    print(f"original_perplexity : {summary['original_perplexity']:.4f}")
    print(
        "token neutro        : "
        f"{summary['replacement_token']!r} "
        f"(id={summary['replacement_token_id']})"
    )
    print()

    header = (
        f"{'classe':<10} | {'tokens':>6} | {'rho médio':>9} | "
        f"{'loss':>10} | {'delta':>10} | {'delta/tok':>10} | {'ppl':>10}"
    )
    print(header)
    print("-" * len(header))

    for route in ROUTES:
        print(
            f"{route:<10} | "
            f"{summary[f'{route}_token_count']:>6} | "
            f"{summary[f'{route}_mean_rho']:>9.4f} | "
            f"{summary[f'{route}_removed_loss']:>10.4f} | "
            f"{summary[f'{route}_delta_loss']:>10.4f} | "
            f"{summary[f'{route}_delta_loss_per_token']:>10.4f} | "
            f"{summary[f'{route}_removed_perplexity']:>10.4f}"
        )

    print("=" * 100)

    if not summary["criterion_applicable"]:
        print("Critério bruto: N/A - faltam tokens cheap ou expensive neste prompt.")
    elif summary["criterion_passed_raw"]:
        print("Critério bruto: PASSOU - expensive aumentou a loss mais do que cheap.")
    else:
        print("Critério bruto: NÃO PASSOU - expensive não superou cheap neste prompt.")

    if summary["criterion_applicable"]:
        if summary["criterion_passed_per_token"]:
            print("Critério por token: PASSOU - expensive teve maior delta/tok.")
        else:
            print("Critério por token: NÃO PASSOU - expensive não teve maior delta/tok.")

    print()


def _format_optional(value, decimals: int = 4) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return "sim" if value else "não"

    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"

    return str(value)


def print_balanced_ablation_report(summary: dict):
    print()
    print("Validação balanceada por ablation")
    print("=" * 100)
    print(f"original_loss       : {summary['original_loss']:.4f}")
    print(f"original_perplexity : {summary['original_perplexity']:.4f}")
    print(f"k                   : {summary['k']}")
    print(f"status              : {summary['balanced_status']}")
    print(
        "token neutro        : "
        f"{summary['replacement_token']!r} "
        f"(id={summary['replacement_token_id']})"
    )
    print()

    if summary["k"] == 0:
        print("Resultado N/A: este prompt não teve tokens classificados como expensive.")
        print()
        return

    header = (
        f"{'grupo':<12} | {'loss':>10} | {'delta':>10} | "
        f"{'delta/tok':>10} | preview"
    )
    print(header)
    print("-" * len(header))

    for group_name in ("expensive", "cheap", "medium"):
        print(
            f"{group_name:<12} | "
            f"{_format_optional(summary[f'{group_name}_loss']):>10} | "
            f"{_format_optional(summary[f'{group_name}_delta_loss']):>10} | "
            f"{_format_optional(summary[f'{group_name}_delta_per_token']):>10} | "
            f"{summary[f'{group_name}_tokens_preview']}"
        )

    print("-" * len(header))
    print(
        f"{'random':<12} | "
        f"{'':>10} | "
        f"{'':>10} | "
        f"{_format_optional(summary['random_mean_delta_per_token']):>10} | "
        f"std={_format_optional(summary['random_std_delta_per_token'])}, "
        f"trials={summary['random_trials_completed']}"
    )
    print("=" * 100)
    print(
        "expensive > cheap       : "
        f"{_format_optional(summary['expensive_gt_cheap'])}"
    )
    print(
        "expensive > random_mean : "
        f"{_format_optional(summary['expensive_gt_random_mean'])}"
    )
    print()
