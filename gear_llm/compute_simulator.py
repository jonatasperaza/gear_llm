from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from gear_llm.analyzer import analyze_prompt, analyze_prompt_with_model
from gear_llm.config import ModelConfig, RouterConfig
from gear_llm.model_loader import load_model_and_tokenizer
from gear_llm.report import save_csv


@dataclass
class ComputeCostConfig:
    cheap_cost: float = 0.35
    medium_cost: float = 0.70
    expensive_cost: float = 1.00


def route_cost(route: str, config: ComputeCostConfig) -> float:
    if route == "cheap":
        return config.cheap_cost

    if route == "medium":
        return config.medium_cost

    if route == "expensive":
        return config.expensive_cost

    raise ValueError(f"Rota desconhecida: {route}")


def simulate_compute_from_rows(
    rows: list[dict],
    cost_config: ComputeCostConfig | None = None,
    prompt: str | None = None,
    prompt_name: str | None = None,
) -> dict:
    """
    Estima custo computacional teórico a partir das rotas por token.

    O baseline assume custo 1.0 para todos os tokens. A simulação assume que
    cheap, medium e expensive usam frações diferentes de computação.
    """

    if cost_config is None:
        cost_config = ComputeCostConfig()

    total_tokens = len(rows)
    counts = Counter(row["route"] for row in rows)

    baseline_cost = float(total_tokens)
    simulated_cost = sum(
        route_cost(row["route"], cost_config)
        for row in rows
    )
    saved_cost = baseline_cost - simulated_cost
    saved_percent = (
        (saved_cost / baseline_cost) * 100
        if baseline_cost > 0
        else 0.0
    )
    avg_cost_per_token = (
        simulated_cost / total_tokens
        if total_tokens > 0
        else 0.0
    )

    return {
        "prompt_name": prompt_name or "",
        "prompt": prompt or "",
        "total_tokens": total_tokens,
        "cheap_count": counts.get("cheap", 0),
        "medium_count": counts.get("medium", 0),
        "expensive_count": counts.get("expensive", 0),
        "cheap_cost": cost_config.cheap_cost,
        "medium_cost": cost_config.medium_cost,
        "expensive_cost": cost_config.expensive_cost,
        "baseline_cost": baseline_cost,
        "simulated_cost": simulated_cost,
        "saved_cost": saved_cost,
        "saved_percent": saved_percent,
        "avg_cost_per_token": avg_cost_per_token,
    }


def run_compute_sim_with_model(
    prompt: str,
    model,
    tokenizer,
    device: str,
    router_config: RouterConfig,
    cost_config: ComputeCostConfig | None = None,
) -> dict:
    rows = analyze_prompt_with_model(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        router_config=router_config,
    )

    return simulate_compute_from_rows(
        rows=rows,
        cost_config=cost_config,
        prompt=prompt,
    )


def run_compute_sim(
    prompt: str,
    model_config: ModelConfig,
    router_config: RouterConfig,
    cost_config: ComputeCostConfig | None = None,
) -> dict:
    rows = analyze_prompt(
        prompt=prompt,
        model_config=model_config,
        router_config=router_config,
    )

    return simulate_compute_from_rows(
        rows=rows,
        cost_config=cost_config,
        prompt=prompt,
    )


def run_compute_sim_benchmark(
    prompts: dict[str, str],
    model_config: ModelConfig,
    router_config: RouterConfig,
    cost_config: ComputeCostConfig | None = None,
) -> list[dict]:
    model, tokenizer, device = load_model_and_tokenizer(model_config.model_name)
    summaries = []

    for name, prompt in prompts.items():
        summary = run_compute_sim_with_model(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            router_config=router_config,
            cost_config=cost_config,
        )
        summary["prompt_name"] = name
        summaries.append(summary)

    return summaries


def print_compute_sim_report(summary: dict):
    print()
    print("Simulação de economia computacional")
    print("=" * 100)

    if summary.get("prompt_name"):
        print(f"prompt              : {summary['prompt_name']}")

    print(f"total_tokens        : {summary['total_tokens']}")
    print(f"cheap_count         : {summary['cheap_count']}")
    print(f"medium_count        : {summary['medium_count']}")
    print(f"expensive_count     : {summary['expensive_count']}")
    print()
    print(f"baseline_cost       : {summary['baseline_cost']:.4f}")
    print(f"simulated_cost      : {summary['simulated_cost']:.4f}")
    print(f"saved_cost          : {summary['saved_cost']:.4f}")
    print(f"saved_percent       : {summary['saved_percent']:.2f}%")
    print(f"avg_cost_per_token  : {summary['avg_cost_per_token']:.4f}")
    print("=" * 100)
    print()


def print_compute_sim_benchmark_report(summaries: list[dict]):
    print()
    print("Benchmark de economia computacional")
    print("=" * 100)

    header = (
        f"{'prompt':<16} | {'tokens':>6} | {'cheap':>5} | "
        f"{'medium':>6} | {'exp':>4} | {'sim_cost':>9} | "
        f"{'saved':>8} | {'saved %':>8}"
    )
    print(header)
    print("-" * len(header))

    for summary in summaries:
        print(
            f"{summary['prompt_name']:<16} | "
            f"{summary['total_tokens']:>6} | "
            f"{summary['cheap_count']:>5} | "
            f"{summary['medium_count']:>6} | "
            f"{summary['expensive_count']:>4} | "
            f"{summary['simulated_cost']:>9.2f} | "
            f"{summary['saved_cost']:>8.2f} | "
            f"{summary['saved_percent']:>7.2f}%"
        )

    print("=" * 100)
    print()


def save_compute_sim_csv(summary: dict, path: str | Path):
    save_csv([summary], str(path))


def save_compute_sim_rows(summaries: list[dict], path: str | Path):
    save_csv(summaries, str(path))
