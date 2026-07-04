import argparse

from gear_llm.compute_simulator import (
    ComputeCostConfig,
    print_compute_sim_report,
    run_compute_sim,
    save_compute_sim_csv,
)
from gear_llm.config import ModelConfig, RouterConfig


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: simulação de economia computacional por rota."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Texto que será analisado e simulado.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ModelConfig.model_name,
        help="Modelo do Hugging Face.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/compute_sim.csv",
        help="Caminho para salvar o resumo em CSV.",
    )
    parser.add_argument(
        "--cheap-cost",
        type=float,
        default=0.35,
        help="Custo teórico de tokens cheap.",
    )
    parser.add_argument(
        "--medium-cost",
        type=float,
        default=0.70,
        help="Custo teórico de tokens medium.",
    )
    parser.add_argument(
        "--expensive-cost",
        type=float,
        default=1.00,
        help="Custo teórico de tokens expensive.",
    )

    args = parser.parse_args()

    model_config = ModelConfig(model_name=args.model)
    router_config = RouterConfig()
    cost_config = ComputeCostConfig(
        cheap_cost=args.cheap_cost,
        medium_cost=args.medium_cost,
        expensive_cost=args.expensive_cost,
    )

    summary = run_compute_sim(
        prompt=args.prompt,
        model_config=model_config,
        router_config=router_config,
        cost_config=cost_config,
    )

    print_compute_sim_report(summary)
    save_compute_sim_csv(summary, args.csv)
    print(f"CSV salvo em: {args.csv}")


if __name__ == "__main__":
    main()
