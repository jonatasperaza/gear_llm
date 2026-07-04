import argparse

from gear_llm.ablation import (
    print_ablation_report,
    print_balanced_ablation_report,
    run_ablation,
    run_balanced_ablation,
    save_ablation_csv,
)
from gear_llm.config import ModelConfig, RouterConfig


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: validação por ablation de tokens roteados."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Texto que será analisado e perturbado.",
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
        default=None,
        help="Caminho para salvar o resumo em CSV.",
    )
    parser.add_argument(
        "--neutral-text",
        type=str,
        default=" ",
        help="Texto neutro usado para escolher o token substituto.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Roda ablation balanceada controlando o número de tokens.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=20,
        help="Número de rodadas do baseline aleatório na ablation balanceada.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do baseline aleatório.",
    )

    args = parser.parse_args()

    model_config = ModelConfig(model_name=args.model)
    router_config = RouterConfig()

    if args.balanced:
        summary = run_balanced_ablation(
            prompt=args.prompt,
            model_config=model_config,
            router_config=router_config,
            neutral_text=args.neutral_text,
            random_trials=args.random_trials,
            seed=args.seed,
        )
        csv_path = args.csv or "results/balanced_ablation.csv"
        print_balanced_ablation_report(summary)
    else:
        summary = run_ablation(
            prompt=args.prompt,
            model_config=model_config,
            router_config=router_config,
            neutral_text=args.neutral_text,
        )
        csv_path = args.csv or "results/ablation.csv"
        print_ablation_report(summary)

    save_ablation_csv(summary, csv_path)
    print(f"CSV salvo em: {csv_path}")


if __name__ == "__main__":
    main()
