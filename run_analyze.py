import argparse

from gear_llm.analyzer import analyze_prompt
from gear_llm.config import ModelConfig, RouterConfig
from gear_llm.report import print_report, save_csv


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: Geometric-Entropy Adaptive Routing"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Texto que será analisado pelo modelo.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="Modelo do Hugging Face.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Caminho opcional para salvar resultado em CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de tokens para mostrar no terminal.",
    )

    args = parser.parse_args()

    model_config = ModelConfig(model_name=args.model)
    router_config = RouterConfig()

    rows = analyze_prompt(
        prompt=args.prompt,
        model_config=model_config,
        router_config=router_config,
    )

    print_report(rows, limit=args.limit)

    if args.csv:
        save_csv(rows, args.csv)
        print(f"CSV salvo em: {args.csv}")


if __name__ == "__main__":
    main()