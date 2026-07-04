import argparse

from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    adaptive_generate,
    print_adaptive_report,
    save_adaptive_history,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: Adaptive Dual-Model Generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt inicial para geração.",
    )
    parser.add_argument(
        "--cheap-model",
        type=str,
        default=AdaptiveGenerationConfig.cheap_model_name,
        help="Modelo barato.",
    )
    parser.add_argument(
        "--expensive-model",
        type=str,
        default=AdaptiveGenerationConfig.expensive_model_name,
        help="Modelo caro usado em fallback.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Número máximo de tokens novos.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura usada para calcular probabilidades.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.35,
        help="Entropia máxima para aceitar o token barato.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.20,
        help="Margem mínima top1-top2 para aceitar o token barato.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/adaptive_generation.csv",
        help="Caminho para salvar o histórico por token.",
    )

    args = parser.parse_args()

    config = AdaptiveGenerationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        entropy_threshold=args.entropy_threshold,
        margin_threshold=args.margin_threshold,
    )

    _, history, summary = adaptive_generate(
        prompt=args.prompt,
        config=config,
    )

    print_adaptive_report(summary)
    save_adaptive_history(history, args.csv)
    print(f"CSV salvo em: {args.csv}")


if __name__ == "__main__":
    main()
