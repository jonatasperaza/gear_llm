import argparse

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.quality_benchmark import (
    print_quality_benchmark_report,
    run_quality_benchmark,
    save_quality_benchmark,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: Quality-vs-Cost Benchmark."
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
        help="Modelo caro usado como referência.",
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
        help="Temperatura usada para probabilidades/logits.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/quality_benchmark.csv",
        help="Caminho para salvar o CSV.",
    )

    args = parser.parse_args()

    rows = run_quality_benchmark(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print_quality_benchmark_report(rows)
    save_quality_benchmark(rows, args.csv)
    print(f"CSV salvo em: {args.csv}")


if __name__ == "__main__":
    main()
