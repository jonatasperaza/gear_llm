import argparse

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.config import (
    DEVICE_CHOICES,
    PROMPT_FORMAT_CHOICES,
    TORCH_DTYPE_CHOICES,
)
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
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default="auto",
        help="Device para carregar os dois modelos.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=TORCH_DTYPE_CHOICES,
        default="auto",
        help="dtype dos pesos dos dois modelos.",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=PROMPT_FORMAT_CHOICES,
        default="auto",
        help="Formato do prompt: raw, chat ou auto.",
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
        device=args.device,
        torch_dtype=args.torch_dtype,
        prompt_format=args.prompt_format,
    )

    print_quality_benchmark_report(rows)
    save_quality_benchmark(rows, args.csv)
    print(f"CSV salvo em: {args.csv}")


if __name__ == "__main__":
    main()
