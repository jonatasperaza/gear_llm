import argparse

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.config import (
    DEVICE_CHOICES,
    PROMPT_FORMAT_CHOICES,
    TORCH_DTYPE_CHOICES,
)
from gear_llm.dataset_benchmark import (
    parse_categories,
    print_dataset_benchmark_report,
    run_dataset_benchmark,
    save_dataset_benchmark_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset benchmark do GEAR-LLM para o hybrid router."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/prompts.jsonl",
        help="Caminho do JSONL de prompts.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Categorias separadas por virgula.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite total de prompts apos filtro.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximo de tokens novos por geracao.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Pasta onde os CSVs serao salvos.",
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
        help="Modelo caro.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default="auto",
        help="Device para carregar os dois modelos.",
    )
    parser.add_argument(
        "--cheap-device",
        type=str,
        default=None,
        help="Device opcional para o modelo barato, ex: cuda:0.",
    )
    parser.add_argument(
        "--expensive-device",
        type=str,
        default=None,
        help="Device opcional para o modelo caro, ex: cuda:1.",
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
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura de geracao.",
    )
    args = parser.parse_args()

    rows, summary_rows, matrix_rows = run_dataset_benchmark(
        dataset_path=args.dataset,
        categories=parse_categories(args.categories),
        limit=args.limit,
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        cheap_device=args.cheap_device,
        expensive_device=args.expensive_device,
        torch_dtype=args.torch_dtype,
        prompt_format=args.prompt_format,
    )
    print_dataset_benchmark_report(summary_rows, matrix_rows)
    detailed_csv, summary_csv, matrix_csv = save_dataset_benchmark_outputs(
        rows=rows,
        summary_rows=summary_rows,
        matrix_rows=matrix_rows,
        output_dir=args.output_dir,
    )

    print(f"{'dataset_csv':<15} -> {detailed_csv}")
    print(f"{'summary_csv':<15} -> {summary_csv}")
    print(f"{'matrix_csv':<15} -> {matrix_csv}")


if __name__ == "__main__":
    main()
