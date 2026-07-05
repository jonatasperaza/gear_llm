import argparse

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.config import DEVICE_CHOICES, TORCH_DTYPE_CHOICES
from gear_llm.latency_benchmark import (
    print_latency_benchmark_report,
    run_latency_benchmark,
    save_latency_benchmark_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Latency Benchmark real do GEAR-LLM."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximo de tokens novos por geracao.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura de geracao.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Quantidade de execucoes de aquecimento por prompt/modo.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=3,
        help="Quantidade de execucoes medidas por prompt/modo.",
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
        "--torch-dtype",
        type=str,
        choices=TORCH_DTYPE_CHOICES,
        default="auto",
        help="dtype dos pesos dos dois modelos.",
    )
    args = parser.parse_args()

    rows, summary_rows, winner_rows = run_latency_benchmark(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    print_latency_benchmark_report(summary_rows)
    detailed_csv, summary_csv, winners_csv = save_latency_benchmark_outputs(
        rows=rows,
        summary_rows=summary_rows,
        winner_rows=winner_rows,
        output_dir=args.output_dir,
    )
    print(f"{'latency_csv':<15} -> {detailed_csv}")
    print(f"{'summary_csv':<15} -> {summary_csv}")
    print(f"{'winners_csv':<15} -> {winners_csv}")


if __name__ == "__main__":
    main()
