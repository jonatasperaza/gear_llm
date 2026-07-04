import argparse

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.guard_tuning import (
    print_guard_tuning_report,
    run_guard_tuning,
    save_guard_tuning,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: Guard Tuning para Adaptive Dual-Model Generation."
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
        help="Número máximo de tokens novos por prompt.",
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
        default="results/guard_tuning.csv",
        help="Caminho para salvar o CSV detalhado.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/guard_tuning_summary.csv",
        help="Caminho para salvar o CSV agregado por configuração.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Limita a quantidade de configurações testadas. Útil para smoke tests.",
    )

    args = parser.parse_args()

    rows, summary_rows = run_guard_tuning(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_configs=args.max_configs,
    )

    print_guard_tuning_report(summary_rows)
    save_guard_tuning(rows, args.csv)
    save_guard_tuning(summary_rows, args.summary_csv)
    print(f"CSV detalhado salvo em: {args.csv}")
    print(f"CSV agregado salvo em: {args.summary_csv}")


if __name__ == "__main__":
    main()
