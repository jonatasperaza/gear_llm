import argparse

from gear_llm.speculative_generator import SpeculativeGenerationConfig
from gear_llm.speculative_tuning import (
    print_speculative_tuning_report,
    run_speculative_tuning,
    save_speculative_tuning,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: Speculative Tuning para Adaptive Speculative Decoding."
    )
    parser.add_argument(
        "--cheap-model",
        type=str,
        default=SpeculativeGenerationConfig.cheap_model_name,
        help="Modelo barato usado para gerar drafts.",
    )
    parser.add_argument(
        "--expensive-model",
        type=str,
        default=SpeculativeGenerationConfig.expensive_model_name,
        help="Modelo caro usado como verificador e referência.",
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
        help="Temperatura usada nas distribuições.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Limita a quantidade de configurações testadas. Útil para smoke tests.",
    )
    parser.add_argument(
        "--config-filter",
        type=str,
        default=None,
        help="Lista de nomes de configs separados por vírgula para rodar.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/speculative_tuning.csv",
        help="Caminho para salvar o CSV detalhado.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/speculative_tuning_summary.csv",
        help="Caminho para salvar o CSV agregado por configuração.",
    )

    args = parser.parse_args()

    rows, summary_rows = run_speculative_tuning(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_configs=args.max_configs,
        config_filter=args.config_filter,
    )

    print_speculative_tuning_report(summary_rows)
    save_speculative_tuning(rows, args.csv)
    save_speculative_tuning(summary_rows, args.summary_csv)
    print(f"CSV detalhado salvo em: {args.csv}")
    print(f"CSV agregado salvo em: {args.summary_csv}")


if __name__ == "__main__":
    main()
