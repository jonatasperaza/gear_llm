import argparse

from gear_llm.mode_oracle import (
    print_mode_oracle_report,
    run_mode_oracle,
    save_mode_oracle_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Mode Oracle Benchmark offline para o GEAR-LLM."
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="results/dataset_benchmark.csv",
        help="CSV detalhado gerado pelo dataset benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Pasta onde os CSVs serao salvos.",
    )
    args = parser.parse_args()

    try:
        (
            oracle_rows,
            summary_rows,
            confidence_summary_rows,
            comparison_rows,
            metrics,
        ) = run_mode_oracle(dataset_csv=args.dataset_csv)
    except FileNotFoundError as error:
        print(error)
        return

    print_mode_oracle_report(summary_rows, confidence_summary_rows, metrics)
    oracle_csv, summary_csv, confidence_csv, comparison_csv = (
        save_mode_oracle_outputs(
            oracle_rows=oracle_rows,
            summary_rows=summary_rows,
            confidence_summary_rows=confidence_summary_rows,
            comparison_rows=comparison_rows,
            output_dir=args.output_dir,
        )
    )
    print(f"{'oracle_csv':<15} -> {oracle_csv}")
    print(f"{'summary_csv':<15} -> {summary_csv}")
    print(f"{'confidence_csv':<15} -> {confidence_csv}")
    print(f"{'compare_csv':<15} -> {comparison_csv}")


if __name__ == "__main__":
    main()
