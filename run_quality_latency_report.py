import argparse

from gear_llm.quality_latency_report import (
    build_quality_latency_report,
    print_quality_latency_report,
    save_quality_latency_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Combina benchmarks de qualidade e latencia do GEAR-LLM."
    )
    parser.add_argument(
        "--latency-winners",
        type=str,
        default="results/latency_winners.csv",
        help="CSV com vencedores de latencia.",
    )
    parser.add_argument(
        "--latency-summary",
        type=str,
        default="results/latency_benchmark_summary.csv",
        help="CSV summary do latency benchmark.",
    )
    parser.add_argument(
        "--quality-csv",
        type=str,
        default="results/quality_benchmark.csv",
        help="CSV do quality benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Pasta onde os CSVs serao salvos.",
    )
    args = parser.parse_args()

    report_rows, summary_rows = build_quality_latency_report(
        latency_winners_csv=args.latency_winners,
        latency_summary_csv=args.latency_summary,
        quality_csv=args.quality_csv,
    )
    print_quality_latency_report(report_rows)
    report_csv, summary_csv = save_quality_latency_outputs(
        report_rows=report_rows,
        summary_rows=summary_rows,
        output_dir=args.output_dir,
    )
    print(f"{'report_csv':<15} -> {report_csv}")
    print(f"{'summary_csv':<15} -> {summary_csv}")


if __name__ == "__main__":
    main()
