import argparse

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.config import (
    DEVICE_CHOICES,
    PROMPT_FORMAT_CHOICES,
    TORCH_DTYPE_CHOICES,
)
from gear_llm.task_evaluation import (
    build_runtime_profile_outputs,
    build_task_quality_latency_outputs,
    print_runtime_profile_report,
    print_task_evaluation_overall_report,
    print_task_evaluation_report,
    print_task_quality_latency_report,
    run_task_evaluation,
    save_runtime_profile_outputs,
    save_task_quality_latency_outputs,
    save_task_evaluation_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: task-specific quality evaluation."
    )
    parser.add_argument(
        "--cheap-model",
        type=str,
        default=AdaptiveGenerationConfig.cheap_model_name,
        help="Cheap model.",
    )
    parser.add_argument(
        "--expensive-model",
        type=str,
        default=AdaptiveGenerationConfig.expensive_model_name,
        help="Expensive model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default="auto",
        help="Device used by both models.",
    )
    parser.add_argument(
        "--cheap-device",
        type=str,
        default=None,
        help="Optional device for the cheap model, e.g. cuda:0.",
    )
    parser.add_argument(
        "--expensive-device",
        type=str,
        default=None,
        help="Optional device for the expensive model, e.g. cuda:1.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=TORCH_DTYPE_CHOICES,
        default="auto",
        help="Torch dtype used by both models.",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=PROMPT_FORMAT_CHOICES,
        default="auto",
        help="Prompt format: raw, chat, or auto.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature used by generation modes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where CSV files are saved.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_tasks.jsonl",
        help="Task evaluation JSONL dataset.",
    )
    parser.add_argument(
        "--include-latency",
        action="store_true",
        help="Measure generation latency and save task quality-latency reports.",
    )
    parser.add_argument(
        "--profile-runtime",
        action="store_true",
        help="Collect detailed runtime profile CSVs for each task/mode.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup generations per task/mode when --include-latency is set.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=1,
        help="Measured generations per task/mode when --include-latency is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tasks after category/difficulty filters.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated task categories, for example: math,logic,code.",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        default=None,
        help="Comma-separated difficulties: easy,medium,hard.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=(
            "Comma-separated modes: expensive_only,cheap_only,"
            "adaptive_calibrated,adaptive_guarded_v3,adaptive_code_quality,"
            "speculative_adaptive,prompt_router_v1,hybrid."
        ),
    )

    args = parser.parse_args()

    rows, summary_rows, difficulty_rows, overall_rows = run_task_evaluation(
        dataset_path=args.dataset,
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        cheap_device=args.cheap_device,
        expensive_device=args.expensive_device,
        torch_dtype=args.torch_dtype,
        prompt_format=args.prompt_format,
        include_latency=args.include_latency,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        limit=args.limit,
        categories=args.categories,
        difficulties=args.difficulties,
        modes=args.modes,
        profile_runtime=args.profile_runtime,
    )
    print_task_evaluation_report(summary_rows)
    print_task_evaluation_overall_report(overall_rows)
    detailed_csv, summary_csv, difficulty_csv, overall_csv = (
        save_task_evaluation_outputs(
            rows=rows,
            summary_rows=summary_rows,
            difficulty_rows=difficulty_rows,
            overall_rows=overall_rows,
            output_dir=args.output_dir,
        )
    )
    print(f"task_csv        -> {detailed_csv}")
    print(f"task_summary    -> {summary_csv}")
    print(f"task_difficulty -> {difficulty_csv}")
    print(f"task_overall    -> {overall_csv}")

    if args.include_latency:
        (
            latency_rows,
            latency_summary_rows,
            latency_by_category_rows,
            latency_by_difficulty_rows,
        ) = build_task_quality_latency_outputs(rows)
        print_task_quality_latency_report(latency_summary_rows)
        (
            report_csv,
            latency_summary_csv,
            latency_category_csv,
            latency_difficulty_csv,
        ) = save_task_quality_latency_outputs(
            report_rows=latency_rows,
            summary_rows=latency_summary_rows,
            by_category_rows=latency_by_category_rows,
            by_difficulty_rows=latency_by_difficulty_rows,
            output_dir=args.output_dir,
        )
        print(f"task_ql_report  -> {report_csv}")
        print(f"task_ql_summary -> {latency_summary_csv}")
        print(f"task_ql_category-> {latency_category_csv}")
        print(f"task_ql_diff    -> {latency_difficulty_csv}")

    if args.profile_runtime:
        profile_rows, profile_summary_rows = build_runtime_profile_outputs(rows)
        print_runtime_profile_report(profile_summary_rows)
        profile_csv, profile_summary_csv = save_runtime_profile_outputs(
            profile_rows=profile_rows,
            summary_rows=profile_summary_rows,
            output_dir=args.output_dir,
        )
        print(f"runtime_profile -> {profile_csv}")
        print(f"runtime_summary -> {profile_summary_csv}")


if __name__ == "__main__":
    main()
