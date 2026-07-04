import argparse
from pathlib import Path

from gear_llm.ablation import (
    run_ablation_with_model,
    run_balanced_ablation_with_model,
    save_ablation_rows,
)
from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    adaptive_generate_with_models,
    load_adaptive_models,
    save_adaptive_summary_rows,
)
from gear_llm.analyzer import analyze_prompt_with_model
from gear_llm.compute_simulator import (
    ComputeCostConfig,
    print_compute_sim_benchmark_report,
    save_compute_sim_rows,
    simulate_compute_from_rows,
)
from gear_llm.config import ModelConfig, RouterConfig
from gear_llm.model_loader import load_model_and_tokenizer
from gear_llm.report import save_csv
from gear_llm.teacher_calibration import (
    TeacherCalibrationConfig,
    load_teacher_models,
    run_teacher_calibration_with_models,
    save_teacher_grid,
    save_teacher_rows,
    threshold_grid_search,
)


PROMPTS = {
    "easy": "Explique em uma frase o que é água.",
    "math": "Explique por que a inversa de f(x)=5x+1 é (x-1)/5.",
    "logic_negation": (
        "Se não chover e apenas se o vento parar, então podemos sair; "
        "exceto se houver alerta."
    ),
    "code": (
        "Escreva uma função Python: if x % 2 == 0, retorne x / 2; "
        "caso contrário, retorne 3 * x + 1."
    ),
    "long_simple": (
        "O dia começou calmo. As pessoas caminharam pela praça, "
        "compraram pão, conversaram sobre o tempo e voltaram para casa. "
        "Nada urgente aconteceu, apenas uma sequência simples de eventos."
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark simples do GEAR-LLM em vários tipos de prompt."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ModelConfig.model_name,
        help="Modelo do Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Pasta onde os CSVs serão salvos.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Também roda validação por ablation nos prompts principais.",
    )
    parser.add_argument(
        "--balanced-ablation",
        action="store_true",
        help="Também roda ablation balanceada nos prompts principais.",
    )
    parser.add_argument(
        "--compute-sim",
        action="store_true",
        help="Também roda simulação de economia computacional.",
    )
    parser.add_argument(
        "--adaptive-generate",
        action="store_true",
        help="Também roda Adaptive Dual-Model Generation nos prompts principais.",
    )
    parser.add_argument(
        "--adaptive-compare-thresholds",
        action="store_true",
        help="Compara thresholds antigos e calibrados na geração adaptativa.",
    )
    parser.add_argument(
        "--teacher-calibration",
        action="store_true",
        help="Também roda calibração offline cheap-vs-teacher.",
    )
    parser.add_argument(
        "--ablation-csv",
        type=str,
        default=None,
        help="CSV opcional para salvar o benchmark de ablation.",
    )
    parser.add_argument(
        "--neutral-text",
        type=str,
        default=" ",
        help="Texto neutro usado pela ablation.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=20,
        help="Número de rodadas do baseline aleatório na ablation balanceada.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do baseline aleatório.",
    )
    parser.add_argument(
        "--cheap-cost",
        type=float,
        default=0.35,
        help="Custo teórico de tokens cheap na simulação.",
    )
    parser.add_argument(
        "--medium-cost",
        type=float,
        default=0.70,
        help="Custo teórico de tokens medium na simulação.",
    )
    parser.add_argument(
        "--expensive-cost",
        type=float,
        default=1.00,
        help="Custo teórico de tokens expensive na simulação.",
    )
    parser.add_argument(
        "--cheap-model",
        type=str,
        default=AdaptiveGenerationConfig.cheap_model_name,
        help="Modelo barato para geração adaptativa.",
    )
    parser.add_argument(
        "--expensive-model",
        type=str,
        default=AdaptiveGenerationConfig.expensive_model_name,
        help="Modelo caro para geração adaptativa.",
    )
    parser.add_argument(
        "--adaptive-max-new-tokens",
        type=int,
        default=80,
        help="Máximo de tokens novos na geração adaptativa.",
    )
    parser.add_argument(
        "--adaptive-temperature",
        type=float,
        default=0.7,
        help="Temperatura da geração adaptativa.",
    )
    parser.add_argument(
        "--adaptive-entropy-threshold",
        type=float,
        default=AdaptiveGenerationConfig.entropy_threshold,
        help="Entropia máxima para aceitar o modelo barato.",
    )
    parser.add_argument(
        "--adaptive-margin-threshold",
        type=float,
        default=0.20,
        help="Margem mínima top1-top2 para aceitar o modelo barato.",
    )
    parser.add_argument(
        "--teacher-max-steps",
        type=int,
        default=40,
        help="Número máximo de passos gerados pelo teacher.",
    )
    parser.add_argument(
        "--teacher-top-k",
        type=int,
        default=5,
        help="Top-k do teacher para topk_match.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.7,
        help="Temperatura usada na calibração teacher.",
    )

    args = parser.parse_args()

    model_config = ModelConfig(model_name=args.model)
    router_config = RouterConfig()
    cost_config = ComputeCostConfig(
        cheap_cost=args.cheap_cost,
        medium_cost=args.medium_cost,
        expensive_cost=args.expensive_cost,
    )
    adaptive_config = AdaptiveGenerationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.adaptive_max_new_tokens,
        temperature=args.adaptive_temperature,
        entropy_threshold=args.adaptive_entropy_threshold,
        margin_threshold=args.adaptive_margin_threshold,
    )
    teacher_config = TeacherCalibrationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_steps=args.teacher_max_steps,
        top_k=args.teacher_top_k,
        temperature=args.teacher_temperature,
    )
    output_dir = Path(args.output_dir)

    model, tokenizer, device = load_model_and_tokenizer(model_config.model_name)
    adaptive_models = None
    teacher_models = None

    if args.adaptive_generate or args.adaptive_compare_thresholds:
        adaptive_models = load_adaptive_models(adaptive_config)

    if args.teacher_calibration:
        teacher_models = adaptive_models or load_teacher_models(teacher_config)

    ablation_rows = []
    balanced_ablation_rows = []
    compute_sim_rows = []
    adaptive_generation_rows = []
    adaptive_threshold_comparison_rows = []
    teacher_calibration_rows = []
    teacher_grid_rows = []

    for name, prompt in PROMPTS.items():
        rows = analyze_prompt_with_model(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            router_config=router_config,
        )

        csv_path = output_dir / f"{name}.csv"
        save_csv(rows, str(csv_path))
        print(f"{name:<15} -> {csv_path}")

        if args.compute_sim:
            compute_summary = simulate_compute_from_rows(
                rows=rows,
                cost_config=cost_config,
                prompt=prompt,
                prompt_name=name,
            )
            compute_sim_rows.append(compute_summary)
            print(
                f"{'compute_sim':<15} -> {name}: "
                f"saved={compute_summary['saved_percent']:.2f}%, "
                f"avg_cost/token={compute_summary['avg_cost_per_token']:.4f}"
            )

        if args.ablation:
            summary = run_ablation_with_model(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                router_config=router_config,
                neutral_text=args.neutral_text,
            )
            summary["prompt_name"] = name
            ablation_rows.append(summary)

            if summary["criterion_status"] == "not_applicable":
                status = "N/A"
            elif summary["criterion_passed_raw"]:
                status = "PASSOU"
            else:
                status = "NÃO PASSOU"

            print(
                f"{'ablation':<15} -> {name}: "
                f"cheap_delta={summary['cheap_delta_loss']:.4f}, "
                f"expensive_delta={summary['expensive_delta_loss']:.4f}, "
                f"cheap_delta/tok={summary['cheap_delta_loss_per_token']:.4f}, "
                f"expensive_delta/tok={summary['expensive_delta_loss_per_token']:.4f} "
                f"({status})"
            )

        if args.balanced_ablation:
            balanced_summary = run_balanced_ablation_with_model(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                router_config=router_config,
                neutral_text=args.neutral_text,
                random_trials=args.random_trials,
                seed=args.seed,
            )
            balanced_summary["prompt_name"] = name
            balanced_ablation_rows.append(balanced_summary)

            status = balanced_summary["balanced_status"]
            print(
                f"{'balanced':<15} -> {name}: "
                f"k={balanced_summary['k']}, "
                f"expensive/tok={balanced_summary['expensive_delta_per_token']}, "
                f"cheap/tok={balanced_summary['cheap_delta_per_token']}, "
                f"random_mean/tok={balanced_summary['random_mean_delta_per_token']} "
                f"({status})"
            )

        if args.adaptive_generate:
            (
                adaptive_cheap_model,
                adaptive_expensive_model,
                adaptive_tokenizer,
                adaptive_device,
            ) = adaptive_models
            _, _, adaptive_summary = adaptive_generate_with_models(
                prompt=prompt,
                cheap_model=adaptive_cheap_model,
                expensive_model=adaptive_expensive_model,
                tokenizer=adaptive_tokenizer,
                device=adaptive_device,
                config=adaptive_config,
            )
            adaptive_summary["prompt_name"] = name
            adaptive_generation_rows.append(adaptive_summary)
            print(
                f"{'adaptive':<15} -> {name}: "
                f"cheap={adaptive_summary['cheap_percent']:.2f}%, "
                f"expensive_calls={adaptive_summary['expensive_model_calls']}, "
                f"saved={adaptive_summary['estimated_saved_percent']:.2f}%"
            )

        if args.adaptive_compare_thresholds:
            (
                adaptive_cheap_model,
                adaptive_expensive_model,
                adaptive_tokenizer,
                adaptive_device,
            ) = adaptive_models
            comparison_configs = [
                (
                    "old_0.45_0.20",
                    AdaptiveGenerationConfig(
                        cheap_model_name=args.cheap_model,
                        expensive_model_name=args.expensive_model,
                        max_new_tokens=args.adaptive_max_new_tokens,
                        temperature=args.adaptive_temperature,
                        entropy_threshold=0.45,
                        margin_threshold=0.20,
                    ),
                ),
                (
                    "calibrated_0.35_0.20",
                    AdaptiveGenerationConfig(
                        cheap_model_name=args.cheap_model,
                        expensive_model_name=args.expensive_model,
                        max_new_tokens=args.adaptive_max_new_tokens,
                        temperature=args.adaptive_temperature,
                        entropy_threshold=0.35,
                        margin_threshold=0.20,
                    ),
                ),
            ]

            for config_name, comparison_config in comparison_configs:
                _, _, comparison_summary = adaptive_generate_with_models(
                    prompt=prompt,
                    cheap_model=adaptive_cheap_model,
                    expensive_model=adaptive_expensive_model,
                    tokenizer=adaptive_tokenizer,
                    device=adaptive_device,
                    config=comparison_config,
                )
                adaptive_threshold_comparison_rows.append(
                    {
                        "prompt_name": name,
                        "config_name": config_name,
                        "entropy_threshold": comparison_config.entropy_threshold,
                        "margin_threshold": comparison_config.margin_threshold,
                        "total_generated_tokens": comparison_summary[
                            "total_generated_tokens"
                        ],
                        "cheap_accepted_tokens": comparison_summary[
                            "cheap_accepted_tokens"
                        ],
                        "expensive_model_calls": comparison_summary[
                            "expensive_model_calls"
                        ],
                        "cheap_percent": comparison_summary["cheap_percent"],
                        "estimated_saved_percent": comparison_summary[
                            "estimated_saved_percent"
                        ],
                        "generated_text": comparison_summary["generated_text"],
                    }
                )

        if args.teacher_calibration:
            (
                teacher_cheap_model,
                teacher_expensive_model,
                teacher_tokenizer,
                teacher_device,
            ) = teacher_models
            teacher_rows, teacher_summary, teacher_grid = (
                run_teacher_calibration_with_models(
                    prompt=prompt,
                    cheap_model=teacher_cheap_model,
                    expensive_model=teacher_expensive_model,
                    tokenizer=teacher_tokenizer,
                    device=teacher_device,
                    config=teacher_config,
                    prompt_name=name,
                )
            )
            teacher_calibration_rows.extend(teacher_rows)
            teacher_grid_rows.extend(teacher_grid)

            viable_grid = [
                row
                for row in teacher_grid
                if row["precision_accept"] is not None
                and row["estimated_saved_percent"] > 0
            ]
            best = None

            if viable_grid:
                best = max(
                    viable_grid,
                    key=lambda row: (
                        row["precision_accept"],
                        row["estimated_saved_percent"],
                    ),
                )

            if best:
                best_text = (
                    f"best_precision={best['precision_accept']:.2%}, "
                    f"saved={best['estimated_saved_percent']:.2f}%"
                )
            else:
                best_text = "sem threshold com economia positiva"

            print(
                f"{'teacher':<15} -> {name}: "
                f"exact={teacher_summary['exact_match_rate']:.2%}, "
                f"topk={teacher_summary['topk_match_rate']:.2%}, "
                f"{best_text}"
            )

    if args.ablation:
        ablation_csv = (
            Path(args.ablation_csv)
            if args.ablation_csv
            else output_dir / "ablation_benchmark.csv"
        )
        save_ablation_rows(ablation_rows, ablation_csv)
        print(f"{'ablation_csv':<15} -> {ablation_csv}")

    if args.balanced_ablation:
        balanced_csv = output_dir / "balanced_ablation_benchmark.csv"
        save_ablation_rows(balanced_ablation_rows, balanced_csv)
        print(f"{'balanced_csv':<15} -> {balanced_csv}")

    if args.compute_sim:
        compute_csv = output_dir / "compute_sim_benchmark.csv"
        save_compute_sim_rows(compute_sim_rows, compute_csv)
        print_compute_sim_benchmark_report(compute_sim_rows)
        print(f"{'compute_csv':<15} -> {compute_csv}")

    if args.adaptive_generate:
        adaptive_csv = output_dir / "adaptive_generation_benchmark.csv"
        save_adaptive_summary_rows(adaptive_generation_rows, adaptive_csv)
        print(f"{'adaptive_csv':<15} -> {adaptive_csv}")

    if args.adaptive_compare_thresholds:
        comparison_csv = output_dir / "adaptive_threshold_comparison.csv"
        save_csv(adaptive_threshold_comparison_rows, str(comparison_csv))
        print()
        print("Comparação de thresholds adaptativos")
        print("=" * 100)
        header = (
            f"{'prompt':<16} | {'old saved':>9} | {'cal saved':>9} | "
            f"{'old calls':>9} | {'cal calls':>9} | "
            f"{'old cheap':>9} | {'cal cheap':>9}"
        )
        print(header)
        print("-" * len(header))

        for name in PROMPTS:
            old_row = next(
                row
                for row in adaptive_threshold_comparison_rows
                if row["prompt_name"] == name
                and row["config_name"] == "old_0.45_0.20"
            )
            calibrated_row = next(
                row
                for row in adaptive_threshold_comparison_rows
                if row["prompt_name"] == name
                and row["config_name"] == "calibrated_0.35_0.20"
            )
            print(
                f"{name:<16} | "
                f"{old_row['estimated_saved_percent']:>8.2f}% | "
                f"{calibrated_row['estimated_saved_percent']:>8.2f}% | "
                f"{old_row['expensive_model_calls']:>9} | "
                f"{calibrated_row['expensive_model_calls']:>9} | "
                f"{old_row['cheap_percent']:>8.2f}% | "
                f"{calibrated_row['cheap_percent']:>8.2f}%"
            )

        print("=" * 100)
        print(
            "Observação: esta comparação é online; se os textos gerados divergem, "
            "as decisões futuras também podem divergir."
        )
        print(f"{'threshold_csv':<15} -> {comparison_csv}")

    if args.teacher_calibration:
        teacher_csv = output_dir / "teacher_calibration.csv"
        teacher_grid_csv = output_dir / "teacher_threshold_grid.csv"
        aggregate_grid = threshold_grid_search(
            rows=teacher_calibration_rows,
            config=teacher_config,
            prompt_name="ALL",
        )
        teacher_grid_rows.extend(aggregate_grid)
        save_teacher_rows(teacher_calibration_rows, teacher_csv)
        save_teacher_grid(teacher_grid_rows, teacher_grid_csv)
        print(f"{'teacher_csv':<15} -> {teacher_csv}")
        print(f"{'teacher_grid':<15} -> {teacher_grid_csv}")


if __name__ == "__main__":
    main()
