import argparse

from gear_llm.teacher_calibration import (
    TeacherCalibrationConfig,
    print_teacher_summary,
    run_teacher_calibration,
    save_teacher_grid,
    save_teacher_rows,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: calibração offline cheap-vs-teacher."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt inicial para a calibração.",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default="",
        help="Nome opcional do prompt nos CSVs.",
    )
    parser.add_argument(
        "--cheap-model",
        type=str,
        default=TeacherCalibrationConfig.cheap_model_name,
        help="Modelo barato.",
    )
    parser.add_argument(
        "--expensive-model",
        type=str,
        default=TeacherCalibrationConfig.expensive_model_name,
        help="Modelo caro/teacher.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Número máximo de passos gerados pelo teacher.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k do modelo caro usado para topk_match.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura usada para transformar logits em probabilidades.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/teacher_calibration.csv",
        help="CSV detalhado por passo.",
    )
    parser.add_argument(
        "--grid-csv",
        type=str,
        default="results/teacher_threshold_grid.csv",
        help="CSV do grid search de thresholds.",
    )

    args = parser.parse_args()

    config = TeacherCalibrationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_steps=args.max_steps,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    rows, summary, grid_rows = run_teacher_calibration(
        prompt=args.prompt,
        config=config,
        prompt_name=args.prompt_name,
    )

    print_teacher_summary(summary, grid_rows)
    save_teacher_rows(rows, args.csv)
    save_teacher_grid(grid_rows, args.grid_csv)
    print(f"CSV detalhado salvo em: {args.csv}")
    print(f"Grid salvo em: {args.grid_csv}")


if __name__ == "__main__":
    main()
