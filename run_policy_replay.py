import argparse

from gear_llm.policy_replay import (
    print_policy_replay_report,
    run_policy_replay,
    save_policy_replay,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: replay offline de políticas de thresholds."
    )
    parser.add_argument(
        "--teacher-csv",
        type=str,
        default="results/teacher_calibration.csv",
        help="CSV gerado pela teacher calibration.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/policy_replay.csv",
        help="Caminho para salvar o replay de políticas.",
    )
    parser.add_argument(
        "--cheap-cost",
        type=float,
        default=0.35,
        help="Custo da consulta ao modelo barato.",
    )
    parser.add_argument(
        "--expensive-cost",
        type=float,
        default=1.00,
        help="Custo da consulta ao modelo caro.",
    )

    args = parser.parse_args()

    try:
        rows = run_policy_replay(
            teacher_csv=args.teacher_csv,
            cheap_cost=args.cheap_cost,
            expensive_cost=args.expensive_cost,
        )
    except FileNotFoundError as error:
        print(error)
        return

    print_policy_replay_report(rows)
    save_policy_replay(rows, args.csv)
    print(f"CSV salvo em: {args.csv}")


if __name__ == "__main__":
    main()
