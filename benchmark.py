import argparse
from pathlib import Path

from gear_llm.ablation import (
    run_ablation_with_model,
    run_balanced_ablation_with_model,
    save_ablation_rows,
)
from gear_llm.analyzer import analyze_prompt_with_model
from gear_llm.config import ModelConfig, RouterConfig
from gear_llm.model_loader import load_model_and_tokenizer
from gear_llm.report import save_csv


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

    args = parser.parse_args()

    model_config = ModelConfig(model_name=args.model)
    router_config = RouterConfig()
    output_dir = Path(args.output_dir)

    model, tokenizer, device = load_model_and_tokenizer(model_config.model_name)
    ablation_rows = []
    balanced_ablation_rows = []

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


if __name__ == "__main__":
    main()
