import argparse

from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    adaptive_generate,
    print_adaptive_report,
    save_adaptive_history,
)
from gear_llm.config import (
    DEVICE_CHOICES,
    PROMPT_FORMAT_CHOICES,
    TORCH_DTYPE_CHOICES,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: Adaptive Dual-Model Generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt inicial para geração.",
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
        help="Modelo caro usado em fallback.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default="auto",
        help="Device para carregar os dois modelos.",
    )
    parser.add_argument(
        "--cheap-device",
        type=str,
        default=None,
        help="Device opcional para o modelo barato, ex: cuda:0.",
    )
    parser.add_argument(
        "--expensive-device",
        type=str,
        default=None,
        help="Device opcional para o modelo caro, ex: cuda:1.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=TORCH_DTYPE_CHOICES,
        default="auto",
        help="dtype dos pesos dos dois modelos.",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=PROMPT_FORMAT_CHOICES,
        default="auto",
        help="Formato do prompt: raw, chat ou auto.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Número máximo de tokens novos.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura usada para calcular probabilidades.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.35,
        help="Entropia máxima para aceitar o token barato.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.20,
        help="Margem mínima top1-top2 para aceitar o token barato.",
    )
    parser.add_argument(
        "--teacher-check-interval",
        type=int,
        default=AdaptiveGenerationConfig.teacher_check_interval,
        help="Intervalo de chamadas periódicas ao modelo caro.",
    )
    parser.add_argument(
        "--disable-periodic-teacher-check",
        action="store_true",
        help="Desliga chamadas periódicas ao modelo caro.",
    )
    parser.add_argument(
        "--disable-repetition-guard",
        action="store_true",
        help="Desliga o guard de repetição.",
    )
    parser.add_argument(
        "--repetition-ngram-size",
        type=int,
        default=3,
        help="Tamanho do n-grama usado pelo guard de repetição.",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=AdaptiveGenerationConfig.repetition_threshold,
        help="Taxa parcial de n-gramas repetidos que aciona fallback.",
    )
    parser.add_argument(
        "--disable-repetition-guard-requires-uncertainty",
        action="store_true",
        help="Permite repetition_guard mesmo sem sinal de incerteza.",
    )
    parser.add_argument(
        "--repetition-guard-entropy-threshold",
        type=float,
        default=AdaptiveGenerationConfig.repetition_guard_entropy_threshold,
        help="Entropia mínima para o repetition_guard passar pelo gate de incerteza.",
    )
    parser.add_argument(
        "--repetition-guard-margin-threshold",
        type=float,
        default=AdaptiveGenerationConfig.repetition_guard_margin_threshold,
        help="Margem abaixo da qual o repetition_guard passa pelo gate de incerteza.",
    )
    parser.add_argument(
        "--repetition-guard-cooldown-tokens",
        type=int,
        default=AdaptiveGenerationConfig.repetition_guard_cooldown_tokens,
        help="Número de tokens de espera após repetition_guard chamar o modelo caro.",
    )
    parser.add_argument(
        "--disable-risk-gated-periodic-check",
        action="store_true",
        help="Desliga o gating por risco das chamadas periódicas ao modelo caro.",
    )
    parser.add_argument(
        "--periodic-entropy-risk-threshold",
        type=float,
        default=AdaptiveGenerationConfig.periodic_entropy_risk_threshold,
        help="Entropia mínima para considerar uma chamada periódica arriscada.",
    )
    parser.add_argument(
        "--periodic-margin-risk-threshold",
        type=float,
        default=AdaptiveGenerationConfig.periodic_margin_risk_threshold,
        help="Margem abaixo da qual uma chamada periódica é considerada arriscada.",
    )
    parser.add_argument(
        "--periodic-repetition-risk-threshold",
        type=float,
        default=AdaptiveGenerationConfig.periodic_repetition_risk_threshold,
        help="Repetição parcial acima da qual uma chamada periódica é arriscada.",
    )
    parser.add_argument(
        "--max-expensive-call-ratio",
        type=float,
        default=AdaptiveGenerationConfig.max_expensive_call_ratio,
        help="Razão máxima de chamadas caras antes de bloquear fallback periódico puro.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/adaptive_generation.csv",
        help="Caminho para salvar o histórico por token.",
    )

    args = parser.parse_args()

    config = AdaptiveGenerationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        device=args.device,
        cheap_device=args.cheap_device,
        expensive_device=args.expensive_device,
        torch_dtype=args.torch_dtype,
        prompt_format=args.prompt_format,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        entropy_threshold=args.entropy_threshold,
        margin_threshold=args.margin_threshold,
        teacher_check_interval=args.teacher_check_interval,
        enable_periodic_teacher_check=not args.disable_periodic_teacher_check,
        enable_repetition_guard=not args.disable_repetition_guard,
        repetition_ngram_size=args.repetition_ngram_size,
        repetition_threshold=args.repetition_threshold,
        risk_gated_periodic_check=not args.disable_risk_gated_periodic_check,
        periodic_entropy_risk_threshold=args.periodic_entropy_risk_threshold,
        periodic_margin_risk_threshold=args.periodic_margin_risk_threshold,
        periodic_repetition_risk_threshold=(
            args.periodic_repetition_risk_threshold
        ),
        max_expensive_call_ratio=args.max_expensive_call_ratio,
        repetition_guard_requires_uncertainty=(
            not args.disable_repetition_guard_requires_uncertainty
        ),
        repetition_guard_entropy_threshold=(
            args.repetition_guard_entropy_threshold
        ),
        repetition_guard_margin_threshold=args.repetition_guard_margin_threshold,
        repetition_guard_cooldown_tokens=args.repetition_guard_cooldown_tokens,
    )

    _, history, summary = adaptive_generate(
        prompt=args.prompt,
        config=config,
    )

    print_adaptive_report(summary)
    save_adaptive_history(history, args.csv)
    print(f"CSV salvo em: {args.csv}")


if __name__ == "__main__":
    main()
