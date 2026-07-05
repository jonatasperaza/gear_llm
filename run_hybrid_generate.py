import argparse

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.config import (
    DEVICE_CHOICES,
    PROMPT_FORMAT_CHOICES,
    TORCH_DTYPE_CHOICES,
)
from gear_llm.hybrid_router import (
    hybrid_generate_with_models,
    load_hybrid_models,
)


def main():
    parser = argparse.ArgumentParser(
        description="Geracao hibrida do GEAR-LLM com selecao automatica de modo."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt usado na geracao.",
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
        help="Modelo caro.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default="auto",
        help="Device para carregar os dois modelos.",
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
        help="Maximo de tokens novos.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura de geracao.",
    )
    args = parser.parse_args()

    cheap_model, expensive_model, tokenizer, device = load_hybrid_models(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        torch_dtype=args.torch_dtype,
        prompt_format=args.prompt_format,
    )
    summary = hybrid_generate_with_models(
        prompt=args.prompt,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt_format=args.prompt_format,
    )

    print()
    print("Hybrid Mode Router")
    print("=" * 100)
    print(f"prompt_type            : {summary['prompt_type']}")
    print(f"selected_mode          : {summary['selected_mode']}")
    print(f"estimated_saved_percent: {summary['estimated_saved_percent']:.2f}%")
    print(f"expensive_model_calls  : {summary['expensive_model_calls']}")
    print("-" * 100)
    print(summary["generated_text"])
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
