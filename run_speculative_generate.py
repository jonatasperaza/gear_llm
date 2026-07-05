import argparse

from gear_llm.config import DEVICE_CHOICES, TORCH_DTYPE_CHOICES
from gear_llm.speculative_generator import (
    SpeculativeGenerationConfig,
    print_speculative_report,
    save_speculative_blocks,
    save_speculative_tokens,
    speculative_generate,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAR-LLM: Adaptive Speculative Decoding."
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
        default=SpeculativeGenerationConfig.cheap_model_name,
        help="Modelo barato usado para gerar rascunhos.",
    )
    parser.add_argument(
        "--expensive-model",
        type=str,
        default=SpeculativeGenerationConfig.expensive_model_name,
        help="Modelo caro usado para verificar os blocos.",
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
        "--max-new-tokens",
        type=int,
        default=SpeculativeGenerationConfig.max_new_tokens,
        help="Número máximo de tokens novos.",
    )
    parser.add_argument(
        "--draft-length",
        type=int,
        default=SpeculativeGenerationConfig.draft_length,
        help="Tamanho inicial do bloco de rascunho.",
    )
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=SpeculativeGenerationConfig.verify_top_k,
        help="Top-k do modelo caro usado para aceitar tokens do rascunho.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=SpeculativeGenerationConfig.temperature,
        help="Temperatura usada nas distribuições.",
    )
    parser.add_argument(
        "--min-draft-length",
        type=int,
        default=SpeculativeGenerationConfig.min_draft_length,
        help="Menor tamanho permitido para o draft adaptativo.",
    )
    parser.add_argument(
        "--max-draft-length",
        type=int,
        default=SpeculativeGenerationConfig.max_draft_length,
        help="Maior tamanho permitido para o draft adaptativo.",
    )
    parser.add_argument(
        "--blocks-csv",
        type=str,
        default="results/speculative_blocks.csv",
        help="CSV para salvar histórico por bloco.",
    )
    parser.add_argument(
        "--tokens-csv",
        type=str,
        default="results/speculative_tokens.csv",
        help="CSV para salvar histórico por token.",
    )

    args = parser.parse_args()

    config = SpeculativeGenerationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        draft_length=args.draft_length,
        verify_top_k=args.verify_top_k,
        temperature=args.temperature,
        min_draft_length=args.min_draft_length,
        max_draft_length=args.max_draft_length,
    )

    _, block_rows, token_rows, summary = speculative_generate(
        prompt=args.prompt,
        config=config,
    )

    print_speculative_report(summary)
    save_speculative_blocks(block_rows, args.blocks_csv)
    save_speculative_tokens(token_rows, args.tokens_csv)
    print(f"CSV de blocos salvo em: {args.blocks_csv}")
    print(f"CSV de tokens salvo em: {args.tokens_csv}")


if __name__ == "__main__":
    main()
