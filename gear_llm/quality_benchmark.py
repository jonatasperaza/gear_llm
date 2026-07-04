import difflib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch

from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    adaptive_generate_with_models,
    load_adaptive_models,
)
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


def normalize_word(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text


def word_set(text: str) -> set[str]:
    normalized = normalize_word(text)
    return set(re.findall(r"\b\w+\b", normalized))


def sequence_similarity(text: str, reference: str) -> float:
    return difflib.SequenceMatcher(None, text, reference).ratio()


def jaccard_similarity(text: str, reference: str) -> float:
    left = word_set(text)
    right = word_set(reference)

    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def repeated_ngram_rate(text: str, n: int) -> float:
    words = text.split()

    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)

    return repeated / len(ngrams)


@torch.no_grad()
def generate_greedy_with_model(
    prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    prompt_length = input_ids.shape[-1]
    safe_temperature = max(temperature, 1e-6)

    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits[0, -1].float() / safe_temperature
        token_id = int(torch.argmax(logits).detach().cpu())
        token_tensor = torch.tensor([[token_id]], device=device)
        input_ids = torch.cat([input_ids, token_tensor], dim=-1)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    generated_ids = input_ids[0, prompt_length:]
    generated_text = tokenizer.decode(
        generated_ids,
        clean_up_tokenization_spaces=False,
        skip_special_tokens=True,
    )

    return generated_text, int(generated_ids.numel())


def estimated_saved_percent(
    total_generated_tokens: int,
    cheap_calls: int,
    expensive_calls: int,
    cheap_cost: float = 0.35,
    expensive_cost: float = 1.00,
) -> float:
    baseline_cost = total_generated_tokens * expensive_cost

    if baseline_cost == 0:
        return 0.0

    simulated_cost = cheap_calls * cheap_cost + expensive_calls * expensive_cost
    return 100 * (baseline_cost - simulated_cost) / baseline_cost


def build_quality_row(
    prompt_name: str,
    mode: str,
    generated_text: str,
    reference_text: str,
    total_generated_tokens: int,
    cheap_accepted_tokens: int,
    expensive_model_calls: int,
    saved_percent: float,
) -> dict:
    return {
        "prompt_name": prompt_name,
        "mode": mode,
        "generated_text": generated_text,
        "total_generated_tokens": total_generated_tokens,
        "cheap_accepted_tokens": cheap_accepted_tokens,
        "expensive_model_calls": expensive_model_calls,
        "estimated_saved_percent": saved_percent,
        "similarity_to_expensive": sequence_similarity(
            generated_text,
            reference_text,
        ),
        "jaccard_to_expensive": jaccard_similarity(
            generated_text,
            reference_text,
        ),
        "repeated_3gram_rate": repeated_ngram_rate(generated_text, 3),
        "repeated_4gram_rate": repeated_ngram_rate(generated_text, 4),
    }


def run_quality_benchmark(
    prompts: dict[str, str] | None = None,
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> list[dict]:
    if prompts is None:
        prompts = PROMPTS

    base_config = AdaptiveGenerationConfig(
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    cheap_model, expensive_model, tokenizer, device = load_adaptive_models(base_config)
    rows = []

    for prompt_name, prompt in prompts.items():
        expensive_text, expensive_tokens = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        rows.append(
            build_quality_row(
                prompt_name=prompt_name,
                mode="expensive_only",
                generated_text=expensive_text,
                reference_text=expensive_text,
                total_generated_tokens=expensive_tokens,
                cheap_accepted_tokens=0,
                expensive_model_calls=expensive_tokens,
                saved_percent=0.0,
            )
        )

        cheap_text, cheap_tokens = generate_greedy_with_model(
            prompt=prompt,
            model=cheap_model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        cheap_saved = estimated_saved_percent(
            total_generated_tokens=cheap_tokens,
            cheap_calls=cheap_tokens,
            expensive_calls=0,
        )

        rows.append(
            build_quality_row(
                prompt_name=prompt_name,
                mode="cheap_only",
                generated_text=cheap_text,
                reference_text=expensive_text,
                total_generated_tokens=cheap_tokens,
                cheap_accepted_tokens=cheap_tokens,
                expensive_model_calls=0,
                saved_percent=cheap_saved,
            )
        )

        adaptive_modes = (
            (
                "adaptive_calibrated",
                AdaptiveGenerationConfig(
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    entropy_threshold=0.35,
                    margin_threshold=0.20,
                    enable_periodic_teacher_check=False,
                    enable_repetition_guard=False,
                ),
            ),
            (
                "adaptive_guarded",
                AdaptiveGenerationConfig(
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    entropy_threshold=0.35,
                    margin_threshold=0.20,
                    teacher_check_interval=16,
                    enable_periodic_teacher_check=True,
                    enable_repetition_guard=True,
                    repetition_ngram_size=3,
                    repetition_threshold=0.25,
                    risk_gated_periodic_check=False,
                    max_expensive_call_ratio=1.00,
                ),
            ),
            (
                "adaptive_guarded_v2",
                AdaptiveGenerationConfig(
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    entropy_threshold=0.35,
                    margin_threshold=0.20,
                    teacher_check_interval=16,
                    enable_periodic_teacher_check=True,
                    enable_repetition_guard=True,
                    repetition_ngram_size=3,
                    repetition_threshold=0.25,
                    risk_gated_periodic_check=True,
                    periodic_entropy_risk_threshold=0.25,
                    periodic_margin_risk_threshold=0.35,
                    periodic_repetition_risk_threshold=0.05,
                    max_expensive_call_ratio=0.40,
                ),
            ),
        )

        for mode, adaptive_config in adaptive_modes:
            _, _, adaptive_summary = adaptive_generate_with_models(
                prompt=prompt,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                device=device,
                config=adaptive_config,
            )

            rows.append(
                build_quality_row(
                    prompt_name=prompt_name,
                    mode=mode,
                    generated_text=adaptive_summary["generated_text"],
                    reference_text=expensive_text,
                    total_generated_tokens=adaptive_summary[
                        "total_generated_tokens"
                    ],
                    cheap_accepted_tokens=adaptive_summary[
                        "cheap_accepted_tokens"
                    ],
                    expensive_model_calls=adaptive_summary[
                        "expensive_model_calls"
                    ],
                    saved_percent=adaptive_summary["estimated_saved_percent"],
                )
            )

    return rows


def print_quality_benchmark_report(rows: list[dict]):
    print()
    print("Quality-vs-Cost Benchmark")
    print("=" * 120)
    header = (
        f"{'prompt':<16} | {'mode':<20} | {'saved %':>8} | "
        f"{'seq sim':>8} | {'jaccard':>8} | {'rep3':>8} | "
        f"{'rep4':>8} | {'calls':>5}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['prompt_name']:<16} | "
            f"{row['mode']:<20} | "
            f"{row['estimated_saved_percent']:>7.2f}% | "
            f"{row['similarity_to_expensive']:>8.4f} | "
            f"{row['jaccard_to_expensive']:>8.4f} | "
            f"{row['repeated_3gram_rate']:>8.4f} | "
            f"{row['repeated_4gram_rate']:>8.4f} | "
            f"{row['expensive_model_calls']:>5}"
        )

    print("=" * 120)
    print()


def save_quality_benchmark(rows: list[dict], path: str | Path):
    save_csv(rows, str(path))
