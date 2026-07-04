import re
import unicodedata

from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    adaptive_generate_with_models,
)
from gear_llm.speculative_generator import (
    SpeculativeGenerationConfig,
    load_speculative_models,
    speculative_generate_with_models,
)


HYBRID_MODES = (
    "adaptive_calibrated",
    "adaptive_guarded_v3",
    "speculative_adaptive",
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", normalize_text(text)))


def classify_prompt(prompt: str) -> str:
    """
    Classifica o prompt com heuristicas simples e transparentes.

    A ordem evita falsos positivos comuns: codigo vem antes de matematica,
    e logica exige sinal forte ou mais de um marcador logico.
    """

    normalized = normalize_text(prompt)
    words = _word_set(prompt)

    code_words = {
        "def",
        "function",
        "return",
        "if",
        "else",
        "for",
        "while",
        "class",
        "import",
        "const",
        "let",
        "var",
        "codigo",
    }
    if words & code_words:
        return "code"

    logic_words = {
        "nao",
        "se",
        "apenas",
        "exceto",
        "mas",
        "porem",
        "entao",
        "bloqueado",
        "valido",
        "invalido",
        "permitido",
        "proibido",
    }
    strong_logic_words = {
        "exceto",
        "porem",
        "entao",
        "bloqueado",
        "valido",
        "invalido",
        "permitido",
        "proibido",
    }
    logic_hits = words & logic_words
    if len(logic_hits) >= 2 or logic_hits & strong_logic_words:
        return "logic"

    math_phrases = (
        "f(x)",
        "inversa",
        "equacao",
        "formula",
        "calcule",
        "explique por que",
    )
    math_symbols = set("=+-*/^")
    if any(phrase in normalized for phrase in math_phrases):
        return "math"
    if any(symbol in prompt for symbol in math_symbols):
        return "math"

    word_count = len(re.findall(r"\b\w+\b", normalized))
    if word_count > 40:
        return "long_simple"

    return "general"


def choose_mode(prompt_type: str) -> str:
    if prompt_type == "logic":
        return "adaptive_guarded_v3"
    if prompt_type == "math":
        return "speculative_adaptive"
    return "adaptive_calibrated"


def adaptive_calibrated_config(
    cheap_model_name: str,
    expensive_model_name: str,
    max_new_tokens: int,
    temperature: float,
) -> AdaptiveGenerationConfig:
    return AdaptiveGenerationConfig(
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        entropy_threshold=0.35,
        margin_threshold=0.20,
        enable_periodic_teacher_check=False,
        enable_repetition_guard=False,
    )


def adaptive_guarded_v3_config(
    cheap_model_name: str,
    expensive_model_name: str,
    max_new_tokens: int,
    temperature: float,
) -> AdaptiveGenerationConfig:
    return AdaptiveGenerationConfig(
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
        repetition_guard_requires_uncertainty=True,
        repetition_guard_entropy_threshold=0.25,
        repetition_guard_margin_threshold=0.35,
        repetition_guard_cooldown_tokens=8,
    )


def speculative_adaptive_config(
    cheap_model_name: str,
    expensive_model_name: str,
    max_new_tokens: int,
    temperature: float,
) -> SpeculativeGenerationConfig:
    return SpeculativeGenerationConfig(
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        draft_length=6,
        verify_top_k=3,
        min_draft_length=2,
        max_draft_length=8,
    )


def load_hybrid_models(
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
):
    config = speculative_adaptive_config(
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return load_speculative_models(config)


def generate_with_mode(
    prompt: str,
    mode: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> dict:
    if mode not in HYBRID_MODES:
        raise ValueError(f"Modo desconhecido: {mode}")

    if mode == "speculative_adaptive":
        config = speculative_adaptive_config(
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        _, _, _, summary = speculative_generate_with_models(
            prompt=prompt,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            tokenizer=tokenizer,
            device=device,
            config=config,
        )
    else:
        if mode == "adaptive_guarded_v3":
            config = adaptive_guarded_v3_config(
                cheap_model_name=cheap_model_name,
                expensive_model_name=expensive_model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            config = adaptive_calibrated_config(
                cheap_model_name=cheap_model_name,
                expensive_model_name=expensive_model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        _, _, summary = adaptive_generate_with_models(
            prompt=prompt,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            tokenizer=tokenizer,
            device=device,
            config=config,
        )
        total = summary["total_generated_tokens"]
        cheap_accepted = summary["cheap_accepted_tokens"]
        summary = {
            **summary,
            "cheap_generated_tokens": total,
            "expensive_corrected_tokens": total - cheap_accepted,
            "acceptance_rate": cheap_accepted / total if total else 0.0,
        }

    return {
        **summary,
        "mode": mode,
    }


def hybrid_generate_with_models(
    prompt: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> dict:
    prompt_type = classify_prompt(prompt)
    selected_mode = choose_mode(prompt_type)
    summary = generate_with_mode(
        prompt=prompt,
        mode=selected_mode,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return {
        **summary,
        "prompt_type": prompt_type,
        "selected_mode": selected_mode,
    }
