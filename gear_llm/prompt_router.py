import re
import unicodedata
from pathlib import Path


def _normalize_prompt(prompt: str) -> str:
    text = prompt.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


CHEAP_FEATURES = (
    ("sort_list", r"\bsort(?:ed|ing)?\b.*\blist\b|\blist\b.*\bsort"),
    ("sum_dictionary_items", r"\bsum\b.*\b(dictionary|dict)\b"),
    ("circumference_or_perimeter", r"\b(circumference|perimeter)\b"),
    ("first_repeated_character", r"\bfirst repeated character\b"),
    ("convert_string_to_list", r"\b(convert|change)\b.*\bstring\b.*\blist\b"),
    ("kth_element", r"\bk(?:th)?\s+(?:element|item)\b"),
    ("first_element_each_sublist", r"\bfirst element\b.*\bsublist"),
    ("n_largest_integers", r"\b(?:n|k)\s+largest\b.*\b(integer|number|element)"),
    (
        "sum_non_repeated_elements",
        r"\bsum\b.*\b(non[- ]?repeated|not repeated|unique)\b",
    ),
)

EXPENSIVE_FEATURES = (
    ("raised_to_power", r"\b(raised to|power|fifth power|fourth power)\b"),
    (
        "sum_squares_even_naturals",
        r"\bsum\b.*\bsquares?\b.*\bfirst\b.*\beven\b.*\bnatural",
    ),
    ("tuple_adjacency", r"\bt[_ ]?i\b.*\bt[_ ]?i\s*\+\s*1|\badjacent\b.*\btuple"),
    ("remove_lowercase_substrings", r"\bremove\b.*\blowercase\b.*\bsubstring"),
    ("lateral_surface_area", r"\blateral surface area\b"),
    ("closest_smaller_number", r"\bclosest\b.*\bsmaller\b.*\bnumber"),
    ("next_perfect_square", r"\bnext\b.*\bperfect square\b"),
    ("count_equal_three_ints", r"\bcount\b.*\bequal\b.*\bthree\b.*\b(integer|number)"),
    (
        "max_product_increasing_subsequence",
        r"\bmax(?:imum)? product\b.*\bincreasing\b.*\bsubsequence",
    ),
    (
        "largest_sum_repeated_array",
        r"\blargest sum\b.*\bcontiguous\b.*\barray\b.*\brepeated\b.*\bk\b",
    ),
)


EXPENSIVE_FEATURES_V2 = (
    ("fifth_power", r"\bfifth power\b|\braised to the fifth power\b"),
    ("fourth_power", r"\bfourth power\b|\braised to the fourth power\b"),
    (
        "sum_squares_even_naturals",
        r"\bsum of squares\b.*\bfirst\b.*\beven\b.*\bnatural numbers\b",
    ),
    ("tuple_product_adjacency", r"\bt[_ ]?\{?i\}?\s*\*\s*t[_ ]?\{?i\s*\+\s*1\}?"),
    ("tuple_length_n", r"\btuple\b.*\blength n\b|\btuple of length n\b"),
    ("i_th_element", r"\bi[- ]th element\b"),
    ("remove_lowercase_substrings", r"\bremove\b.*\blowercase\b.*\bsubstrings?\b"),
    ("lateral_surface_area", r"\blateral surface area\b"),
    ("closest_smaller_number", r"\bclosest smaller number\b"),
    ("next_perfect_square", r"\bnext perfect square\b"),
    (
        "count_equal_three_numbers",
        r"\bcount\b.*\bnumber of equal numbers\b.*\bfrom three\b",
    ),
)


def _matched_features(prompt: str, features: tuple[tuple[str, str], ...]) -> list[str]:
    return [name for name, pattern in features if re.search(pattern, prompt)]


def classify_prompt_router_v1(prompt: str) -> dict:
    """Route a whole prompt to cheap_only or expensive_only before generation."""
    normalized = _normalize_prompt(prompt)
    expensive_matches = _matched_features(normalized, EXPENSIVE_FEATURES)
    cheap_matches = _matched_features(normalized, CHEAP_FEATURES)

    if expensive_matches:
        return {
            "selected_mode": "expensive_only",
            "reason": "matched_expensive_features",
            "matched_features": expensive_matches,
        }

    if cheap_matches:
        return {
            "selected_mode": "cheap_only",
            "reason": "matched_cheap_features",
            "matched_features": cheap_matches,
        }

    return {
        "selected_mode": "expensive_only",
        "reason": "default_conservative",
        "matched_features": [],
    }


def classify_prompt_router_v2(prompt: str) -> dict:
    """More aggressive prompt-level router: cheap_only by default."""
    normalized = _normalize_prompt(prompt)
    expensive_matches = _matched_features(normalized, EXPENSIVE_FEATURES_V2)

    if expensive_matches:
        return {
            "selected_mode": "expensive_only",
            "reason": "matched_high_confidence_expensive_features",
            "matched_features": expensive_matches,
        }

    return {
        "selected_mode": "cheap_only",
        "reason": "default_cheap",
        "matched_features": [],
    }


def load_prompt_router_ml_model(path: str | Path):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            "prompt_router_ml_v1 requires a trained model file. "
            f"Not found: {model_path}. Train one with "
            "python scripts/train_prompt_router_ml.py "
            "--router-dataset results/prompt_router_dataset.csv "
            "--output-model results/prompt_router_ml_v1.joblib"
        )

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "prompt_router_ml_v1 requires joblib/scikit-learn. "
            "Install project requirements first: pip install -r requirements.txt"
        ) from exc

    return joblib.load(model_path)


def classify_prompt_router_ml_v1(prompt: str, model) -> dict:
    prediction = str(model.predict([prompt])[0])
    if prediction not in {"cheap_only", "expensive_only"}:
        raise ValueError(
            "prompt_router_ml_v1 model predicted unsupported mode: "
            f"{prediction}. Expected cheap_only or expensive_only."
        )

    matched_features = []
    if hasattr(model, "predict_proba"):
        try:
            classes = list(model.classes_)
            probabilities = model.predict_proba([prompt])[0]
            confidence = float(probabilities[classes.index(prediction)])
            matched_features.append(f"confidence={confidence:.4f}")
        except Exception:
            pass

    return {
        "selected_mode": prediction,
        "reason": "ml_prediction",
        "matched_features": matched_features,
    }
