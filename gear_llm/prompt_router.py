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


def load_prompt_router_ml_v2_artifacts(path: str | Path):
    """Load a v2 model.joblib and its policy_meta.json from the same directory.

    Returns (model, policy_meta_dict).
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            "prompt_router_ml_v2 requires a trained model file. "
            f"Not found: {model_path}. Train one with "
            "python scripts/train_prompt_router_v2.py "
            "--train-csv results/router_dataset_v2/train_features.csv "
            "--val-csv results/router_dataset_v2/val_features.csv"
        )
    meta_path = model_path.parent / "policy_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "prompt_router_ml_v2 requires policy_meta.json next to the model. "
            f"Not found: {meta_path}"
        )

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "prompt_router_ml_v2 requires joblib/scikit-learn. "
            "Install project requirements first: pip install -r requirements.txt"
        ) from exc

    model = joblib.load(model_path)
    import json

    with meta_path.open("r", encoding="utf-8") as file:
        policy_meta = json.load(file)

    return model, policy_meta


def classify_prompt_router_ml_v2(
    prompt: str,
    model,
    policy_meta: dict,
    probing_features: dict | None = None,
) -> dict:
    """Route using the v2 pipeline (TF-IDF + probing features).

    If *probing_features* is None, the model receives only the prompt text
    (TF-IDF path). When probing features from a cheap-model forward pass are
    available, they are injected as additional columns so the full feature set
    is used, matching how the model was trained.

    Returns the standard router info dict with selected_mode in
    {"cheap_only", "expensive_only"}.
    """
    from gear_llm.probing_features import PROBING_FEATURE_KEYS
    import pandas as pd

    mode = policy_meta.get("mode", "l2d")
    threshold = float(policy_meta.get("threshold", 0.5))
    if mode not in {"l2d", "classifier"}:
        raise ValueError(f"Unsupported prompt_router_ml_v2 mode: {mode}")
    feature_keys = tuple(
        policy_meta.get("feature_keys") or PROBING_FEATURE_KEYS
    )

    # Build a single-row DataFrame matching the training schema.
    row = {"prompt": prompt}
    if probing_features is not None:
        for key in feature_keys:
            row[key] = float(probing_features.get(key, 0.0))
    else:
        # No probing features available: fill with zeros so the ColumnTransformer
        # can still consume the row.
        for key in feature_keys:
            row[key] = 0.0
    df = pd.DataFrame([row])

    if mode == "l2d":
        delta_pred = float(model.predict(df)[0])
        selected = "expensive_only" if delta_pred > threshold else "cheap_only"
        return {
            "selected_mode": selected,
            "reason": "ml_v2_l2d",
            "matched_features": [f"delta_pred={delta_pred:.4f}", f"threshold={threshold:.4f}"],
        }
    else:
        # classifier mode.
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            probs = model.predict_proba(df)[0]
            if "expensive_only" in classes:
                idx = classes.index("expensive_only")
                score = float(probs[idx])
            else:
                score = 0.0
            selected = "expensive_only" if score >= threshold else "cheap_only"
            return {
                "selected_mode": selected,
                "reason": "ml_v2_classifier",
                "matched_features": [f"expensive_prob={score:.4f}", f"threshold={threshold:.4f}"],
            }
        # Fallback: hard predict.
        prediction = str(model.predict(df)[0])
        if prediction not in {"cheap_only", "expensive_only"}:
            prediction = "cheap_only"
        return {
            "selected_mode": prediction,
            "reason": "ml_v2_classifier_hard",
            "matched_features": [],
        }
