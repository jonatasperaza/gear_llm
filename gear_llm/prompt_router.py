import re
import unicodedata


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
