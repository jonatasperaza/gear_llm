import math
import re
import unicodedata

import torch
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Calcula a entropia normalizada dos logits de próxima posição.

    logits shape:
        [seq_len, vocab_size]

    retorna:
        [seq_len]
    """

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    entropy = -(probs * log_probs).sum(dim=-1)

    vocab_size = logits.shape[-1]
    max_entropy = math.log(vocab_size)

    return entropy / max_entropy


def align_next_token_metric(metric: torch.Tensor) -> torch.Tensor:
    """
    Alinha uma métrica de logits causais com o token observado.

    Em modelos causais, logits[t-1] descreve a distribuição usada para
    prever input_ids[t]. O primeiro token não tem contexto anterior.
    """

    aligned = torch.zeros_like(metric)

    if metric.numel() > 1:
        aligned[1:] = metric[:-1]

    return aligned


def curvature_from_hidden(hidden: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Curvatura semântica local:

    kappa_t = ||h_t - 2h_{t-1} + h_{t-2}|| / ||h_t||
    """

    hidden = hidden.float()
    seq_len = hidden.shape[0]
    curvature = torch.zeros(seq_len, device=hidden.device)

    if seq_len < 3:
        return curvature

    second_diff = hidden[2:] - 2 * hidden[1:-1] + hidden[:-2]
    numerator = torch.norm(second_diff, dim=-1)
    denominator = torch.norm(hidden[2:], dim=-1) + eps

    curvature[2:] = numerator / denominator

    curvature = torch.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

    return curvature


def novelty_from_hidden(
    hidden: torch.Tensor,
    window: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calcula novidade geométrica simples por cosseno.

    Ideia:
    - pega o vetor atual h_t
    - compara com a média dos vetores anteriores
    - se for muito diferente, novidade é alta

    hidden shape:
        [seq_len, hidden_dim]

    retorna:
        [seq_len]
    """

    hidden = hidden.float()
    seq_len = hidden.shape[0]
    novelty = torch.zeros(seq_len, device=hidden.device)

    for t in range(1, seq_len):
        start = max(0, t - window)
        context = hidden[start:t]

        context_mean = context.mean(dim=0)

        h_norm = F.normalize(hidden[t], dim=0, eps=eps)
        c_norm = F.normalize(context_mean, dim=0, eps=eps)

        cosine_similarity = torch.sum(h_norm * c_norm)

        # Cosseno vai de -1 até 1.
        # 1 significa muito parecido.
        # Então 1 - cosseno significa "diferença".
        novelty[t] = (1 - cosine_similarity).clamp(0, 2)

    return novelty


def minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normaliza um vetor para ficar entre 0 e 1.
    """

    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_min = torch.min(x)
    x_max = torch.max(x)

    if torch.abs(x_max - x_min) < eps:
        return torch.zeros_like(x)

    return (x - x_min) / (x_max - x_min + eps)


def robust_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalização mais robusta contra outliers.
    Usa percentis aproximados para evitar que um token absurdo distorça tudo.
    """

    if x.numel() == 0:
        return x

    x = x.float()
    finite_values = x[torch.isfinite(x)]

    if finite_values.numel() == 0:
        return torch.zeros_like(x)

    if x.numel() < 4:
        return minmax_normalize(x, eps=eps)

    low = torch.quantile(finite_values, 0.05)
    high = torch.quantile(finite_values, 0.95)

    fallback = torch.median(finite_values)
    x = torch.where(torch.isfinite(x), x, fallback)
    x = torch.clamp(x, low, high)

    if torch.abs(high - low) < eps:
        return torch.zeros_like(x)

    return ((x - low) / (high - low + eps)).clamp(0, 1)


def surprisal_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calcula o surprisal bruto de cada token real.

    surprisal_t = -log P(token_t | contexto anterior)

    logits shape:
        [seq_len, vocab_size]

    input_ids shape:
        [seq_len]

    retorna:
        [seq_len]

    Observação:
    - O primeiro token não tem contexto anterior, então recebe 0.
    - logits[t-1] prevê input_ids[t].
    """

    seq_len = input_ids.shape[0]
    surprisal = torch.zeros(seq_len, device=logits.device)

    if seq_len < 2:
        return surprisal

    log_probs = F.log_softmax(logits[:-1], dim=-1)

    target_ids = input_ids[1:]

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    raw_surprisal = -token_log_probs

    surprisal[1:] = raw_surprisal

    surprisal = torch.nan_to_num(surprisal, nan=0.0, posinf=0.0, neginf=0.0)

    return surprisal


STRUCTURAL_OPERATORS = set("+-*/=^%")
STRUCTURAL_BRACKETS = set("()[]{}")
STRUCTURAL_PUNCTUATION = {",", ":", ";"}
STRUCTURAL_SYMBOLS = STRUCTURAL_OPERATORS | STRUCTURAL_BRACKETS | STRUCTURAL_PUNCTUATION
LOGICAL_WORDS = {
    "não",
    "nao",
    "se",
    "apenas",
    "exceto",
    "mas",
    "porém",
    "porem",
    "ou",
    "e",
    "então",
    "entao",
}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _clean_token(token: str) -> str:
    token = token.replace("Ġ", " ").replace("▁", " ").strip().lower()
    return unicodedata.normalize("NFC", token)


def _raw_word(token: str) -> str:
    token = _clean_token(token)
    return re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", token)


def _word_key(token: str) -> str:
    return _strip_accents(_raw_word(token))


def _is_logical_word(token: str) -> bool:
    raw_word = _raw_word(token)

    if raw_word in LOGICAL_WORDS:
        return True

    # Evita confundir "é" com o conectivo lógico "e".
    if len(raw_word) == 1:
        return False

    return _strip_accents(raw_word) in LOGICAL_WORDS


def _starts_new_text_unit(token: str) -> bool:
    return token.startswith((" ", "\n", "\t", "Ġ", "▁"))


def _mark_split_logical_words(tokens: list[str], values: list[float]):
    max_parts = 6

    for start in range(len(tokens)):
        combined = ""
        indices: list[int] = []

        for index in range(start, min(len(tokens), start + max_parts)):
            if index > start and _starts_new_text_unit(tokens[index]):
                break

            piece = _raw_word(tokens[index])

            if not piece:
                break

            combined += piece
            indices.append(index)

            if _is_logical_word(combined):
                for logical_index in indices:
                    values[logical_index] = 1.0
                break


def _looks_numeric(token: str) -> bool:
    token = _clean_token(token)
    return bool(re.search(r"\d", token))


def _has_structural_symbol(token: str) -> bool:
    return any(char in STRUCTURAL_SYMBOLS for char in token)


def structural_importance_from_tokens(
    tokens: list[str],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Marca tokens estruturalmente importantes para raciocínio.

    A saída é simples e interpretável:
    - 1.0 para operadores, delimitadores, pontuação estrutural,
      palavras lógicas, números e variáveis curtas em contexto matemático.
    - 0.0 para os demais tokens.
    """

    values: list[float] = []
    cleaned_tokens = [_clean_token(token) for token in tokens]

    for token in tokens:
        is_structural = (
            _has_structural_symbol(token)
            or _looks_numeric(token)
            or _is_logical_word(token)
        )

        values.append(1.0 if is_structural else 0.0)

    _mark_split_logical_words(tokens, values)

    for index, token in enumerate(cleaned_tokens):
        raw_word = _raw_word(token)
        word = _word_key(token)

        is_ascii_variable = (
            len(raw_word) == 1
            and raw_word.isascii()
            and word.isalpha()
        )

        if values[index] > 0 or not is_ascii_variable:
            continue

        start = max(0, index - 2)
        stop = min(len(tokens), index + 3)
        local_context = "".join(cleaned_tokens[start:stop])

        has_math_context = (
            any(
                char in STRUCTURAL_OPERATORS or char in STRUCTURAL_BRACKETS
                for char in local_context
            )
            or any(_looks_numeric(cleaned_tokens[i]) for i in range(start, stop))
        )

        if has_math_context:
            values[index] = 1.0

    return torch.tensor(values, dtype=torch.float32, device=device)
