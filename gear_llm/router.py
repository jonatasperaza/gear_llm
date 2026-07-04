import torch

from gear_llm.config import RouterConfig
from gear_llm.metrics import robust_normalize


def compute_rho(
    entropy: torch.Tensor,
    surprisal: torch.Tensor,
    novelty: torch.Tensor,
    curvature: torch.Tensor,
    structural_importance: torch.Tensor,
    config: RouterConfig,
) -> torch.Tensor:
    """
    Calcula o score rho de cada token.

    rho = combinação de:
    - entropia
    - surprisal
    - novidade geométrica
    - curvatura semântica
    - importância estrutural
    """

    entropy_norm = entropy.clamp(0, 1)
    surprisal_norm = robust_normalize(surprisal)
    novelty_norm = robust_normalize(novelty)
    curvature_norm = robust_normalize(curvature)
    structural_norm = structural_importance.float().clamp(0, 1)

    rho = (
        config.entropy_weight * entropy_norm
        + config.surprisal_weight * surprisal_norm
        + config.novelty_weight * novelty_norm
        + config.curvature_weight * curvature_norm
        + config.structural_weight * structural_norm
    )

    return rho.clamp(0, 1)


def classify_route(rho_value: float, config: RouterConfig) -> str:
    if rho_value < config.cheap_threshold:
        return "cheap"

    if rho_value < config.expensive_threshold:
        return "medium"

    return "expensive"


def is_word_continuation(token: str, previous_token: str | None) -> bool:
    """
    Detecta tokens que parecem continuação de uma palavra.

    Exemplos:
    Expl + ique
    in + vers + a
    """

    if previous_token is None:
        return False

    if token.startswith(" "):
        return False

    current_clean = token.strip()
    previous_clean = previous_token.strip()

    if not current_clean or not previous_clean:
        return False

    return current_clean.isalpha() and previous_clean.isalpha()

def adjust_rho_for_tokenization(
    rho: torch.Tensor,
    tokens: list[str],
    continuation_factor: float,
) -> torch.Tensor:
    """
    Reduz o custo de tokens que são continuação de palavra.
    """

    adjusted = rho.clone()

    for index, token in enumerate(tokens):
        previous_token = tokens[index - 1] if index > 0 else None

        if is_word_continuation(token, previous_token):
            adjusted[index] = adjusted[index] * continuation_factor

    return adjusted.clamp(0, 1)
