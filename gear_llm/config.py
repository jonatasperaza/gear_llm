from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"


@dataclass
class RouterConfig:
    entropy_weight: float = 0.18
    surprisal_weight: float = 0.22
    novelty_weight: float = 0.18
    curvature_weight: float = 0.17
    structural_weight: float = 0.25

    cheap_threshold: float = 0.35
    expensive_threshold: float = 0.70

    novelty_window: int = 8
    continuation_factor: float = 0.70
