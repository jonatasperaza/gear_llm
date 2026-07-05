from dataclasses import dataclass


DEFAULT_CHEAP_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_EXPENSIVE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

DEVICE_CHOICES = ("auto", "cpu", "cuda")
TORCH_DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")
PROMPT_FORMAT_CHOICES = ("raw", "chat", "auto")


@dataclass
class ModelConfig:
    model_name: str = DEFAULT_CHEAP_MODEL
    device: str = "auto"
    torch_dtype: str = "auto"


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
