import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str = "auto") -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"device inválido: {device}. Use auto, cpu ou cuda."
        )

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device=cuda solicitado, mas CUDA não está disponível.")

    return device


def get_device(device: str = "auto") -> str:
    return resolve_device(device)


def resolve_torch_dtype(torch_dtype: str = "auto", device: str = "auto") -> torch.dtype:
    resolved_device = resolve_device(device)

    if torch_dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(
            "torch_dtype inválido: "
            f"{torch_dtype}. Use auto, float32, float16 ou bfloat16."
        )

    if torch_dtype == "auto":
        return torch.float16 if resolved_device == "cuda" else torch.float32

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[torch_dtype]


def torch_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, str(dtype).replace("torch.", ""))


def load_causal_lm_model(
    model_name: str,
    device: str = "auto",
    torch_dtype: str = "auto",
):
    resolved_device = resolve_device(device)
    dtype = resolve_torch_dtype(torch_dtype, resolved_device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

    model.to(resolved_device)
    model.eval()
    model.gear_device = resolved_device
    model.gear_torch_dtype = torch_dtype_name(dtype)

    return model


def get_model_runtime_metadata(model, fallback_device: str = "auto") -> dict:
    resolved_device = getattr(model, "gear_device", None) or resolve_device(
        fallback_device
    )
    dtype_name = getattr(model, "gear_torch_dtype", "")

    if not dtype_name:
        try:
            dtype_name = torch_dtype_name(next(model.parameters()).dtype)
        except StopIteration:
            dtype_name = ""

    return {
        "device": resolved_device,
        "torch_dtype": dtype_name,
    }


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    torch_dtype: str = "auto",
):
    resolved_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_causal_lm_model(
        model_name=model_name,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )

    return model, tokenizer, resolved_device
