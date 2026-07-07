import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gear_llm.config import PROMPT_FORMAT_CHOICES


def resolve_device(device: str = "auto") -> str:
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", str(device)):
        raise ValueError(
            f"device inválido: {device}. Use auto, cpu, cuda ou cuda:N."
        )

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"device={device} solicitado, mas CUDA não está disponível.")
        if ":" in str(device):
            index = int(str(device).split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"device={device} solicitado, mas há apenas "
                    f"{torch.cuda.device_count()} CUDA device(s)."
                )

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
        return torch.float16 if resolved_device.startswith("cuda") else torch.float32

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


def resolve_prompt_format(prompt_format: str = "auto") -> str:
    if prompt_format not in PROMPT_FORMAT_CHOICES:
        raise ValueError(
            "prompt_format inválido: "
            f"{prompt_format}. Use raw, chat ou auto."
        )

    return prompt_format


def resolve_effective_prompt_format(tokenizer, prompt_format: str = "auto") -> str:
    resolved_prompt_format = resolve_prompt_format(prompt_format)

    if resolved_prompt_format == "raw":
        return "raw"

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    if resolved_prompt_format == "chat":
        if not has_chat_template:
            raise ValueError(
                "prompt_format=chat solicitado, mas o tokenizer não possui "
                "chat_template."
            )
        return "chat"

    return "chat" if has_chat_template else "raw"


def resolve_split_devices(
    device: str = "auto",
    cheap_device: str | None = None,
    expensive_device: str | None = None,
) -> tuple[str, str, str]:
    if cheap_device is None and expensive_device is None:
        resolved = resolve_device(device)
        return resolved, resolved, resolved

    default_device = resolve_device(device)
    resolved_cheap = resolve_device(cheap_device or default_device)
    resolved_expensive = resolve_device(expensive_device or default_device)
    primary_device = resolved_cheap
    return resolved_cheap, resolved_expensive, primary_device


def encode_prompt(
    prompt: str,
    tokenizer,
    device: str,
    prompt_format: str = "auto",
) -> tuple[dict, str]:
    effective_prompt_format = resolve_effective_prompt_format(
        tokenizer,
        prompt_format,
    )

    if effective_prompt_format == "chat":
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            encoded = {"input_ids": input_ids}
    else:
        encoded = tokenizer(prompt, return_tensors="pt")

    encoded = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(encoded).items()
    }

    return encoded, effective_prompt_format


def load_tokenizer_pair(
    cheap_model_name: str,
    expensive_model_name: str,
):
    cheap_tokenizer = AutoTokenizer.from_pretrained(cheap_model_name)

    if cheap_model_name == expensive_model_name:
        expensive_tokenizer = cheap_tokenizer
    else:
        expensive_tokenizer = AutoTokenizer.from_pretrained(expensive_model_name)

    cheap_tokenizer.gear_cheap_tokenizer = cheap_tokenizer
    cheap_tokenizer.gear_expensive_tokenizer = expensive_tokenizer
    cheap_tokenizer.gear_cheap_model_name = cheap_model_name
    cheap_tokenizer.gear_expensive_model_name = expensive_model_name

    return cheap_tokenizer, expensive_tokenizer


def get_cheap_tokenizer(tokenizer):
    return getattr(tokenizer, "gear_cheap_tokenizer", tokenizer)


def get_expensive_tokenizer(tokenizer):
    return getattr(tokenizer, "gear_expensive_tokenizer", tokenizer)


def tokenizers_have_compatible_vocab(left_tokenizer, right_tokenizer) -> bool:
    if left_tokenizer is right_tokenizer:
        return True

    try:
        return left_tokenizer.get_vocab() == right_tokenizer.get_vocab()
    except Exception:
        return False


def prompt_format_metadata(
    tokenizer,
    prompt_format: str = "auto",
) -> dict:
    cheap_tokenizer = get_cheap_tokenizer(tokenizer)
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)

    return {
        "prompt_format": prompt_format,
        "effective_prompt_format_cheap": resolve_effective_prompt_format(
            cheap_tokenizer,
            prompt_format,
        ),
        "effective_prompt_format_expensive": resolve_effective_prompt_format(
            expensive_tokenizer,
            prompt_format,
        ),
    }


def ensure_shared_prompt_encoding(
    prompt: str,
    tokenizer,
    device: str,
    prompt_format: str = "auto",
    cheap_device: str | None = None,
    expensive_device: str | None = None,
) -> tuple[dict, str, str]:
    cheap_tokenizer = get_cheap_tokenizer(tokenizer)
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)
    resolved_cheap_device = resolve_device(cheap_device or device)
    resolved_expensive_device = resolve_device(expensive_device or device)

    cheap_encoded, effective_cheap = encode_prompt(
        prompt=prompt,
        tokenizer=cheap_tokenizer,
        device=resolved_cheap_device,
        prompt_format=prompt_format,
    )

    if expensive_tokenizer is cheap_tokenizer:
        return cheap_encoded, effective_cheap, effective_cheap

    if not tokenizers_have_compatible_vocab(cheap_tokenizer, expensive_tokenizer):
        raise ValueError(
            "Os modos adaptive/speculative exigem tokenizers com vocabulário "
            "compatível. Use modelos com tokenizer compatível ou rode modos "
            "single-model."
        )

    expensive_encoded, effective_expensive = encode_prompt(
        prompt=prompt,
        tokenizer=expensive_tokenizer,
        device=resolved_expensive_device,
        prompt_format=prompt_format,
    )

    cheap_ids = cheap_encoded["input_ids"].detach().cpu()
    expensive_ids = expensive_encoded["input_ids"].detach().cpu()
    if not torch.equal(cheap_ids, expensive_ids):
        raise ValueError(
            "Os modos adaptive/speculative exigem a mesma sequência de input_ids "
            "após formatação do prompt. Os tokenizers parecem compatíveis, mas "
            "o chat template ou a tokenização produziram entradas diferentes."
        )

    return cheap_encoded, effective_cheap, effective_expensive


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
