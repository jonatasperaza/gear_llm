import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str):
    device = get_device()

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    model.to(device)
    model.eval()

    return model, tokenizer, device
