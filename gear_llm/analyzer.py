import torch

from gear_llm.config import ModelConfig, RouterConfig
from gear_llm.metrics import (
    align_next_token_metric,
    curvature_from_hidden,
    entropy_from_logits,
    novelty_from_hidden,
    structural_importance_from_tokens,
    surprisal_from_logits,
)
from gear_llm.model_loader import load_model_and_tokenizer
from gear_llm.router import (
    adjust_rho_for_tokenization,
    classify_route,
    compute_rho,
)


@torch.no_grad()
def analyze_prompt_with_model(
    prompt: str,
    model,
    tokenizer,
    device: str,
    router_config: RouterConfig,
):
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    outputs = model(
        **encoded,
        output_hidden_states=True,
        return_dict=True,
    )

    input_ids = encoded["input_ids"][0]

    # logits:
    # [batch, seq_len, vocab_size]
    logits = outputs.logits[0]

    # hidden_states é uma tupla:
    # camada 0, camada 1, camada 2...
    # vamos usar a última camada por enquanto
    last_hidden = outputs.hidden_states[-1][0]

    # Em modelos causais, logits[t-1] prevê input_ids[t].
    entropy = align_next_token_metric(entropy_from_logits(logits))
    surprisal = surprisal_from_logits(
        logits=logits,
        input_ids=input_ids,
    )
    novelty = novelty_from_hidden(
        last_hidden,
        window=router_config.novelty_window,
    )
    curvature = curvature_from_hidden(last_hidden)

    tokens = [
        tokenizer.decode(
            [token_id.item()],
            clean_up_tokenization_spaces=False,
        )
        for token_id in input_ids
    ]

    structural_importance = structural_importance_from_tokens(
        tokens,
        device=device,
    )

    rho = compute_rho(
        entropy=entropy,
        surprisal=surprisal,
        novelty=novelty,
        curvature=curvature,
        structural_importance=structural_importance,
        config=router_config,
    )

    rho = adjust_rho_for_tokenization(
        rho=rho,
        tokens=tokens,
        continuation_factor=router_config.continuation_factor,
    )

    rows = []

    for index, token_id in enumerate(input_ids):
        token_text = tokens[index]

        rho_value = float(rho[index].detach().cpu())
        route = classify_route(rho_value, router_config)

        rows.append(
            {
                "index": index,
                "token": token_text,
                "entropy": float(entropy[index].detach().cpu()),
                "surprisal": float(surprisal[index].detach().cpu()),
                "novelty": float(novelty[index].detach().cpu()),
                "curvature": float(curvature[index].detach().cpu()),
                "structural_importance": float(
                    structural_importance[index].detach().cpu()
                ),
                "rho": rho_value,
                "route": route,
            }
        )

    return rows


@torch.no_grad()
def analyze_prompt(
    prompt: str,
    model_config: ModelConfig,
    router_config: RouterConfig,
):
    model, tokenizer, device = load_model_and_tokenizer(model_config.model_name)

    return analyze_prompt_with_model(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        router_config=router_config,
    )
