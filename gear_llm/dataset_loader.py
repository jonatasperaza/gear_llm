import json
from pathlib import Path


def load_prompts(path: str) -> list[dict]:
    prompts = []
    dataset_path = Path(path)

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            item = json.loads(stripped)
            missing = {"id", "category", "prompt"} - set(item)
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(
                    f"Linha {line_number} de {dataset_path} sem campos: {fields}"
                )

            prompts.append(item)

    return prompts


def filter_prompts(
    prompts: list[dict],
    categories: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    selected = prompts

    if categories:
        allowed = set(categories)
        selected = [item for item in selected if item["category"] in allowed]

    if limit is not None:
        selected = selected[:limit]

    return selected
