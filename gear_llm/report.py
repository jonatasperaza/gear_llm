import csv
from collections import Counter
from pathlib import Path


def clean_token(token: str) -> str:
    token = token.replace("\n", "\\n")
    token = token.replace("\t", "\\t")

    if token == " ":
        return "<space>"

    return token


def _short_token(token: str, max_len: int = 18) -> str:
    token = clean_token(token)

    if len(token) > max_len:
        return token[: max_len - 3] + "..."

    return token


def _print_route_stats(rows: list[dict]):
    if not rows:
        print("Sem tokens para resumir.")
        return

    total = len(rows)
    counts = Counter(row["route"] for row in rows)
    rho_mean = sum(row["rho"] for row in rows) / total

    print("Resumo")
    print("-" * 100)

    for route in ("cheap", "medium", "expensive"):
        count = counts.get(route, 0)
        percentage = 100 * count / total
        print(f"{route:<9}: {count:>4} tokens ({percentage:>6.2f}%)")

    print(f"rho médio : {rho_mean:>10.4f}")
    print()


def _print_rank(title: str, rows: list[dict]):
    print(title)
    print("-" * 100)

    for row in rows:
        token = _short_token(row["token"], max_len=16)
        print(
            f"{row['index']:>4} | "
            f"{token:<16} | "
            f"rho={row['rho']:.4f} | "
            f"route={row['route']}"
        )

    print()


def print_report(rows: list[dict], limit: int | None = None):
    display_rows = rows[:limit] if limit is not None else rows

    print()
    print("Resultado da análise")
    print("=" * 100)

    header = (
        f"{'idx':>4} | {'token':<18} | {'entropy':>8} | "
        f"{'surp':>8} | {'novelty':>8} | {'curv':>8} | "
        f"{'struct':>8} | {'rho':>8} | route"
    )
    print(header)
    print("-" * len(header))

    for row in display_rows:
        token = _short_token(row["token"])

        print(
            f"{row['index']:>4} | "
            f"{token:<18} | "
            f"{row['entropy']:>8.4f} | "
            f"{row['surprisal']:>8.4f} | "
            f"{row['novelty']:>8.4f} | "
            f"{row['curvature']:>8.4f} | "
            f"{row['structural_importance']:>8.4f} | "
            f"{row['rho']:>8.4f} | "
            f"{row['route']}"
        )

    print("=" * 100)
    print()

    _print_route_stats(rows)

    most_expensive = sorted(rows, key=lambda row: row["rho"], reverse=True)[:5]
    cheapest = sorted(rows, key=lambda row: row["rho"])[:5]

    _print_rank("Top 5 tokens mais caros", most_expensive)
    _print_rank("Top 5 tokens mais baratos", cheapest)


def save_csv(rows: list[dict], path: str):
    if not rows:
        return

    output_path = Path(path)

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
