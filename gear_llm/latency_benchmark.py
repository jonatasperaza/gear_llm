import time
from collections import defaultdict
from pathlib import Path

import torch

from gear_llm.adaptive_generator import AdaptiveGenerationConfig
from gear_llm.hybrid_router import (
    choose_mode,
    classify_prompt,
    generate_with_mode,
    load_hybrid_models,
)
from gear_llm.model_loader import (
    get_cheap_tokenizer,
    get_expensive_tokenizer,
    get_model_runtime_metadata,
    prompt_format_metadata,
)
from gear_llm.quality_benchmark import (
    PROMPTS,
    estimated_saved_percent,
    generate_greedy_with_model,
)
from gear_llm.report import save_csv


LATENCY_MODES = (
    "cheap_only",
    "expensive_only",
    "adaptive_calibrated",
    "adaptive_guarded_v3",
    "speculative_adaptive",
    "hybrid",
)


def _cuda_devices(*devices: str) -> list[str]:
    unique = []
    for device in devices:
        if str(device).startswith("cuda") and device not in unique:
            unique.append(device)
    return unique


def _synchronize_cuda_devices(devices: list[str]):
    for device in devices:
        torch.cuda.synchronize(device)


def _reset_peak_memory_stats(devices: list[str]):
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def _max_memory_allocated(devices: list[str]) -> int:
    return sum(int(torch.cuda.max_memory_allocated(device)) for device in devices)


def run_latency_benchmark(
    prompts: dict[str, str] | None = None,
    cheap_model_name: str = AdaptiveGenerationConfig.cheap_model_name,
    expensive_model_name: str = AdaptiveGenerationConfig.expensive_model_name,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    device: str = "auto",
    cheap_device: str | None = None,
    expensive_device: str | None = None,
    torch_dtype: str = "auto",
    prompt_format: str = "auto",
    models=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    if prompts is None:
        prompts = PROMPTS

    if models is None:
        cheap_model, expensive_model, tokenizer, device = load_hybrid_models(
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            cheap_device=cheap_device,
            expensive_device=expensive_device,
            torch_dtype=torch_dtype,
            prompt_format=prompt_format,
        )
    else:
        cheap_model, expensive_model, tokenizer, device = models

    cheap_runtime_metadata = get_model_runtime_metadata(
        cheap_model,
        fallback_device=device,
    )
    expensive_runtime_metadata = get_model_runtime_metadata(
        expensive_model,
        fallback_device=device,
    )
    if cheap_runtime_metadata["torch_dtype"] != expensive_runtime_metadata["torch_dtype"]:
        raise ValueError(
            "cheap_model e expensive_model precisam usar o mesmo dtype. "
            f"cheap={cheap_runtime_metadata}, expensive={expensive_runtime_metadata}"
        )

    runtime_torch_dtype = cheap_runtime_metadata["torch_dtype"]
    runtime_prompt_metadata = prompt_format_metadata(tokenizer, prompt_format)
    runtime_device = (
        cheap_runtime_metadata["device"]
        if cheap_runtime_metadata["device"] == expensive_runtime_metadata["device"]
        else "split"
    )

    rows = []

    for prompt_name, prompt in prompts.items():
        for mode in LATENCY_MODES:
            for _ in range(warmup_runs):
                _run_generation(
                    prompt=prompt,
                    mode=mode,
                    cheap_model=cheap_model,
                    expensive_model=expensive_model,
                    tokenizer=tokenizer,
                    device=cheap_runtime_metadata["device"],
                    cheap_device=cheap_runtime_metadata["device"],
                    expensive_device=expensive_runtime_metadata["device"],
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    prompt_format=prompt_format,
                )

        mode_order = list(LATENCY_MODES)
        for run_index in range(measured_runs):
            measured_order = (
                mode_order if run_index % 2 == 0 else list(reversed(mode_order))
            )
            for mode in measured_order:
                rows.append(
                    _measure_generation(
                        prompt_name=prompt_name,
                        prompt=prompt,
                        mode=mode,
                        run_index=run_index,
                        cheap_model=cheap_model,
                        expensive_model=expensive_model,
                        tokenizer=tokenizer,
                        device=cheap_runtime_metadata["device"],
                        cheap_device=cheap_runtime_metadata["device"],
                        expensive_device=expensive_runtime_metadata["device"],
                        cheap_model_name=cheap_model_name,
                        expensive_model_name=expensive_model_name,
                        runtime_torch_dtype=runtime_torch_dtype,
                        runtime_device=runtime_device,
                        runtime_cheap_device=cheap_runtime_metadata["device"],
                        runtime_expensive_device=expensive_runtime_metadata["device"],
                        runtime_prompt_metadata=runtime_prompt_metadata,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        prompt_format=prompt_format,
                    )
                )

    summary_rows = summarize_latency_rows(rows)
    winner_rows = build_latency_winners(summary_rows)

    return rows, summary_rows, winner_rows


def _measure_generation(
    prompt_name: str,
    prompt: str,
    mode: str,
    run_index: int,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    cheap_device: str,
    expensive_device: str,
    cheap_model_name: str,
    expensive_model_name: str,
    runtime_torch_dtype: str,
    runtime_device: str,
    runtime_cheap_device: str,
    runtime_expensive_device: str,
    runtime_prompt_metadata: dict,
    max_new_tokens: int,
    temperature: float,
    prompt_format: str,
) -> dict:
    cuda_devices = _cuda_devices(cheap_device, expensive_device)

    if cuda_devices:
        torch.cuda.empty_cache()
        _reset_peak_memory_stats(cuda_devices)
        _synchronize_cuda_devices(cuda_devices)

    total_start = time.perf_counter()
    prompt_type = ""
    selected_mode = ""
    generation_mode = mode
    route_time = 0.0

    if mode == "hybrid":
        route_start = time.perf_counter()
        prompt_type = classify_prompt(prompt)
        selected_mode = choose_mode(prompt_type, prompt)
        generation_mode = selected_mode
        route_time = time.perf_counter() - route_start

    generation_start = time.perf_counter()
    result = run_selected_generation_mode(
        prompt=prompt,
        selected_mode=generation_mode,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        cheap_device=cheap_device,
        expensive_device=expensive_device,
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_format=prompt_format,
    )

    if cuda_devices:
        _synchronize_cuda_devices(cuda_devices)

    generation_time = time.perf_counter() - generation_start
    total_time = time.perf_counter() - total_start
    overhead_time = max(0.0, total_time - route_time - generation_time)
    generated_tokens = result["generated_tokens"]
    tokens_per_second = generated_tokens / total_time if total_time else 0.0
    memory_peak_bytes = (
        _max_memory_allocated(cuda_devices) if cuda_devices else 0
    )

    return {
        "prompt_name": prompt_name,
        "mode": mode,
        "cheap_model_name": cheap_model_name,
        "expensive_model_name": expensive_model_name,
        "selected_mode": selected_mode,
        "prompt_type": prompt_type,
        "run_index": run_index,
        "device": runtime_device,
        "cheap_device": runtime_cheap_device,
        "expensive_device": runtime_expensive_device,
        "torch_dtype": runtime_torch_dtype,
        "prompt_format": runtime_prompt_metadata["prompt_format"],
        "effective_prompt_format_cheap": runtime_prompt_metadata[
            "effective_prompt_format_cheap"
        ],
        "effective_prompt_format_expensive": runtime_prompt_metadata[
            "effective_prompt_format_expensive"
        ],
        "total_time_seconds": total_time,
        "route_time_seconds": route_time,
        "generation_time_seconds": generation_time,
        "overhead_time_seconds": overhead_time,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "estimated_saved_percent": result["estimated_saved_percent"],
        "expensive_model_calls": result["expensive_model_calls"],
        "cheap_accepted_tokens": result.get("cheap_accepted_tokens", ""),
        "acceptance_rate": result.get("acceptance_rate", ""),
        "memory_peak_bytes": memory_peak_bytes,
        "memory_peak_mb": memory_peak_bytes / (1024 * 1024),
        "generated_text": result["generated_text"],
    }


def _run_generation(
    prompt: str,
    mode: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    cheap_device: str,
    expensive_device: str,
    cheap_model_name: str,
    expensive_model_name: str,
    max_new_tokens: int,
    temperature: float,
    prompt_format: str,
) -> dict:
    if mode == "hybrid":
        prompt_type = classify_prompt(prompt)
        selected_mode = choose_mode(prompt_type, prompt)
        result = run_selected_generation_mode(
            prompt=prompt,
            selected_mode=selected_mode,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            tokenizer=tokenizer,
            device=device,
            cheap_device=cheap_device,
            expensive_device=expensive_device,
            cheap_model_name=cheap_model_name,
            expensive_model_name=expensive_model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
        )
        return {
            **result,
            "selected_mode": selected_mode,
            "prompt_type": prompt_type,
        }

    return run_selected_generation_mode(
        prompt=prompt,
        selected_mode=mode,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=device,
        cheap_device=cheap_device,
        expensive_device=expensive_device,
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_format=prompt_format,
    )


def run_selected_generation_mode(
    prompt: str,
    selected_mode: str,
    cheap_model,
    expensive_model,
    tokenizer,
    device: str,
    cheap_device: str,
    expensive_device: str,
    cheap_model_name: str,
    expensive_model_name: str,
    max_new_tokens: int,
    temperature: float,
    prompt_format: str,
) -> dict:
    if selected_mode == "cheap_only":
        cheap_tokenizer = get_cheap_tokenizer(tokenizer)
        generated_text, generated_tokens = generate_greedy_with_model(
            prompt=prompt,
            model=cheap_model,
            tokenizer=cheap_tokenizer,
            device=cheap_device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
        )
        return {
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "estimated_saved_percent": estimated_saved_percent(
                total_generated_tokens=generated_tokens,
                cheap_calls=generated_tokens,
                expensive_calls=0,
            ),
            "expensive_model_calls": 0,
            "cheap_accepted_tokens": generated_tokens,
            "acceptance_rate": 1.0 if generated_tokens else 0.0,
        }

    if selected_mode == "expensive_only":
        expensive_tokenizer = get_expensive_tokenizer(tokenizer)
        generated_text, generated_tokens = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=expensive_tokenizer,
            device=expensive_device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
        )
        return {
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "estimated_saved_percent": 0.0,
            "expensive_model_calls": generated_tokens,
            "cheap_accepted_tokens": "",
            "acceptance_rate": "",
        }

    summary = generate_with_mode(
        prompt=prompt,
        mode=selected_mode,
        cheap_model=cheap_model,
        expensive_model=expensive_model,
        tokenizer=tokenizer,
        device=cheap_device,
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_format=prompt_format,
    )
    return _summary_result(summary=summary)


def _summary_result(
    summary: dict,
) -> dict:
    return {
        "generated_text": summary["generated_text"],
        "generated_tokens": summary["total_generated_tokens"],
        "estimated_saved_percent": summary["estimated_saved_percent"],
        "expensive_model_calls": summary["expensive_model_calls"],
        "cheap_accepted_tokens": summary.get("cheap_accepted_tokens", ""),
        "acceptance_rate": summary.get("acceptance_rate", ""),
    }


def summarize_latency_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in rows:
        groups[(row["prompt_name"], row["mode"])].append(row)

    summary_rows = []
    for (prompt_name, mode), group_rows in sorted(groups.items()):
        first_row = group_rows[0]
        avg_total_time = _average(
            row["total_time_seconds"] for row in group_rows
        )
        selected_modes = sorted(
            {
                str(row.get("selected_mode", ""))
                for row in group_rows
                if row.get("selected_mode", "")
            }
        )
        prompt_types = sorted(
            {
                str(row.get("prompt_type", ""))
                for row in group_rows
                if row.get("prompt_type", "")
            }
        )
        summary_rows.append(
            {
                "prompt_name": prompt_name,
                "mode": mode,
                "cheap_model_name": first_row.get("cheap_model_name", ""),
                "expensive_model_name": first_row.get(
                    "expensive_model_name",
                    "",
                ),
                "device": first_row.get("device", ""),
                "cheap_device": first_row.get("cheap_device", ""),
                "expensive_device": first_row.get("expensive_device", ""),
                "torch_dtype": first_row.get("torch_dtype", ""),
                "prompt_format": first_row.get("prompt_format", ""),
                "effective_prompt_format_cheap": first_row.get(
                    "effective_prompt_format_cheap",
                    "",
                ),
                "effective_prompt_format_expensive": first_row.get(
                    "effective_prompt_format_expensive",
                    "",
                ),
                "selected_mode": "+".join(selected_modes),
                "prompt_type": "+".join(prompt_types),
                "run_count": len(group_rows),
                "avg_total_time_seconds": avg_total_time,
                "avg_tokens_per_second": _average(
                    row["tokens_per_second"] for row in group_rows
                ),
                "avg_estimated_saved_percent": _average(
                    row["estimated_saved_percent"] for row in group_rows
                ),
                "avg_expensive_model_calls": _average(
                    row["expensive_model_calls"] for row in group_rows
                ),
                "avg_memory_peak_mb": _average(
                    row["memory_peak_mb"] for row in group_rows
                ),
            }
        )

    expensive_baselines = {
        row["prompt_name"]: row["avg_total_time_seconds"]
        for row in summary_rows
        if row["mode"] == "expensive_only"
    }

    for row in summary_rows:
        expensive_baseline = expensive_baselines.get(row["prompt_name"], 0.0)
        avg_total_time = row["avg_total_time_seconds"]
        latency_ratio = (
            avg_total_time / expensive_baseline
            if expensive_baseline
            else 0.0
        )
        real_speedup = (
            100 * (expensive_baseline - avg_total_time) / expensive_baseline
            if expensive_baseline
            else 0.0
        )

        row["expensive_baseline_seconds"] = expensive_baseline
        row["latency_ratio_vs_expensive"] = latency_ratio
        row["real_speedup_vs_expensive_percent"] = real_speedup

    return summary_rows


def build_latency_winners(summary_rows: list[dict]) -> list[dict]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)

    for row in summary_rows:
        by_prompt[row["prompt_name"]].append(row)

    winner_rows = []
    for prompt_name, rows in sorted(by_prompt.items()):
        fastest_including_cheap = min(
            rows,
            key=lambda row: row["avg_total_time_seconds"],
        )
        rows_excluding_cheap = [
            row for row in rows if row["mode"] != "cheap_only"
        ]
        fastest_excluding_cheap = min(
            rows_excluding_cheap,
            key=lambda row: row["avg_total_time_seconds"],
        )
        expensive_only = next(
            (row for row in rows if row["mode"] == "expensive_only"),
            None,
        )
        expensive_seconds = (
            expensive_only["avg_total_time_seconds"] if expensive_only else 0.0
        )
        fastest_excluding_seconds = fastest_excluding_cheap[
            "avg_total_time_seconds"
        ]
        best_speedup = (
            100
            * (expensive_seconds - fastest_excluding_seconds)
            / expensive_seconds
            if expensive_seconds
            else 0.0
        )

        winner_rows.append(
            {
                "prompt_name": prompt_name,
                "cheap_model_name": fastest_excluding_cheap.get(
                    "cheap_model_name",
                    "",
                ),
                "expensive_model_name": fastest_excluding_cheap.get(
                    "expensive_model_name",
                    "",
                ),
                "device": fastest_excluding_cheap.get("device", ""),
                "cheap_device": fastest_excluding_cheap.get("cheap_device", ""),
                "expensive_device": fastest_excluding_cheap.get(
                    "expensive_device",
                    "",
                ),
                "torch_dtype": fastest_excluding_cheap.get(
                    "torch_dtype",
                    "",
                ),
                "prompt_format": fastest_excluding_cheap.get(
                    "prompt_format",
                    "",
                ),
                "effective_prompt_format_cheap": fastest_excluding_cheap.get(
                    "effective_prompt_format_cheap",
                    "",
                ),
                "effective_prompt_format_expensive": (
                    fastest_excluding_cheap.get(
                        "effective_prompt_format_expensive",
                        "",
                    )
                ),
                "fastest_mode_including_cheap": fastest_including_cheap[
                    "mode"
                ],
                "fastest_mode_excluding_cheap": fastest_excluding_cheap[
                    "mode"
                ],
                "expensive_only_seconds": expensive_seconds,
                "fastest_seconds_excluding_cheap": fastest_excluding_seconds,
                "best_real_speedup_excluding_cheap_percent": best_speedup,
            }
        )

    return winner_rows


def print_latency_benchmark_report(summary_rows: list[dict]):
    print()
    print("Latency Benchmark Summary")
    print("=" * 136)
    header = (
        f"{'prompt':<16} | {'mode':<22} | {'avg sec':>9} | "
        f"{'tok/s':>9} | {'saved %':>8} | {'calls':>7} | "
        f"{'real speedup':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['prompt_name']:<16} | "
            f"{row['mode']:<22} | "
            f"{row['avg_total_time_seconds']:>9.4f} | "
            f"{row['avg_tokens_per_second']:>9.2f} | "
            f"{row['avg_estimated_saved_percent']:>7.2f}% | "
            f"{row['avg_expensive_model_calls']:>7.2f} | "
            f"{row['real_speedup_vs_expensive_percent']:>11.2f}%"
        )

    print("=" * 136)
    print_latency_sanity_warnings(summary_rows)
    print()


def print_latency_sanity_warnings(summary_rows: list[dict]):
    by_prompt_mode = {
        (row["prompt_name"], row["mode"]): row for row in summary_rows
    }
    warnings = []

    for row in summary_rows:
        if row["mode"] != "hybrid":
            continue
        if row.get("selected_mode") != "speculative_adaptive":
            continue

        direct = by_prompt_mode.get(
            (row["prompt_name"], "speculative_adaptive")
        )
        if not direct:
            continue

        hybrid_time = row["avg_total_time_seconds"]
        direct_time = direct["avg_total_time_seconds"]
        if direct_time and hybrid_time > 2 * direct_time:
            warnings.append(
                (
                    row["prompt_name"],
                    hybrid_time,
                    direct_time,
                    hybrid_time / direct_time,
                )
            )

    if not warnings:
        return

    print()
    print("Latency sanity warnings")
    print("-" * 80)
    for prompt_name, hybrid_time, direct_time, ratio in warnings:
        print(
            "WARNING: hybrid selected speculative_adaptive for "
            f"{prompt_name}, but hybrid avg time {hybrid_time:.4f}s is "
            f"{ratio:.2f}x direct speculative_adaptive {direct_time:.4f}s."
        )


def save_latency_benchmark_outputs(
    rows: list[dict],
    summary_rows: list[dict],
    winner_rows: list[dict],
    output_dir: str | Path = "results",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detailed_csv = output_path / "latency_benchmark.csv"
    summary_csv = output_path / "latency_benchmark_summary.csv"
    winners_csv = output_path / "latency_winners.csv"

    save_csv(rows, str(detailed_csv))
    save_csv(summary_rows, str(summary_csv))
    save_csv(winner_rows, str(winners_csv))

    return detailed_csv, summary_csv, winners_csv


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
