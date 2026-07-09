import argparse
from pathlib import Path

from gear_llm.ablation import (
    run_ablation_with_model,
    run_balanced_ablation_with_model,
    save_ablation_rows,
)
from gear_llm.adaptive_generator import (
    AdaptiveGenerationConfig,
    adaptive_generate_with_models,
    load_adaptive_models,
    save_adaptive_summary_rows,
)
from gear_llm.analyzer import analyze_prompt_with_model
from gear_llm.compute_simulator import (
    ComputeCostConfig,
    print_compute_sim_benchmark_report,
    save_compute_sim_rows,
    simulate_compute_from_rows,
)
from gear_llm.config import (
    DEVICE_CHOICES,
    PROMPT_FORMAT_CHOICES,
    TORCH_DTYPE_CHOICES,
    ModelConfig,
    RouterConfig,
)
from gear_llm.dataset_benchmark import (
    parse_categories,
    print_dataset_benchmark_report,
    run_dataset_benchmark,
    save_dataset_benchmark_outputs,
)
from gear_llm.guard_tuning import (
    print_guard_tuning_report,
    run_guard_tuning,
    save_guard_tuning,
)
from gear_llm.hybrid_router import (
    adaptive_code_quality_config,
    choose_mode,
    classify_prompt,
    generate_with_mode,
    load_hybrid_models,
)
from gear_llm.latency_benchmark import (
    print_latency_benchmark_report,
    run_latency_benchmark,
    save_latency_benchmark_outputs,
)
from gear_llm.model_loader import (
    get_expensive_tokenizer,
    get_model_runtime_metadata,
    load_model_and_tokenizer,
)
from gear_llm.mode_oracle import (
    print_mode_oracle_report,
    run_mode_oracle,
    save_mode_oracle_outputs,
)
from gear_llm.policy_replay import (
    print_policy_replay_report,
    run_policy_replay,
    save_policy_replay,
)
from gear_llm.quality_benchmark import (
    generate_greedy_with_model,
    print_quality_benchmark_report,
    run_quality_benchmark,
    save_quality_benchmark,
    sequence_similarity,
)
from gear_llm.report import save_csv
from gear_llm.speculative_generator import (
    SpeculativeGenerationConfig,
    load_speculative_models,
    save_speculative_summary_rows,
    speculative_generate_with_models,
)
from gear_llm.speculative_tuning import (
    print_speculative_tuning_report,
    run_speculative_tuning,
    save_speculative_tuning,
)
from gear_llm.teacher_calibration import (
    TeacherCalibrationConfig,
    load_teacher_models,
    run_teacher_calibration_with_models,
    save_teacher_grid,
    save_teacher_rows,
    threshold_grid_search,
)
from gear_llm.task_evaluation import (
    build_runtime_profile_outputs,
    build_task_quality_latency_outputs,
    print_runtime_profile_report,
    print_task_evaluation_overall_report,
    print_task_evaluation_report,
    print_task_quality_latency_report,
    run_task_evaluation,
    save_runtime_profile_outputs,
    save_task_quality_latency_outputs,
    save_task_evaluation_outputs,
)


PROMPTS = {
    "easy": "Explique em uma frase o que é água.",
    "math": "Explique por que a inversa de f(x)=5x+1 é (x-1)/5.",
    "logic_negation": (
        "Se não chover e apenas se o vento parar, então podemos sair; "
        "exceto se houver alerta."
    ),
    "code": (
        "Escreva uma função Python: if x % 2 == 0, retorne x / 2; "
        "caso contrário, retorne 3 * x + 1."
    ),
    "long_simple": (
        "O dia começou calmo. As pessoas caminharam pela praça, "
        "compraram pão, conversaram sobre o tempo e voltaram para casa. "
        "Nada urgente aconteceu, apenas uma sequência simples de eventos."
    ),
}


def _adaptive_benchmark_row(
    prompt_name: str,
    mode: str,
    summary: dict,
    reference_text: str,
) -> dict:
    total = summary["total_generated_tokens"]
    cheap_accepted = summary["cheap_accepted_tokens"]

    return {
        "prompt_name": prompt_name,
        "mode": mode,
        "prompt_format": summary.get("prompt_format", ""),
        "effective_prompt_format_cheap": summary.get(
            "effective_prompt_format_cheap",
            "",
        ),
        "effective_prompt_format_expensive": summary.get(
            "effective_prompt_format_expensive",
            "",
        ),
        "generated_text": summary["generated_text"],
        "total_generated_tokens": total,
        "cheap_generated_tokens": total,
        "cheap_accepted_tokens": cheap_accepted,
        "expensive_corrected_tokens": total - cheap_accepted,
        "expensive_model_calls": summary["expensive_model_calls"],
        "acceptance_rate": cheap_accepted / total if total else 0.0,
        "estimated_saved_percent": summary["estimated_saved_percent"],
        "similarity_to_expensive": sequence_similarity(
            summary["generated_text"],
            reference_text,
        ),
    }


def _speculative_benchmark_row(
    prompt_name: str,
    summary: dict,
    reference_text: str,
) -> dict:
    return {
        "prompt_name": prompt_name,
        "mode": "speculative_adaptive",
        "prompt_format": summary.get("prompt_format", ""),
        "effective_prompt_format_cheap": summary.get(
            "effective_prompt_format_cheap",
            "",
        ),
        "effective_prompt_format_expensive": summary.get(
            "effective_prompt_format_expensive",
            "",
        ),
        "generated_text": summary["generated_text"],
        "total_generated_tokens": summary["total_generated_tokens"],
        "cheap_generated_tokens": summary["cheap_generated_tokens"],
        "cheap_accepted_tokens": summary["cheap_accepted_tokens"],
        "expensive_corrected_tokens": summary["expensive_corrected_tokens"],
        "expensive_model_calls": summary["expensive_model_calls"],
        "acceptance_rate": summary["acceptance_rate"],
        "estimated_saved_percent": summary["estimated_saved_percent"],
        "similarity_to_expensive": sequence_similarity(
            summary["generated_text"],
            reference_text,
        ),
    }


def print_speculative_benchmark_report(rows: list[dict]):
    print()
    print("Speculative Benchmark")
    print("=" * 120)
    header = (
        f"{'prompt':<16} | {'mode':<22} | {'saved %':>8} | "
        f"{'calls':>5} | {'total':>5} | {'cheap ok':>8} | "
        f"{'corr':>5} | {'accept':>8} | {'sim':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['prompt_name']:<16} | "
            f"{row['mode']:<22} | "
            f"{row['estimated_saved_percent']:>7.2f}% | "
            f"{row['expensive_model_calls']:>5} | "
            f"{row['total_generated_tokens']:>5} | "
            f"{row['cheap_accepted_tokens']:>8} | "
            f"{row['expensive_corrected_tokens']:>5} | "
            f"{row['acceptance_rate']:>7.2%} | "
            f"{row['similarity_to_expensive']:>7.4f}"
        )

    print("=" * 120)
    print()


def run_speculative_benchmark(
    prompts: dict[str, str],
    cheap_model_name: str,
    expensive_model_name: str,
    max_new_tokens: int,
    temperature: float,
    draft_length: int,
    verify_top_k: int,
    min_draft_length: int,
    max_draft_length: int,
    device: str = "auto",
    cheap_device: str | None = None,
    expensive_device: str | None = None,
    torch_dtype: str = "auto",
    prompt_format: str = "auto",
    models=None,
) -> list[dict]:
    speculative_config = SpeculativeGenerationConfig(
        cheap_model_name=cheap_model_name,
        expensive_model_name=expensive_model_name,
        device=device,
        cheap_device=cheap_device,
        expensive_device=expensive_device,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        draft_length=draft_length,
        verify_top_k=verify_top_k,
        temperature=temperature,
        prompt_format=prompt_format,
        min_draft_length=min_draft_length,
        max_draft_length=max_draft_length,
    )

    if models is None:
        cheap_model, expensive_model, tokenizer, device = load_speculative_models(
            speculative_config
        )
    else:
        cheap_model, expensive_model, tokenizer, device = models

    rows = []
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)
    cheap_runtime = get_model_runtime_metadata(cheap_model, fallback_device=device)
    expensive_runtime = get_model_runtime_metadata(
        expensive_model,
        fallback_device=device,
    )

    for prompt_name, prompt in prompts.items():
        reference_text, _ = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=expensive_tokenizer,
            device=expensive_runtime["device"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
        )
        adaptive_modes = (
            (
                "adaptive_calibrated",
                AdaptiveGenerationConfig(
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    device=cheap_runtime["device"],
                    cheap_device=cheap_runtime["device"],
                    expensive_device=expensive_runtime["device"],
                    prompt_format=prompt_format,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    entropy_threshold=0.35,
                    margin_threshold=0.20,
                    enable_periodic_teacher_check=False,
                    enable_repetition_guard=False,
                ),
            ),
            (
                "adaptive_guarded_v3",
                AdaptiveGenerationConfig(
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    device=cheap_runtime["device"],
                    cheap_device=cheap_runtime["device"],
                    expensive_device=expensive_runtime["device"],
                    prompt_format=prompt_format,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    entropy_threshold=0.35,
                    margin_threshold=0.20,
                    teacher_check_interval=16,
                    enable_periodic_teacher_check=True,
                    enable_repetition_guard=True,
                    repetition_ngram_size=3,
                    repetition_threshold=0.25,
                    risk_gated_periodic_check=True,
                    periodic_entropy_risk_threshold=0.25,
                    periodic_margin_risk_threshold=0.35,
                    periodic_repetition_risk_threshold=0.05,
                    max_expensive_call_ratio=0.40,
                    repetition_guard_requires_uncertainty=True,
                    repetition_guard_entropy_threshold=0.25,
                    repetition_guard_margin_threshold=0.35,
                    repetition_guard_cooldown_tokens=8,
                ),
            ),
            (
                "adaptive_code_quality",
                adaptive_code_quality_config(
                    cheap_model_name=cheap_model_name,
                    expensive_model_name=expensive_model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    device=cheap_runtime["device"],
                    cheap_device=cheap_runtime["device"],
                    expensive_device=expensive_runtime["device"],
                    torch_dtype=torch_dtype,
                    prompt_format=prompt_format,
                ),
            ),
        )

        for mode, adaptive_config in adaptive_modes:
            _, _, adaptive_summary = adaptive_generate_with_models(
                prompt=prompt,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                device=cheap_runtime["device"],
                config=adaptive_config,
            )
            rows.append(
                _adaptive_benchmark_row(
                    prompt_name=prompt_name,
                    mode=mode,
                    summary=adaptive_summary,
                    reference_text=reference_text,
                )
            )

        _, _, _, speculative_summary = speculative_generate_with_models(
            prompt=prompt,
            cheap_model=cheap_model,
            expensive_model=expensive_model,
            tokenizer=tokenizer,
            device=cheap_runtime["device"],
            config=speculative_config,
        )
        rows.append(
            _speculative_benchmark_row(
                prompt_name=prompt_name,
                summary=speculative_summary,
                reference_text=reference_text,
            )
        )

    return rows


def _hybrid_benchmark_row(
    prompt_name: str,
    prompt_type: str,
    mode: str,
    selected_mode: str,
    summary: dict,
    reference_text: str,
) -> dict:
    return {
        "prompt_name": prompt_name,
        "prompt_type": prompt_type,
        "mode": mode,
        "selected_mode": selected_mode,
        "prompt_format": summary.get("prompt_format", ""),
        "effective_prompt_format_cheap": summary.get(
            "effective_prompt_format_cheap",
            "",
        ),
        "effective_prompt_format_expensive": summary.get(
            "effective_prompt_format_expensive",
            "",
        ),
        "generated_text": summary["generated_text"],
        "total_generated_tokens": summary["total_generated_tokens"],
        "cheap_generated_tokens": summary["cheap_generated_tokens"],
        "cheap_accepted_tokens": summary["cheap_accepted_tokens"],
        "expensive_corrected_tokens": summary["expensive_corrected_tokens"],
        "expensive_model_calls": summary["expensive_model_calls"],
        "acceptance_rate": summary["acceptance_rate"],
        "estimated_saved_percent": summary["estimated_saved_percent"],
        "similarity_to_expensive": sequence_similarity(
            summary["generated_text"],
            reference_text,
        ),
    }


def print_hybrid_benchmark_report(rows: list[dict]):
    print()
    print("Hybrid Mode Benchmark")
    print("=" * 132)
    header = (
        f"{'prompt':<16} | {'type':<12} | {'mode':<22} | "
        f"{'selected':<22} | {'saved %':>8} | {'calls':>5} | "
        f"{'accept':>8} | {'sim':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['prompt_name']:<16} | "
            f"{row['prompt_type']:<12} | "
            f"{row['mode']:<22} | "
            f"{row['selected_mode']:<22} | "
            f"{row['estimated_saved_percent']:>7.2f}% | "
            f"{row['expensive_model_calls']:>5} | "
            f"{row['acceptance_rate']:>7.2%} | "
            f"{row['similarity_to_expensive']:>7.4f}"
        )

    print("=" * 132)
    print()


def run_hybrid_benchmark(
    prompts: dict[str, str],
    cheap_model_name: str,
    expensive_model_name: str,
    max_new_tokens: int,
    temperature: float,
    device: str = "auto",
    cheap_device: str | None = None,
    expensive_device: str | None = None,
    torch_dtype: str = "auto",
    prompt_format: str = "auto",
    models=None,
) -> list[dict]:
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

    rows = []
    base_modes = (
        "adaptive_calibrated",
        "adaptive_guarded_v3",
        "adaptive_code_quality",
        "speculative_adaptive",
    )
    expensive_tokenizer = get_expensive_tokenizer(tokenizer)
    cheap_runtime = get_model_runtime_metadata(cheap_model, fallback_device=device)
    expensive_runtime = get_model_runtime_metadata(
        expensive_model,
        fallback_device=device,
    )

    for prompt_name, prompt in prompts.items():
        prompt_type = classify_prompt(prompt)
        selected_mode = choose_mode(prompt_type, prompt)
        reference_text, _ = generate_greedy_with_model(
            prompt=prompt,
            model=expensive_model,
            tokenizer=expensive_tokenizer,
            device=expensive_runtime["device"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_format=prompt_format,
        )

        mode_summaries = {}
        for mode in base_modes:
            summary = generate_with_mode(
                prompt=prompt,
                mode=mode,
                cheap_model=cheap_model,
                expensive_model=expensive_model,
                tokenizer=tokenizer,
                device=cheap_runtime["device"],
                cheap_model_name=cheap_model_name,
                expensive_model_name=expensive_model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_format=prompt_format,
            )
            mode_summaries[mode] = summary
            rows.append(
                _hybrid_benchmark_row(
                    prompt_name=prompt_name,
                    prompt_type=prompt_type,
                    mode=mode,
                    selected_mode=mode,
                    summary=summary,
                    reference_text=reference_text,
                )
            )

        rows.append(
            _hybrid_benchmark_row(
                prompt_name=prompt_name,
                prompt_type=prompt_type,
                mode="hybrid",
                selected_mode=selected_mode,
                summary=mode_summaries[selected_mode],
                reference_text=reference_text,
            )
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark simples do GEAR-LLM em vários tipos de prompt."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ModelConfig.model_name,
        help="Modelo do Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Pasta onde os CSVs serão salvos.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Também roda validação por ablation nos prompts principais.",
    )
    parser.add_argument(
        "--balanced-ablation",
        action="store_true",
        help="Também roda ablation balanceada nos prompts principais.",
    )
    parser.add_argument(
        "--compute-sim",
        action="store_true",
        help="Também roda simulação de economia computacional.",
    )
    parser.add_argument(
        "--adaptive-generate",
        action="store_true",
        help="Também roda Adaptive Dual-Model Generation nos prompts principais.",
    )
    parser.add_argument(
        "--adaptive-compare-thresholds",
        action="store_true",
        help="Compara thresholds antigos e calibrados na geração adaptativa.",
    )
    parser.add_argument(
        "--teacher-calibration",
        action="store_true",
        help="Também roda calibração offline cheap-vs-teacher.",
    )
    parser.add_argument(
        "--policy-replay",
        action="store_true",
        help="Roda replay offline de políticas usando teacher_calibration.csv.",
    )
    parser.add_argument(
        "--quality-benchmark",
        action="store_true",
        help="Roda benchmark Quality-vs-Cost dos modos de geração.",
    )
    parser.add_argument(
        "--guard-tuning",
        action="store_true",
        help="Roda busca de configurações para o adaptive_guarded.",
    )
    parser.add_argument(
        "--speculative-generate",
        action="store_true",
        help="Roda benchmark de Adaptive Speculative Decoding.",
    )
    parser.add_argument(
        "--speculative-tuning",
        action="store_true",
        help="Roda tuning de parâmetros do Adaptive Speculative Decoding.",
    )
    parser.add_argument(
        "--hybrid-benchmark",
        action="store_true",
        help="Compara adaptive, guarded, speculative e o roteador híbrido.",
    )
    parser.add_argument(
        "--dataset-benchmark",
        action="store_true",
        help="Roda benchmark do hybrid router em um dataset JSONL.",
    )
    parser.add_argument(
        "--mode-oracle",
        action="store_true",
        help="Calcula o melhor modo por prompt usando dataset_benchmark.csv.",
    )
    parser.add_argument(
        "--latency-benchmark",
        action="store_true",
        help="Mede latencia real dos modos de geracao.",
    )
    parser.add_argument(
        "--task-evaluation",
        action="store_true",
        help="Roda avaliacao task-specific de math, logic e code.",
    )
    parser.add_argument(
        "--include-latency",
        action="store_true",
        help="Inclui latencia real na avaliacao task-specific.",
    )
    parser.add_argument(
        "--profile-runtime",
        action="store_true",
        help="Coleta perfil detalhado de runtime na avaliacao task-specific.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmups por task/modo na avaliacao task-specific com latencia.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=1,
        help="Medicoes por task/modo na avaliacao task-specific com latencia.",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        default=None,
        help="Dificuldades separadas por vírgula para --task-evaluation.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=(
            "Modos separados por vírgula para --task-evaluation, incluindo "
            "prompt_router_v1 e prompt_router_v2."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/prompts.jsonl",
        help="Caminho do dataset JSONL usado por --dataset-benchmark.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Categorias separadas por vírgula para filtrar dataset/task evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite total de prompts/tasks após filtro.",
    )
    parser.add_argument(
        "--ablation-csv",
        type=str,
        default=None,
        help="CSV opcional para salvar o benchmark de ablation.",
    )
    parser.add_argument(
        "--neutral-text",
        type=str,
        default=" ",
        help="Texto neutro usado pela ablation.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=20,
        help="Número de rodadas do baseline aleatório na ablation balanceada.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do baseline aleatório.",
    )
    parser.add_argument(
        "--cheap-cost",
        type=float,
        default=0.35,
        help="Custo teórico de tokens cheap na simulação.",
    )
    parser.add_argument(
        "--medium-cost",
        type=float,
        default=0.70,
        help="Custo teórico de tokens medium na simulação.",
    )
    parser.add_argument(
        "--expensive-cost",
        type=float,
        default=1.00,
        help="Custo teórico de tokens expensive na simulação.",
    )
    parser.add_argument(
        "--cheap-model",
        type=str,
        default=AdaptiveGenerationConfig.cheap_model_name,
        help="Modelo barato para geração adaptativa.",
    )
    parser.add_argument(
        "--expensive-model",
        type=str,
        default=AdaptiveGenerationConfig.expensive_model_name,
        help="Modelo caro para geração adaptativa.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default="auto",
        help="Device para carregar modelos: auto, cpu ou cuda.",
    )
    parser.add_argument(
        "--cheap-device",
        type=str,
        default=None,
        help="Device opcional para o modelo barato, ex: cuda:0.",
    )
    parser.add_argument(
        "--expensive-device",
        type=str,
        default=None,
        help="Device opcional para o modelo caro, ex: cuda:1.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=TORCH_DTYPE_CHOICES,
        default="auto",
        help="dtype dos pesos: auto, float32, float16 ou bfloat16.",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=PROMPT_FORMAT_CHOICES,
        default="auto",
        help="Formato do prompt para geracao: raw, chat ou auto.",
    )
    parser.add_argument(
        "--adaptive-max-new-tokens",
        type=int,
        default=80,
        help="Máximo de tokens novos na geração adaptativa.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override do máximo de tokens novos para rotinas speculative/tuning/latency.",
    )
    parser.add_argument(
        "--latency-warmup-runs",
        type=int,
        default=1,
        help="Execucoes de aquecimento por prompt/modo no latency benchmark.",
    )
    parser.add_argument(
        "--latency-measured-runs",
        type=int,
        default=3,
        help="Execucoes medidas por prompt/modo no latency benchmark.",
    )
    parser.add_argument(
        "--adaptive-temperature",
        type=float,
        default=0.7,
        help="Temperatura da geração adaptativa.",
    )
    parser.add_argument(
        "--speculative-draft-length",
        type=int,
        default=SpeculativeGenerationConfig.draft_length,
        help="Tamanho inicial do draft speculative.",
    )
    parser.add_argument(
        "--speculative-verify-top-k",
        type=int,
        default=SpeculativeGenerationConfig.verify_top_k,
        help="Top-k do modelo caro para verificar tokens do draft.",
    )
    parser.add_argument(
        "--speculative-min-draft-length",
        type=int,
        default=SpeculativeGenerationConfig.min_draft_length,
        help="Menor tamanho permitido para o draft speculative adaptativo.",
    )
    parser.add_argument(
        "--speculative-max-draft-length",
        type=int,
        default=SpeculativeGenerationConfig.max_draft_length,
        help="Maior tamanho permitido para o draft speculative adaptativo.",
    )
    parser.add_argument(
        "--adaptive-entropy-threshold",
        type=float,
        default=AdaptiveGenerationConfig.entropy_threshold,
        help="Entropia máxima para aceitar o modelo barato.",
    )
    parser.add_argument(
        "--adaptive-margin-threshold",
        type=float,
        default=0.20,
        help="Margem mínima top1-top2 para aceitar o modelo barato.",
    )
    parser.add_argument(
        "--teacher-check-interval",
        type=int,
        default=AdaptiveGenerationConfig.teacher_check_interval,
        help="Intervalo de chamadas periódicas ao modelo caro.",
    )
    parser.add_argument(
        "--disable-periodic-teacher-check",
        action="store_true",
        help="Desliga chamadas periódicas ao modelo caro.",
    )
    parser.add_argument(
        "--disable-repetition-guard",
        action="store_true",
        help="Desliga o guard de repetição.",
    )
    parser.add_argument(
        "--repetition-ngram-size",
        type=int,
        default=3,
        help="Tamanho do n-grama usado pelo guard de repetição.",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=AdaptiveGenerationConfig.repetition_threshold,
        help="Taxa parcial de n-gramas repetidos que aciona fallback.",
    )
    parser.add_argument(
        "--disable-repetition-guard-requires-uncertainty",
        action="store_true",
        help="Permite repetition_guard mesmo sem sinal de incerteza.",
    )
    parser.add_argument(
        "--repetition-guard-entropy-threshold",
        type=float,
        default=AdaptiveGenerationConfig.repetition_guard_entropy_threshold,
        help="Entropia mínima para o repetition_guard passar pelo gate de incerteza.",
    )
    parser.add_argument(
        "--repetition-guard-margin-threshold",
        type=float,
        default=AdaptiveGenerationConfig.repetition_guard_margin_threshold,
        help="Margem abaixo da qual o repetition_guard passa pelo gate de incerteza.",
    )
    parser.add_argument(
        "--repetition-guard-cooldown-tokens",
        type=int,
        default=AdaptiveGenerationConfig.repetition_guard_cooldown_tokens,
        help="Número de tokens de espera após repetition_guard chamar o modelo caro.",
    )
    parser.add_argument(
        "--disable-risk-gated-periodic-check",
        action="store_true",
        help="Desliga o gating por risco das chamadas periódicas ao modelo caro.",
    )
    parser.add_argument(
        "--periodic-entropy-risk-threshold",
        type=float,
        default=AdaptiveGenerationConfig.periodic_entropy_risk_threshold,
        help="Entropia mínima para considerar uma chamada periódica arriscada.",
    )
    parser.add_argument(
        "--periodic-margin-risk-threshold",
        type=float,
        default=AdaptiveGenerationConfig.periodic_margin_risk_threshold,
        help="Margem abaixo da qual uma chamada periódica é considerada arriscada.",
    )
    parser.add_argument(
        "--periodic-repetition-risk-threshold",
        type=float,
        default=AdaptiveGenerationConfig.periodic_repetition_risk_threshold,
        help="Repetição parcial acima da qual uma chamada periódica é arriscada.",
    )
    parser.add_argument(
        "--max-expensive-call-ratio",
        type=float,
        default=AdaptiveGenerationConfig.max_expensive_call_ratio,
        help="Razão máxima de chamadas caras antes de bloquear fallback periódico puro.",
    )
    parser.add_argument(
        "--teacher-max-steps",
        type=int,
        default=40,
        help="Número máximo de passos gerados pelo teacher.",
    )
    parser.add_argument(
        "--teacher-top-k",
        type=int,
        default=5,
        help="Top-k do teacher para topk_match.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=0.7,
        help="Temperatura usada na calibração teacher.",
    )
    parser.add_argument(
        "--guard-max-configs",
        type=int,
        default=None,
        help="Limita a quantidade de configs no guard tuning. Útil para smoke tests.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Limita a quantidade de configs no speculative tuning.",
    )
    parser.add_argument(
        "--config-filter",
        type=str,
        default=None,
        help="Lista de configs speculative separadas por vírgula para tuning.",
    )

    args = parser.parse_args()

    model_config = ModelConfig(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    router_config = RouterConfig()
    cost_config = ComputeCostConfig(
        cheap_cost=args.cheap_cost,
        medium_cost=args.medium_cost,
        expensive_cost=args.expensive_cost,
    )
    adaptive_config = AdaptiveGenerationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        device=args.device,
        cheap_device=args.cheap_device,
        expensive_device=args.expensive_device,
        torch_dtype=args.torch_dtype,
        prompt_format=args.prompt_format,
        max_new_tokens=args.adaptive_max_new_tokens,
        temperature=args.adaptive_temperature,
        entropy_threshold=args.adaptive_entropy_threshold,
        margin_threshold=args.adaptive_margin_threshold,
        teacher_check_interval=args.teacher_check_interval,
        enable_periodic_teacher_check=not args.disable_periodic_teacher_check,
        enable_repetition_guard=not args.disable_repetition_guard,
        repetition_ngram_size=args.repetition_ngram_size,
        repetition_threshold=args.repetition_threshold,
        risk_gated_periodic_check=not args.disable_risk_gated_periodic_check,
        periodic_entropy_risk_threshold=args.periodic_entropy_risk_threshold,
        periodic_margin_risk_threshold=args.periodic_margin_risk_threshold,
        periodic_repetition_risk_threshold=(
            args.periodic_repetition_risk_threshold
        ),
        max_expensive_call_ratio=args.max_expensive_call_ratio,
        repetition_guard_requires_uncertainty=(
            not args.disable_repetition_guard_requires_uncertainty
        ),
        repetition_guard_entropy_threshold=(
            args.repetition_guard_entropy_threshold
        ),
        repetition_guard_margin_threshold=args.repetition_guard_margin_threshold,
        repetition_guard_cooldown_tokens=args.repetition_guard_cooldown_tokens,
    )
    teacher_config = TeacherCalibrationConfig(
        cheap_model_name=args.cheap_model,
        expensive_model_name=args.expensive_model,
        max_steps=args.teacher_max_steps,
        top_k=args.teacher_top_k,
        temperature=args.teacher_temperature,
    )
    output_dir = Path(args.output_dir)
    speculative_max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else args.adaptive_max_new_tokens
    )
    task_dataset = args.dataset
    if args.task_evaluation and args.dataset == "data/prompts.jsonl":
        task_dataset = "data/eval_tasks.jsonl"

    model_work_requested = any(
        (
            args.ablation,
            args.balanced_ablation,
            args.compute_sim,
            args.adaptive_generate,
            args.adaptive_compare_thresholds,
            args.teacher_calibration,
        )
    )

    if not model_work_requested and (
        args.policy_replay
        or args.quality_benchmark
        or args.guard_tuning
        or args.speculative_generate
        or args.speculative_tuning
        or args.hybrid_benchmark
        or args.dataset_benchmark
        or args.mode_oracle
        or args.latency_benchmark
        or args.task_evaluation
    ):
        if args.policy_replay:
            teacher_csv = output_dir / "teacher_calibration.csv"
            policy_csv = output_dir / "policy_replay.csv"

            try:
                policy_rows = run_policy_replay(teacher_csv=teacher_csv)
            except FileNotFoundError as error:
                print(error)
                return

            print_policy_replay_report(policy_rows)
            save_policy_replay(policy_rows, policy_csv)
            print(f"{'policy_csv':<15} -> {policy_csv}")

        if args.quality_benchmark:
            quality_csv = output_dir / "quality_benchmark.csv"
            quality_rows = run_quality_benchmark(
                prompts=PROMPTS,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=args.adaptive_max_new_tokens,
                temperature=args.adaptive_temperature,
                device=args.device,
                cheap_device=args.cheap_device,
                expensive_device=args.expensive_device,
                torch_dtype=args.torch_dtype,
                prompt_format=args.prompt_format,
            )
            print_quality_benchmark_report(quality_rows)
            save_quality_benchmark(quality_rows, quality_csv)
            print(f"{'quality_csv':<15} -> {quality_csv}")

        if args.guard_tuning:
            guard_csv = output_dir / "guard_tuning.csv"
            guard_summary_csv = output_dir / "guard_tuning_summary.csv"
            guard_rows, guard_summary_rows = run_guard_tuning(
                prompts=PROMPTS,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=args.adaptive_max_new_tokens,
                temperature=args.adaptive_temperature,
                max_configs=args.guard_max_configs,
            )
            print_guard_tuning_report(guard_summary_rows)
            save_guard_tuning(guard_rows, guard_csv)
            save_guard_tuning(guard_summary_rows, guard_summary_csv)
            print(f"{'guard_csv':<15} -> {guard_csv}")
            print(f"{'guard_summary':<15} -> {guard_summary_csv}")

        if args.speculative_generate:
            speculative_csv = output_dir / "speculative_benchmark.csv"
            speculative_rows = run_speculative_benchmark(
                prompts=PROMPTS,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=speculative_max_new_tokens,
                temperature=args.adaptive_temperature,
                draft_length=args.speculative_draft_length,
                verify_top_k=args.speculative_verify_top_k,
                min_draft_length=args.speculative_min_draft_length,
                max_draft_length=args.speculative_max_draft_length,
                device=args.device,
                cheap_device=args.cheap_device,
                expensive_device=args.expensive_device,
                torch_dtype=args.torch_dtype,
                prompt_format=args.prompt_format,
            )
            print_speculative_benchmark_report(speculative_rows)
            save_speculative_summary_rows(speculative_rows, speculative_csv)
            print(f"{'speculative_csv':<15} -> {speculative_csv}")

        if args.speculative_tuning:
            tuning_csv = output_dir / "speculative_tuning.csv"
            tuning_summary_csv = output_dir / "speculative_tuning_summary.csv"
            tuning_rows, tuning_summary_rows = run_speculative_tuning(
                prompts=PROMPTS,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=speculative_max_new_tokens,
                temperature=args.adaptive_temperature,
                max_configs=args.max_configs,
                config_filter=args.config_filter,
            )
            print_speculative_tuning_report(tuning_summary_rows)
            save_speculative_tuning(tuning_rows, tuning_csv)
            save_speculative_tuning(tuning_summary_rows, tuning_summary_csv)
            print(f"{'spec_tuning_csv':<15} -> {tuning_csv}")
            print(f"{'spec_tuning_sum':<15} -> {tuning_summary_csv}")

        if args.hybrid_benchmark:
            hybrid_csv = output_dir / "hybrid_benchmark.csv"
            hybrid_rows = run_hybrid_benchmark(
                prompts=PROMPTS,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=speculative_max_new_tokens,
                temperature=args.adaptive_temperature,
                device=args.device,
                cheap_device=args.cheap_device,
                expensive_device=args.expensive_device,
                torch_dtype=args.torch_dtype,
                prompt_format=args.prompt_format,
            )
            print_hybrid_benchmark_report(hybrid_rows)
            save_csv(hybrid_rows, str(hybrid_csv))
            print(f"{'hybrid_csv':<15} -> {hybrid_csv}")

        if args.dataset_benchmark:
            dataset_rows, dataset_summary_rows, dataset_matrix_rows = (
                run_dataset_benchmark(
                    dataset_path=args.dataset,
                    categories=parse_categories(args.categories),
                    limit=args.limit,
                    cheap_model_name=args.cheap_model,
                    expensive_model_name=args.expensive_model,
                    max_new_tokens=speculative_max_new_tokens,
                    temperature=args.adaptive_temperature,
                    device=args.device,
                    cheap_device=args.cheap_device,
                    expensive_device=args.expensive_device,
                    torch_dtype=args.torch_dtype,
                    prompt_format=args.prompt_format,
                )
            )
            print_dataset_benchmark_report(
                dataset_summary_rows,
                dataset_matrix_rows,
            )
            detailed_csv, summary_csv, matrix_csv = save_dataset_benchmark_outputs(
                rows=dataset_rows,
                summary_rows=dataset_summary_rows,
                matrix_rows=dataset_matrix_rows,
                output_dir=output_dir,
            )
            print(f"{'dataset_csv':<15} -> {detailed_csv}")
            print(f"{'summary_csv':<15} -> {summary_csv}")
            print(f"{'matrix_csv':<15} -> {matrix_csv}")

        if args.mode_oracle:
            dataset_csv = output_dir / "dataset_benchmark.csv"

            try:
                (
                    oracle_rows,
                    oracle_summary_rows,
                    oracle_confidence_rows,
                    oracle_compare_rows,
                    oracle_metrics,
                ) = run_mode_oracle(dataset_csv=dataset_csv)
            except FileNotFoundError as error:
                print(error)
                return

            print_mode_oracle_report(
                oracle_summary_rows,
                oracle_confidence_rows,
                oracle_metrics,
            )
            (
                oracle_csv,
                oracle_summary_csv,
                oracle_confidence_csv,
                oracle_compare_csv,
            ) = (
                save_mode_oracle_outputs(
                    oracle_rows=oracle_rows,
                    summary_rows=oracle_summary_rows,
                    confidence_summary_rows=oracle_confidence_rows,
                    comparison_rows=oracle_compare_rows,
                    output_dir=output_dir,
                )
            )
            print(f"{'oracle_csv':<15} -> {oracle_csv}")
            print(f"{'oracle_summary':<15} -> {oracle_summary_csv}")
            print(f"{'oracle_conf':<15} -> {oracle_confidence_csv}")
            print(f"{'oracle_compare':<15} -> {oracle_compare_csv}")

        if args.latency_benchmark:
            latency_rows, latency_summary_rows, latency_winner_rows = (
                run_latency_benchmark(
                    prompts=PROMPTS,
                    cheap_model_name=args.cheap_model,
                    expensive_model_name=args.expensive_model,
                    max_new_tokens=speculative_max_new_tokens,
                    temperature=args.adaptive_temperature,
                    warmup_runs=args.latency_warmup_runs,
                    measured_runs=args.latency_measured_runs,
                    device=args.device,
                    cheap_device=args.cheap_device,
                    expensive_device=args.expensive_device,
                    torch_dtype=args.torch_dtype,
                    prompt_format=args.prompt_format,
                )
            )
            print_latency_benchmark_report(latency_summary_rows)
            latency_csv, latency_summary_csv, latency_winners_csv = (
                save_latency_benchmark_outputs(
                    rows=latency_rows,
                    summary_rows=latency_summary_rows,
                    winner_rows=latency_winner_rows,
                    output_dir=output_dir,
                )
            )
            print(f"{'latency_csv':<15} -> {latency_csv}")
            print(f"{'latency_sum':<15} -> {latency_summary_csv}")
            print(f"{'latency_win':<15} -> {latency_winners_csv}")

        if args.task_evaluation:
            (
                task_rows,
                task_summary_rows,
                task_difficulty_rows,
                task_overall_rows,
            ) = run_task_evaluation(
                dataset_path=task_dataset,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=speculative_max_new_tokens,
                temperature=args.adaptive_temperature,
                device=args.device,
                cheap_device=args.cheap_device,
                expensive_device=args.expensive_device,
                torch_dtype=args.torch_dtype,
                prompt_format=args.prompt_format,
                include_latency=args.include_latency,
                warmup_runs=args.warmup_runs,
                measured_runs=args.measured_runs,
                limit=args.limit,
                categories=args.categories,
                difficulties=args.difficulties,
                modes=args.modes,
                profile_runtime=args.profile_runtime,
            )
            print_task_evaluation_report(task_summary_rows)
            print_task_evaluation_overall_report(task_overall_rows)
            (
                task_csv,
                task_summary_csv,
                task_difficulty_csv,
                task_overall_csv,
            ) = save_task_evaluation_outputs(
                rows=task_rows,
                summary_rows=task_summary_rows,
                difficulty_rows=task_difficulty_rows,
                overall_rows=task_overall_rows,
                output_dir=output_dir,
            )
            print(f"{'task_csv':<15} -> {task_csv}")
            print(f"{'task_summary':<15} -> {task_summary_csv}")
            print(f"{'task_diff':<15} -> {task_difficulty_csv}")
            print(f"{'task_overall':<15} -> {task_overall_csv}")

            if args.include_latency:
                (
                    latency_rows,
                    latency_summary_rows,
                    latency_by_category_rows,
                    latency_by_difficulty_rows,
                ) = build_task_quality_latency_outputs(task_rows)
                print_task_quality_latency_report(latency_summary_rows)
                (
                    report_csv,
                    latency_summary_csv,
                    latency_category_csv,
                    latency_difficulty_csv,
                ) = save_task_quality_latency_outputs(
                    report_rows=latency_rows,
                    summary_rows=latency_summary_rows,
                    by_category_rows=latency_by_category_rows,
                    by_difficulty_rows=latency_by_difficulty_rows,
                    output_dir=output_dir,
                )
                print(f"{'task_ql_report':<15} -> {report_csv}")
                print(f"{'task_ql_sum':<15} -> {latency_summary_csv}")
                print(f"{'task_ql_cat':<15} -> {latency_category_csv}")
                print(f"{'task_ql_diff':<15} -> {latency_difficulty_csv}")

            if args.profile_runtime:
                profile_rows, profile_summary_rows = build_runtime_profile_outputs(
                    task_rows
                )
                print_runtime_profile_report(profile_summary_rows)
                profile_csv, profile_summary_csv = save_runtime_profile_outputs(
                    profile_rows=profile_rows,
                    summary_rows=profile_summary_rows,
                    output_dir=output_dir,
                )
                print(f"{'runtime_prof':<15} -> {profile_csv}")
                print(f"{'runtime_sum':<15} -> {profile_summary_csv}")

        return

    model, tokenizer, device = load_model_and_tokenizer(
        model_config.model_name,
        device=model_config.device,
        torch_dtype=model_config.torch_dtype,
    )
    adaptive_models = None
    teacher_models = None

    if args.adaptive_generate or args.adaptive_compare_thresholds:
        adaptive_models = load_adaptive_models(adaptive_config)

    if args.teacher_calibration:
        teacher_models = adaptive_models or load_teacher_models(teacher_config)

    ablation_rows = []
    balanced_ablation_rows = []
    compute_sim_rows = []
    adaptive_generation_rows = []
    adaptive_threshold_comparison_rows = []
    teacher_calibration_rows = []
    teacher_grid_rows = []

    for name, prompt in PROMPTS.items():
        rows = analyze_prompt_with_model(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            router_config=router_config,
        )

        csv_path = output_dir / f"{name}.csv"
        save_csv(rows, str(csv_path))
        print(f"{name:<15} -> {csv_path}")

        if args.compute_sim:
            compute_summary = simulate_compute_from_rows(
                rows=rows,
                cost_config=cost_config,
                prompt=prompt,
                prompt_name=name,
            )
            compute_sim_rows.append(compute_summary)
            print(
                f"{'compute_sim':<15} -> {name}: "
                f"saved={compute_summary['saved_percent']:.2f}%, "
                f"avg_cost/token={compute_summary['avg_cost_per_token']:.4f}"
            )

        if args.ablation:
            summary = run_ablation_with_model(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                router_config=router_config,
                neutral_text=args.neutral_text,
            )
            summary["prompt_name"] = name
            ablation_rows.append(summary)

            if summary["criterion_status"] == "not_applicable":
                status = "N/A"
            elif summary["criterion_passed_raw"]:
                status = "PASSOU"
            else:
                status = "NÃO PASSOU"

            print(
                f"{'ablation':<15} -> {name}: "
                f"cheap_delta={summary['cheap_delta_loss']:.4f}, "
                f"expensive_delta={summary['expensive_delta_loss']:.4f}, "
                f"cheap_delta/tok={summary['cheap_delta_loss_per_token']:.4f}, "
                f"expensive_delta/tok={summary['expensive_delta_loss_per_token']:.4f} "
                f"({status})"
            )

        if args.balanced_ablation:
            balanced_summary = run_balanced_ablation_with_model(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                router_config=router_config,
                neutral_text=args.neutral_text,
                random_trials=args.random_trials,
                seed=args.seed,
            )
            balanced_summary["prompt_name"] = name
            balanced_ablation_rows.append(balanced_summary)

            status = balanced_summary["balanced_status"]
            print(
                f"{'balanced':<15} -> {name}: "
                f"k={balanced_summary['k']}, "
                f"expensive/tok={balanced_summary['expensive_delta_per_token']}, "
                f"cheap/tok={balanced_summary['cheap_delta_per_token']}, "
                f"random_mean/tok={balanced_summary['random_mean_delta_per_token']} "
                f"({status})"
            )

        if args.adaptive_generate:
            (
                adaptive_cheap_model,
                adaptive_expensive_model,
                adaptive_tokenizer,
                adaptive_device,
            ) = adaptive_models
            _, _, adaptive_summary = adaptive_generate_with_models(
                prompt=prompt,
                cheap_model=adaptive_cheap_model,
                expensive_model=adaptive_expensive_model,
                tokenizer=adaptive_tokenizer,
                device=adaptive_device,
                config=adaptive_config,
            )
            adaptive_summary["prompt_name"] = name
            adaptive_generation_rows.append(adaptive_summary)
            print(
                f"{'adaptive':<15} -> {name}: "
                f"cheap={adaptive_summary['cheap_percent']:.2f}%, "
                f"expensive_calls={adaptive_summary['expensive_model_calls']}, "
                f"saved={adaptive_summary['estimated_saved_percent']:.2f}%"
            )

        if args.adaptive_compare_thresholds:
            (
                adaptive_cheap_model,
                adaptive_expensive_model,
                adaptive_tokenizer,
                adaptive_device,
            ) = adaptive_models
            comparison_configs = [
                (
                    "old_0.45_0.20",
                    AdaptiveGenerationConfig(
                        cheap_model_name=args.cheap_model,
                        expensive_model_name=args.expensive_model,
                        max_new_tokens=args.adaptive_max_new_tokens,
                        temperature=args.adaptive_temperature,
                        entropy_threshold=0.45,
                        margin_threshold=0.20,
                        enable_periodic_teacher_check=False,
                        enable_repetition_guard=False,
                    ),
                ),
                (
                    "calibrated_0.35_0.20",
                    AdaptiveGenerationConfig(
                        cheap_model_name=args.cheap_model,
                        expensive_model_name=args.expensive_model,
                        max_new_tokens=args.adaptive_max_new_tokens,
                        temperature=args.adaptive_temperature,
                        entropy_threshold=0.35,
                        margin_threshold=0.20,
                        enable_periodic_teacher_check=False,
                        enable_repetition_guard=False,
                    ),
                ),
            ]

            for config_name, comparison_config in comparison_configs:
                _, _, comparison_summary = adaptive_generate_with_models(
                    prompt=prompt,
                    cheap_model=adaptive_cheap_model,
                    expensive_model=adaptive_expensive_model,
                    tokenizer=adaptive_tokenizer,
                    device=adaptive_device,
                    config=comparison_config,
                )
                adaptive_threshold_comparison_rows.append(
                    {
                        "prompt_name": name,
                        "config_name": config_name,
                        "entropy_threshold": comparison_config.entropy_threshold,
                        "margin_threshold": comparison_config.margin_threshold,
                        "total_generated_tokens": comparison_summary[
                            "total_generated_tokens"
                        ],
                        "cheap_accepted_tokens": comparison_summary[
                            "cheap_accepted_tokens"
                        ],
                        "expensive_model_calls": comparison_summary[
                            "expensive_model_calls"
                        ],
                        "cheap_percent": comparison_summary["cheap_percent"],
                        "estimated_saved_percent": comparison_summary[
                            "estimated_saved_percent"
                        ],
                        "generated_text": comparison_summary["generated_text"],
                    }
                )

        if args.teacher_calibration:
            (
                teacher_cheap_model,
                teacher_expensive_model,
                teacher_tokenizer,
                teacher_device,
            ) = teacher_models
            teacher_rows, teacher_summary, teacher_grid = (
                run_teacher_calibration_with_models(
                    prompt=prompt,
                    cheap_model=teacher_cheap_model,
                    expensive_model=teacher_expensive_model,
                    tokenizer=teacher_tokenizer,
                    device=teacher_device,
                    config=teacher_config,
                    prompt_name=name,
                )
            )
            teacher_calibration_rows.extend(teacher_rows)
            teacher_grid_rows.extend(teacher_grid)

            viable_grid = [
                row
                for row in teacher_grid
                if row["precision_accept"] is not None
                and row["estimated_saved_percent"] > 0
            ]
            best = None

            if viable_grid:
                best = max(
                    viable_grid,
                    key=lambda row: (
                        row["precision_accept"],
                        row["estimated_saved_percent"],
                    ),
                )

            if best:
                best_text = (
                    f"best_precision={best['precision_accept']:.2%}, "
                    f"saved={best['estimated_saved_percent']:.2f}%"
                )
            else:
                best_text = "sem threshold com economia positiva"

            print(
                f"{'teacher':<15} -> {name}: "
                f"exact={teacher_summary['exact_match_rate']:.2%}, "
                f"topk={teacher_summary['topk_match_rate']:.2%}, "
                f"{best_text}"
            )

    if args.ablation:
        ablation_csv = (
            Path(args.ablation_csv)
            if args.ablation_csv
            else output_dir / "ablation_benchmark.csv"
        )
        save_ablation_rows(ablation_rows, ablation_csv)
        print(f"{'ablation_csv':<15} -> {ablation_csv}")

    if args.balanced_ablation:
        balanced_csv = output_dir / "balanced_ablation_benchmark.csv"
        save_ablation_rows(balanced_ablation_rows, balanced_csv)
        print(f"{'balanced_csv':<15} -> {balanced_csv}")

    if args.compute_sim:
        compute_csv = output_dir / "compute_sim_benchmark.csv"
        save_compute_sim_rows(compute_sim_rows, compute_csv)
        print_compute_sim_benchmark_report(compute_sim_rows)
        print(f"{'compute_csv':<15} -> {compute_csv}")

    if args.adaptive_generate:
        adaptive_csv = output_dir / "adaptive_generation_benchmark.csv"
        save_adaptive_summary_rows(adaptive_generation_rows, adaptive_csv)
        print(f"{'adaptive_csv':<15} -> {adaptive_csv}")

    if args.adaptive_compare_thresholds:
        comparison_csv = output_dir / "adaptive_threshold_comparison.csv"
        save_csv(adaptive_threshold_comparison_rows, str(comparison_csv))
        print()
        print("Comparação de thresholds adaptativos")
        print("=" * 100)
        header = (
            f"{'prompt':<16} | {'old saved':>9} | {'cal saved':>9} | "
            f"{'old calls':>9} | {'cal calls':>9} | "
            f"{'old cheap':>9} | {'cal cheap':>9}"
        )
        print(header)
        print("-" * len(header))

        for name in PROMPTS:
            old_row = next(
                row
                for row in adaptive_threshold_comparison_rows
                if row["prompt_name"] == name
                and row["config_name"] == "old_0.45_0.20"
            )
            calibrated_row = next(
                row
                for row in adaptive_threshold_comparison_rows
                if row["prompt_name"] == name
                and row["config_name"] == "calibrated_0.35_0.20"
            )
            print(
                f"{name:<16} | "
                f"{old_row['estimated_saved_percent']:>8.2f}% | "
                f"{calibrated_row['estimated_saved_percent']:>8.2f}% | "
                f"{old_row['expensive_model_calls']:>9} | "
                f"{calibrated_row['expensive_model_calls']:>9} | "
                f"{old_row['cheap_percent']:>8.2f}% | "
                f"{calibrated_row['cheap_percent']:>8.2f}%"
            )

        print("=" * 100)
        print(
            "Observação: esta comparação é online; se os textos gerados divergem, "
            "as decisões futuras também podem divergir."
        )
        print(f"{'threshold_csv':<15} -> {comparison_csv}")

    if args.teacher_calibration:
        teacher_csv = output_dir / "teacher_calibration.csv"
        teacher_grid_csv = output_dir / "teacher_threshold_grid.csv"
        aggregate_grid = threshold_grid_search(
            rows=teacher_calibration_rows,
            config=teacher_config,
            prompt_name="ALL",
        )
        teacher_grid_rows.extend(aggregate_grid)
        save_teacher_rows(teacher_calibration_rows, teacher_csv)
        save_teacher_grid(teacher_grid_rows, teacher_grid_csv)
        print(f"{'teacher_csv':<15} -> {teacher_csv}")
        print(f"{'teacher_grid':<15} -> {teacher_grid_csv}")

    if args.policy_replay and model_work_requested:
        teacher_csv = output_dir / "teacher_calibration.csv"
        policy_csv = output_dir / "policy_replay.csv"

        try:
            policy_rows = run_policy_replay(teacher_csv=teacher_csv)
        except FileNotFoundError as error:
            print(error)
            return

        print_policy_replay_report(policy_rows)
        save_policy_replay(policy_rows, policy_csv)
        print(f"{'policy_csv':<15} -> {policy_csv}")

    if args.quality_benchmark and model_work_requested:
        quality_csv = output_dir / "quality_benchmark.csv"
        quality_rows = run_quality_benchmark(
            prompts=PROMPTS,
            cheap_model_name=args.cheap_model,
            expensive_model_name=args.expensive_model,
            max_new_tokens=args.adaptive_max_new_tokens,
            temperature=args.adaptive_temperature,
            device=args.device,
            cheap_device=args.cheap_device,
            expensive_device=args.expensive_device,
            torch_dtype=args.torch_dtype,
            prompt_format=args.prompt_format,
        )
        print_quality_benchmark_report(quality_rows)
        save_quality_benchmark(quality_rows, quality_csv)
        print(f"{'quality_csv':<15} -> {quality_csv}")

    if args.guard_tuning and model_work_requested:
        guard_csv = output_dir / "guard_tuning.csv"
        guard_summary_csv = output_dir / "guard_tuning_summary.csv"
        guard_rows, guard_summary_rows = run_guard_tuning(
            prompts=PROMPTS,
            cheap_model_name=args.cheap_model,
            expensive_model_name=args.expensive_model,
            max_new_tokens=args.adaptive_max_new_tokens,
            temperature=args.adaptive_temperature,
            max_configs=args.guard_max_configs,
        )
        print_guard_tuning_report(guard_summary_rows)
        save_guard_tuning(guard_rows, guard_csv)
        save_guard_tuning(guard_summary_rows, guard_summary_csv)
        print(f"{'guard_csv':<15} -> {guard_csv}")
        print(f"{'guard_summary':<15} -> {guard_summary_csv}")

    if args.speculative_generate and model_work_requested:
        speculative_csv = output_dir / "speculative_benchmark.csv"
        speculative_rows = run_speculative_benchmark(
            prompts=PROMPTS,
            cheap_model_name=args.cheap_model,
            expensive_model_name=args.expensive_model,
            max_new_tokens=speculative_max_new_tokens,
            temperature=args.adaptive_temperature,
            draft_length=args.speculative_draft_length,
            verify_top_k=args.speculative_verify_top_k,
            min_draft_length=args.speculative_min_draft_length,
            max_draft_length=args.speculative_max_draft_length,
            device=args.device,
            cheap_device=args.cheap_device,
            expensive_device=args.expensive_device,
            torch_dtype=args.torch_dtype,
            prompt_format=args.prompt_format,
            models=adaptive_models,
        )
        print_speculative_benchmark_report(speculative_rows)
        save_speculative_summary_rows(speculative_rows, speculative_csv)
        print(f"{'speculative_csv':<15} -> {speculative_csv}")

    if args.speculative_tuning and model_work_requested:
        tuning_csv = output_dir / "speculative_tuning.csv"
        tuning_summary_csv = output_dir / "speculative_tuning_summary.csv"
        tuning_rows, tuning_summary_rows = run_speculative_tuning(
            prompts=PROMPTS,
            cheap_model_name=args.cheap_model,
            expensive_model_name=args.expensive_model,
            max_new_tokens=speculative_max_new_tokens,
            temperature=args.adaptive_temperature,
            max_configs=args.max_configs,
            config_filter=args.config_filter,
        )
        print_speculative_tuning_report(tuning_summary_rows)
        save_speculative_tuning(tuning_rows, tuning_csv)
        save_speculative_tuning(tuning_summary_rows, tuning_summary_csv)
        print(f"{'spec_tuning_csv':<15} -> {tuning_csv}")
        print(f"{'spec_tuning_sum':<15} -> {tuning_summary_csv}")

    if args.hybrid_benchmark and model_work_requested:
        hybrid_csv = output_dir / "hybrid_benchmark.csv"
        hybrid_rows = run_hybrid_benchmark(
            prompts=PROMPTS,
            cheap_model_name=args.cheap_model,
            expensive_model_name=args.expensive_model,
            max_new_tokens=speculative_max_new_tokens,
            temperature=args.adaptive_temperature,
            device=args.device,
            cheap_device=args.cheap_device,
            expensive_device=args.expensive_device,
            torch_dtype=args.torch_dtype,
            prompt_format=args.prompt_format,
            models=adaptive_models,
        )
        print_hybrid_benchmark_report(hybrid_rows)
        save_csv(hybrid_rows, str(hybrid_csv))
        print(f"{'hybrid_csv':<15} -> {hybrid_csv}")

    if args.dataset_benchmark and model_work_requested:
        dataset_rows, dataset_summary_rows, dataset_matrix_rows = (
            run_dataset_benchmark(
                dataset_path=args.dataset,
                categories=parse_categories(args.categories),
                limit=args.limit,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=speculative_max_new_tokens,
                temperature=args.adaptive_temperature,
                device=args.device,
                cheap_device=args.cheap_device,
                expensive_device=args.expensive_device,
                torch_dtype=args.torch_dtype,
                prompt_format=args.prompt_format,
                models=adaptive_models,
            )
        )
        print_dataset_benchmark_report(
            dataset_summary_rows,
            dataset_matrix_rows,
        )
        detailed_csv, summary_csv, matrix_csv = save_dataset_benchmark_outputs(
            rows=dataset_rows,
            summary_rows=dataset_summary_rows,
            matrix_rows=dataset_matrix_rows,
            output_dir=output_dir,
        )
        print(f"{'dataset_csv':<15} -> {detailed_csv}")
        print(f"{'summary_csv':<15} -> {summary_csv}")
        print(f"{'matrix_csv':<15} -> {matrix_csv}")

    if args.mode_oracle and model_work_requested:
        dataset_csv = output_dir / "dataset_benchmark.csv"

        try:
            (
                oracle_rows,
                oracle_summary_rows,
                oracle_confidence_rows,
                oracle_compare_rows,
                oracle_metrics,
            ) = run_mode_oracle(dataset_csv=dataset_csv)
        except FileNotFoundError as error:
            print(error)
            return

        print_mode_oracle_report(
            oracle_summary_rows,
            oracle_confidence_rows,
            oracle_metrics,
        )
        (
            oracle_csv,
            oracle_summary_csv,
            oracle_confidence_csv,
            oracle_compare_csv,
        ) = (
            save_mode_oracle_outputs(
                oracle_rows=oracle_rows,
                summary_rows=oracle_summary_rows,
                confidence_summary_rows=oracle_confidence_rows,
                comparison_rows=oracle_compare_rows,
                output_dir=output_dir,
            )
        )
        print(f"{'oracle_csv':<15} -> {oracle_csv}")
        print(f"{'oracle_summary':<15} -> {oracle_summary_csv}")
        print(f"{'oracle_conf':<15} -> {oracle_confidence_csv}")
        print(f"{'oracle_compare':<15} -> {oracle_compare_csv}")

    if args.latency_benchmark and model_work_requested:
        latency_rows, latency_summary_rows, latency_winner_rows = (
            run_latency_benchmark(
                prompts=PROMPTS,
                cheap_model_name=args.cheap_model,
                expensive_model_name=args.expensive_model,
                max_new_tokens=speculative_max_new_tokens,
                temperature=args.adaptive_temperature,
                warmup_runs=args.latency_warmup_runs,
                measured_runs=args.latency_measured_runs,
                device=args.device,
                cheap_device=args.cheap_device,
                expensive_device=args.expensive_device,
                torch_dtype=args.torch_dtype,
                prompt_format=args.prompt_format,
                models=adaptive_models,
            )
        )
        print_latency_benchmark_report(latency_summary_rows)
        latency_csv, latency_summary_csv, latency_winners_csv = (
            save_latency_benchmark_outputs(
                rows=latency_rows,
                summary_rows=latency_summary_rows,
                winner_rows=latency_winner_rows,
                output_dir=output_dir,
            )
        )
        print(f"{'latency_csv':<15} -> {latency_csv}")
        print(f"{'latency_sum':<15} -> {latency_summary_csv}")
        print(f"{'latency_win':<15} -> {latency_winners_csv}")

    if args.task_evaluation and model_work_requested:
        (
            task_rows,
            task_summary_rows,
            task_difficulty_rows,
            task_overall_rows,
        ) = run_task_evaluation(
            dataset_path=task_dataset,
            cheap_model_name=args.cheap_model,
            expensive_model_name=args.expensive_model,
            max_new_tokens=speculative_max_new_tokens,
            temperature=args.adaptive_temperature,
            device=args.device,
            cheap_device=args.cheap_device,
            expensive_device=args.expensive_device,
            torch_dtype=args.torch_dtype,
            prompt_format=args.prompt_format,
            include_latency=args.include_latency,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            limit=args.limit,
            categories=args.categories,
            difficulties=args.difficulties,
            modes=args.modes,
            profile_runtime=args.profile_runtime,
        )
        print_task_evaluation_report(task_summary_rows)
        print_task_evaluation_overall_report(task_overall_rows)
        (
            task_csv,
            task_summary_csv,
            task_difficulty_csv,
            task_overall_csv,
        ) = save_task_evaluation_outputs(
            rows=task_rows,
            summary_rows=task_summary_rows,
            difficulty_rows=task_difficulty_rows,
            overall_rows=task_overall_rows,
            output_dir=output_dir,
        )
        print(f"{'task_csv':<15} -> {task_csv}")
        print(f"{'task_summary':<15} -> {task_summary_csv}")
        print(f"{'task_diff':<15} -> {task_difficulty_csv}")
        print(f"{'task_overall':<15} -> {task_overall_csv}")

        if args.include_latency:
            (
                latency_rows,
                latency_summary_rows,
                latency_by_category_rows,
                latency_by_difficulty_rows,
            ) = build_task_quality_latency_outputs(task_rows)
            print_task_quality_latency_report(latency_summary_rows)
            (
                report_csv,
                latency_summary_csv,
                latency_category_csv,
                latency_difficulty_csv,
            ) = save_task_quality_latency_outputs(
                report_rows=latency_rows,
                summary_rows=latency_summary_rows,
                by_category_rows=latency_by_category_rows,
                by_difficulty_rows=latency_by_difficulty_rows,
                output_dir=output_dir,
            )
            print(f"{'task_ql_report':<15} -> {report_csv}")
            print(f"{'task_ql_sum':<15} -> {latency_summary_csv}")
            print(f"{'task_ql_cat':<15} -> {latency_category_csv}")
            print(f"{'task_ql_diff':<15} -> {latency_difficulty_csv}")

        if args.profile_runtime:
            profile_rows, profile_summary_rows = build_runtime_profile_outputs(
                task_rows
            )
            print_runtime_profile_report(profile_summary_rows)
            profile_csv, profile_summary_csv = save_runtime_profile_outputs(
                profile_rows=profile_rows,
                summary_rows=profile_summary_rows,
                output_dir=output_dir,
            )
            print(f"{'runtime_prof':<15} -> {profile_csv}")
            print(f"{'runtime_sum':<15} -> {profile_summary_csv}")


if __name__ == "__main__":
    main()
