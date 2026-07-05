# Resultados Atuais

Snapshot dos resultados experimentais atuais do GEAR-LLM.

Data do snapshot: 2026-07-04.

Modelos usados:

- barato: `HuggingFaceTB/SmolLM2-135M-Instruct`
- caro: `HuggingFaceTB/SmolLM2-360M-Instruct`

Aviso: estes resultados são iniciais, dependem dos prompts de benchmark e ainda não representam aceleração real do modelo. A economia é uma estimativa de custo teórico.

## Teacher Calibration

A calibração offline indicou que a configuração calibrada abaixo é mais conservadora que a política antiga:

```text
entropy_threshold = 0.35
margin_threshold  = 0.20
```

Resultado agregado observado:

```text
precision_accept = 97.90%
false_accept     = 2.10%
saved_percent    = 36.50%
```

Interpretação: quando o modelo barato é aceito sob esses thresholds, ele tende a concordar muito bem com o modelo caro nos contextos calibrados.

## Guarded v3 vs v2

O `adaptive_guarded_v2` mostrou um problema importante: em alguns prompts, especialmente `easy`, o guard de repetição chamava o modelo caro demais e podia gerar economia negativa.

O `adaptive_guarded_v3` adicionou:

- separação entre fallbacks obrigatórios e opcionais;
- budget cap para fallbacks opcionais;
- repetition guard condicionado à incerteza;
- cooldown para repetition guard.

Comparação principal em `results/quality_benchmark.csv`:

| prompt | v2 saved | v2 calls | v3 saved | v3 calls | observação |
| --- | ---: | ---: | ---: | ---: | --- |
| easy | -2.50% | 54 | 53.75% | 9 | v3 removeu o caso de economia negativa |
| math | 26.25% | 31 | 46.25% | 15 | v3 reduziu chamadas caras |
| logic_negation | 3.75% | 49 | 42.50% | 18 | v3 preservou parte da melhora com muito menos custo |
| code | 50.00% | 12 | 50.00% | 12 | custo equivalente |
| long_simple | 18.75% | 37 | 21.25% | 35 | leve melhora de custo |

## Speculative Benchmark

Configuração speculative atual:

```text
initial_draft_length = 6
verify_top_k         = 3
min_draft_length     = 2
max_draft_length     = 8
```

Resumo de `results/speculative_benchmark.csv`:

| prompt | mode | saved | expensive calls | acceptance | similarity |
| --- | --- | ---: | ---: | ---: | ---: |
| easy | speculative_adaptive | 45.19% | 12 | 96.25% | 0.1050 |
| math | speculative_adaptive | 33.25% | 17 | 90.00% | 0.5940 |
| logic_negation | speculative_adaptive | 14.31% | 22 | 81.25% | 0.0550 |
| code | speculative_adaptive | 49.06% | 11 | 97.50% | 0.5820 |
| long_simple | speculative_adaptive | 11.69% | 22 | 76.25% | 0.0360 |

Leitura rápida:

- `math`: speculative melhorou similaridade contra `adaptive_calibrated` neste snapshot.
- `code`: speculative manteve economia alta.
- `logic_negation` e `long_simple`: speculative foi pior em similaridade, motivando o hybrid router.

## Hybrid Benchmark

O hybrid router escolhe:

```text
math          -> speculative_adaptive
logic         -> adaptive_guarded_v3
code/general  -> adaptive_calibrated
```

Resumo de `results/hybrid_benchmark.csv`:

| prompt | prompt_type | selected_mode | saved | expensive calls | similarity |
| --- | --- | --- | ---: | ---: | ---: |
| easy | general | adaptive_calibrated | 57.50% | 6 | 0.0856 |
| math | math | speculative_adaptive | 33.25% | 17 | 0.5940 |
| logic_negation | logic | adaptive_guarded_v3 | 42.50% | 18 | 0.4557 |
| code | code | adaptive_calibrated | 50.00% | 12 | 0.6597 |
| long_simple | general | adaptive_calibrated | 21.25% | 35 | 0.1190 |

Critério atingido:

- evita o pior caso do speculative em `logic_negation`;
- evita speculative em `long_simple`;
- mantém o ganho do speculative no prompt `math`;
- mantém políticas simples e interpretáveis.

## Próximos Passos

- ampliar o conjunto de prompts;
- separar prompts por idioma e domínio;
- medir latência real com KV cache;
- comparar com speculative decoding mais fiel ao algoritmo clássico;
- usar métricas de qualidade mais fortes que similaridade textual superficial;
- investigar thresholds específicos por tipo de prompt.

## CPU Latency Benchmark

The latency benchmark measures real wall-clock generation time, not only estimated compute savings.

Configuration:

- Device: CPU
- max_new_tokens: 32
- warmup_runs: 1
- measured_runs: 5
- Cheap model: HuggingFaceTB/SmolLM2-135M-Instruct
- Expensive model: HuggingFaceTB/SmolLM2-360M-Instruct

Main findings:

- `cheap_only` is always the fastest mode, but it is not a quality-equivalent baseline.
- Among modes that use the expensive model, adaptive routing is the most stable strategy on CPU.
- `adaptive_calibrated` and `adaptive_guarded_v3` consistently produce real speedups over `expensive_only`.
- `speculative_adaptive` is competitive on `code`, but slower than `expensive_only` on `logic_negation`, `long_simple`, and `math`.
- Hybrid routing has negligible routing overhead, but its latency depends entirely on the selected generation mode.
- Quality-oriented routing and latency-oriented routing can disagree.

Best non-cheap modes by prompt:

| Prompt | Best mode excluding cheap_only | Real speedup vs expensive_only |
|---|---:|---:|
| code | speculative_adaptive | 33.19% |
| easy | adaptive_guarded_v3 | 37.19% |
| logic_negation | adaptive_calibrated | 22.79% |
| long_simple | adaptive_calibrated | 11.56% |
| math | hybrid | 36.56% |

Interpretation:

The CPU benchmark suggests that adaptive token-level routing is currently more robust than the speculative implementation. Speculative decoding remains experimental and may require larger model gaps, optimized block verification, and KV-cache-aware implementation before becoming globally competitive.

## GPU Latency Benchmark

### SmolLM2 135M -> 360M

The CUDA latency benchmark was validated with:

- `torch.cuda.is_available() = True`
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- max_new_tokens: 32
- warmup_runs: 1
- measured_runs: 5
- Cheap model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Expensive model: `HuggingFaceTB/SmolLM2-360M-Instruct`

Main result:

With this small model pair on GPU, `expensive_only` was the fastest mode when excluding `cheap_only` for every benchmark prompt.

Best non-cheap modes by prompt:

| Prompt | Best mode excluding cheap_only |
|---|---|
| code | expensive_only |
| easy | expensive_only |
| logic_negation | expensive_only |
| long_simple | expensive_only |
| math | expensive_only |

Interpretation:

- On CPU, `adaptive_calibrated` and `adaptive_guarded_v3` produced real speedups over `expensive_only`.
- On GPU, with small models, `expensive_only` wins because the expensive model is still cheap enough and adaptive modes add Python/model-switching overhead.
- Hybrid routing has very small decision overhead, but the selected generation mode can still be slower than `expensive_only`.
- `speculative_adaptive` is not competitive in the current GPU benchmark.
- The next validation should use model pairs with a larger performance gap, such as `Qwen2.5-0.5B -> Qwen2.5-1.5B` or `Qwen2.5-1.5B -> Qwen2.5-3B`, preferably with configurable model support across benchmarks.

### Qwen2.5 0.5B -> 3B

This benchmark tested a larger model gap on the same limited GPU class:

- `torch.cuda.is_available() = True`
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- device: `cuda`
- dtype: `float16`
- max_new_tokens: 32
- warmup_runs: 1
- measured_runs: 3
- Cheap model: `Qwen/Qwen2.5-0.5B-Instruct`
- Expensive model: `Qwen/Qwen2.5-3B-Instruct`

Best non-cheap modes by prompt:

| Prompt | Best mode excluding cheap_only | Real speedup vs expensive_only |
|---|---|---:|
| code | hybrid | 76.96% |
| easy | adaptive_calibrated | 68.82% |
| logic_negation | adaptive_guarded_v3 | 66.63% |
| long_simple | speculative_adaptive | 58.27% |
| math | speculative_adaptive | 71.42% |

Interpretation:

- This is the first strong GPU latency result for the project.
- Unlike `SmolLM2-135M -> SmolLM2-360M`, where `expensive_only` won, the larger `Qwen2.5-0.5B -> Qwen2.5-3B` gap makes adaptive routing advantageous.
- This supports the hypothesis that real speedup depends strongly on the gap between models and on the real cost of the expensive model on the target hardware.
- Peak memory was close to 7 GB, above the RTX 3050 Laptop GPU's 6 GB dedicated VRAM, so the run likely involved Windows shared GPU/system memory.
- This is a valid result for limited hardware, but it is not equivalent to fitting the full run inside dedicated VRAM.
- `speculative_adaptive` became competitive again on `long_simple` and `math` once the expensive model became costly enough.

## Quality-Latency Report

### Qwen2.5 0.5B -> 3B

This report combines real latency and quality metrics for the same model pair and runtime configuration:

- Cheap model: `Qwen/Qwen2.5-0.5B-Instruct`
- Expensive model: `Qwen/Qwen2.5-3B-Instruct`
- Device: `cuda`
- torch_dtype: `float16`
- prompt_format: `auto` with effective chat template
- max_new_tokens: 64
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

| Prompt | Latency winner | Real speedup | Similarity | Jaccard | Rep3 | Quality-latency score |
|---|---|---:|---:|---:|---:|---:|
| code | adaptive_calibrated | 84.74% | 0.5517 | 0.5294 | 0.0000 | 0.8959 |
| easy | adaptive_guarded_v3 | 56.41% | 0.2440 | 0.3023 | 0.0000 | 0.4606 |
| logic_negation | adaptive_guarded_v3 | 76.20% | 0.0547 | 0.3556 | 0.0263 | 0.3209 |
| long_simple | adaptive_guarded_v3 | 68.68% | 0.0467 | 0.2034 | 0.0000 | 0.2692 |
| math | adaptive_guarded_v3 | 83.11% | 0.2011 | 0.2750 | 0.0000 | 0.4776 |

Interpretation:

- This is the first combined quality-latency report for the project.
- Latency shows strong real speedups across all prompts.
- `code` is the strongest case: high speedup with relatively good similarity and Jaccard overlap.
- `easy` and `math` have strong speedups, but only moderate to low textual similarity.
- `logic_negation` and `long_simple` show that textual similarity can be low even when latency speedup is high, so semantic and task-specific evaluation are needed.
- The project has demonstrated real speedup, but it has not yet demonstrated quality equivalence to `expensive_only`.
- The next step is task-specific evaluation: unit tests for code, symbolic or expected-answer checks for math, labels for logic, and LLM-judge or human review for open-ended text.
