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

## Task Quality-Latency Benchmark

### Qwen2.5 0.5B -> 3B

This benchmark evaluates task-level correctness and real wall-clock latency on the same run. Unlike lexical similarity to `expensive_only`, the evaluator checks simple expected answers for math, labels for logic, and local unit tests for code.

The repeated timing run adds warmup and multiple measured generations per task/mode to reduce dependence on a single laptop CUDA timing sample.

Configuration:

- Cheap model: `Qwen/Qwen2.5-0.5B-Instruct`
- Expensive model: `Qwen/Qwen2.5-3B-Instruct`
- Device: `cuda`
- torch_dtype: `float16`
- prompt_format: `auto` with effective chat template
- max_new_tokens: 128
- warmup_runs: 1
- measured_runs: 3
- Dataset: `data/eval_tasks.jsonl`
- Dataset size: 45 tasks
  - 15 math
  - 15 logic
  - 15 code
  - balanced across easy, medium, and hard

Overall results:

| Mode | Pass rate | Avg speedup | Avg calls | Avg time | Std time |
|---|---:|---:|---:|---:|---:|
| expensive_only | 91.11% | 0.00% | 16.64 | 13.594s | 24.238s |
| cheap_only | 73.33% | 55.32% | 0.00 | 4.508s | 6.094s |
| adaptive_calibrated | 86.67% | 45.70% | 2.09 | 5.959s | 8.213s |
| adaptive_guarded_v3 | 86.67% | 41.58% | 2.71 | 6.916s | 9.828s |
| speculative_adaptive | 80.00% | 15.48% | 4.40 | 6.539s | 9.671s |
| hybrid | 86.67% | 46.66% | 2.64 | 5.881s | 9.359s |

Interpretation:

- This is the strongest quality-latency result so far.
- `hybrid` was narrowly the best routed mode by quality-latency trade-off.
- `adaptive_calibrated` was extremely close, with the same pass rate and fewer expensive calls.
- `hybrid` and the adaptive modes reached 86.67% pass rate, compared with 91.11% for `expensive_only` and 73.33% for `cheap_only`.
- `hybrid` and the adaptive modes preserved about 95.13% of the `expensive_only` pass rate.
- `cheap_only` was faster, but lost much more accuracy.
- `speculative_adaptive` was weaker in this task benchmark.

Limitations:

- The full run took about 2h10 on the local laptop.
- Standard deviations are high.
- The reported standard deviation reflects task/output variability and laptop CUDA variability, not only repeated-run noise.
- Memory pressure and shared GPU memory may affect timings.
- The dataset has only 45 tasks.
- The code evaluator is a local subprocess harness, not a production sandbox.
- Math evaluation is still normalization/contains-based, not symbolic.
- Logic evaluation is keyword-rule-based.
- Results are specific to `Qwen2.5-0.5B -> Qwen2.5-3B` on this CUDA setup.

This benchmark improves the quality signal beyond lexical similarity, but it does not prove quality is solved. It should be treated as preliminary evidence that task-aware routed generation can approach expensive-only correctness while reducing real latency in a favorable setting, not as a production benchmark.

## External MBPP Code Benchmark — Qwen2.5-Coder 0.5B → 3B, Split GPU

This benchmark evaluates external MBPP-style code tasks from `data/external_eval_tasks_90.jsonl`. It uses a split-GPU setup with a Qwen2.5-Coder cheap/expensive pair.

Configuration:

- Dataset: `data/external_eval_tasks_90.jsonl`
- Category: `code`
- Task count: 30
- Cheap model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Expensive model: `Qwen/Qwen2.5-Coder-3B-Instruct`
- cheap_device: `cuda:0`
- expensive_device: `cuda:1`
- device: `split`
- torch_dtype: `float16`
- prompt_format: `auto`
- max_new_tokens: 256
- warmup_runs: 1
- measured_runs: 3

Results:

| Mode | Pass rate | Avg time | Avg speedup | Median speedup | Avg estimated saved | Avg expensive calls |
|---|---:|---:|---:|---:|---:|---:|
| expensive_only | 60.00% | 13.183s | 0.00% | - | 0.00% | 133.77 |
| cheap_only | 46.67% | 5.232s | 34.62% | - | 65.00% | 0.00 |
| adaptive_guarded_v3 | 53.33% | 6.081s | 24.58% | 41.19% | 58.32% | 9.77 |
| hybrid | 53.33% | 6.065s | 24.83% | 41.24% | 58.32% | 9.77 |

Interpretation:

- Updating `hybrid` to route code prompts to `adaptive_guarded_v3` improved hybrid pass rate from 50.00% to 53.33%.
- `hybrid` preserved 88.9% of the `expensive_only` pass rate.
- `hybrid` reduced average expensive-model calls from 133.77 to 9.77, approximately a 92.7% reduction.
- `hybrid` achieved 24.83% average per-task speedup and 41.24% median per-task speedup.
- This is promising external evidence for routed code generation, but it is not definitive: the sample has only 30 tasks, and confidence intervals are still wide.
- The result should be interpreted as preliminary external evidence, not as a final generalization claim.
