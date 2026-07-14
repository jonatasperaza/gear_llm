# Experimentos do GEAR-LLM

Este documento resume as fases experimentais implementadas no GEAR-LLM e o que cada uma tenta validar.

## 1. Token Analysis e `rho`

A primeira fase mede cada token do prompt e calcula um score `rho` para estimar sua criticidade.

O score combina:

- **Entropia**: incerteza da distribuição de próximo token.
- **Surprisal**: quão inesperado foi o token atual dado o contexto anterior.
- **Novidade geométrica**: distância do hidden state atual em relação ao histórico recente.
- **Curvatura semântica**: mudança de direção no espaço de hidden states.
- **Importância estrutural**: boost para tokens matemáticos, numéricos, lógicos, operadores, pontuação estrutural e delimitadores.

Forma geral:

```text
rho =
  entropy_weight    * entropy_norm
+ surprisal_weight  * surprisal_norm
+ novelty_weight    * novelty_norm
+ curvature_weight  * curvature_norm
+ structural_weight * structural_importance
```

O resultado por token é salvo em CSV com rota:

```text
cheap | medium | expensive
```

## 2. Balanced Ablation

A ablation simples substitui tokens de uma classe e mede o aumento de loss. A versão balanceada torna a comparação mais justa:

- define `k` como o número de tokens `expensive`;
- compara grupos com o mesmo tamanho;
- remove/substitui:
  - `top_k_expensive`
  - `bottom_k_cheap`
  - `top_k_medium`
  - `random_k`
- calcula `delta_loss` e `delta_loss_per_token`;
- roda múltiplos baselines aleatórios.

Critério desejado:

```text
expensive_delta_per_token > cheap_delta_per_token
expensive_delta_per_token > random_mean_delta_per_token
```

## 3. Teacher Calibration

A teacher calibration compara o modelo barato com o modelo caro no mesmo contexto.

Para cada passo de geração, mede:

- entropia normalizada do modelo barato;
- probabilidade top-1;
- probabilidade top-2;
- margem top-1 menos top-2;
- se o top-1 do barato bate com o top-1 do caro;
- se o top-1 do barato está no top-k do caro.

Depois faz grid search de thresholds:

```text
entropy_threshold: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55
margin_threshold : 0.10, 0.15, 0.20, 0.25, 0.30
```

Objetivo: encontrar thresholds que aceitem o modelo barato quando ele tende a concordar com o modelo caro.

## 4. Policy Replay

O replay de políticas usa os contextos já coletados pela teacher calibration.

Isso evita path-dependence da geração online: todas as políticas são comparadas nos mesmos passos e nos mesmos logits salvos.

Políticas atuais:

- `old_0.45_0.20`
- `calibrated_0.35_0.20`
- `strict_0.30_0.20`
- `loose_0.50_0.15`

Métricas:

- accept rate;
- exact precision accept;
- top-k precision accept;
- false accept rate;
- estimated saved percent.

## 5. Quality Benchmark

O benchmark de qualidade compara modos de geração contra `expensive_only`.

Modos principais:

- `cheap_only`
- `expensive_only`
- `adaptive_calibrated`
- `adaptive_guarded`
- `adaptive_guarded_v2`
- `adaptive_guarded_v3`

Métricas:

- economia estimada;
- chamadas ao modelo caro;
- similaridade com `expensive_only` via `difflib.SequenceMatcher`;
- Jaccard de palavras normalizadas;
- taxa de repetição de 3-gramas;
- taxa de repetição de 4-gramas.

## 6. Speculative Tuning

O speculative decoding usa o modelo barato para gerar um bloco de tokens de rascunho e o modelo caro para verificar esse bloco em uma única passagem.

A fase de tuning testa combinações de:

- `initial_draft_length`
- `verify_top_k`
- `min_draft_length`
- `max_draft_length`

Score usado:

```text
score =
    similarity_to_expensive
  + 0.25 * jaccard_to_expensive
  - 0.50 * repeated_3gram_rate
  + 0.20 * max(0, estimated_saved_percent / 100)
  - 0.75 * max(0, -estimated_saved_percent / 100)
```

Configuração atual escolhida:

```text
initial_draft_length = 6
verify_top_k = 3
min_draft_length = 2
max_draft_length = 8
```

## 7. Hybrid Benchmark

O hybrid router escolhe automaticamente um modo de geração com base no tipo de prompt.

Classificações:

- `logic`
- `math`
- `code`
- `long_simple`
- `general`

Política atual:

```text
logic       -> adaptive_guarded_v3
math        -> adaptive_calibrated
code        -> adaptive_code_quality
long_simple -> adaptive_calibrated
general     -> adaptive_calibrated
```

Prompts curtos e diretos, fora de código e matemática, podem usar
`speculative_adaptive`. Matemática não usa speculative por padrão porque o mode
oracle anterior não encontrou evidência suficientemente forte para essa regra.

O benchmark compara:

- `adaptive_calibrated`
- `adaptive_guarded_v3`
- `adaptive_code_quality`
- `speculative_adaptive`
- `hybrid`

Objetivo: evitar casos ruins do speculative em lógica e prompts longos simples, mantendo ganhos em matemática.

## 8. Latency Benchmark

O latency benchmark mede tempo real de geração para comparar a economia teórica com desempenho observado.

Modos medidos:

- `cheap_only`
- `expensive_only`
- `adaptive_calibrated`
- `adaptive_guarded_v3`
- `adaptive_code_quality`
- `speculative_adaptive`
- `hybrid`

Métricas:

- tempo total por execução;
- tokens gerados;
- tokens por segundo;
- economia estimada;
- chamadas ao modelo caro;
- tokens aceitos pelo caminho barato, quando aplicável;
- taxa de aceitação, quando aplicável;
- pico de memória CUDA, quando disponível.

O benchmark usa `time.perf_counter()` e sincroniza CUDA antes/depois da geração quando `torch.cuda.is_available()`.

Comando de smoke test:

```powershell
python run_latency_benchmark.py --max-new-tokens 8 --warmup-runs 0 --measured-runs 1
```

## 9. External Dataset Evaluation

External Dataset Evaluation builds larger task-specific JSONL files compatible with `run_task_evaluation.py`.

The builder is local-file first: it does not require internet during benchmark execution. Place public benchmark files under:

```text
data/external_sources/gsm8k/test.jsonl or test.json
data/external_sources/mbpp/mbpp.jsonl or sanitized-mbpp.json
data/external_sources/logiqa/test.jsonl or test.json
```

Then build a normalized dataset:

```powershell
python scripts/build_external_eval_tasks.py `
  --output data/external_eval_tasks.jsonl `
  --math-source gsm8k `
  --code-source mbpp `
  --logic-source logiqa `
  --math-limit 200 `
  --code-limit 200 `
  --logic-limit 200 `
  --seed 42
```

Recommended local run modes:

Correctness-only larger run:

```powershell
python run_task_evaluation.py `
  --dataset data/external_eval_tasks.jsonl `
  --modes expensive_only,cheap_only,adaptive_calibrated,hybrid `
  --max-new-tokens 128
```

Latency subset:

```powershell
python run_task_evaluation.py `
  --dataset data/external_eval_tasks.jsonl `
  --limit 60 `
  --modes expensive_only,cheap_only,adaptive_calibrated,hybrid `
  --include-latency `
  --warmup-runs 1 `
  --measured-runs 3 `
  --max-new-tokens 128
```

For smoke tests without external downloads, use the bundled sample fixtures:

```powershell
python scripts/build_external_eval_tasks.py `
  --output data/external_eval_tasks_smoke.jsonl `
  --math-source sample `
  --code-source sample `
  --logic-source sample `
  --math-limit 2 `
  --code-limit 2 `
  --logic-limit 2 `
  --seed 42
```

## 10. Prompt-Level Routing

Runtime profiling showed a structural limitation of token-level routing: the
cheap model runs on nearly every token, and an expensive fallback adds another
forward pass. Prompt-level routing avoids that sum by choosing `cheap_only` or
`expensive_only` before generation.

Implemented policies:

- `prompt_router_v1`: conservative manual heuristics.
- `prompt_router_v2`: cheap by default, with high-confidence expensive rules.
- `prompt_router_ml_v1`: TF-IDF plus Logistic Regression trained from oracle
  labels.

The original 30-task MBPP sample made v2 look very strong: it matched the
oracle at 70% pass rate while retaining about 39% real speedup. On a different
100-task seed, however, it selected cheap for 99 prompts and reached only 48%,
showing that the manual rules overfit the analysis sample.

## 11. Learned Router Evaluation

The ML router dataset joins `cheap_only` and `expensive_only` task scores. The
current target is `oracle_score_label`: use expensive only when its score is
higher than the cheap score.

The seed123 model reproduced the oracle in-sample, but seed999 exposed 20
overlapping prompts. On the remaining 80 unseen prompts, the default classifier
predicted no true `expensive_only` case. ROC-AUC was 0.5545 and Average Precision
was 0.2047, indicating weak ranking signal from TF-IDF alone.

Threshold simulations at 0.40 and 0.43 are exploratory because they were tuned
on the same unseen set being reported. The next valid protocol is one fixed,
non-overlapping split over all 427 MBPP tasks: train for fitting, validation for
feature/class-weight/threshold choices, and a final untouched test set.

Canonical Kaggle artifacts are stored under `results/kaggle/`. See
`results/kaggle/README.md` for seed and provenance details.

## 12. Fixed MBPP Split and Prompt Probing

The next router protocol uses all 427 sanitized MBPP tasks with one persisted,
non-overlapping split:

- train: 257 tasks;
- validation: 85 tasks;
- test: 85 tasks.

`scripts/build_mbpp_split.py` stratifies by the existing MBPP difficulty
heuristic and writes `data/mbpp_split_manifest.jsonl`. The dataset builder
validates every split JSONL against this manifest before generation.

`gear_llm/probing_features.py` extracts prompt-level features from one cheap
and one expensive prompt prefill: normalized entropy, margin, top-1
probability, robust surprisal, geometric novelty/curvature, structural density,
top-k agreement, approximate KL, rank, token count, and word count. Tensor
comparisons are device-safe for `cuda:0`/`cuda:1`, while the full dual-GPU test
remains to be run on Kaggle.

The probing cost is real. Inference with agreement features touches the
expensive model before route selection, so latency reports include this
decision time and expose the probing forward counts separately.

## 13. Learning-to-Defer Router v2

`scripts/build_router_dataset_v2.py` generates cheap and expensive answers,
evaluates both, derives oracle labels and writes probing features with
incremental deduplicated checkpoints. Model/device/dtype and synchronized
generation/probing times are recorded for reproducibility.

`scripts/train_prompt_router_v2.py` supports two policies:

- `classifier`: Logistic Regression or dense GBDT over TF-IDF plus probing;
- `l2d`: Ridge regression over `delta_score`, followed by expected-cost
  threshold calibration.

Estimator regularization, class weight and route threshold are selected on
validation. `scripts/eval_router_v2.py` then evaluates the frozen policy on the
test CSV and optionally includes the archived TF-IDF v1 as a historical
reference.

Local smoke validation has passed with distinct train/validation files, model
serialization, one-shot test reporting and task-evaluation dispatch. These
smoke rows are not scientific results. The complete 427-task feature build and
the untouched 85-task test evaluation remain pending on Kaggle.
