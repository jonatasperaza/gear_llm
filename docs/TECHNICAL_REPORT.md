# GEAR-LLM: Preliminary Results on Latency-Aware Cheap/Expensive LLM Routing

## Abstract

GEAR-LLM is an experimental Python, PyTorch, and Hugging Face project that investigates routing between a cheap language model and a more expensive language model. The goal is to reduce latency and theoretical compute cost without always relying on the expensive model for every token or every prompt.

The current prototype includes offline token analysis, teacher calibration,
adaptive dual-model generation, speculative decoding, prompt-level routing,
task-specific correctness evaluation, latency benchmarking, and runtime
profiling. The strongest latency result is a CUDA benchmark using
Qwen2.5-0.5B-Instruct as the cheap model and Qwen2.5-3B-Instruct as the
expensive model with chat formatting enabled. Prompt-level experiments also
demonstrate model complementarity, but the learned router does not yet
generalize reliably to unseen prompts.

GEAR-LLM demonstrated large real wall-clock speedups in a specific GPU setting when the expensive model was sufficiently costly relative to the cheap model and hardware. Quality preservation is not yet proven robustly; current similarity/Jaccard metrics are weak proxies for semantic correctness.

## 1. Motivation

Large language models are powerful, but they are also expensive to run. Smaller models are faster and cheaper, but they may fail on harder reasoning, formatting, or instruction-following cases.

The central question behind GEAR-LLM is simple: when is it worth calling the expensive model?

The hypothesis is that not every token or prompt needs the same level of computation. Some text is predictable, repetitive, or low-risk. Other text contains logic, negation, math, code, structural symbols, or semantic turns where the cheap model may be less reliable. If the system can identify when the cheap model is likely sufficient, it may reduce cost and latency while preserving enough quality for useful workloads.

## 2. What GEAR-LLM is

GEAR-LLM is a research prototype for studying cheap/expensive model routing. It currently supports several experimental components:

- **Cheap model:** the smaller model used as the default low-cost path.
- **Expensive model:** the larger model used as the higher-quality fallback or verifier.
- **Token analysis / rho:** an offline analysis phase that scores tokens using entropy, surprisal, novelty, curvature, and structural importance.
- **adaptive_calibrated:** token-by-token generation that accepts the cheap model when calibrated entropy and margin thresholds indicate confidence.
- **adaptive_guarded_v3:** an adaptive generator with repetition guards, periodic teacher checks, uncertainty gating, cooldowns, and a budget cap for optional fallbacks.
- **adaptive_code_quality:** a conservative token-level profile for structured code generation.
- **speculative_adaptive:** a block-based draft-and-verify generator where the cheap model drafts multiple tokens and the expensive model verifies the block.
- **hybrid router:** a heuristic router that chooses among token-level and speculative generation policies.
- **prompt_router_v1/v2:** manual policies that choose cheap-only or expensive-only before generation.
- **prompt_router_ml_v1:** TF-IDF plus Logistic Regression trained from cheap/expensive oracle labels.
- **prompt_router_ml_v2:** fixed-split prompt router with classifier and learning-to-defer candidates over TF-IDF and prompt-probing features. The completed 427-task protocol selected the TF-IDF-only classifier on validation.
- **teacher calibration:** an offline procedure that compares cheap-model predictions against expensive-model predictions on the same context.
- **mode oracle:** an offline benchmark that estimates which generation mode had the best quality-cost score for each prompt.
- **task evaluation:** expected-answer checks for math, labels for logic, and local subprocess tests for generated Python code.
- **latency benchmark:** a wall-clock benchmark that measures actual generation time, tokens per second, memory peaks, and speedup against expensive_only.
- **runtime profiling:** separates cheap forward, expensive forward, routing, guard, prompt-formatting, decoding, and evaluation time.
- **quality-latency report:** a combined report that joins latency winners with quality proxy metrics.

The project is deliberately simple and didactic. It does not yet implement optimized KV-cache sharing, production serving, or fully optimized speculative decoding.

## 3. What this project does NOT claim

GEAR-LLM should be read as preliminary experimental research, not as a production inference engine.

It does not claim:

- Quality equivalent to the expensive model.
- Universal superiority over expensive-only inference.
- Replacement for optimized speculative decoding implementations.
- Validation at large production scales such as 7B -> 70B.
- A production-ready serving stack.
- Hardware-independent results.
- Router thresholds that generalize across all domains or model families.

The results depend on hardware, model pair, prompt format, dtype, implementation overhead, tokenizer behavior, and the specific prompts used in the benchmark.

## 4. Related Work

GEAR-LLM sits near several established research and engineering areas:

- Speculative decoding, where a smaller draft model proposes tokens and a larger model verifies them.
- Model routing and cascades, where easy examples are handled by cheaper models and harder examples are escalated.
- Adaptive computation and early-exit methods, where the amount of computation varies by input or token.
- Mixture-of-experts routing, where different components are selected dynamically.
- Cost-quality trade-off evaluation for inference systems.

This report does not include formal citations yet. References should be added before formal publication.

## 5. Experimental Setup

The main hardware setting documented here is:

- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Device: CUDA when available.
- Dtype: float16 for the main Qwen GPU runs.
- Prompt format: auto/chat for Qwen Instruct models.
- Main quality-latency max tokens: 64.

The main model pairs tested were:

- HuggingFaceTB/SmolLM2-135M-Instruct -> HuggingFaceTB/SmolLM2-360M-Instruct.
- Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen2.5-1.5B-Instruct.
- Qwen/Qwen2.5-0.5B-Instruct -> Qwen/Qwen2.5-3B-Instruct.

The Qwen Instruct runs use chat template formatting through the tokenizer when available. This matters because raw prompts can produce poor instruction-following behavior for chat/instruct models.

The Qwen2.5-0.5B -> Qwen2.5-3B CUDA runs reported memory peaks around 7 GB in the benchmark summaries. This is above the 6 GB dedicated VRAM of the RTX 3050 Laptop GPU, so the run likely involved Windows shared GPU memory. This should be treated as a real result on constrained hardware, but not as equivalent to a run fully contained in dedicated VRAM.

## 6. Latency Results

### 6.1 CPU findings

Saved CPU results were present under `results/cpu_latency/`. In that run, adaptive_calibrated and adaptive_guarded_v3 showed real speedups on some prompts, and hybrid also won one prompt. However, speculative_adaptive was not robust globally. This supports one of the project lessons: estimated compute savings and real wall-clock speedups are related, but they are not the same thing.

### 6.2 GPU with small model gap

Saved GPU results for SmolLM2-135M-Instruct -> SmolLM2-360M-Instruct were present under `results/gpu_latency/`. In that setting, expensive_only was the fastest non-cheap mode for all five prompts.

This negative result is important. When the expensive model is still small enough, the overhead of routing, switching, Python control flow, and dual-model execution can dominate the theoretical savings.

### 6.3 GPU with medium model gap

Saved GPU results for Qwen2.5-0.5B-Instruct -> Qwen2.5-1.5B-Instruct were present under `results/gpu_latency_qwen_0_5b_1_5b/`. This model pair started to show adaptive wins in some prompts, especially code and logic_negation, but expensive_only still won other prompts.

This suggests that the model gap matters, but the 0.5B -> 1.5B gap was not enough to make routing reliably superior in the saved benchmark.

### 6.4 GPU with larger model gap: Qwen2.5-0.5B -> Qwen2.5-3B

The strongest latency result currently comes from:

- Cheap model: Qwen/Qwen2.5-0.5B-Instruct.
- Expensive model: Qwen/Qwen2.5-3B-Instruct.
- Device: CUDA.
- Dtype: float16.
- Prompt format: auto/chat.
- Max new tokens: 64.

The following table uses `results/gpu_latency_qwen_0_5b_3b_64_chat/latency_winners.csv`.

| Prompt | Best non-cheap mode | Real speedup vs expensive_only |
|---|---:|---:|
| code | adaptive_calibrated | 84.74% |
| easy | adaptive_guarded_v3 | 56.41% |
| logic_negation | adaptive_guarded_v3 | 76.20% |
| long_simple | adaptive_guarded_v3 | 68.68% |
| math | adaptive_guarded_v3 | 83.11% |

This is the strongest latency result in the project so far. The gain appeared when the expensive model became sufficiently costly relative to the cheap model and the hardware. It suggests that model gap and hardware characteristics are central factors in whether cheap/expensive routing produces real speedup.

The memory caveat matters. The measured memory peaks were near 7 GB, above the dedicated 6 GB VRAM of the RTX 3050 Laptop GPU. This result is still useful for a constrained laptop setup, but it should not be interpreted as a clean dedicated-VRAM inference result.

## 7. Quality-Latency Report

The quality-latency report combines the latency winner for each prompt with simple quality proxy metrics. The following table uses `results/quality_latency_report.csv`.

| Prompt | Latency winner | Speedup | Similarity | Jaccard | Rep3 | Quality-latency score |
|---|---:|---:|---:|---:|---:|---:|
| code | adaptive_calibrated | 84.74% | 0.5517 | 0.5294 | 0.0000 | 0.8959 |
| easy | adaptive_guarded_v3 | 56.41% | 0.2440 | 0.3023 | 0.0000 | 0.4606 |
| logic_negation | adaptive_guarded_v3 | 76.20% | 0.0547 | 0.3556 | 0.0263 | 0.3209 |
| long_simple | adaptive_guarded_v3 | 68.68% | 0.0467 | 0.2034 | 0.0000 | 0.2692 |
| math | adaptive_guarded_v3 | 83.11% | 0.2011 | 0.2750 | 0.0000 | 0.4776 |

Code is the strongest case so far: it combines high speedup with reasonable similarity and Jaccard scores against expensive_only.

Easy and math show strong speedups, but only moderate or low textual similarity. Logic_negation and long_simple show especially clearly that text similarity is not enough to evaluate semantic correctness. A response may be correct but phrased differently, or textually similar but logically wrong.

This report demonstrates a quality-latency trade-off. It does not demonstrate quality equivalent to expensive_only.

## Task-Specific Quality-Latency Evaluation

The next evaluation path moves beyond lexical similarity to `expensive_only`. It uses a small task-oriented dataset with expected answers for math, labels for logic, and local unit tests for code.

Configuration:

- Cheap model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Expensive model: `Qwen/Qwen2.5-3B-Instruct`.
- Device: CUDA.
- torch_dtype: float16.
- Prompt format: auto/chat.
- Max new tokens: 128.
- Dataset: `data/eval_tasks.jsonl`.
- Dataset size: 45 tasks, with 15 math, 15 logic, and 15 code tasks balanced across easy, medium, and hard.

| Mode | Pass rate | Avg real speedup | Avg estimated saved | Avg expensive calls | Avg time |
|---|---:|---:|---:|---:|---:|
| expensive_only | 91.11% | 0.00% | 0.00% | 16.64 | 9.365s |
| cheap_only | 73.33% | 60.00% | 65.00% | 0.00 | 2.292s |
| adaptive_calibrated | 86.67% | 51.66% | 54.32% | 2.09 | 3.232s |
| adaptive_guarded_v3 | 86.67% | 47.34% | 53.69% | 2.71 | 3.751s |
| speculative_adaptive | 80.00% | 17.56% | 20.11% | 4.40 | 4.289s |
| hybrid | 86.67% | 52.60% | 51.03% | 2.64 | 3.276s |

This is the strongest preliminary evidence so far that routed modes can preserve much of task-level correctness while reducing real latency in this specific model and hardware setting. `hybrid` was the strongest quality-latency mode in this run, while `adaptive_calibrated` was very close and used fewer expensive calls on average.

The routed adaptive/hybrid modes reached 86.67% pass rate, compared with 91.11% for `expensive_only` and 73.33% for `cheap_only`. In this run, `hybrid` preserved about 95.13% of the `expensive_only` pass rate while achieving 52.60% average real speedup. `cheap_only` was faster, but lost substantially more accuracy, and `speculative_adaptive` was weaker on this task benchmark.

The result is encouraging, but it should not be overread. It is based on one measured run per task/mode, a small 45-task dataset, laptop CUDA timing, a local subprocess code evaluator, normalization-based math checks, and keyword-based logic labels. It improves the quality signal beyond lexical similarity, but it does not prove quality is solved.

## External MBPP Code Benchmark — Qwen2.5-Coder 0.5B → 3B, Split GPU

This benchmark extends the task-specific evaluation path to external MBPP-style code tasks. It uses a coder-specialized cheap/expensive pair and places the two models on separate CUDA devices.

Configuration:

- Dataset: `data/external_eval_tasks_90.jsonl`.
- Category: `code`.
- Task count: 30.
- Cheap model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- Expensive model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- cheap_device: `cuda:0`.
- expensive_device: `cuda:1`.
- Device: split.
- torch_dtype: float16.
- Prompt format: auto.
- max_new_tokens: 256.
- warmup_runs: 1.
- measured_runs: 3.

| Mode | Pass rate | Avg total time | Avg speedup | Median speedup | Avg estimated saved | Avg expensive calls |
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
- For this MBPP code run, `adaptive_guarded_v3` is the best routed policy because it preserves more pass rate than `adaptive_calibrated` while still maintaining real speedup.
- This is promising external evidence, but it is not definitive. The sample has only 30 tasks, confidence intervals are wide, and code evaluation still depends on the local subprocess harness and extracted generated functions.
- The result should be treated as preliminary external evidence, not as a final generalization claim.

## Larger MBPP Correctness-Only Follow-up

This follow-up used the same MBPP-style code evaluation path on a larger subset. The CSV files for this run were lost, so the result below was recovered from the printed Kaggle output. That makes it less auditable than the CSV-backed benchmark above, but still useful for tracking the direction of the experiment.

Configuration:

- Category: `code`.
- Cheap model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- Expensive model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- cheap_device: `cuda:0`.
- expensive_device: `cuda:1`.
- torch_dtype: float16.
- prompt_format: auto.
- max_new_tokens: 256.
- Modes: `expensive_only`, `cheap_only`, `adaptive_guarded_v3`, `hybrid`.

Recovered results:

| Mode | Pass rate | Avg score | Avg estimated saved | Avg expensive calls |
|---|---:|---:|---:|---:|
| expensive_only | 55.00% | 0.580 | 0.00% | 145.66 |
| cheap_only | 39.00% | 0.403 | 65.00% | 0.00 |
| adaptive_guarded_v3 | 46.00% | 0.470 | 58.14% | 10.30 |
| hybrid | 46.00% | 0.470 | 58.14% | 10.30 |

Interpretation:

- `hybrid` preserved 83.6% of the `expensive_only` pass rate.
- `hybrid` improved over `cheap_only` by 7 percentage points.
- `hybrid` reduced average expensive-model calls from 145.66 to 10.30, about a 92.9% reduction.
- This larger run is weaker than the 30-task MBPP result, where `hybrid` preserved 88.9% of the expensive-only pass rate.
- The guarded hybrid policy strongly reduces expensive calls, but quality preservation is not robust enough yet on larger MBPP subsets.

## 8. Prompt-Level Routing and Generalization

Runtime profiling exposed a structural cost in the current token-level
prototype: cheap forwards run throughout generation, and an expensive fallback
adds another model forward. Prompt-level routing instead chooses one model
before generation.

On the original 30-task MBPP sample, `prompt_router_v2` selected cheap for 21
tasks and expensive for 9. It reached 70% pass rate and 0.7444 average score,
matching the oracle, while averaging 5.107 seconds versus 13.234 seconds for
`expensive_only`. Average real speedup was 39.32%.

That result did not generalize to the seed123 100-task sample:

| Mode | Pass rate | Avg score | Estimated saved | Route distribution |
|---|---:|---:|---:|---|
| expensive_only | 52.00% | 0.5475 | 0.00% | 100 expensive |
| cheap_only | 47.00% | 0.4968 | 65.00% | 100 cheap |
| prompt_router_v2 | 48.00% | 0.5068 | 64.35% | 99 cheap / 1 expensive |

The strict oracle reached 61%, confirming complementarity: cheap failed while
expensive passed on 14 tasks, while cheap passed and expensive failed on 9.
The manual router captured only one of the 14 strict expensive-needed cases.

`prompt_router_ml_v1` was then trained on seed123 using TF-IDF and Logistic
Regression. It reproduced the 61% oracle pass rate in-sample, with 53.95%
estimated savings, but this does not constitute held-out evidence.

On seed999, the ML router reached 49%, compared with 54% for expensive and 46%
for cheap. Twenty seed999 prompts overlap seed123. On the 80 unseen prompts,
the ML router fell to the cheap baseline at 46.25% and missed all 13 oracle
expensive cases. ROC-AUC was 0.5545 and Average Precision was 0.2047.

A threshold of 0.40 simulated 53.75% pass rate and 38.19% estimated savings on
the unseen subset, but that threshold was selected on the same data and is not
a final generalization result.

### 8.1 Fixed-Split Prompt Router ML v2

To remove sample overlap and test-set threshold tuning, the next protocol used
all 427 sanitized MBPP tasks in one persisted split: 257 train, 85 validation,
and 85 held-out test tasks. Four candidates combined classifier or
learning-to-defer objectives with TF-IDF-only or TF-IDF plus 26 probing
features. The validation rule required at least 95% preservation of the
expensive-only pass rate and average score, then minimized expensive routes.

Validation selected the TF-IDF-only classifier at threshold `0.0761947`. The
frozen policy was evaluated once on test:

| Policy | Pass rate | Avg score | Expensive routes | PR-AUC | Expensive recall |
|---|---:|---:|---:|---:|---:|
| cheap_only | 41.18% | 0.4314 | 0/85 | - | - |
| expensive_only | 49.41% | 0.5098 | 85/85 | - | - |
| prompt_router_ml_v2 | **50.59%** | **0.5294** | 50/85 | 0.4486 | 72.22% |
| oracle | 58.82% | 0.6118 | task-dependent | - | 100.00% |

V2 passed 43 tasks, versus 42 for `expensive_only`, while choosing cheap for
35 prompts. The route mix gives 26.76% theoretical savings; the artifact's
token-count cost proxy improved by only 3.26%. Expensive precision was 26.00%.
Probing was not selected on validation, so its current form did not improve the
quality/cost trade-off.

### 8.2 Frozen-Policy Latency Follow-up

The frozen no-probing policy was subsequently timed on the same 85 tasks with
one warmup and three measured runs in the split-GPU Kaggle setup. The detailed
CSVs were lost; these aggregates were recovered from terminal output:

| Mode | Pass rate | Avg real speedup | Avg calls | Avg time | Std time |
|---|---:|---:|---:|---:|---:|
| expensive_only | 49.41% | 0.00% | 150.96 | 15.438s | 10.324s |
| prompt_router_ml_v2 | **50.59%** | **21.47%** | 76.52 | **9.709s** | 7.657s |

The result is preliminary evidence that the prompt-level policy can avoid the
double-forward overhead of token-level routing and produce a real latency gain.
The 21.47% figure is the mean of per-task speedups, not the ratio of aggregate
mean times. High variance and missing task-level artifacts require a clean
replication before stronger claims.

The current evidence supports model complementarity and the computational
structure of prompt-level routing. The fixed-split result is preliminary
held-out evidence, but one 85-task test is not enough to establish broad
generalization. Canonical artifacts are stored under `results/router_dataset_v2/`
and `results/router_v2/frozen_validation_policy/`.

## 9. Key Findings

- Estimated savings and real latency are not the same.
- CPU and GPU behave differently.
- Small model gaps may not compensate for routing overhead.
- Larger model gaps can produce large real wall-clock speedups.
- Prompt formatting matters for Instruct models.
- Hybrid routing overhead is negligible; the selected generation mode dominates runtime.
- Speculative decoding is promising but not robust globally in the current implementation.
- Prompt-level routing avoids the cheap-plus-expensive forward cost within one generation.
- The fixed-split TF-IDF router slightly exceeded `expensive_only` on one held-out MBPP test, but its expensive-route precision was low.
- Quality evaluation remains the main unresolved issue.

## 10. Limitations

The known limitations are substantial:

- **Switching cost:** loading and using two models can add latency, memory pressure, and scheduling overhead.
- **Shared GPU memory:** the strongest Qwen2.5-0.5B -> 3B result likely involved Windows shared GPU memory because the peak exceeded dedicated VRAM.
- **Small and overlapping datasets:** the seed123/seed999 MBPP samples overlap by 20 prompts.
- **Weak quality metrics:** difflib similarity, Jaccard similarity, and repeated n-gram rates are weak proxies for semantic correctness.
- **Limited task-specific correctness:** code has a local test harness, but math is not symbolic and logic still relies on keyword rules.
- **No large-scale 7B -> 70B validation:** results have not been validated on production-scale model gaps.
- **No optimized KV-cache implementation:** the current prototype does not share or optimize KV-cache state between models.
- **Speculative implementation is not production optimized:** it is a first research version, not a high-performance speculative decoding engine.
- **Possible router overfitting:** the hybrid router uses manually chosen heuristics that may overfit the current prompts.
- **Weak learned-router representation:** v1 TF-IDF missed every true expensive case among 80 unseen MBPP prompts; fixed-split v2 improved recall to 72.22% but precision remained 26.00%.
- **Windows laptop hardware limitations:** results depend on the RTX 3050 Laptop GPU, Windows memory behavior, and local runtime overhead.

Quality preservation is not yet proven robustly; current similarity/Jaccard metrics are weak proxies for semantic correctness.

## 11. External Critique Response

An external critique identified five concerns that are important for interpreting the current results. The project response is below.

### 11.1 Routing overhead may exceed benefits

**Critique summary:** Routing between a cheap and expensive model can add Python control-flow overhead, model-switching overhead, memory pressure, and scheduling cost. The overhead can exceed the saved computation.

**Current status:** Confirmed and central.

**Evidence from current experiments:**

- The SmolLM2 small-gap GPU results showed `expensive_only` winning against routed non-cheap modes.
- The `Qwen2.5-0.5B -> Qwen2.5-3B` CUDA results showed routed modes winning when the expensive model was sufficiently costly relative to the cheap model and hardware.

**Planned mitigation:**

- Document model-gap dependency explicitly.
- Benchmark on more hardware and model pairs.
- Optimize runtime later, after validating the algorithmic trade-off.

### 11.2 Local token-level signals do not capture global reasoning difficulty

**Critique summary:** Entropy, margin, top-k agreement, and related confidence signals are local token-level measures. They can miss cases where the next token is easy to predict but the reasoning needed to reach the answer is hard.

**Current status:** Valid limitation.

**Evidence from current experiments:**

- Entropy and margin are local token-level signals in the adaptive generator.
- Task-specific evaluation was added to measure final correctness on math, logic, and code tasks.

**Planned mitigation:**

- Add prompt-level and task-level risk features.
- Extend the learned router beyond TF-IDF with cheap-model confidence, prompt
  structure, category, difficulty, pass/fail outcomes, and latency inputs.

### 11.3 Lexical quality metrics are weak

**Critique summary:** Similarity, Jaccard overlap, and repetition rates do not reliably measure semantic correctness.

**Current status:** Partially mitigated.

**Evidence from current experiments:**

- The quality-latency report still uses lexical similarity and Jaccard as lightweight proxies.
- Task evaluation now includes code tests, math expected-answer checks, and logic labels.

**Planned mitigation:**

- Add symbolic math checks.
- Improve logic labels and task design.
- Expand the dataset.
- Add LLM-judge or human review for open-ended text.

### 11.4 Manual heuristics and thresholds may not generalize

**Critique summary:** Hand-written thresholds and prompt rules may not transfer across domains, model pairs, languages, or hardware.

**Current status:** Confirmed and investigating.

**Evidence from current experiments:**

- The hybrid router still uses hand-written rules.
- `prompt_router_v2` matched the oracle on 30 observed MBPP prompts but reached
  only 48% on a different 100-task sample.
- The TF-IDF router missed all 13 oracle-expensive cases among 80 unseen
  seed999 prompts.

**Planned mitigation:**

- Use a fixed non-overlapping train/validation/test protocol.
- Add cheap-model confidence and structural features beyond prompt TF-IDF.
- Select class weight and routing threshold on validation, with explicit
  expensive-recall reporting.

### 11.5 Budget caps may trade away quality

**Critique summary:** Budget caps keep latency and cost controlled, but they may block optional expensive-model calls that would improve output quality.

**Current status:** Intentional trade-off, but too rigid.

**Evidence from current experiments:**

- The budget cap prevents optional expensive fallbacks after the configured call ratio is reached.
- This helps latency/cost control, but can limit quality, especially in structured outputs such as code.

**Planned mitigation:**

- Expose policy modes: `latency`, `balanced`, and `quality`.
- Make budget caps dynamic by category and difficulty.
- Tune budget behavior using task-quality-latency data.

The current value proposition should be stated as:

> "On a constrained GPU setup with Qwen2.5-0.5B -> Qwen2.5-3B, GEAR-LLM showed preliminary evidence that routed modes can preserve most task-level correctness while reducing average wall-clock latency. This does not imply universal usefulness; the method is useful only under specific model-gap, hardware, and quality-tolerance conditions."

## 12. Next Steps

The fixed split protocol has been completed. Train/validation/test contained
257/85/85 non-overlapping tasks. Validation-only selection froze a TF-IDF-only
classifier at threshold `0.0761947`; probing was therefore not used by the
final policy.

On the one-shot held-out test, v2 reached 50.59% pass rate and 0.5294 average
score, compared with 49.41% and 0.5098 for `expensive_only`. It chose the cheap
model for 35/85 prompts, reached 72.22% recall and 26.00% precision for the
expensive-needed class, with PR-AUC 0.4486 and ROC-AUC 0.6517. The route mix
implies 26.76% theoretical savings, while the token-count cost proxy improved
only 3.26%. The later recovered Kaggle timing reported 21.47% average per-task
real speedup, but its detailed CSV artifacts were not preserved.

The next phase should broaden quality and systems validation:

- Repeat the frozen-policy latency benchmark and preserve all task-level artifacts.
- Validate it on a new external code dataset without retuning on the consumed test split.
- Revisit probing features only through a new validation protocol.
- Report expensive recall, PR-AUC, task score, route percentage, and real latency together.
- Improve task-specific evaluation: symbolic math, stronger logic labels, and
  human or LLM review for open text.
- Expand datasets across categories and languages.
- Test Qwen2.5-1.5B -> Qwen2.5-3B if hardware permits.
- Test on hardware with enough dedicated VRAM to avoid shared-memory effects.
- Compare against stronger baselines and optimized speculative decoding implementations.

## 13. Reproducibility

Latency benchmark:

```powershell
.\.venv-cuda\Scripts\python.exe run_latency_benchmark.py `
  --cheap-model Qwen/Qwen2.5-0.5B-Instruct `
  --expensive-model Qwen/Qwen2.5-3B-Instruct `
  --device cuda `
  --torch-dtype float16 `
  --max-new-tokens 64 `
  --warmup-runs 1 `
  --measured-runs 3 `
  --prompt-format auto
```

Quality benchmark:

```powershell
.\.venv-cuda\Scripts\python.exe run_quality_benchmark.py `
  --cheap-model Qwen/Qwen2.5-0.5B-Instruct `
  --expensive-model Qwen/Qwen2.5-3B-Instruct `
  --device cuda `
  --torch-dtype float16 `
  --max-new-tokens 64 `
  --prompt-format auto
```

Quality-latency report:

```powershell
.\.venv-cuda\Scripts\python.exe run_quality_latency_report.py
```

Evaluate the archived prompt router model:

```powershell
.\.venv-cuda\Scripts\python.exe run_task_evaluation.py `
  --dataset data/external_eval_tasks_90.jsonl `
  --categories code `
  --modes prompt_router_ml_v1 `
  --prompt-router-model results/kaggle/prompt_router_ml_v1/seed123_train/model.joblib
```

Build and train the fixed-split router v2:

```powershell
.\.venv-cuda\Scripts\python.exe scripts/build_router_dataset_v2.py `
  --cheap-model Qwen/Qwen2.5-Coder-0.5B-Instruct `
  --expensive-model Qwen/Qwen2.5-Coder-3B-Instruct `
  --device cuda --torch-dtype float16 `
  --max-new-tokens 256

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss classifier `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/classifier_probing

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss classifier --no-probing `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/classifier_tfidf

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss l2d `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/l2d_probing

.\.venv-cuda\Scripts\python.exe scripts/train_prompt_router_v2.py `
  --loss l2d --no-probing `
  --train-csv results/router_dataset_v2/train_features.csv `
  --val-csv results/router_dataset_v2/val_features.csv `
  --output-dir results/router_v2/l2d_tfidf

.\.venv-cuda\Scripts\python.exe scripts/select_router_v2_policy.py `
  --validation-csv results/router_dataset_v2/val_features.csv `
  --candidates-root results/router_v2 `
  --output-dir results/router_v2/frozen_validation_policy
```

Earlier Kaggle CSVs and provenance notes are stored under `results/kaggle/`.
The completed fixed-split artifacts are under `results/router_dataset_v2/` and
`results/router_v2/frozen_validation_policy/`.

## 14. Conclusion

GEAR-LLM demonstrated a real latency win in a specific, reproducible setting.
The strongest latency evidence is Qwen2.5-0.5B -> Qwen2.5-3B on CUDA with chat
formatting. Prompt-level MBPP experiments also show real complementarity
between Qwen2.5-Coder-0.5B and 3B.

This does not close the thesis. Manual rules and the first TF-IDF router failed
to generalize reliably; fixed-split v2 produced a small held-out improvement,
but still needs external and artifact-preserving latency replication. The project
should be treated as preliminary experimental research into cheap/expensive
model routing, not as a production-ready inference engine.
