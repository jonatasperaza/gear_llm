# Known Problems and Limitations

## Status Legend

- **Open**: ainda nao resolvido.
- **Investigating**: em analise ou com experimento em andamento.
- **Partially mitigated**: mitigado parcialmente, mas nao resolvido.
- **Mitigated**: resolvido o suficiente para o MVP atual.
- **Deferred**: adiado para fase futura.

## 1. Switching Cost Between Models

**Status:** Partially mitigated

### Problem

The current cost model assumes that using the cheap model plus occasional expensive model calls is cheaper than using the expensive model alone. In real systems, switching between models may introduce extra latency, memory pressure, and scheduling overhead.

### Impact

Theoretical savings may not translate into real wall-clock speedup.

### Current Mitigation

The project now includes `latency_benchmark.py`, which measures wall-clock time, tokens per second, expensive model calls, and CUDA peak memory when available.

Current observations:

- On CPU, `adaptive_calibrated` and `adaptive_guarded_v3` can produce real speedups over `expensive_only`.
- On GPU with `HuggingFaceTB/SmolLM2-135M-Instruct -> HuggingFaceTB/SmolLM2-360M-Instruct`, `expensive_only` was the fastest non-cheap mode for `code`, `easy`, `logic_negation`, `long_simple`, and `math`.
- Hybrid routing has very small decision overhead, but the selected generation mode can still be slower than `expensive_only`.

### Next Steps

- Keep comparing theoretical savings with real execution time.
- Add latency reports for larger model gaps and different hardware.
- Measure memory pressure when both models are loaded.
- Separate routing overhead, model-switching overhead, and generation time in future reports.

## 2. Small Gap Between Cheap and Expensive Models

**Status:** Open

### Problem

The current MVP uses `HuggingFaceTB/SmolLM2-135M-Instruct` as the cheap model and `HuggingFaceTB/SmolLM2-360M-Instruct` as the expensive model. This is useful for local experimentation, but the model size gap is small compared to realistic production scenarios.

### Impact

Results may not generalize to larger pairs such as `0.5B -> 7B`, `1.5B -> 14B`, or `3B -> 32B`.

### Current Mitigation

The documentation states that the project is experimental. The GPU latency benchmark confirms that this small gap is not enough to make adaptive routing faster than `expensive_only` on an NVIDIA GeForce RTX 3050 6GB Laptop GPU.

### Next Steps

- Test larger model pairs when hardware allows, such as `Qwen2.5-0.5B -> Qwen2.5-1.5B` or `Qwen2.5-1.5B -> Qwen2.5-3B`.
- Add configurable model-pair support to all benchmark entry points.
- Keep the benchmark model-agnostic.
- Report model sizes clearly in all result files.

## 3. Manual Thresholds and Hard-Coded Routing

**Status:** Investigating

### Problem

The project uses manually selected thresholds and heuristics, such as
`entropy_threshold`, `margin_threshold`, `teacher_check_interval`,
`verify_top_k`, hybrid rules, and the keyword patterns in
`prompt_router_v1/v2`.

### Impact

Static rules may not adapt well across domains such as code, math, logic, summaries, and creative writing.

### Current Mitigation

The project includes teacher calibration, policy replay, guard tuning,
speculative tuning, mode oracle benchmark, and `prompt_router_ml_v1` using
TF-IDF plus Logistic Regression. The learned router reproduced its seed123
oracle in-sample but failed to identify any true expensive route among 80
unseen seed999 prompts.

`prompt_router_ml_v2` now provides a fixed train/validation/test protocol,
probing features and validation-only selection for classifier or
learning-to-defer policies. Its full 427-task dataset has not been generated,
so generalization is not yet demonstrated.

### Next Steps

- Generate the complete fixed-split feature dataset on Kaggle.
- Freeze the selected validation policy before touching test.
- Compare classifier and L2D by expensive recall, PR-AUC, downstream score and real latency.

## 4. Text Coherence When Mixing Models

**Status:** Partially mitigated

### Problem

Dual-model token-by-token generation may mix stylistic and semantic tendencies from different models, potentially causing inconsistent or incoherent text.

### Impact

The generated answer may become unstable, repetitive, or stylistically inconsistent.

### Current Mitigation

Adaptive Guard v3 reduces excessive optional fallbacks and adds budget/cooldown logic. Speculative decoding verifies cheap-model drafts with the expensive model.

### Next Steps

- Evaluate speculative decoding as a way to reduce token-by-token model switching, but only when latency and quality benchmarks justify it.
- Add quality checks for repetition and semantic drift.
- Compare generated text against `expensive_only` more robustly.
- Consider periodic full-context verification.

## 5. Local Confidence Is Not the Same as Reasoning Difficulty

**Status:** Partially mitigated

### Problem

Entropy, top-1 probability, margin, and top-k agreement are local token-level signals. A token can be easy to predict even when the reasoning needed to reach it is difficult.

### Impact

The router may save compute in places where deeper reasoning was needed, especially in math, code, and logic.

### Current Mitigation

The project includes `rho` metrics and `prompt_router_ml_v2` now aggregates
cheap-model entropy, margin, surprisal, novelty, curvature, structural density
and prompt shape. Optional cheap-vs-expensive agreement features add stronger
signals but require an expensive prompt prefill.

### Next Steps

- Measure whether the new features improve held-out PR-AUC and expensive recall.
- Ablate expensive-agreement features against cheap-only probing to quantify their latency cost.
- Add task-level risk features for math and logic after the MBPP protocol is frozen.

## 6. Weak Quality Metrics

**Status:** Partially mitigated

### Problem

The project currently uses `difflib` similarity, word-level Jaccard similarity, and repeated n-gram rates. These are useful for MVP experiments but do not reliably measure correctness.

### Impact

A response can be correct but textually different from `expensive_only`, or textually similar but factually wrong.

### Current Mitigation

The project uses multiple simple metrics instead of relying on only one.

### Next Steps

- Add task-specific evaluation:
  - code: execute tests when possible;
  - math: compare against expected answers or symbolic checks;
  - logic: use labeled expected outcomes;
  - general text: use LLM-as-judge or human review.
- Add datasets with expected answers.

## 7. KV Cache and Memory Management Are Not Optimized

**Status:** Open

### Problem

The current implementation does not optimize KV cache sharing, reuse, or memory layout between cheap and expensive models.

### Impact

Real inference performance may be much worse than theoretical cost estimates.

### Current Mitigation

None yet. The current implementation intentionally avoids KV cache optimization to keep the MVP simple.

### Next Steps

- Measure memory usage first.
- Add KV cache-aware benchmarking.
- Investigate whether speculative verification can reuse cached states.
- Consider integration with optimized inference engines later.

## 8. Python Overhead and Lack of Optimized Kernels

**Status:** Open

### Problem

The current implementation runs as a Python research prototype. It is not optimized for production inference.

### Impact

Python loop overhead may dominate for small models and short generations.

### Current Mitigation

The current goal is experimentation, not production serving. The latency benchmark now exposes when adaptive Python-level control flow is slower than direct `expensive_only` generation, especially on GPU with small models.

### Next Steps

- Identify bottlenecks.
- Compare CPU and GPU bottlenecks separately.
- Consider batching, `torch.compile`, vLLM, llama.cpp, or custom kernels only after the algorithmic direction is validated.

## 9. Dataset Size and Diversity

**Status:** Partially mitigated

### Problem

The project has a 60-prompt general dataset, 45 task-specific manual tasks, a
90-task external set, and 100-task MBPP samples. These are still small relative
to the problem, and independently sampled MBPP subsets can overlap.

### Impact

Results may overfit to the current prompts and categories.

### Current Mitigation

Dataset benchmark, mode oracle, external task evaluation, Wilson confidence
intervals, and MBPP oracle datasets now exist. A seed123/seed999 comparison
revealed 20 repeated prompts, so the nominal 100-task test contained only 80
genuinely unseen tasks.

### Next Steps

- Create one persisted split manifest over all 427 MBPP tasks.
- Use approximately 257 train, 85 validation, and 85 untouched test tasks.
- Add expected answers where possible.
- Add adversarial prompts.
- Add prompts in English and Portuguese.
- Track benchmark results over time.

## 10. Speculative Decoding Is Not Yet Robust Globally

**Status:** Investigating

### Problem

Speculative decoding performs well in some cases but poorly in others, especially when local top-k acceptance does not preserve global answer quality.

### Impact

The mode should not be used as the default global strategy yet.

### Current Mitigation

Hybrid routing no longer selects `speculative_adaptive` automatically unless the router explicitly chooses it based on policy. The GPU latency benchmark shows that `speculative_adaptive` is not competitive with `expensive_only` for the current SmolLM2 135M -> 360M pair.

### Next Steps

- Continue speculative tuning.
- Improve acceptance rules.
- Add prompt-level and block-level risk features.
- Compare against standard speculative decoding baselines.
- Re-test speculative decoding with larger model gaps and optimized verification.

## 11. Budget Cap Limits Code Quality

**Status:** Open

### Problem

The `adaptive_calibrated` mode uses a default `max_expensive_call_ratio=0.40` (40%). For code-heavy prompts, this cap is too restrictive because code tokens are structurally important (operators, delimiters, logic words) and often require the expensive model.

### Example

A 32-token code generation may need 20-25 expensive tokens (operators, syntax, logic), but the 40% cap limits expensive calls to only 13 tokens. After reaching the cap, the system is forced to use the cheap model for tokens that genuinely need the expensive model, resulting in broken code.

### Impact

- Broken syntax and logic in generated code
- Cheap model "invents" operators or words when forced to fill expensive slots
- Code quality degrades significantly for longer generations

### Current Mitigation

Hybrid router now selects `adaptive_code_quality` for code. This improves task
pass rate in some MBPP runs, but its stricter fallbacks can remove most of the
latency advantage.

### Next Steps

- Compare the conservative code profile against prompt-level routing before
  increasing its budget further.
- Add dynamic budget based on prompt type and measured task difficulty.
- Prefer a single-model prompt route when confidence is sufficient, avoiding
  the cheap-plus-expensive token-level cost entirely.

## 12. Code Detection Is Python-Centric and Language-Blind

**Status:** Open

### Problem

The `classify_prompt` function uses hardcoded `code_words` that are mostly Python/JavaScript-centric. Languages like Java, C#, C++, Rust, Go, SQL, and HTML/CSS are not detected as code.

### Example

```
"Faça uma classe em Java com método público que retorna String"
```

No `code_words` match → classified as `general` instead of `code` → wrong mode selected.

### Impact

- Prompts for Java, C#, Rust, Go, SQL, HTML fall through to `general` mode
- Wrong mode means wrong routing strategy and potentially worse quality
- Users writing code in non-Python languages get suboptimal treatment

### Current Mitigation

None. The current word list covers Python and JavaScript well.

### Next Steps

- Add universal code keywords (`public`, `private`, `void`, `struct`, `func`, `fn`)
- Add language-specific keywords (Java, C#, Rust, Go, SQL)
- Detect code structure (braces `{}`, semicolons `;`, indentation patterns)
- Consider regex patterns for common code structures
- Add Portuguese equivalents (`classe`, `método`, `função`)

## 13. Prompt Router Generalization and Split Leakage

**Status:** Open

### Problem

Manual prompt-router rules matched the oracle on the original 30-task MBPP
sample but selected expensive for only one of 100 tasks under seed123. The
TF-IDF learned router then reproduced the seed123 oracle in-sample, but the
seed999 sample overlapped seed123 by 20 prompts.

On the 80 unseen prompts, the ML router missed all 13 oracle-expensive cases.
ROC-AUC was 0.5545 and Average Precision was 0.2047, showing weak separation
from prompt text alone.

### Impact

- In-sample router quality substantially overstates generalization.
- Random seeds do not guarantee independent datasets.
- Accuracy is dominated by the majority `cheap_only` class.
- Threshold tuning on the reported test subset creates another optimistic bias.

### Current Mitigation

Kaggle artifacts are archived by seed under `results/kaggle/`, and reported
unseen metrics explicitly exclude the 20 overlapping prompts. The model remains
experimental and is not presented as a validated routing policy.

A persistent 427-task manifest and non-overlapping 257/85/85 JSONLs now exist.
The builder rejects split/manifest mismatches, training selects settings on
validation, and the final evaluator is separated from training. Only local
smoke data has been generated so far.

### Next Steps

- Generate all 427 cheap/expensive outcomes and probing features.
- Freeze the policy selected on validation before evaluating final test once.
- Report expensive recall, PR-AUC, task score, route percentage, and latency.
- Evaluate richer features than TF-IDF, including cheap-model uncertainty and
  prompt structure.

## External Critique Response

This section records five concerns raised by an external critique and the current project response.

### 1. Routing Overhead May Exceed Benefits

**Critique summary:** Cheap/expensive routing can add Python control-flow overhead, model-switching overhead, memory pressure, and scheduling cost. In some settings, this overhead may be larger than the saved model compute.

**Current status:** Confirmed and central.

**Evidence from current experiments:**

- The SmolLM2 small-gap GPU benchmark showed `expensive_only` winning against routed non-cheap modes.
- The `Qwen2.5-0.5B -> Qwen2.5-3B` CUDA benchmark showed routed modes winning when the expensive model was sufficiently costly relative to the cheap model and hardware.

**Planned mitigation:**

- Document model-gap dependency clearly in results and reports.
- Benchmark on more hardware and model pairs.
- Optimize runtime later, after the algorithmic trade-off is better validated.

### 2. Local Token Signals Do Not Capture Global Reasoning Difficulty

**Critique summary:** Entropy, margin, top-k agreement, and related token-level confidence signals are local. They can miss cases where a token is easy to predict but the underlying reasoning is hard.

**Current status:** Valid limitation.

**Evidence from current experiments:**

- The online adaptive generator currently relies heavily on local cheap-model entropy and margin.
- Task-specific evaluation was added to measure final correctness instead of relying only on local agreement or lexical similarity.

**Planned mitigation:**

- Add prompt-level and task-level risk features.
- Train a learned router from oracle data and task-evaluation outcomes.
- Use prompt features, cheap-model confidence, task category, difficulty, and final pass/fail signals as router inputs.

### 3. Lexical Quality Metrics Are Weak

**Critique summary:** Similarity, Jaccard overlap, and repeated n-gram rates are weak proxies for semantic correctness. A generated answer can be correct but textually different, or similar but wrong.

**Current status:** Partially mitigated.

**Evidence from current experiments:**

- The quality-latency report still uses lexical similarity and Jaccard as lightweight proxies.
- Task evaluation now includes code tests, math expected-answer checks, and logic labels.

**Planned mitigation:**

- Add symbolic math checks.
- Use stronger logic labels and more carefully designed logical tasks.
- Expand the task dataset.
- Add LLM-judge or human review for open-ended text.

### 4. Manual Heuristics and Thresholds May Not Generalize

**Critique summary:** Hand-written thresholds and routing rules may work on the current prompts but fail across new domains, model families, languages, or hardware.

**Current status:** Confirmed and investigating.

**Evidence from current experiments:**

- The hybrid router still uses hand-written prompt rules.
- `prompt_router_v2` matched the oracle on 30 observed MBPP tasks but reached
  only 48% on a different 100-task sample.
- `prompt_router_ml_v1` reproduced its training oracle but missed every true
  expensive case among 80 unseen prompts.
- On the fixed 427-task protocol, the frozen `prompt_router_ml_v2` improved
  held-out pass rate from 49.41% (`expensive_only`) to 50.59%, but expensive
  precision remained only 26.00% and the probing variants were rejected by
  validation-only policy selection.

**Planned mitigation:**

- Keep interpretable baselines, but evaluate them on a fixed non-overlapping
  train/validation/test protocol.
- Add cheap-model confidence, prompt structure, oracle labels, task outcomes,
  and latency features.
- Optimize threshold and class weight on validation, emphasizing
  expensive-route recall.
- Measure real latency of the frozen no-probing policy and validate the route
  rule on a new external code dataset before further tuning.

### 5. Budget Caps May Trade Away Quality

**Critique summary:** Budget caps protect latency and cost, but they can block optional calls to the expensive model that would have improved quality.

**Current status:** Intentional trade-off, but too rigid.

**Evidence from current experiments:**

- The budget cap prevents optional expensive fallbacks after the configured call ratio is reached.
- This is useful for latency/cost control, but it can limit quality, especially in structured outputs such as code.

**Planned mitigation:**

- Expose policy modes:
  - `latency`
  - `balanced`
  - `quality`
- Make the budget dynamic by category and difficulty.
- Revisit caps with task-quality-latency data rather than using one fixed policy everywhere.

The current value proposition should be stated as:

> "On a constrained GPU setup with Qwen2.5-0.5B -> Qwen2.5-3B, GEAR-LLM showed preliminary evidence that routed modes can preserve most task-level correctness while reducing average wall-clock latency. This does not imply universal usefulness; the method is useful only under specific model-gap, hardware, and quality-tolerance conditions."

## Current Priority Order

1. Generate the full 427-task router-v2 dataset on Kaggle using the fixed manifest.
2. Train classifier and L2D variants on train and select one policy on validation.
3. Evaluate that frozen policy once on the untouched test split.
4. Measure task correctness and real latency, including both probing prefills.
5. Ablate cheap-only probing against cheap-plus-expensive agreement features.
6. Improve math, logic, and open-text quality evaluation.
7. Revisit token-level/speculative decoding with optimized KV-cache-aware
   baselines.
8. Test additional model gaps and dedicated-VRAM hardware.

## Current Project Status

GEAR-LLM is an experimental research prototype. It currently validates routing ideas and cost/quality trade-offs, but it is not yet a production-ready inference engine.
