# Known Problems and Limitations

## Status Legend

- **Open**: ainda nao resolvido.
- **Investigating**: em analise ou com experimento em andamento.
- **Partially mitigated**: mitigado parcialmente, mas nao resolvido.
- **Mitigated**: resolvido o suficiente para o MVP atual.
- **Deferred**: adiado para fase futura.

## 1. Switching Cost Between Models

**Status:** Open

### Problem

The current cost model assumes that using the cheap model plus occasional expensive model calls is cheaper than using the expensive model alone. In real systems, switching between models may introduce extra latency, memory pressure, and scheduling overhead.

### Impact

Theoretical savings may not translate into real wall-clock speedup.

### Current Mitigation

The project estimates cost using `cheap_cost` and `expensive_cost`, but does not yet measure real latency or memory.

### Next Steps

- Implement `latency_benchmark.py`.
- Measure total generation time, tokens per second, and expensive model calls.
- Measure memory usage when both models are loaded.
- Compare theoretical savings with real execution time.

## 2. Small Gap Between Cheap and Expensive Models

**Status:** Open

### Problem

The current MVP uses `HuggingFaceTB/SmolLM2-135M-Instruct` as the cheap model and `HuggingFaceTB/SmolLM2-360M-Instruct` as the expensive model. This is useful for local experimentation, but the model size gap is small compared to realistic production scenarios.

### Impact

Results may not generalize to larger pairs such as `0.5B -> 7B`, `1.5B -> 14B`, or `3B -> 32B`.

### Current Mitigation

The documentation states that the project is experimental.

### Next Steps

- Test larger model pairs when hardware allows.
- Keep the benchmark model-agnostic.
- Report model sizes clearly in all result files.

## 3. Manual Thresholds and Hard-Coded Routing

**Status:** Partially mitigated

### Problem

The project uses manually selected thresholds and heuristics, such as `entropy_threshold`, `margin_threshold`, `teacher_check_interval`, `verify_top_k`, and hybrid routing rules.

### Impact

Static rules may not adapt well across domains such as code, math, logic, summaries, and creative writing.

### Current Mitigation

The project includes teacher calibration, policy replay, guard tuning, speculative tuning, and mode oracle benchmark.

### Next Steps

- Build a learned router using the oracle results as labels.
- Start with simple interpretable models such as logistic regression or decision trees.
- Use prompt features and cheap-model metrics as router inputs.

## 4. Text Coherence When Mixing Models

**Status:** Partially mitigated

### Problem

Dual-model token-by-token generation may mix stylistic and semantic tendencies from different models, potentially causing inconsistent or incoherent text.

### Impact

The generated answer may become unstable, repetitive, or stylistically inconsistent.

### Current Mitigation

Adaptive Guard v3 reduces excessive optional fallbacks and adds budget/cooldown logic. Speculative decoding verifies cheap-model drafts with the expensive model.

### Next Steps

- Prefer speculative decoding over token-by-token model switching.
- Add quality checks for repetition and semantic drift.
- Compare generated text against `expensive_only` more robustly.
- Consider periodic full-context verification.

## 5. Local Confidence Is Not the Same as Reasoning Difficulty

**Status:** Open

### Problem

Entropy, top-1 probability, margin, and top-k agreement are local token-level signals. A token can be easy to predict even when the reasoning needed to reach it is difficult.

### Impact

The router may save compute in places where deeper reasoning was needed, especially in math, code, and logic.

### Current Mitigation

The project includes `rho` metrics such as surprisal, novelty, curvature, and structural importance in offline analysis.

### Next Steps

- Reintroduce geometric and structural features into online routing.
- Track hidden-state curvature and novelty from the cheap model.
- Add prompt-level risk features for math, code, and logic.
- Validate routing decisions with task-specific correctness checks.

## 6. Weak Quality Metrics

**Status:** Open

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

The current goal is experimentation, not production serving.

### Next Steps

- Measure latency first.
- Identify bottlenecks.
- Consider batching, `torch.compile`, vLLM, llama.cpp, or custom kernels only after the algorithmic direction is validated.

## 9. Dataset Size and Diversity

**Status:** Partially mitigated

### Problem

The project now has a 60-prompt JSONL dataset, but this is still small and manually created.

### Impact

Results may overfit to the current prompts and categories.

### Current Mitigation

Dataset benchmark and mode oracle now exist.

### Next Steps

- Expand to 100+ prompts per category.
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

Hybrid routing no longer selects `speculative_adaptive` automatically unless the router explicitly chooses it based on policy.

### Next Steps

- Continue speculative tuning.
- Improve acceptance rules.
- Add prompt-level and block-level risk features.
- Compare against standard speculative decoding baselines.

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

Hybrid router selects `adaptive_calibrated` for code, but the mode still applies the 40% budget cap.

### Next Steps

- Set `max_expensive_call_ratio=0.70` or higher for code prompts
- Or remove budget cap entirely for `adaptive_calibrated` (hybrid router already handles mode selection)
- Add dynamic budget based on prompt type detection

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

## Current Priority Order

1. Validate the latest hybrid router changes.
2. Implement a real latency benchmark.
3. Compare theoretical savings against real wall-clock speed.
4. Add learned routing from oracle data.
5. Improve quality evaluation beyond text similarity.
6. Revisit speculative decoding with better acceptance criteria.
7. Investigate KV cache and production-serving concerns.
8. Fix budget cap limiting code generation quality.

## Current Project Status

GEAR-LLM is an experimental research prototype. It currently validates routing ideas and cost/quality trade-offs, but it is not yet a production-ready inference engine.
