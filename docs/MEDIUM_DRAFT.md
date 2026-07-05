# I Built a Small Experimental LLM Router — and Found When It Actually Gets Faster

## The idea

What if not every token needed the large model?

That is the question behind GEAR-LLM, a small experimental project I built to study cheap/expensive routing for language model inference. Instead of always sending the whole generation through the more expensive model, the system tries to use a smaller model when the next step looks easy and call the larger model when the situation looks riskier.

The project started with token-level analysis: entropy, surprisal, geometric novelty, curvature, and structural importance. Over time, it grew into several online generation modes:

- **adaptive_calibrated:** use the cheap model when entropy and probability margin look safe.
- **adaptive_guarded_v3:** add repetition checks, uncertainty gates, cooldowns, and a budget cap.
- **speculative_adaptive:** let the cheap model draft a block of tokens, then ask the expensive model to verify it.
- **hybrid router:** choose a mode based on simple prompt heuristics.

The goal was not to build a production inference engine. The goal was to answer a narrower experimental question: can cheap/expensive routing produce real wall-clock speedups, not just theoretical savings?

This is not a claim that cheap/expensive routing is a new research area. Speculative decoding, model cascades, adaptive computation, and routing systems already exist. What I wanted to test here was narrower: in a small open-source prototype, on my own hardware, when does this idea actually produce real wall-clock speedup instead of just theoretical savings?

## The first surprise

The first important result was negative.

Estimated savings were not the same as real speedup.

On GPU with a small model gap, using HuggingFaceTB/SmolLM2-135M-Instruct as the cheap model and HuggingFaceTB/SmolLM2-360M-Instruct as the expensive model, expensive_only was still the fastest non-cheap mode in the saved benchmark.

That was useful. It showed that routing overhead matters. If the expensive model is still cheap enough, the cost of switching between models, running Python control flow, managing two models, and doing extra checks can erase the theoretical benefit.

In other words: routing is not free.

## The second surprise

The second surprise came when the model gap got larger.

Using:

- Cheap model: Qwen/Qwen2.5-0.5B-Instruct.
- Expensive model: Qwen/Qwen2.5-3B-Instruct.
- Device: CUDA.
- Dtype: float16.
- Prompt format: auto/chat.
- GPU: RTX 3050 6GB Laptop GPU.
- Max new tokens: 64.

the routed modes started to show large real wall-clock speedups against expensive_only.

| Prompt | Best non-cheap mode | Real speedup vs expensive_only |
|---|---:|---:|
| code | adaptive_calibrated | 84.74% |
| easy | adaptive_guarded_v3 | 56.41% |
| logic_negation | adaptive_guarded_v3 | 76.20% |
| long_simple | adaptive_guarded_v3 | 68.68% |
| math | adaptive_guarded_v3 | 83.11% |

This is the main result I would state carefully:

GEAR-LLM demonstrated large real wall-clock speedups in a specific GPU setting when the expensive model was sufficiently costly relative to the cheap model and hardware.

That does not mean the method is universally better. It means the routing idea became practically interesting once the expensive model was costly enough.

There is also an important hardware caveat. The RTX 3050 Laptop GPU has 6 GB of dedicated VRAM, while the benchmark showed memory peaks near 7 GB. On Windows, this likely involved shared GPU memory. So the result is valid for this constrained laptop setup, but it is not the same as a clean fully-dedicated-VRAM benchmark.

## What worked

Adaptive routing worked best in the strongest run. In the Qwen2.5-0.5B -> Qwen2.5-3B benchmark, the best non-cheap mode was usually adaptive_guarded_v3 or adaptive_calibrated.

Hybrid routing overhead was small in this benchmark. The router itself was not the expensive part; runtime was mostly dominated by the selected generation mode.
Prompt formatting mattered. For Qwen Instruct models, using the tokenizer chat template produced more coherent instruction-style outputs than raw prompting.

Model gap mattered a lot. The small SmolLM2 gap did not produce GPU speedups, the Qwen 0.5B -> 1.5B gap produced partial wins, and the Qwen 0.5B -> 3B gap produced the strongest latency result.

The code prompt was the strongest quality-latency case in the current report:

| Prompt | Latency winner | Speedup | Similarity | Jaccard | Rep3 | Quality-latency score |
|---|---:|---:|---:|---:|---:|---:|
| code | adaptive_calibrated | 84.74% | 0.5517 | 0.5294 | 0.0000 | 0.8959 |

That is not proof of correctness, but it is the cleanest current example where speed and textual similarity both look encouraging.

## What did not work yet

Speculative decoding was unstable across settings. It was competitive in some earlier runs, especially when the expensive model became more costly, but it was not robust enough to use as the default global strategy.

Quality evaluation is still the weakest part of the project. The current benchmark uses difflib similarity, word-level Jaccard similarity, and repeated n-gram rates. These are useful for quick research feedback, but they are not semantic correctness.

This matters especially for logic, math, and long-form answers. A response can be textually different from expensive_only and still be correct. It can also be textually similar and still be wrong.

The quality-latency report shows the gap clearly:

| Prompt | Latency winner | Speedup | Similarity | Jaccard | Rep3 | Quality-latency score |
|---|---:|---:|---:|---:|---:|---:|
| code | adaptive_calibrated | 84.74% | 0.5517 | 0.5294 | 0.0000 | 0.8959 |
| easy | adaptive_guarded_v3 | 56.41% | 0.2440 | 0.3023 | 0.0000 | 0.4606 |
| logic_negation | adaptive_guarded_v3 | 76.20% | 0.0547 | 0.3556 | 0.0263 | 0.3209 |
| long_simple | adaptive_guarded_v3 | 68.68% | 0.0467 | 0.2034 | 0.0000 | 0.2692 |
| math | adaptive_guarded_v3 | 83.11% | 0.2011 | 0.2750 | 0.0000 | 0.4776 |

The speedups are strong. The quality evidence is not strong enough yet.

Shared memory may also influence the results. The best benchmark was run on a 6 GB laptop GPU, and memory peaks were near 7 GB. That makes the result practical and interesting, but also hardware-specific.

## The most honest conclusion

This does not prove that the method is universally better. It proves that under the right model gap and hardware constraints, cheap/expensive routing can produce large real wall-clock speedups.

For now, I would describe GEAR-LLM as a preliminary research prototype. It has shown that the latency side of the idea can work in at least one concrete GPU setting. It has not yet shown that quality is preserved robustly enough for production use.

## What comes next

The next step is better evaluation.

For code, generated functions should be executed against tests.

For math, answers should be checked against expected outputs or symbolic checks.

For logic, prompts need labeled expected outcomes.

For open-ended text, the project needs either human review or a carefully designed LLM-as-judge setup.

After that, the router should become less hand-written. The current hybrid router uses heuristics. A better version could learn from the mode oracle results and use prompt features, cheap-model confidence signals, latency estimates, and quality-risk features.

I also want to test more model pairs, especially Qwen2.5-1.5B -> Qwen2.5-3B and larger gaps on hardware with enough dedicated VRAM. That would help separate algorithmic effects from laptop memory behavior.

The larger research question is still open:

Can we build routers that know when a cheap model is enough, when a larger model is necessary, and when the extra latency is worth paying?

GEAR-LLM is an early attempt to make that question measurable.

For now, the most useful result is not “this router solves inference.” It is more specific: routing only became interesting when the expensive model was expensive enough for the hardware. That boundary is measurable, and that is what I want to keep exploring.

The benchmark scripts, CSV outputs, and result summaries are included in the repository so the numbers can be inspected instead of only quoted.

## Link to repo

[Repository link here](https://github.com/jonatasperaza/gear-llm.git)
