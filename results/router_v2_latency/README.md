# Prompt Router ML v2 Latency Run

These aggregate results were recovered from the printed Kaggle terminal output.
The detailed CSV files produced by the benchmark were not preserved, so this
directory does not contain task-level timings and cannot support a per-task
audit.

## Configuration

- Dataset: `data/mbpp_test_85.jsonl`
- Tasks: 85 held-out MBPP code tasks
- Cheap model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Expensive model: `Qwen/Qwen2.5-Coder-3B-Instruct`
- Devices: cheap on `cuda:0`, expensive on `cuda:1`
- Dtype: `float16`
- Prompt format: `auto`
- Max new tokens: 256
- Warmup runs: 1
- Measured runs: 3
- Modes: `expensive_only`, `prompt_router_ml_v2`

## Recovered Result

| Mode | Pass rate | Avg score | Avg speedup | Avg saved | Avg calls | Avg time | Std time |
|---|---:|---:|---:|---:|---:|---:|---:|
| expensive_only | 49.41% | 0.510 | 0.00% | 0.00% | 150.96 | 15.438s | 10.324s |
| prompt_router_ml_v2 | 50.59% | 0.529 | 21.47% | 26.76% | 76.52 | 9.709s | 7.657s |

`avg_real_speedup_vs_expensive_percent` is the benchmark's mean of per-task
speedups. It is not the ratio of the two aggregate mean times, which weights
tasks differently.

The run confirms the same 43/85 versus 42/85 correctness result reported by the
one-shot fixed-split evaluation. It also provides preliminary evidence of a real
latency reduction for the frozen no-probing policy. Because the detailed CSVs
were lost and timing variance is high, this result should be replicated with
the raw artifacts preserved.
