# Prompt Router v2 — Test Report

**One-shot evaluation on the held-out test split.**
Per PROBLEMS.md #13, this script must be run at most once with the frozen policy; re-running with different knobs invalidates the generalization claim.

- test csv: `results\router_dataset_v2\test_features.csv`
- test rows: 85
- policy mode: `classifier`
- threshold: `0.07619472357836586`
- probing features: `False`
- v1 is a historical TF-IDF reference; unless retrained on the fixed train split, it is not a protocol-matched baseline.

## Comparison table

| Policy | Pass rate | Avg score | Expensive routes | Cost | PR-AUC | ROC-AUC | Exp recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| cheap_only | 0.4118 | 0.4314 | — | 2529.75 | — | — | — |
| expensive_only | 0.4941 | 0.5098 | — | 2235.00 | — | — | — |
| prompt_router_ml_v1 | 0.4588 | 0.4824 | 8/85 | 2334.95 | 0.5188 | 0.7056 | 0.2778 |
| prompt_router_ml_v2 | 0.5059 | 0.5294 | 50/85 | 2162.25 | 0.4486 | 0.6517 | 0.7222 |

## Interpretation

1. **Quality**: v2 passed 43/85 tasks, versus 42/85 for `expensive_only`; the one-task difference is preliminary.
2. **Cost**: v2 selected cheap for 35 prompts. The route mix implies 26.76% theoretical savings, while the token-count cost proxy improved 3.26%; neither is wall-clock speedup.
3. **Routing quality**: expensive recall was 72.22%, but precision was only 26.00%, with PR-AUC 0.4486 and ROC-AUC 0.6517.
4. **Probing**: validation selected the TF-IDF-only candidate, so the current probing representation did not improve the selected quality/cost trade-off.
