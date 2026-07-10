# Kaggle Experiment Archive

This directory contains the canonical Kaggle artifacts for the prompt-level
routing experiments. Files under the top-level `results/` directory are mutable
outputs from local commands; the files here are archived by experiment and
should not be overwritten by smoke tests.

## Layout

### `prompt_router_v2/seed42_30_latency`

- 30 MBPP tasks sampled with the original analysis seed.
- Qwen2.5-Coder-0.5B-Instruct -> Qwen2.5-Coder-3B-Instruct.
- Split GPU: cheap on `cuda:0`, expensive on `cuda:1`.
- `float16`, chat template through `prompt_format=auto`.
- `max_new_tokens=256`, warmup 1, measured runs 3.
- Canonical files are the `task_quality_latency_*.csv` files.

The original archive also contained `task_evaluation*.csv` files copied from
the 100-task run. Those contaminated duplicates were intentionally removed.

### `prompt_router_v2/seed123_100_correctness`

- 100 MBPP tasks sampled with seed 123.
- Correctness-only comparison of `expensive_only`, `cheap_only`, and
  `prompt_router_v2`.
- Same model pair and split-GPU configuration as above.

### `prompt_router_ml_v1/seed123_train`

- Router dataset built from the seed123 cheap/expensive results.
- TF-IDF plus Logistic Regression trained with `oracle_score_label`.
- The serialized model was created with scikit-learn 1.6.1.
- `training_metrics.csv` reports the internal 70/30 split; `model.joblib` was
  fitted again on all 100 tasks after that evaluation.

### `prompt_router_ml_v1/seed999_test`

- 100-task correctness-only evaluation using a second MBPP sample.
- `oracle_dataset.csv` contains cheap/expensive scores and oracle labels.
- Twenty prompts overlap with seed123. Generalization claims must use the 80
  prompts not present in the seed123 router dataset.

## Important Interpretation

The manual v2 router matched the oracle on the original 30-task sample but did
not generalize to seed123. The ML router reproduced the seed123 oracle
in-sample, but on the 80 unseen seed999 tasks its default threshold identified
none of the 13 `expensive_only` oracle cases.

These results support prompt-level model complementarity, not a solved routing
policy. The next benchmark should use one fixed, non-overlapping train,
validation, and test split over the full MBPP source.
