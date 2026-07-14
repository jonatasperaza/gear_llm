import json
import tempfile
import unittest
from pathlib import Path

import torch

from gear_llm.probing_features import (
    PROBING_FEATURE_KEYS,
    _last_position_agreement,
    _per_position_top1_match,
    _per_position_topk_kl,
    _per_position_topk_match,
)
from scripts.build_router_dataset_v2 import (
    derive_labels_and_costs,
    load_checkpoint_rows,
    validate_split_tasks,
)
from gear_llm.task_evaluation import resolve_task_modes
from scripts.train_prompt_router_v2 import train_classifier, train_l2d


class FixedSplitTests(unittest.TestCase):
    def test_manifest_has_announced_non_overlapping_split(self):
        path = Path("data/mbpp_split_manifest.jsonl")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_split = {
            split: {
                row["mbpp_task_id"]
                for row in rows
                if row["split"] == split
            }
            for split in ("train", "val", "test")
        }
        self.assertEqual(len(by_split["train"]), 257)
        self.assertEqual(len(by_split["val"]), 85)
        self.assertEqual(len(by_split["test"]), 85)
        self.assertFalse(by_split["train"] & by_split["val"])
        self.assertFalse(by_split["train"] & by_split["test"])
        self.assertFalse(by_split["val"] & by_split["test"])

    def test_manifest_mismatch_is_rejected(self):
        tasks = [{"id": "x", "mbpp_task_id": 10}]
        with self.assertRaises(ValueError):
            validate_split_tasks("train", tasks, {10: "test"})


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_loader_keeps_latest_unique_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            records = [
                {"task_id": "a", "score": 0.0},
                {"task_id": "b", "score": 0.5},
                {"task_id": "a", "score": 1.0},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            loaded = load_checkpoint_rows(path)
            self.assertEqual(len(loaded), 2)
            by_id = {row["task_id"]: row for row in loaded}
            self.assertEqual(by_id["a"]["score"], 1.0)


class LabelTests(unittest.TestCase):
    def test_strict_oracle_defers_only_when_expensive_recovers_failure(self):
        values = derive_labels_and_costs(0.0, 1.0, False, True)
        self.assertEqual(values["oracle_strict_label"], "expensive_only")
        self.assertEqual(values["oracle_score_label"], "expensive_only")
        self.assertGreater(values["delta_cost"], 0.0)

    def test_v2_is_available_but_not_enabled_without_explicit_model(self):
        self.assertNotIn("prompt_router_ml_v2", resolve_task_modes())
        self.assertEqual(
            resolve_task_modes("prompt_router_ml_v2"),
            ["prompt_router_ml_v2"],
        )


class ProbingTensorTests(unittest.TestCase):
    def test_feature_schema_contains_rank_and_shape(self):
        self.assertIn("agree_last_cheap_rank_in_exp_topk", PROBING_FEATURE_KEYS)
        self.assertIn("n_tokens", PROBING_FEATURE_KEYS)

    def test_agreement_helpers_on_cpu(self):
        torch.manual_seed(7)
        cheap = torch.randn(4, 17)
        expensive = torch.randn(4, 17)
        exact = _per_position_top1_match(cheap, expensive)
        topk = _per_position_topk_match(cheap, expensive, top_k=3)
        kl = _per_position_topk_kl(cheap, expensive, top_k=3)
        last = _last_position_agreement(cheap[-1], expensive[-1], 1.0, 3)
        self.assertEqual(tuple(exact.shape), (4,))
        self.assertEqual(tuple(topk.shape), (4,))
        self.assertEqual(tuple(kl.shape), (4,))
        self.assertTrue(torch.all(kl >= 0))
        self.assertIn("agree_last_cheap_rank_in_exp_topk", last)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_agreement_helpers_across_two_cuda_devices(self):
        cheap = torch.randn(3, 19, device="cuda:0")
        expensive = torch.randn(3, 19, device="cuda:1")
        self.assertEqual(
            tuple(_per_position_top1_match(cheap, expensive).shape),
            (3,),
        )
        self.assertEqual(
            tuple(_per_position_topk_match(cheap, expensive, 3).shape),
            (3,),
        )
        self.assertEqual(
            tuple(_per_position_topk_kl(cheap, expensive, 3).shape),
            (3,),
        )


class TrainingPipelineTests(unittest.TestCase):
    @staticmethod
    def _data(size: int) -> dict:
        rows = []
        features = []
        prompts = []
        labels = []
        deltas = []
        cheap_costs = []
        expensive_costs = []
        for index in range(size):
            expensive = index % 3 == 0
            prompt = (
                f"Write function {index} with nested tuple arithmetic"
                if expensive
                else f"Write simple list helper {index}"
            )
            label = "expensive_only" if expensive else "cheap_only"
            delta = 1.0 if expensive else 0.0
            row = {
                "task_id": f"task_{index}",
                "mbpp_task_id": index,
                "prompt": prompt,
                "oracle_score_label": label,
                "cheap_score": 0.0 if expensive else 1.0,
                "expensive_score": 1.0,
                "route_cheap_cost": 50.35 if expensive else 0.35,
                "route_expensive_cost": 1.0,
                "delta_score": delta,
            }
            rows.append(row)
            prompts.append(prompt)
            labels.append(label)
            deltas.append(delta)
            cheap_costs.append(row["route_cheap_cost"])
            expensive_costs.append(row["route_expensive_cost"])
            features.append(
                {
                    key: float((index + offset) % 5) / 5.0
                    for offset, key in enumerate(PROBING_FEATURE_KEYS)
                }
            )
        return {
            "prompts": prompts,
            "feat_rows": features,
            "labels_score": labels,
            "labels_strict": labels,
            "delta_score": deltas,
            "route_cheap_cost": cheap_costs,
            "route_expensive_cost": expensive_costs,
            "delta_cost": [
                cheap - expensive
                for cheap, expensive in zip(cheap_costs, expensive_costs)
            ],
            "rows": rows,
        }

    def test_classifier_and_l2d_fit_on_distinct_splits(self):
        train = self._data(18)
        validation = self._data(9)
        classifier, metrics, _, meta = train_classifier(
            train,
            validation,
            "classifier",
            True,
            [None, "balanced"],
            [0.1, 1.0],
        )
        self.assertTrue(hasattr(classifier, "predict"))
        self.assertIn("threshold", meta)
        self.assertEqual(metrics["n_total"], 9)

        regressor, metrics, _, meta, frontier = train_l2d(
            train,
            validation,
            True,
            [0.1, 1.0],
        )
        self.assertTrue(hasattr(regressor, "predict"))
        self.assertEqual(meta["mode"], "l2d")
        self.assertTrue(frontier)
        self.assertEqual(metrics["n_total"], 9)


if __name__ == "__main__":
    unittest.main()
