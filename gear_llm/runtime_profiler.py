import time
from contextlib import contextmanager

import torch


TIME_FIELDS = (
    "total_generation_time_seconds",
    "cheap_forward_time_seconds",
    "expensive_forward_time_seconds",
    "router_decision_time_seconds",
    "guard_time_seconds",
    "tokenizer_decode_time_seconds",
    "prompt_format_time_seconds",
    "evaluation_time_seconds",
)

COUNT_FIELDS = (
    "number_of_cheap_forwards",
    "number_of_expensive_forwards",
    "number_of_router_decisions",
    "number_of_tokens_generated",
)


class RuntimeProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.times = {field: 0.0 for field in TIME_FIELDS}
        self.counts = {field: 0 for field in COUNT_FIELDS}

    def add_time(self, field: str, value: float):
        if self.enabled:
            self.times[field] = self.times.get(field, 0.0) + float(value)

    def increment(self, field: str, value: int = 1):
        if self.enabled:
            self.counts[field] = self.counts.get(field, 0) + int(value)

    @contextmanager
    def timed(self, field: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(field, time.perf_counter() - start)

    @contextmanager
    def forward(self, role: str, device: str):
        if not self.enabled:
            yield
            return

        if role not in {"cheap", "expensive"}:
            raise ValueError(f"Unknown forward role: {role}")

        self._sync_if_cuda(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync_if_cuda(device)
            elapsed = time.perf_counter() - start
            self.add_time(f"{role}_forward_time_seconds", elapsed)
            self.increment(f"number_of_{role}_forwards")

    def summary(self, generated_tokens: int = 0) -> dict:
        if generated_tokens:
            self.counts["number_of_tokens_generated"] = int(generated_tokens)

        cheap_forwards = self.counts["number_of_cheap_forwards"]
        expensive_forwards = self.counts["number_of_expensive_forwards"]
        tokens = self.counts["number_of_tokens_generated"]
        total_time = self.times["total_generation_time_seconds"]
        cheap_time = self.times["cheap_forward_time_seconds"]
        expensive_time = self.times["expensive_forward_time_seconds"]

        return {
            **self.times,
            **self.counts,
            "average_time_per_generated_token": (
                total_time / tokens if tokens else 0.0
            ),
            "average_cheap_forward_time": (
                cheap_time / cheap_forwards if cheap_forwards else 0.0
            ),
            "average_expensive_forward_time": (
                expensive_time / expensive_forwards if expensive_forwards else 0.0
            ),
            "routing_overhead_time_seconds": max(
                0.0,
                total_time - cheap_time - expensive_time,
            ),
        }

    @staticmethod
    def _sync_if_cuda(device: str):
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)


def maybe_profiler(enabled: bool) -> RuntimeProfiler | None:
    return RuntimeProfiler(enabled=True) if enabled else None
