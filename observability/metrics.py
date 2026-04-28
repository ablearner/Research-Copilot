"""Lightweight in-process metrics collector for Kepler."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class MetricsCollector:
    """Thread-safe counters and histograms for operational metrics."""

    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _start_time: float = field(default_factory=time.monotonic)

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None = None) -> str:
        if not labels:
            return name
        suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{suffix}}}"

    def increment(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    @contextmanager
    def timer(self, name: str, labels: dict[str, str] | None = None) -> Generator[None, None, None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.observe(name, time.monotonic() - start, labels)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            histograms: dict[str, dict[str, Any]] = {}
            for k, v in self._histograms.items():
                if not v:
                    continue
                sv = sorted(v)
                n = len(sv)
                histograms[k] = {
                    "count": n,
                    "sum": sum(sv),
                    "mean": round(sum(sv) / n, 4),
                    "min": round(sv[0], 4),
                    "p50": round(sv[n // 2], 4),
                    "p95": round(sv[int(n * 0.95)], 4),
                    "p99": round(sv[min(int(n * 0.99), n - 1)], 4),
                    "max": round(sv[-1], 4),
                }
        return {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "counters": counters,
            "histograms": histograms,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._start_time = time.monotonic()


# Global singleton
metrics = MetricsCollector()
