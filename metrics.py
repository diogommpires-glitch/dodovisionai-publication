"""metrics.py — v9.5.x Prometheus exposition.

Minimal zero-dependency metrics exporter that speaks Prometheus text
format. We don't want to pull in `prometheus_client` just for this, so
we re-implement counters / histograms with the minimum API our routes
need.

Exposed at `GET /metrics`.

Usage from application code:

    from metrics import REGISTRY
    REGISTRY.counter("pad_verdicts_total", labels={"verdict": "LIVE"}).inc()
    REGISTRY.histogram("pad_branch_latency_ms",
                       labels={"branch": "moire"}).observe(47.2)

All metric names follow Prometheus naming conventions
(https://prometheus.io/docs/practices/naming/).
"""
from __future__ import annotations

import threading
import time
from typing import Mapping, Optional


class _Counter:
    """Monotonic counter."""

    __slots__ = ("name", "help", "values", "_lock")

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self.values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: Optional[Mapping[str, str]]) -> tuple:
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    def inc(self, amount: float = 1.0, labels: Optional[Mapping[str, str]] = None) -> None:
        k = self._key(labels)
        with self._lock:
            self.values[k] = self.values.get(k, 0.0) + amount

    def get(self, labels: Optional[Mapping[str, str]] = None) -> float:
        return self.values.get(self._key(labels), 0.0)

    def expose(self) -> str:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if not self.values:
            out.append(f"{self.name} 0")
            return "\n".join(out)
        for k, v in sorted(self.values.items()):
            if k:
                lbl = ",".join(f'{n}="{val}"' for n, val in k)
                out.append(f"{self.name}{{{lbl}}} {v}")
            else:
                out.append(f"{self.name} {v}")
        return "\n".join(out)


class _Gauge:
    """Latest-value gauge — unlike a counter it can go up or down."""

    __slots__ = ("name", "help", "values", "_lock")

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self.values: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: Optional[Mapping[str, str]]) -> tuple:
        return tuple(sorted(labels.items())) if labels else ()

    def set(self, value: float, labels: Optional[Mapping[str, str]] = None) -> None:
        with self._lock:
            self.values[self._key(labels)] = float(value)

    def expose(self) -> str:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        if not self.values:
            out.append(f"{self.name} 0")
            return "\n".join(out)
        for k, v in sorted(self.values.items()):
            if k:
                lbl = ",".join(f'{n}="{val}"' for n, val in k)
                out.append(f"{self.name}{{{lbl}}} {v}")
            else:
                out.append(f"{self.name} {v}")
        return "\n".join(out)


class _Histogram:
    """Latency / size histogram with cumulative buckets.

    Matches Prometheus client_python semantics: each observation lands in
    *every* bucket whose upper-bound is ≥ the sample.
    """

    # Millisecond-oriented default buckets — suits our per-branch latency case
    DEFAULT_BUCKETS_MS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

    __slots__ = ("name", "help", "buckets", "counts", "sums", "_lock")

    def __init__(self, name: str, help_text: str,
                 buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS) -> None:
        self.name = name
        self.help = help_text
        self.buckets = tuple(sorted(buckets))
        self.counts: dict[tuple, list[int]] = {}
        self.sums: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: Optional[Mapping[str, str]]) -> tuple:
        return tuple(sorted(labels.items())) if labels else ()

    def observe(self, value: float,
                labels: Optional[Mapping[str, str]] = None) -> None:
        k = self._key(labels)
        with self._lock:
            row = self.counts.setdefault(k, [0] * (len(self.buckets) + 1))
            self.sums[k] = self.sums.get(k, 0.0) + value
            for i, upper in enumerate(self.buckets):
                if value <= upper:
                    row[i] += 1
            row[-1] += 1  # +Inf bucket (count)

    def expose(self) -> str:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for k, row in sorted(self.counts.items()):
            lbl_pairs = list(k)
            lbl_base = ",".join(f'{n}="{val}"' for n, val in lbl_pairs)
            for i, upper in enumerate(self.buckets):
                le = f'le="{upper}"'
                bundle = f'{{{lbl_base},{le}}}' if lbl_base else f'{{{le}}}'
                out.append(f"{self.name}_bucket{bundle} {row[i]}")
            le_inf = 'le="+Inf"'
            bundle = f'{{{lbl_base},{le_inf}}}' if lbl_base else f'{{{le_inf}}}'
            out.append(f"{self.name}_bucket{bundle} {row[-1]}")
            if lbl_base:
                out.append(f"{self.name}_count{{{lbl_base}}} {row[-1]}")
                out.append(f"{self.name}_sum{{{lbl_base}}} {self.sums[k]}")
            else:
                out.append(f"{self.name}_count {row[-1]}")
                out.append(f"{self.name}_sum {self.sums[k]}")
        return "\n".join(out)


class _Registry:
    """Simple in-process registry."""

    def __init__(self) -> None:
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()

    def counter(self, name: str, help_text: str = "",
                labels: Optional[Mapping[str, str]] = None) -> _Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = _Counter(name, help_text or name)
        c = self._counters[name]
        if labels:
            # Return a helper bound to these labels
            class _Bound:
                __slots__ = ("_c",)
                def __init__(_self, c):
                    _self._c = c
                def inc(_self, amount=1.0):
                    c.inc(amount, labels)
                def get(_self):
                    return c.get(labels)
            return _Bound(c)  # type: ignore
        return c

    def gauge(self, name: str, help_text: str = "",
              labels: Optional[Mapping[str, str]] = None) -> _Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = _Gauge(name, help_text or name)
        g = self._gauges[name]
        if labels:
            class _Bound:
                __slots__ = ("_g",)
                def __init__(_self, g): _self._g = g
                def set(_self, v): g.set(v, labels)
            return _Bound(g)  # type: ignore
        return g

    def histogram(self, name: str, help_text: str = "",
                  labels: Optional[Mapping[str, str]] = None) -> _Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = _Histogram(name, help_text or name)
        h = self._histograms[name]
        if labels:
            class _Bound:
                __slots__ = ("_h",)
                def __init__(_self, h): _self._h = h
                def observe(_self, v): h.observe(v, labels)
            return _Bound(h)  # type: ignore
        return h

    def expose(self) -> str:
        lines = []
        # Process info first
        lines.append("# HELP process_uptime_seconds Seconds since process start")
        lines.append("# TYPE process_uptime_seconds gauge")
        lines.append(f"process_uptime_seconds {time.time() - self.started_at}")
        for c in self._counters.values():
            lines.append(c.expose())
        for g in self._gauges.values():
            lines.append(g.expose())
        for h in self._histograms.values():
            lines.append(h.expose())
        return "\n".join(lines) + "\n"


REGISTRY = _Registry()


# ---------------------------------------------------------------------
# PAD-specific helpers  (added 2026-04-22 for per-branch calibration
# visibility + reliability EMA tracking).
# ---------------------------------------------------------------------

_REL_EMA: "dict[str, float]" = {}
_REL_ALPHA = 0.1


def update_branch_reliability(branch: str, reliability: float) -> None:
    """Observe a new reliability sample from one branch score; updates an
    exponential moving average and pushes the value into a Prometheus gauge
    ``pad_branch_reliability_ema{branch=...}``. Non-blocking, cheap."""
    try:
        prev = _REL_EMA.get(branch)
        ema = reliability if prev is None else (1 - _REL_ALPHA) * prev + _REL_ALPHA * float(reliability)
        _REL_EMA[branch] = ema
        REGISTRY.gauge("pad_branch_reliability_ema",
                       "Exponential moving avg of branch reliability (alpha=0.1)",
                       labels={"branch": branch}).set(ema)
    except Exception:
        pass


def refresh_calibrator_gauges() -> None:
    """Read the current CalibratorSet on disk and expose per-branch state
    as Prometheus gauges. Safe to call from /metrics handlers and
    periodically. Degrades silently if pad.calibrate is unavailable."""
    try:
        from pad.calibrate import calibrator_status
        st = calibrator_status()
    except Exception:
        return
    REGISTRY.gauge("pad_calibrators_loaded",
                   "1 if pad/models/calibrators.pkl is currently loaded").set(
        1.0 if st.get("loaded") else 0.0)
    for branch, meta in (st.get("branches") or {}).items():
        choice = meta.get("choice", "?")
        REGISTRY.gauge("pad_calibrator_degenerate",
                       "1 if the branch calibrator fell back to identity",
                       labels={"branch": branch}).set(
            1.0 if meta.get("degenerate") else 0.0)
        if meta.get("ece_after") is not None:
            try:
                REGISTRY.gauge("pad_calibrator_ece_after",
                               "ECE of the branch calibrator after fitting",
                               labels={"branch": branch}).set(
                    float(meta["ece_after"]))
            except Exception:
                pass
        REGISTRY.gauge("pad_calibrator_choice_code",
                       "0=platt, 1=isotonic, 2=identity_nosignal, -1=other",
                       labels={"branch": branch}).set(
            0.0 if choice == "platt"
            else 1.0 if choice == "isotonic"
            else 2.0 if choice == "identity_nosignal"
            else -1.0)


__all__ = ["REGISTRY", "update_branch_reliability", "refresh_calibrator_gauges"]
