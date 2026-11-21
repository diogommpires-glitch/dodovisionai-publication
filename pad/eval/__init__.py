"""v9.4.1 step 7 — PAD evaluation harness.

Walks dataset directories under `pad/datasets/` (EULA-gated, not
shipped), scores every sample through the enabled PAD branches,
writes a JSONL of (label, attack_class, p_attack, branch_scores)
records, and hands off to `pad.report.main` for the ISO/IEC 30107-3
summary / DET curve / per-class APCER render.

Designed for "empty report with the correct shape" — when no datasets
are present each loader prints a clean skip message and the harness
still writes `scored.jsonl` (empty) + `summary.md` with zero samples.
"""
from __future__ import annotations

from .run_eval import run_eval, main  # noqa: F401
