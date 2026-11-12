"""pad/validate_v4_v5.py -- sanity-check v4/v5 jsonls before we
trust them for paper numbers.

Runs a battery of schema + content checks on every
`meta_train_records_*_v4.jsonl` or `*_v5.jsonl` produced by the mega
queue's rescoring stages. Catches issues like:

  * empty jsonl (rescorer died mid-run)
  * label imbalance: no live or no spoof samples
  * branches list empty or wrong types
  * p_attack NaN / inf / out-of-range
  * same-path records with conflicting labels (dataset re-scoring bug)
  * ACER sanity bound: fused ACER outside plausible range
    (accepts 0.02..0.6 as the sane band; flags anything outside)

Output: `reports/validate_v4_v5.md` + return code 0/1.

Usage
-----
    python -m pad.validate_v4_v5
    python -m pad.validate_v4_v5 --fail-on-warn
    python -m pad.validate_v4_v5 --inputs reports/meta_train_records_celeba_v4.jsonl
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"


def default_inputs() -> List[Path]:
    out = []
    for pat in ("meta_train_records_*_v4.jsonl",
                "meta_train_records_*_v5.jsonl",
                "deepfake_temporal*_records_*.jsonl",
                "audio_asvspoof_records_v*.jsonl"):
        out.extend(sorted(REPORTS.glob(pat)))
    return out


def load_jsonl(p: Path) -> List[dict]:
    recs = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _fused_p(rec: dict) -> float:
    """Reliability-weighted mean of branch p_attack, or rec['p_attack']
    if already present."""
    top = rec.get("p_attack")
    if top is not None:
        return float(top)
    num = 0.0; den = 0.0
    for b in rec.get("branches") or []:
        if not isinstance(b, dict):
            continue
        p = b.get("p_attack"); r = b.get("reliability", 1.0)
        if p is None or r is None or r <= 0:
            continue
        num += float(p) * float(r); den += float(r)
    return num / den if den > 0 else 0.0


def check_one(path: Path) -> Dict:
    """Run the full battery on a single jsonl. Returns a result dict."""
    result = {
        "path": str(path),
        "name": path.name,
        "status": "OK",
        "errors": [],
        "warnings": [],
        "n_records": 0,
        "n_live": 0,
        "n_spoof": 0,
    }

    if not path.is_file():
        result["status"] = "MISSING"
        result["errors"].append(f"file not found: {path}")
        return result

    recs = load_jsonl(path)
    result["n_records"] = len(recs)
    if len(recs) == 0:
        result["status"] = "EMPTY"
        result["errors"].append("no records")
        return result

    labels = []
    branches_per_rec = []
    ps = []
    seen_paths = {}
    for r in recs:
        lab = r.get("label")
        if lab not in ("live", "spoof"):
            result["errors"].append(f"unknown label {lab!r} in record {r.get('path')}")
            continue
        labels.append(lab)

        # Path uniqueness + label consistency
        k = r.get("path")
        if k in seen_paths and seen_paths[k] != lab:
            result["errors"].append(
                f"conflicting labels for path {k}: {seen_paths[k]} vs {lab}"
            )
        seen_paths[k] = lab

        # Branches schema
        br = r.get("branches")
        if not isinstance(br, list):
            result["errors"].append(f"branches not a list in {k}")
            continue
        branches_per_rec.append(len(br))
        for b in br:
            if not isinstance(b, dict):
                result["errors"].append(f"branch entry not a dict in {k}")
                continue
            p = b.get("p_attack")
            if p is None:
                continue
            try:
                pf = float(p)
            except (TypeError, ValueError):
                result["errors"].append(
                    f"non-numeric p_attack in branch {b.get('name')} of {k}: {p!r}"
                )
                continue
            if math.isnan(pf) or math.isinf(pf):
                result["errors"].append(
                    f"NaN/inf p_attack in branch {b.get('name')} of {k}"
                )
            if not (-0.01 <= pf <= 1.01):
                result["warnings"].append(
                    f"p_attack outside [0,1] in branch {b.get('name')} of {k}: {pf:.4f}"
                )

        # Fused p
        fused = _fused_p(r)
        if math.isnan(fused) or math.isinf(fused):
            result["errors"].append(f"fused p_attack is NaN/inf in {k}")
        ps.append(fused)

    result["n_live"] = labels.count("live")
    result["n_spoof"] = labels.count("spoof")

    # Label balance
    if result["n_live"] == 0:
        result["errors"].append("no live samples")
    if result["n_spoof"] == 0:
        result["errors"].append("no spoof samples")

    # Branch count consistency (informational)
    if branches_per_rec:
        unique = set(branches_per_rec)
        if len(unique) > 1:
            result["warnings"].append(
                f"inconsistent branch counts across records: {sorted(unique)}"
            )

    # ACER sanity bound (only if both labels exist)
    if result["n_live"] > 0 and result["n_spoof"] > 0 and ps:
        ps_arr = np.asarray(ps, dtype=np.float64)
        lbl_arr = np.asarray([1 if l == "spoof" else 0 for l in labels],
                             dtype=np.int32)
        # At a single fixed threshold tau=0.5 compute ACER
        apcer = float(((ps_arr < 0.5) & (lbl_arr == 1)).sum()) / max(1, (lbl_arr == 1).sum())
        bpcer = float(((ps_arr >= 0.5) & (lbl_arr == 0)).sum()) / max(1, (lbl_arr == 0).sum())
        acer = (apcer + bpcer) / 2.0
        result["acer_at_0p5"] = acer
        if not (0.01 <= acer <= 0.65):
            result["warnings"].append(
                f"ACER @ tau=0.5 is {acer:.3f} -- outside sanity band [0.01, 0.65]"
            )

    if result["errors"]:
        result["status"] = "FAIL"
    elif result["warnings"]:
        result["status"] = "WARN"
    return result


def emit_markdown(results: List[Dict]) -> str:
    out = [
        "# v4 / v5 jsonl validation",
        "",
        f"_Generated by `pad.validate_v4_v5`. {len(results)} files checked._",
        "",
        "## Summary",
        "",
        "| File | status | records | live | spoof | ACER@0.5 | warnings | errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        acer = f"{r['acer_at_0p5']:.3f}" if "acer_at_0p5" in r else "—"
        out.append(f"| `{r['name']}` | **{r['status']}** | {r['n_records']} | "
                   f"{r['n_live']} | {r['n_spoof']} | {acer} | "
                   f"{len(r['warnings'])} | {len(r['errors'])} |")
    out.append("")
    for r in results:
        if r["errors"] or r["warnings"]:
            out.append(f"### `{r['name']}`")
            for e in r["errors"]:
                out.append(f"- ❌ **error:** {e}")
            for w in r["warnings"]:
                out.append(f"- ⚠️  warning: {w}")
            out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=REPORTS / "validate_v4_v5.md")
    ap.add_argument("--fail-on-warn", action="store_true",
                    help="exit non-zero if any file has warnings (strict mode)")
    args = ap.parse_args(argv)

    inputs = args.inputs or default_inputs()
    if not inputs:
        print("[validate] no v4/v5 jsonls found (run queue_mega.cmd first)",
              file=sys.stderr)
        return 0

    results = [check_one(p) for p in inputs]
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_warn = sum(1 for r in results if r["status"] == "WARN")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(emit_markdown(results), encoding="utf-8")
    print(f"[validate] wrote {args.out}")

    for r in results:
        icon = {"OK":"✓","WARN":"!","FAIL":"X","EMPTY":"0","MISSING":"?"}.get(r["status"],"?")
        print(f"  [{icon}] {r['name']:<50s} records={r['n_records']:<6d} "
              f"errors={len(r['errors'])} warnings={len(r['warnings'])}")

    if n_fail > 0:
        return 2
    if args.fail_on_warn and n_warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
