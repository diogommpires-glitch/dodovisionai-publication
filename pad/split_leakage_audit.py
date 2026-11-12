"""
General per-branch train/val identity-leakage auditor  (v9.8.1 Audit C5).

The pre-existing `pad/diagnose_split_leakage.py` is patch_depth-specific.
This module generalises it: every trainer in the repo (train_patch_depth,
train_face_xray, train_face_xray_adv, train_audio_adv, train_moire_gbt,
train_temporal_deepfake, train_temporal_mvit) should call
`stamp_split_manifest(...)` after building its train/val splits. The
manifest is a sidecar JSON next to the trained weights:

    pad/models/manifests/<model_name>__split.json
    {
      "model":           "patch_depth_fas_v10b_pgd",
      "trainer":         "pad.train_patch_depth",
      "fit_at":          "2026-05-12T09:24:11",
      "split_strategy":  "identity_stratified",
      "n_train":         449909,
      "n_val":           44496,
      "train_identities_sha256": "f9c3...",   # SHA-256 of sorted identity list
      "val_identities_sha256":   "1b2a...",
      "intersection_count":      0,           # MUST be 0
      "intersection_examples":   [],
    }

`audit_all_manifests(...)` walks `pad/models/manifests/` and asserts
`intersection_count == 0` for every manifest -- a CI test (added in
`pad/tests/test_no_split_leakage.py`) imports this and fails the build
if any branch leaks identities between train and val.

For trainers that haven't been retrofitted yet (the audit-doc lists 4
of them), the diagnostic reports "manifest missing" so the gap is
visible in the report -- not silently zero-leak.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "pad" / "models" / "manifests"


# ─── Trainer-side helpers ──────────────────────────────────────────


def _sha256_of_sorted_identities(identities: Iterable[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(set(identities)):
        h.update((s + "\n").encode("utf-8"))
    return h.hexdigest()


def stamp_split_manifest(
    model: str,
    trainer: str,
    train_identities: Sequence[str],
    val_identities: Sequence[str],
    split_strategy: str = "identity_stratified",
    extra: Optional[Dict] = None,
) -> Dict:
    """Called by each trainer after building train/val splits.

    Writes `pad/models/manifests/<model>__split.json`. Returns the
    manifest dict.

    `train_identities` / `val_identities` should be the per-sample
    identity strings (subject IDs, video IDs, speaker IDs — whatever
    the trainer uses for grouping). Sets are computed inside.
    """
    train_set = set(train_identities)
    val_set = set(val_identities)
    inter = train_set & val_set
    examples = sorted(inter)[:20]
    manifest = {
        "model":                   str(model),
        "trainer":                 str(trainer),
        "fit_at":                  _dt.datetime.utcnow().isoformat(timespec="seconds"),
        "split_strategy":          str(split_strategy),
        "n_train":                 len(train_set),
        "n_val":                   len(val_set),
        "train_identities_sha256": _sha256_of_sorted_identities(train_identities),
        "val_identities_sha256":   _sha256_of_sorted_identities(val_identities),
        "intersection_count":      len(inter),
        "intersection_fraction":   len(inter) / max(1, len(val_set)),
        "intersection_examples":   examples,
    }
    if extra:
        manifest["extra"] = extra

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{model}__split.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[split-audit] wrote {path}  (leak={len(inter)} ids, "
          f"{manifest['intersection_fraction']*100:.2f}% of val)")
    return manifest


# ─── Auditor-side: scan all manifests + report ──────────────────────


KNOWN_TRAINERS = (
    "train_patch_depth",
    "train_face_xray",
    "train_face_xray_adv",
    "train_audio_adv",
    "train_moire_gbt",
    "train_temporal_deepfake",
    "train_temporal_mvit",
)


def audit_all_manifests(manifest_dir: Path = MANIFEST_DIR) -> Dict:
    """Walk every `*__split.json` and report leakage.

    Returns:
      {
        "n_manifests":   int,
        "missing_trainers":  list of trainer names with no manifest,
        "verdicts":      list of per-manifest verdict dicts,
        "all_clean":     bool,
      }
    """
    out = {"n_manifests": 0, "verdicts": [], "missing_trainers": []}
    seen_trainers = set()
    if manifest_dir.exists():
        for path in sorted(manifest_dir.glob("*__split.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                out["verdicts"].append({
                    "manifest": path.name,
                    "verdict":  "INVALID_JSON",
                    "error":    f"{type(e).__name__}: {e}",
                })
                continue
            seen_trainers.add(manifest.get("trainer", "").rsplit(".", 1)[-1])
            leak = int(manifest.get("intersection_count", 0))
            verdict = "PASS" if leak == 0 else "FAIL_LEAK"
            out["verdicts"].append({
                "manifest": path.name,
                "model":    manifest.get("model"),
                "trainer":  manifest.get("trainer"),
                "n_train":  manifest.get("n_train"),
                "n_val":    manifest.get("n_val"),
                "leak":     leak,
                "leak_pct": round(manifest.get("intersection_fraction", 0.0) * 100, 4),
                "verdict":  verdict,
                "examples": manifest.get("intersection_examples", [])[:5],
            })
    out["missing_trainers"] = sorted(set(KNOWN_TRAINERS) - seen_trainers)
    out["n_manifests"] = len(out["verdicts"])
    out["all_clean"] = (
        len(out["missing_trainers"]) == 0
        and all(v.get("verdict") == "PASS" for v in out["verdicts"])
    )
    return out


def _print_table(audit: Dict) -> None:
    print(f"[audit] manifests: {audit['n_manifests']}")
    for v in audit["verdicts"]:
        print(f"  {v['verdict']:9s} {v['model']!s:40s} "
              f"train={v['n_train']:>8} val={v['n_val']:>7} "
              f"leak={v['leak']} ({v['leak_pct']}%)")
    if audit["missing_trainers"]:
        print("[audit] MISSING manifests for trainers:")
        for t in audit["missing_trainers"]:
            print(f"  -- {t}  (run with --diagnostic to fit + stamp)")
    print(f"\n[audit] all_clean={audit['all_clean']}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(REPO_ROOT / "reports" / "split_leakage_audit.json"))
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any manifest reports a leak OR is missing")
    args = p.parse_args(argv)

    audit = audit_all_manifests()
    _print_table(audit)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    audit["generated_at"] = _dt.datetime.utcnow().isoformat(timespec="seconds")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"[audit] wrote -> {args.out}")
    if args.strict and not audit["all_clean"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
