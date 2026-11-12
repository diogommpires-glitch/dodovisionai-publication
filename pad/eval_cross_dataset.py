"""pad/eval_cross_dataset.py — cross-dataset PAD evaluation.

For each configured (branch, test dataset) pair, runs the branch's scorer
over the labelled test folder and writes ISO/IEC 30107-3 metrics:

    reports/cross_dataset/<branch>_on_<dataset>/
        scored.jsonl       # per-sample records
        summary.md         # APCER / BPCER / ACER tables
        summary.json       # machine-readable
        det_curve.svg      # detection-error-tradeoff curve

The paper's Table 2 is literally the concatenation of these summary.md
files for every (branch, dataset) combination you run.

Example layout that makes sense for your current setup:

    # Trained on CelebA-Spoof (D:), evaluate on FF++ crops (E:)
    python -m pad.eval_cross_dataset \
        --branch patch_depth \
        --test-data E:\\pad_datasets\\face_xray \
        --name patch_depth_on_FFpp \
        --limit 4000

    # Trained on FF++ (E:), evaluate on CelebA-Spoof (D:)
    python -m pad.eval_cross_dataset \
        --branch deepfake \
        --test-data D:\\pad_datasets\\patch_depth \
        --name face_xray_on_CelebASpoof \
        --limit 4000

    # Or: evaluate every branch on every dataset (automatic)
    python -m pad.eval_cross_dataset --all

Each run is independent. Re-runs with the same --name are idempotent
(overwrite only the summary, reuse scored.jsonl if present).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent.parent
REPORTS_ROOT = REPO / "reports" / "cross_dataset"


# ── which image-based branches can run, and their scorer callables ──
def _score_with(branch_name: str, img_bgr) -> Dict:
    """Run a single branch on a BGR crop, return a record dict."""
    if branch_name == "moire":
        from pad.branches.moire_branch import score as fn
        s = fn(img_bgr)
    elif branch_name == "patch_depth":
        from pad.branches.patch_depth_branch import score as fn
        s = fn(img_bgr)
    elif branch_name == "deepfake":
        from pad.branches.deepfake_branch import score as fn
        s = fn(pid=0, face_crop_bgr=img_bgr)
    else:
        raise ValueError(f"unsupported branch for image eval: {branch_name}")
    return {
        "name": s.name,
        "p_attack": float(s.p_attack),
        "reliability": float(s.reliability),
        "dominant_class": s.dominant_class,
        "latency_ms": float(getattr(s, "latency_ms", 0.0)),
    }


# ── dataset iteration ──
def _iter_samples(data: Path, limit: Optional[int]) -> List[Tuple[str, Path]]:
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
    live = sorted(p for p in (data / "live").iterdir()
                  if (data / "live").is_dir() and p.suffix.lower() in IMG_EXT) \
        if (data / "live").is_dir() else []
    spoof = sorted(p for p in (data / "spoof").iterdir()
                   if (data / "spoof").is_dir() and p.suffix.lower() in IMG_EXT) \
        if (data / "spoof").is_dir() else []
    items = [("live", p) for p in live] + [("spoof", p) for p in spoof]
    random.Random(42).shuffle(items)
    if limit and limit > 0:
        live_q = [x for x in items if x[0] == "live"][: limit // 2]
        spoof_q = [x for x in items if x[0] == "spoof"][: limit - len(live_q)]
        items = live_q + spoof_q
        random.Random(42).shuffle(items)
    return items


def _score_dataset(branch_name: str, test_data: Path, out_jsonl: Path,
                   limit: Optional[int]) -> int:
    """Score every sample; return number written. Idempotent: reuses an
    existing scored.jsonl with the right length."""
    import cv2
    items = _iter_samples(test_data, limit)
    if not items:
        print(f"[xd] no samples under {test_data}")
        return 0
    # Idempotent reuse
    if out_jsonl.exists():
        try:
            n_existing = sum(1 for _ in open(out_jsonl, encoding="utf-8"))
            if n_existing >= len(items):
                print(f"[xd] reusing existing {out_jsonl} ({n_existing} records)")
                return n_existing
        except Exception:
            pass

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"[xd] scoring {len(items):,} samples with branch={branch_name}")
    try:
        from tqdm import tqdm
        bar = tqdm(items, unit="img", mininterval=2.0)
    except ImportError:
        bar = items

    n_ok = 0
    t0 = time.time()
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for label, path in bar:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            try:
                branch_rec = _score_with(branch_name, img)
            except Exception as e:
                branch_rec = {"name": branch_name, "p_attack": 0.5,
                              "reliability": 0.0, "dominant_class": None,
                              "err": str(e)}
            rec = {
                "path": str(path), "label": label,
                "attack_class": None if label == "live" else "UNKNOWN",
                "p_attack": branch_rec["p_attack"],
                "branches": [branch_rec],
            }
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
    dt = time.time() - t0
    print(f"[xd] wrote {n_ok:,} records in {dt/60:.1f} min -> {out_jsonl}")
    return n_ok


# ── ISO/IEC 30107-3 metrics at BPCER=0.01 ──
def _compute_metrics(records: List[Dict]) -> Dict:
    """APCER @ BPCER=0.01, ACER, plus a naive 0.5-threshold confusion."""
    scores = np.array([r["p_attack"] for r in records], dtype=np.float64)
    labels = np.array([1 if r["label"] == "spoof" else 0 for r in records],
                      dtype=np.int32)
    n_live = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())

    # Sweep thresholds to find BPCER <= 0.01
    sorted_thr = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best = None
    for th in sorted_thr:
        pred = (scores >= th).astype(np.int32)
        bpcer = float((pred[labels == 0] == 1).mean()) if n_live > 0 else 0.0
        apcer = float((pred[labels == 1] == 0).mean()) if n_spoof > 0 else 0.0
        if bpcer <= 0.01:
            if best is None or apcer < best["apcer"]:
                best = {"threshold": float(th), "apcer": apcer, "bpcer": bpcer,
                        "acer": (apcer + bpcer) / 2.0}
    if best is None:
        best = {"threshold": 1.0, "apcer": 1.0, "bpcer": 0.0, "acer": 0.5}

    # Naive @ th=0.5
    pred_50 = (scores >= 0.5).astype(np.int32)
    tp = int(((pred_50 == 1) & (labels == 1)).sum())
    fp = int(((pred_50 == 1) & (labels == 0)).sum())
    fn = int(((pred_50 == 0) & (labels == 1)).sum())
    tn = int(((pred_50 == 0) & (labels == 0)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    # EER — where APCER == BPCER
    eer, eer_th = 1.0, 0.5
    for th in sorted_thr:
        pred = (scores >= th).astype(np.int32)
        b = float((pred[labels == 0] == 1).mean()) if n_live > 0 else 0.0
        a = float((pred[labels == 1] == 0).mean()) if n_spoof > 0 else 0.0
        if abs(a - b) < abs(eer - eer) + 0.001 and (a + b) / 2 < eer:
            eer = (a + b) / 2
            eer_th = float(th)

    return {
        "n_total": len(records), "n_live": n_live, "n_spoof": n_spoof,
        "operating_point_bpcer_01": best,
        "naive_th50": {"accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "eer": {"rate": eer, "threshold": eer_th},
    }


def _det_svg(scores: np.ndarray, labels: np.ndarray) -> str:
    """Minimal standalone-SVG DET curve (APCER vs BPCER)."""
    sorted_thr = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    pts = []
    for th in sorted_thr:
        pred = (scores >= th).astype(np.int32)
        b = float((pred[labels == 0] == 1).mean()) if (labels == 0).any() else 0.0
        a = float((pred[labels == 1] == 0).mean()) if (labels == 1).any() else 0.0
        pts.append((b, a))
    pts.sort()
    w, h = 420, 320
    def _x(v): return 60 + v * (w - 80)
    def _y(v): return (h - 40) - v * (h - 80)
    path = " ".join(f"{'M' if i == 0 else 'L'}{_x(b):.1f},{_y(a):.1f}"
                    for i, (b, a) in enumerate(pts))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            f'<text x="{w//2}" y="20" text-anchor="middle" font-family="monospace" '
            f'font-size="12">DET curve (APCER vs BPCER)</text>'
            f'<line x1="60" y1="{h-40}" x2="{w-20}" y2="{h-40}" stroke="black"/>'
            f'<line x1="60" y1="{h-40}" x2="60"    y2="20"    stroke="black"/>'
            f'<text x="{w//2}" y="{h-8}" text-anchor="middle" font-family="monospace" '
            f'font-size="10">BPCER</text>'
            f'<text x="18" y="{h//2}" text-anchor="middle" font-family="monospace" '
            f'font-size="10" transform="rotate(-90,18,{h//2})">APCER</text>'
            f'<path d="{path}" fill="none" stroke="#a855f7" stroke-width="1.5"/>'
            '</svg>')


def _write_summary(out_dir: Path, branch: str, test_dataset: str,
                   metrics: Dict, records: List[Dict]) -> None:
    op = metrics["operating_point_bpcer_01"]
    eer = metrics["eer"]
    naive = metrics["naive_th50"]
    md = f"""# Cross-dataset evaluation

**Branch**: `{branch}`
**Test dataset**: `{test_dataset}`
**N**: {metrics['n_total']:,}  (live={metrics['n_live']:,}, spoof={metrics['n_spoof']:,})
**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Operating point @ BPCER ≤ 0.01

| Metric | Value |
|---|---:|
| Threshold τ* | **{op['threshold']:.4f}** |
| APCER @ BPCER≤0.01 | **{op['apcer']:.4f}** |
| BPCER | {op['bpcer']:.4f} |
| **ACER** | **{op['acer']:.4f}** |

## Equal-Error Rate

| | Value |
|---|---:|
| EER | **{eer['rate']:.4f}** |
| Threshold | {eer['threshold']:.4f} |

## Naive threshold = 0.5 confusion

| | Predicted live | Predicted spoof |
|---|---:|---:|
| **Actual live** | {naive['tn']} | {naive['fp']} |
| **Actual spoof** | {naive['fn']} | {naive['tp']} |

Accuracy: **{naive['accuracy']:.3f}**

![DET curve](det_curve.svg)
"""
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    scores = np.array([r["p_attack"] for r in records])
    labels = np.array([1 if r["label"] == "spoof" else 0 for r in records])
    (out_dir / "det_curve.svg").write_text(_det_svg(scores, labels),
                                           encoding="utf-8")


# ── main ──
def _one(branch: str, test_data: Path, name: str, limit: Optional[int]) -> Path:
    out_dir = REPORTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "scored.jsonl"
    n = _score_dataset(branch, test_data, jsonl, limit)
    if n == 0:
        return out_dir
    records = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    metrics = _compute_metrics(records)
    _write_summary(out_dir, branch, test_data.name, metrics, records)
    op = metrics["operating_point_bpcer_01"]
    print(f"[xd] {name}: ACER={op['acer']:.4f}  "
          f"APCER@BPCER<=0.01={op['apcer']:.4f}  "
          f"n={metrics['n_total']:,}")
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: cross-dataset evaluation -- fit on one source, eval on another, report ACER/AUC. Wrapped by `pad.cross_dataset_matrix` for the full grid."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", choices=["moire", "patch_depth", "deepfake"])
    ap.add_argument("--test-data", type=Path,
                    help="dir with live/ + spoof/ subfolders")
    ap.add_argument("--name", help="output subdir name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true",
                    help="auto-run every (branch, dataset) pair found "
                         "under E:/D: default locations")
    args = ap.parse_args(argv)

    if args.all:
        defaults = []
        # Already-downloaded datasets (all on E: after the 2026-04-21 move)
        celeba = Path(r"E:\pad_datasets\patch_depth")
        ffpp = Path(r"E:\pad_datasets\face_xray")
        # Fallback to D: in case the user hasn't moved patch_depth yet
        if not celeba.exists():
            legacy = Path(r"D:\pad_datasets\patch_depth")
            if legacy.exists():
                celeba = legacy
        # Presentation-attack (print/mask/screen) datasets -- moire + patch_depth
        msu_mfsd = Path(r"E:\pad_datasets\MSU-MFSD")
        oulu_npu = Path(r"E:\pad_datasets\OULU-NPU")
        replay_attack = Path(r"E:\pad_datasets\Replay-Attack")
        # Deepfake datasets -- face_xray branch only
        celeb_df = Path(r"E:\pad_datasets\Celeb-DF-v2")
        dfdc = Path(r"E:\pad_datasets\DFDC_preview")

        if celeba.exists():
            defaults += [("moire",       celeba, "moire_on_CelebASpoof"),
                         ("patch_depth", celeba, "patch_depth_on_CelebASpoof"),
                         ("deepfake",    celeba, "face_xray_on_CelebASpoof")]
        if ffpp.exists():
            defaults += [("moire",       ffpp, "moire_on_FFpp"),
                         ("patch_depth", ffpp, "patch_depth_on_FFpp"),
                         ("deepfake",    ffpp, "face_xray_on_FFpp")]
        if msu_mfsd.exists():
            defaults += [("moire",       msu_mfsd, "moire_on_MSU-MFSD"),
                         ("patch_depth", msu_mfsd, "patch_depth_on_MSU-MFSD")]
        if oulu_npu.exists():
            defaults += [("moire",       oulu_npu, "moire_on_OULU-NPU"),
                         ("patch_depth", oulu_npu, "patch_depth_on_OULU-NPU"),
                         ("deepfake",    oulu_npu, "face_xray_on_OULU-NPU")]
        if replay_attack.exists():
            defaults += [("moire",       replay_attack, "moire_on_ReplayAttack"),
                         ("patch_depth", replay_attack, "patch_depth_on_ReplayAttack")]
        if celeb_df.exists():
            defaults += [("deepfake",    celeb_df, "face_xray_on_CelebDFv2")]
        if dfdc.exists():
            defaults += [("deepfake",    dfdc, "face_xray_on_DFDC")]
        if not defaults:
            print("[xd] --all found no datasets at default paths")
            return 1
        for branch, data, name in defaults:
            _one(branch, data, name, args.limit)
        return 0

    if not args.branch or not args.test_data:
        ap.error("--branch and --test-data required (or use --all)")
    name = args.name or f"{args.branch}_on_{args.test_data.name}"
    _one(args.branch, args.test_data, name, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
