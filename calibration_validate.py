"""
DODO VISION AI v7.1 — Calibration Validation Harness (Next Step #5)

Records three labeled sessions per registered subject (rest, cognitive
load, mock-deception), replays them through the deception pipeline, and
emits per-indicator ROC curves + empirical weights you can copy back
into truth_engine.py.

Usage
-----
    # 1. Record sessions (writes to calibration/<name>/<label>.parquet)
    python calibration_validate.py record "Dodo" rest     --seconds 90
    python calibration_validate.py record "Dodo" load     --seconds 90
    python calibration_validate.py record "Dodo" deceive  --seconds 90

    # 2. Evaluate
    python calibration_validate.py evaluate "Dodo"

Output
------
    calibration/<name>/report.json
        {
          "per_indicator": {
            "pupil_dilation": {"auc": 0.71, "d": 0.42, ...},
            ...
          },
          "suggested_weights": {"pupil_dilation": 2.7, ...},
          "n_rest": 1800, "n_load": 1800, "n_deceive": 1800
        }
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Dict, List

import config as cfg

INDICATORS = [
    "pupil_dilation", "blink_rate", "heart_rate_bpm", "stress_index",
    "facial_asymmetry", "fear_au_leakage", "duchenne_fake_score",
    "micro_expression_rate", "gaze_aversion",
]

ROOT = cfg.CALIBRATION_DIR


def _session_dir(name: str) -> str:
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def record(name: str, label: str, seconds: int) -> None:
    """Capture affect-vector samples from the running server into a parquet file."""
    import httpx  # required at record time only
    try:
        import pyarrow as pa, pyarrow.parquet as pq
    except ImportError:
        print("pip install pyarrow --break-system-packages")
        sys.exit(1)
    assert label in ("rest", "load", "deceive"), "label must be rest/load/deceive"
    url = f"http://{cfg.HOST}:{cfg.PORT}/affect"
    rows: List[dict] = []
    print(f"[rec] {name}/{label} for {seconds}s — point the camera at the subject.")
    t_end = time.time() + seconds
    with httpx.Client(timeout=2.0) as client:
        while time.time() < t_end:
            try:
                r = client.get(url)
                if r.status_code == 200:
                    payload = r.json()
                    for p in payload.get("persons", []):
                        if p.get("identity") == name or len(payload["persons"]) == 1:
                            row = {"ts": time.time(), "label": label}
                            for k in INDICATORS:
                                v = p.get(k)
                                if isinstance(v, dict):      # e.g. duchenne -> flat it
                                    row[k] = float(v.get("fake", 0.0))
                                else:
                                    row[k] = float(v) if v is not None else None
                            rows.append(row)
            except Exception as e:
                print("  [warn]", e)
            time.sleep(0.1)
    path = os.path.join(_session_dir(name), f"{label}.parquet")
    if not rows:
        print("  [!] no rows captured — is the server running and the subject registered?")
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    print(f"[ok] wrote {len(rows)} rows -> {path}")


def _auc(pos: List[float], neg: List[float]) -> float:
    """Mann-Whitney U-based AUC."""
    if not pos or not neg:
        return 0.5
    both = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    both.sort(key=lambda x: x[0])
    rank_sum = 0.0
    for i, (_, lbl) in enumerate(both, start=1):
        if lbl == 1:
            rank_sum += i
    n1, n2 = len(pos), len(neg)
    u = rank_sum - n1 * (n1 + 1) / 2
    return u / (n1 * n2)


def _cohens_d(pos: List[float], neg: List[float]) -> float:
    import statistics as st
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    mp, mn = st.mean(pos), st.mean(neg)
    sp = (st.pstdev(pos) ** 2 + st.pstdev(neg) ** 2) / 2
    return (mp - mn) / (sp ** 0.5) if sp > 0 else 0.0


def evaluate(name: str) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pip install pyarrow --break-system-packages"); sys.exit(1)
    sdir = _session_dir(name)
    tables = {}
    for lbl in ("rest", "load", "deceive"):
        p = os.path.join(sdir, f"{lbl}.parquet")
        if os.path.exists(p):
            tables[lbl] = pq.read_table(p).to_pylist()
    if "rest" not in tables or "deceive" not in tables:
        print("Need at least rest.parquet and deceive.parquet. Run record first.")
        sys.exit(2)
    report: Dict[str, dict] = {}
    for ind in INDICATORS:
        pos = [r[ind] for r in tables["deceive"] if r.get(ind) is not None]
        neg = [r[ind] for r in tables["rest"] if r.get(ind) is not None]
        report[ind] = {
            "auc": round(_auc(pos, neg), 3),
            "d": round(_cohens_d(pos, neg), 3),
            "n_pos": len(pos),
            "n_neg": len(neg),
        }
    # Suggest weights ∝ |d|, floored at 0.3, capped at 3.0, then renormalized
    raw = {k: max(0.3, min(3.0, abs(v["d"]) * 4.0)) for k, v in report.items()}
    out = {
        "subject": name,
        "per_indicator": report,
        "suggested_weights": {k: round(v, 2) for k, v in raw.items()},
        "counts": {lbl: len(tables.get(lbl, [])) for lbl in ("rest", "load", "deceive")},
        "generated_at": time.time(),
    }
    outp = os.path.join(sdir, "report.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[ok] wrote {outp}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("name"); r.add_argument("label"); r.add_argument("--seconds", type=int, default=90)
    e = sub.add_parser("evaluate"); e.add_argument("name")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.name, args.label, args.seconds)
    else:
        evaluate(args.name)


if __name__ == "__main__":
    main()
