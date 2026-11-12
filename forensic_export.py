"""
DODO VISION AI v7.1 — Per-Person Forensic Export Bundle (Next Step #9)

Complements the session recorder in forensic.py. Where forensic.py captures
a LIVE session as it runs, this module produces a CHAIN-OF-CUSTODY bundle
for a single person_id (or the whole session) that stakeholders can audit
offline, independent of the live server.

Bundle schema: "dodo.forensic.export.v1"

HTTP endpoints mounted via forensic_export.attach(app):
    GET /forensic/export                   -> whole session (JSON)
    GET /forensic/export?person_id=p1      -> scoped to subject
    GET /forensic/export?download=true     -> stream JSON as file
    GET /forensic/manifest?person_id=p1    -> sidecar sha256 manifest
"""
from __future__ import annotations
import hashlib, json, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as cfg
from safe_paths import safe_child

SCHEMA = "dodo.forensic.export.v1"
EXPORTS_DIR = os.path.join(cfg.CALIBRATION_DIR, "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


# ── helpers ──
def _safe(obj: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "..."
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe(v, depth + 1) for k, v in list(obj.items())[:256]}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 5000:
            return [_safe(v, depth + 1) for v in obj[:5000]] + ["...truncated..."]
        return [_safe(v, depth + 1) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    except Exception:
        return ""


def _read_jsonl(path: str, max_rows: int = 20000) -> List[Dict]:
    """v9.8: integrity-aware JSONL reader.

    Tracks both successful parses AND skipped (malformed / truncated)
    lines. JSONL files in this repo are written line-by-line during a
    live session, so a kill-mid-write can leave the LAST line
    partially flushed. We:
        1. Skip blank lines silently (always OK)
        2. Skip un-parseable lines and increment the skip counter
        3. Stash the count in `_read_jsonl.last_skipped` so the caller
           can include it in the forensic bundle's audit field.
    """
    _read_jsonl.last_skipped = 0  # type: ignore[attr-defined]
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    skipped = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    skipped += 1
    except Exception:
        pass
    _read_jsonl.last_skipped = skipped  # type: ignore[attr-defined]
    return rows


# ── collectors (every call tolerates missing modules) ──
def _subject(person_id: Optional[str]) -> Dict[str, Any]:
    if not person_id:
        return {"person_id": None, "display_name": "all_subjects"}
    out = {"person_id": person_id, "display_name": person_id}
    reg = os.path.join(cfg.FACES_DIR, "registry.json")
    if os.path.exists(reg):
        out["registry_hash"] = _sha256(reg)
        try:
            data = json.loads(Path(reg).read_text(encoding="utf-8"))
            if isinstance(data, dict) and person_id in data:
                entry = data[person_id]
                if isinstance(entry, dict):
                    out["display_name"] = entry.get("name", person_id)
        except Exception:
            pass
    return out


def _biometrics(person_id: Optional[str]) -> Dict[str, Any]:
    try:
        import face_identity  # type: ignore
        if person_id and hasattr(face_identity, "pop_cached_state"):
            return _safe(face_identity.pop_cached_state(person_id) or {})
    except Exception:
        pass
    return {}


def _affect(person_id: Optional[str], max_rows: int = 5000) -> List[Dict]:
    """v9.8: also tracks skipped-line count from the source JSONL for
    the bundle's audit metadata (see `_v98_metadata`)."""
    candidates = [
        os.path.join(cfg.PARQUET_LOG_DIR, "affect_mirror.jsonl"),
        os.path.join(cfg.PARQUET_LOG_DIR, "affect.jsonl"),
    ]
    rows: List[Dict] = []
    skipped_total = 0
    for p in candidates:
        if os.path.exists(p):
            rows = _read_jsonl(p, max_rows=max_rows)
            skipped_total = getattr(_read_jsonl, "last_skipped", 0)
            break
    # Stash for the bundle metadata to pick up
    _affect.last_skipped = skipped_total  # type: ignore[attr-defined]
    if person_id:
        rows = [r for r in rows if r.get("person_id") == person_id]
    return [_safe(r) for r in rows]


def _llm(person_id: Optional[str], max_rows: int = 500) -> List[Dict]:
    p = os.path.join(cfg.CALIBRATION_DIR, "llm_training_pairs.jsonl")
    rows = _read_jsonl(p, max_rows=max_rows)
    if person_id:
        tag = str(person_id)
        rows = [r for r in rows
                if tag in json.dumps(r.get("affect", {}), default=str)
                or r.get("person_id") == person_id]
    return [_safe(r) for r in rows]


def _audio(person_id: Optional[str], max_files: int = 500) -> List[Dict]:
    out: List[Dict] = []
    roots = [os.path.join(cfg.SESSIONS_DIR, "audio"),
             os.path.join(cfg.PARQUET_LOG_DIR, "audio")]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root))[:max_files]:
            if person_id and person_id not in name:
                continue
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            st = os.stat(full)
            out.append({
                "path": full,
                "sha256": _sha256(full),
                "bytes": st.st_size,
                "duration_s": round(st.st_size / (cfg.VOICE_SAMPLE_RATE * 2), 3),
                "mtime": st.st_mtime,
            })
    return out


def _radar() -> Dict[str, Any]:
    try:
        import radar  # type: ignore
        if hasattr(radar, "summary"):
            return _safe(radar.summary())
        if hasattr(radar, "get_radar"):
            return _safe(radar.get_radar())
    except Exception:
        pass
    return {}


def _v98_metadata() -> Dict[str, Any]:
    """v9.8: capture the feature-flag + camera-pin + LLM-backend snapshot
    so a forensic bundle is fully reproducible. The reviewer / replay
    operator can read this block and know EXACTLY which model variants,
    which dispatch paths, and which hardware were in play.
    """
    md: Dict[str, Any] = {
        "ff_use_fp16":   os.environ.get("FF_USE_FP16", "0"),
        "ff_blink_cnn":  os.environ.get("FF_BLINK_CNN", "0"),
        "ff_audio_npu":  os.environ.get("FF_AUDIO_NPU", "0"),
        "llm_backend":   os.environ.get("LLM_BACKEND", "(auto)"),
        "vllm_url":      os.environ.get("VLLM_URL", ""),
        "vllm_model":    os.environ.get("VLLM_MODEL", ""),
        "hifps_cam_name": os.environ.get("DODO_HIFPS_CAM_NAME", ""),
        "hifps_force_120": os.environ.get("DODO_HIFPS_FORCE_120", "0"),
        "radar_port":    os.environ.get("DODO_RADAR_PORT", ""),
        "radar_b_port":  os.environ.get("DODO_RADAR_B_PORT", ""),
        "gsr_port":      os.environ.get("DODO_GSR_PORT", ""),
    }
    # Live state from dual_camera + per-branch baselines (so the bundle
    # captures the BASELINE the user accumulated during this session,
    # not just the env config).
    try:
        import dual_camera  # type: ignore
        md["dual_cam_status"] = dual_camera.get_status()
    except Exception:  # nolint:silent-except dual_camera optional in test envs
        pass
    try:
        from pad.branches import patch_depth_branch as _pd  # type: ignore
        md["patch_depth_state"] = {
            "subject_baseline": float(_pd._subject_baseline[0]),
            "baseline_n":       int(_pd._baseline_n[0]),
            "disabled_for_session": bool(_pd._disabled_for_session[0]),
            "secs_high":        (None if _pd._first_high_t[0] is None
                                 else round(time.monotonic() - _pd._first_high_t[0], 1)),
        }
    except Exception:  # nolint:silent-except patch_depth optional in test envs
        pass
    try:
        import rppg as _rppg  # type: ignore
        md["rppg_signal_quality"] = _rppg.get_signal_quality()
    except Exception:  # nolint:silent-except rppg optional in test envs
        pass
    # v9.8: integrity audit -- did any source JSONLs have malformed lines?
    # Surfaced from `_read_jsonl.last_skipped` (set during _affect's read).
    md["jsonl_skipped_lines"] = getattr(_affect, "last_skipped", 0)
    return md


# ── public api ──
def build_bundle(person_id: Optional[str] = None,
                 session_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": time.time(),
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id or os.environ.get("DODO_SESSION_ID", ""),
        "app_version": getattr(cfg, "APP_VERSION", "7.1.0"),
        # v9.8: per-bundle reproducibility metadata
        "v98_metadata": _v98_metadata(),
        "subject": _subject(person_id),
        "biometrics": _biometrics(person_id),
        "affect_timeline": _affect(person_id),
        "llm_messages": _llm(person_id),
        "audio_chunks": _audio(person_id),
        "radar": _radar(),
    }


def write_bundle(person_id: Optional[str] = None,
                 out_dir: Optional[str] = None,
                 session_id: Optional[str] = None) -> Dict[str, Any]:
    out_dir = out_dir or EXPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    bundle = build_bundle(person_id=person_id, session_id=session_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    who = person_id or "all"
    safe_who = "".join(c if c.isalnum() or c in "_-" else "_" for c in who)[:80]
    path = str(safe_child(out_dir, f"forensic_{safe_who}_{stamp}.json"))
    manifest_path = safe_child(out_dir, os.path.basename(path) + ".manifest.json")
    blob = json.dumps(bundle, indent=2, default=str).encode("utf-8")
    with open(path, "wb") as f:
        f.write(blob)
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    manifest = {
        "bundle": os.path.basename(path),
        "path": path,
        "bytes": len(blob),
        "sha256": digest,
        "generated_iso": bundle["generated_iso"],
        "subject": bundle["subject"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def export_qlora_jsonl(
    session_id: str,
    consent: bool,
    scrub_eeg: bool = True,
    min_pairs: int = 1000,
) -> Path:
    """
    v8 — Hardened QLoRA training pair export with explicit consent and EEG privacy.

    Parameters
    ----------
    session_id : str
        Session identifier for filtering pairs.
    consent : bool
        MUST be True (raises ValueError otherwise). Enforces explicit consent.
    scrub_eeg : bool
        If True, apply eeg_privacy.scrub() to affect_vector in each pair.
    min_pairs : int
        Minimum number of clean pairs required. Raises if fewer found.

    Returns
    -------
    path : Path
        Path to the exported JSONL file (sessions/<session_id>/qlora_pairs.jsonl).

    Raises
    ------
    ValueError
        If consent=False or fewer than min_pairs clean pairs.
    """
    if not consent:
        raise ValueError("QLoRA export requires explicit consent=True")
    out_dir = safe_child(cfg.SESSIONS_DIR, session_id)

    # Load training pairs
    llm_pairs_path = os.path.join(cfg.CALIBRATION_DIR, "llm_training_pairs.jsonl")
    if not os.path.exists(llm_pairs_path):
        raise ValueError(f"Training pairs not found: {llm_pairs_path}")

    pairs = _read_jsonl(llm_pairs_path, max_rows=50000)
    if not pairs:
        raise ValueError("No training pairs in llm_training_pairs.jsonl")

    # Filter by session_id
    filtered = [p for p in pairs
                if p.get("session_id") == session_id
                or (session_id in json.dumps(p.get("affect", {}), default=str))]

    if len(filtered) < min_pairs:
        raise ValueError(
            f"Insufficient clean pairs for session {session_id}: "
            f"found {len(filtered)}, need {min_pairs}"
        )

    # Scrub and validate pairs
    clean_pairs = []
    try:
        import eeg_privacy
    except ImportError:
        eeg_privacy = None  # type: ignore

    for pair in filtered:
        av = pair.get("affect_vector")
        if not av or not isinstance(av, dict):
            continue
        if not pair.get("response"):
            continue

        # Scrub EEG if enabled
        if scrub_eeg and eeg_privacy:
            if hasattr(eeg_privacy, "scrub"):
                av = eeg_privacy.scrub(av)
        else:
            # Fallback: drop EEG-like keys
            av = {k: v for k, v in av.items()
                  if not (k.startswith("eeg_") or k.startswith("p300_")
                          or k.startswith("proxy_"))}

        clean_pair = {
            "affect_vector": av,
            "room_context": pair.get("room_context", {}),
            "response": pair.get("response"),
            "person_id": pair.get("person_id"),
            "timestamp": pair.get("timestamp", time.time()),
        }
        clean_pairs.append(clean_pair)

    if len(clean_pairs) < min_pairs:
        raise ValueError(
            f"Insufficient clean pairs after scrubbing: "
            f"have {len(clean_pairs)}, need {min_pairs}"
        )

    # Write to sessions/<session_id>/qlora_pairs.jsonl
    os.makedirs(out_dir, exist_ok=True)
    out_path = safe_child(out_dir, "qlora_pairs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in clean_pairs:
            f.write(json.dumps(_safe(pair), default=str) + "\n")

    return out_path


def attach(app) -> None:
    """Mount FastAPI routes. Call once from main.py: forensic_export.attach(app)."""
    from fastapi import Query
    from fastapi.responses import JSONResponse, FileResponse

    @app.get("/forensic/export")
    def forensic_export(person_id: Optional[str] = Query(default=None),
                        download: bool = Query(default=False)):
        if download:
            manifest = write_bundle(person_id=person_id)
            return FileResponse(manifest["path"],
                                media_type="application/json",
                                filename=manifest["bundle"])
        return JSONResponse(build_bundle(person_id=person_id))

    @app.get("/forensic/manifest")
    def forensic_manifest(person_id: Optional[str] = Query(default=None)):
        manifest = write_bundle(person_id=person_id)
        return JSONResponse(manifest)

    @app.post("/qlora/export")
    def post_qlora_export(
        session_id: str = Query(...),
        consent: bool = Query(default=False),
        scrub_eeg: bool = Query(default=True),
    ):
        """v8 — Export QLoRA training pairs with explicit consent & EEG scrubbing."""
        try:
            out_path = export_qlora_jsonl(
                session_id=session_id,
                consent=consent,
                scrub_eeg=scrub_eeg,
                min_pairs=1000,
            )
            return JSONResponse({
                "ok": True,
                "path": str(out_path),
                "pairs": len([l for l in open(out_path).readlines()]),
            })
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--person-id", default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    m = write_bundle(person_id=a.person_id, out_dir=a.out_dir)
    print(json.dumps(m, indent=2))

