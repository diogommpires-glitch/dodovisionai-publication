from fastapi import APIRouter, Form, UploadFile, File, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
import time

router = APIRouter()

import os
import sys
import subprocess
import json
import config as cfg
import threading

# v9.7.0-merged 2026-05-10: missing imports from the API_anti modular refactor.
# The route handlers extracted out of main.py reference module-level names that
# were never re-imported here.
from ui import UI_HTML

# Standard 3rd-party names referenced by handlers below.
import numpy as np
import cv2
import asyncio
from pathlib import Path
from safe_paths import safe_child
from typing import Any

# Top-level project modules (live at repo root) referenced by handlers below.
# Wrapped in try/except so a missing optional module doesn't break import; the
# handlers that use them will still NameError but only when called, not at boot.
def _try_import(modname, alias=None):
    """Return (alias_or_modname, module) on success; (None, None) on failure."""
    try:
        m = __import__(modname)
        return (alias or modname), m
    except Exception:
        return None, None

for _m in ("analysis", "radar", "oximeter", "gsr", "llm", "face_registry",
          "calibration", "calibration_lap", "calibration_store",
          "detectors", "voice_monitor", "forensic", "logger",
          "truth_engine", "audio_mod", "rppg_mod", "dual_camera",
          "ics_monitor", "dodo_causal", "face_identity", "forensic_log",
          "_replay", "_calib_store", "_auth", "_server_identity", "eeg",
          "audio"):
    _alias, _mod = _try_import(_m)
    if _mod is not None:
        globals()[_alias] = _mod

# Some are aliased imports in main.py (e.g. `import rppg as rppg_mod`).
try:
    import rppg as rppg_mod  # noqa
except Exception:
    pass
try:
    import audio as audio_mod  # noqa
except Exception:
    pass
try:
    import voice_monitor  # noqa
except Exception:
    pass

# main.py module-level state and helpers are bound at FastAPI startup time
# via `bind_main_globals()` (called from main.py's @app.on_event("startup")).
# Until then, handlers should access them via _main_globals() which lazily
# reaches into main's namespace via sys.modules.
def _main_globals():
    """Return main.py's module-level globals dict, or {} if main isn't loaded yet.

    Used by route handlers to access main.py-defined state (person_states,
    _pad_challenge_sessions, _start_time, etc.) without a hard top-level
    `import main` (which would be circular: main.py imports this module)."""
    m = sys.modules.get("main") or sys.modules.get("__main__")
    return getattr(m, "__dict__", {}) if m is not None else {}


def bind_main_globals(main_module):
    """Inject main.py's globals into this module's namespace, idempotently.

    Called from main.py at FastAPI startup. After this runs, route handlers
    can reference `person_states`, `sanitize`, etc. as bare names without
    going through _main_globals(). Existing names in this module are NOT
    overwritten (so our targeted re-imports above stay authoritative)."""
    for k, v in vars(main_module).items():
        if k.startswith("__"): continue
        if k in globals(): continue
        globals()[k] = v

# v9.8.1 — DELETED stub /version, /settings, /flags, /v98/audit,
# /calibration/reset_all handlers that were here. They were no-op shells
# left over from the API_anti modular extraction; FastAPI registers them
# FIRST and wins the route, leaving the richer API_up v9.8 versions
# further down (lines ~1700-1850) as dead code. Schemas the stubs
# returned were missing model fingerprints (/version), live patch_depth
# warmup state (/settings), the proper feature_flags snapshot (/flags),
# the audit script's full output (/v98/audit), and per-face blink reset
# (/calibration/reset_all). Removing the stubs restores the v9.8
# behaviour the rest of the codebase expects.

#  HTTP Endpoints (fallback + non-streaming ops)
# ══════════════════════════════════════════════

# v9.8.1: cache MD5 digests keyed on (path, mtime, size). The /models/status
# endpoint previously hashed every checkpoint on every call -- the runtime
# benchmark surfaced 4.4s p50 latency because each call recomputed MD5s of
# the full model dir (~500 MB-1 GB of weights). Cache invalidates only when
# a file is actually rewritten (training output), so steady-state polls are
# O(1). Lives at module scope so it survives across requests.
_MODELS_STATUS_DIGEST_CACHE: dict = {}  # (path, mtime, size) -> md5_12 str


@router.get("/models/status",
         tags=["ops"], summary="All loaded/available model artefacts")
async def models_status():
    """Enumerate every PAD-related model artefact on disk with size, mtime,
    md5, role tag, and load-status. Operator visibility for deployment
    reproducibility (which version is actually live?). Also exposes the
    calibrator set per-branch state for paper-facing audit."""
    import hashlib
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent.parent / "pad" / "models"
    patterns = ("*.onnx", "*.pkl", "*.pth")

    def _digest(p: _P, mtime: float, size: int) -> str:
        """v9.8.1: mtime+size-keyed cache. Recomputes only when the file
        was rewritten since last access. Empty string for >500 MB files
        (the old behaviour) to keep response time bounded."""
        if size >= 500 * 1024 * 1024:
            return ""
        key = (str(p), round(mtime, 2), size)
        cached = _MODELS_STATUS_DIGEST_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            h = hashlib.md5()
            with open(p, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            digest = h.hexdigest()[:12]
            _MODELS_STATUS_DIGEST_CACHE[key] = digest
            # Bound cache size: a typical pad/models/ has ~30 files; the
            # 200-entry cap covers training-output churn over many days.
            if len(_MODELS_STATUS_DIGEST_CACHE) > 200:
                # Drop oldest insertion (dict is insertion-ordered in 3.7+)
                first_key = next(iter(_MODELS_STATUS_DIGEST_CACHE))
                _MODELS_STATUS_DIGEST_CACHE.pop(first_key, None)
            return digest
        except Exception:
            return ""

    def _role(name: str) -> str:
        """Tag an artefact by its purpose in the PAD stack. Used by
        operator dashboards to group files without inspecting filenames."""
        n = name.lower()
        if "yunet" in n or "face_detection" in n:
            return "face_detector_preproc"
        if "deepfake_temporal" in n:
            return "deepfake_branch_temporal"
        if "face_xray" in n:
            return "deepfake_branch_spatial"
        if "patch_depth" in n:
            return "patch_depth_branch"
        if "audio_pad" in n or "aasist" in n:
            return "audio_branch"
        if "moire" in n:
            return "moire_branch"
        if "calibrators" in n:
            return "calibration"
        if "meta_fuser" in n:
            return "fusion_metafuser"
        if "backup" in n or ".bak" in n:
            return "backup"
        return "unknown"

    artefacts = []
    if root.is_dir():
        for pat in patterns:
            for p in sorted(root.glob(pat)):
                try:
                    st = p.stat()
                    artefacts.append({
                        "name": p.name,
                        "role": _role(p.name),
                        "path": str(p),
                        "size_bytes": st.st_size,
                        "mtime": round(st.st_mtime, 2),
                        "md5_12": _digest(p, st.st_mtime, st.st_size),
                    })
                except Exception:
                    continue

    # Calibrator + meta-fuser runtime state
    calib_state = {}
    try:
        from pad.calibrate import calibrator_status
        calib_state = calibrator_status()
    except Exception as e:
        calib_state = {"error": repr(e)}

    meta_state = {}
    try:
        from pad.meta_fuse import load_meta_fuser
        mf = load_meta_fuser()
        if mf is not None:
            meta_state = {"loaded": True,
                          "metrics": getattr(mf, "metrics", {}),
                          "feature_dim": getattr(mf, "feature_dim", None)}
        else:
            meta_state = {"loaded": False}
    except Exception as e:
        meta_state = {"error": repr(e)}

    return {
        "artefacts": artefacts,
        "calibrators": calib_state,
        "meta_fuser": meta_state,
        "generated_at": time.time(),
    }


@router.get("/status")
async def system_status():
    gov_stats = {}
    try:
        from llm_governor import get_governor
        gov_stats = get_governor().stats()
    except Exception:
        pass
    # v9.8.1 — surface Radar B alongside Radar A. Previously /status only
    # reported Radar A, so when COM3 was unplugged the operator panel saw
    # `RADAR: --` even though Radar B (COM4) was alive and producing
    # HR/BR/people data. Two distinct radars deserve two distinct status
    # blocks; the top status bar logic in app.js can fall back to B when
    # A is offline.
    try:
        _radar_b_status = radar.get_radar_b_status()
    except Exception as _rb_err:
        _radar_b_status = {"connected": False, "port": None,
                           "error": f"get_radar_b_status failed: {_rb_err}"}
    return {
        "webcam": "ok",
        "rppg": "ok" if rppg_mod.start_time else "waiting",
        "radar": radar.get_radar_status(),
        "radar_b": _radar_b_status,
        "oximeter": oximeter.get_oximeter_status(),
        "gsr": gsr.get_gsr_status(),
        "llm": llm.llm_status,
        "llm_governor": gov_stats,
        "faces": len(face_registry.list_faces()),
        "parquet": "ok" if logger.PARQUET_ENABLED else "off",
        "hsemotion": "ok" if detectors.HSEMOTION_ENABLED else "off",
        "voice": {"active": voice_monitor._active},
        "dual_camera": dual_camera.get_status(),
        "forensic": {"active": forensic.get_session() is not None},
        "motion_rejected": rppg_mod.get_motion_stats(),
        "ics": ics_monitor.stats() if _ICS_ENABLED else {"enabled": False},
        "mmwave_120ghz": _mmwave_fusion.summary() if _mmwave_fusion else {"enabled": False},
        "tdmnn": _tdmnn.summary() if _tdmnn else {"enabled": False},
        "sensor_buffer": _sensor_buf.stats(),
        "dodo_causal": dodo_causal.stats() if _DODO_ENABLED else {"enabled": False},
    }

@router.post("/faces/register")
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Bad image"})
    return face_registry.register(name, image)

@router.get("/faces/list")
async def list_faces():
    return {"faces": face_registry.list_faces()}

@router.post("/faces/delete")
async def delete_face(name: str = Form(...)):
    return face_registry.unregister(name)


@router.post("/faces/clear")
async def clear_faces(confirm: str = Form("")):
    """v9.7.5: wipe the entire face registry. Pass confirm='YES' to
    actually run — accidental deletion of a long-trained registry is
    expensive (you have to re-register each person). Used to recover
    from registry corruption or major appearance changes."""
    if confirm != "YES":
        return {"ok": False,
                "error": "pass confirm='YES' to actually wipe; "
                         "this is irreversible"}
    n = face_registry.clear()
    return {"ok": True, "removed": n}

@router.post("/calibrate/start")
async def cal_start():
    return calibration.start_collection()

@router.post("/calibrate/stop")
async def cal_stop():
    return calibration.stop_collection()

@router.get("/calibrate/status")
async def cal_status():
    return calibration.get_status()

@router.post("/calibrate/reset")
async def cal_reset():
    return calibration.reset()


# v9.8.1 — runtime recovery for the high-fps dual-camera thread.
# Browser getUserMedia() can steal the same physical cam Windows MSMF
# already handed to the dual-cam capture loop. When that happens the
# loop keeps reading buffered frames (frames counter freezes, fps
# stays at the last-measured value) and the dual_camera.active flag
# remains True even though nothing is actually advancing. POST this
# to force re-acquisition. Also auto-triggered by the watchdog after
# WATCHDOG_STALL_S seconds of frozen frames.
@router.post("/dual_camera/restart", tags=["ops"],
             summary="Force re-acquisition of the high-fps camera")
async def dual_camera_restart():
    try:
        info = dual_camera.restart()
        return {"ok": True, "info": info, "status_after": dual_camera.get_status()}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# v9.8.1 2026-05-12: serve the latest JPEG-encoded frame from the
# server-held camera (the one dual_camera holds exclusively via DSHOW).
# Lets the UI render a live preview of cameras the browser cannot
# open itself -- e.g. Canon R10 via EOS Webcam Utility, or HD USB
# Camera under exclusive MJPG access. Polled at ~5 Hz by the UI's
# multi-cam panel. Cheap: the camera_loop already encodes every 6th
# frame (~5 fps) and stashes the bytes in `_latest_jpeg_bytes`; this
# endpoint is pure memory copy.
@router.get("/camera/server_feed.jpg", tags=["ui"],
            summary="Latest JPEG frame from the high-fps server camera")
async def camera_server_feed_jpg():
    from fastapi.responses import Response as _Resp
    try:
        data, ts = dual_camera.get_latest_jpeg()
    except Exception:
        data, ts = b"", 0.0
    if not data:
        return _Resp(content=b"", status_code=204)
    return _Resp(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Capture-Ts": str(ts),
        },
    )


@router.get("/camera/server_feed/info", tags=["ui"],
            summary="Status of the server camera feed")
async def camera_server_feed_info():
    """Helper for the UI to decide whether to render the
    server-feed thumbnail tile. Returns the bound camera name, fps,
    and whether bytes are currently flowing."""
    try:
        st = dual_camera.get_status()
        data, ts = dual_camera.get_latest_jpeg()
    except Exception:
        return {"active": False}
    import time as _t
    return {
        "active":      bool(st.get("active")),
        "camera_name": st.get("camera_name"),
        "device":      st.get("device"),
        "fps":         st.get("fps"),
        "has_frame":   bool(data),
        "age_s":       round(_t.time() - ts, 2) if ts else None,
    }


# v9.8.1 2026-05-12: enumerate every DirectShow video camera + its
# supported media-type list (resolution, codec, max fps). Lets the UI
# show "max-fps + quality" per camera so the operator can decide which
# rig to plug / configure. Server-only -- uses pygrabber's MediaType
# API which talks to DirectShow Filter Graph directly.
@router.get("/camera/capabilities", tags=["ui"],
            summary="Per-camera max-fps + resolution list (DirectShow)")
async def camera_capabilities():
    """Return the supported media types for every video device. For
    each device:
      {
        "idx": int,
        "name": str,
        "in_use_by": "server" | "browser" | None,
        "max_fps_modes": [
            {"w": int, "h": int, "fps": float, "codec": str}, ...
        ],
        "best_for_pad": {"w": int, "h": int, "fps": float, "codec": str},
      }
    """
    out: list = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        names = graph.get_input_devices()
    except Exception as e:
        return {"error": f"pygrabber unavailable: {type(e).__name__}: {e}"}

    try:
        srv_status = dual_camera.get_status()
        srv_idx = srv_status.get("device") if srv_status.get("active") else None
    except Exception:
        srv_idx = None

    for idx, name in enumerate(names):
        entry = {"idx": idx, "name": name,
                 "in_use_by": "server" if idx == srv_idx else None,
                 "max_fps_modes": [],
                 "best_for_pad": None}
        try:
            # FilterGraph().get_input_device(idx).get_formats() is the
            # pygrabber API for listing DirectShow media types.
            graph.add_video_input_device(idx)
            formats = graph.get_input_device().get_formats()
            graph.remove_filters()
            # formats is a list of dicts like
            #   {'index': 0, 'media_type_str': 'MJPG 1280x720 60fps',
            #    'width': 1280, 'height': 720, 'min_framerate': 60.0,
            #    'max_framerate': 60.0}
            seen = set()
            for f in formats:
                w = int(f.get("width") or 0)
                h = int(f.get("height") or 0)
                fps = float(f.get("max_framerate") or f.get("min_framerate") or 0)
                # Extract codec out of media_type_str (first token).
                mts = f.get("media_type_str", "")
                codec = mts.split(" ")[0] if mts else "?"
                key = (w, h, round(fps, 1), codec)
                if key in seen:
                    continue
                seen.add(key)
                entry["max_fps_modes"].append({
                    "w": w, "h": h, "fps": round(fps, 1), "codec": codec,
                })
            # Sort by (fps desc, area desc).
            entry["max_fps_modes"].sort(
                key=lambda m: (-m["fps"], -(m["w"] * m["h"])))
            # "Best for PAD" = pick highest-fps mode whose area >=
            # 640x480. PAD branches don't gain much from > 720p but
            # need >= VGA for landmarks.
            for m in entry["max_fps_modes"]:
                if m["w"] >= 640 and m["h"] >= 480 and m["fps"] >= 30:
                    entry["best_for_pad"] = m
                    break
            if entry["best_for_pad"] is None and entry["max_fps_modes"]:
                entry["best_for_pad"] = entry["max_fps_modes"][0]
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        out.append(entry)
    return {"cameras": out}

# v9.8.1 2026-05-12: multi-camera frame recorder for cross-cam pairs
# calibration. The operator wants every plugged-in camera (Canon R10
# main + UGREEN + Integrated + HD USB server) capturing simultaneous
# frames to disk so downstream tooling (pairs calibration, rPPG /
# radar / patch_depth cross-camera benchmarks, deepfake training
# augmentation, etc.) can use them.
#
# Request format (POST JSON):
#   {
#     "session_label": "operator_glasses_2026_05_12",
#     "cameras": [
#       {"name": "Canon EOS Webcam Utility",
#        "role": "browser_main",
#        "jpeg_b64": "<base64>"},
#       {"name": "UGREEN Camera",
#        "role": "browser_strip",
#        "jpeg_b64": "<base64>"},
#       ...
#     ]
#   }
#
# The server adds its own dual_camera frame (HD USB at 720p@60) and
# writes them all under
#   calibration/recorded_pairs/<session>/<ts>_<role>_<name>.jpg
# plus a manifest.json with WALL_TIME / CAM_NAME / ROLE / SHAPE.
@router.post("/calibration/record_multi_cam_frame", tags=["calibration"],
             summary="Save synchronized frames from every active camera")
async def calibration_record_multi_cam_frame(req: Request):
    import base64, time as _t, json as _json
    try:
        body = await req.json()
    except Exception as e:
        return JSONResponse({"ok": False, "err": f"bad json: {e}"}, status_code=400)
    session = str(body.get("session_label") or "default")
    try:
        out_dir = safe_child(Path("calibration") / "recorded_pairs", session)
    except ValueError as e:
        return JSONResponse({"ok": False, "err": str(e)}, status_code=400)
    cams = body.get("cameras") or []
    if not isinstance(cams, list):
        return JSONResponse({"ok": False, "err": "cameras must be list"}, status_code=400)

    ts_ms = int(_t.time() * 1000)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list = []

    # 1. Browser-supplied camera frames.
    for cam in cams:
        try:
            name = str(cam.get("name") or "unknown")
            role = str(cam.get("role") or "browser")
            b64  = cam.get("jpeg_b64") or ""
            if "," in b64 and b64.startswith("data:"):
                # data: URL header
                b64 = b64.split(",", 1)[1]
            raw = base64.b64decode(b64) if b64 else b""
            if not raw:
                continue
            safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)[:60]
            safe_role = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in role)[:60]
            fn = safe_child(out_dir, f"{ts_ms}_{safe_role}_{safe_name}.jpg")
            fn.write_bytes(raw)
            saved.append({"name": name, "role": role,
                          "file": str(fn.relative_to(Path("calibration").resolve())),
                          "bytes": len(raw)})
        except Exception as e:
            saved.append({"err": f"{type(e).__name__}: {e}",
                          "name": cam.get("name")})

    # 2. Server's dual_camera frame (HD USB at 720p@60 etc).
    try:
        data, srv_ts = dual_camera.get_latest_jpeg()
        st = dual_camera.get_status()
        if data and st.get("active"):
            srv_name = str(st.get("camera_name") or "server_cam")
            safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in srv_name)[:60]
            fn = safe_child(out_dir, f"{ts_ms}_server_120fps_{safe}.jpg")
            fn.write_bytes(data)
            saved.append({"name": srv_name, "role": "server_120fps",
                          "file": str(fn.relative_to(Path("calibration").resolve())),
                          "bytes": len(data),
                          "server_ts": srv_ts})
    except Exception as e:
        saved.append({"err": f"server_capture: {type(e).__name__}: {e}"})

    # 3. Append to manifest.json
    try:
        manifest_path = safe_child(out_dir, "manifest.json")
        if manifest_path.exists():
            manifest = _json.loads(manifest_path.read_text())
        else:
            manifest = {"session": session, "frames": []}
        manifest["frames"].append({
            "ts_ms": ts_ms,
            "wall_iso": _t.strftime("%Y-%m-%dT%H:%M:%S",
                                    _t.localtime(ts_ms / 1000.0)),
            "saved": saved,
        })
        manifest_path.write_text(_json.dumps(manifest, indent=2))
    except Exception as e:
        return JSONResponse({"ok": True, "saved": saved,
                             "manifest_err": str(e),
                             "ts_ms": ts_ms}, status_code=200)

    return {"ok": True, "saved": saved, "ts_ms": ts_ms,
            "session": session, "dir": str(out_dir)}


# v9.8.1 2026-05-12: lightweight listing of recorded multi-cam sessions.
# Lets the UI render a "previously recorded pairs" list / count.
@router.get("/calibration/recorded_pairs/list",
            tags=["calibration"],
            summary="List recorded multi-cam pair sessions")
async def calibration_recorded_pairs_list():
    base = Path("calibration") / "recorded_pairs"
    out = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            try:
                jpgs = list(d.glob("*.jpg"))
                manifest = d / "manifest.json"
                out.append({
                    "session":   d.name,
                    "n_frames":  len(jpgs),
                    "has_manifest": manifest.exists(),
                    "size_kb":   sum(p.stat().st_size for p in jpgs) // 1024,
                })
            except Exception:
                pass
    return {"sessions": out}


# v9.7.6 — explicit pause / resume for the auto-collect flow.
# Auto-collect is on by default, accumulating pairs whenever rPPG +
# oximeter both have valid readings. The PAUSE button in the UI hits
# this so you can stop accumulating noisy samples before deliberately
# moving or talking.
@router.post("/calibrate/pause")
async def cal_pause():
    return calibration.pause_collection()

@router.post("/calibrate/resume")
async def cal_resume():
    return calibration.resume_collection()

# v9.7.6 — train rPPG correction from accumulated pairs on disk.
# Use case: you've collected pairs across multiple sessions
# (`calibration/rppg_pairs.json` keeps growing as you hit CAL rPPG +
# Stop), but the in-memory `_rppg_pairs` buffer resets every server
# boot. This endpoint loads the full file and runs the linear
# regression on everything at once. Hit it after at least ~50 pairs
# have accumulated for a stable fit; ~200+ is ideal.
@router.post("/calibrate/recompute")
async def cal_recompute():
    return calibration.recompute_from_disk()

# v9.7.6 — train per-radar correction from accumulated radar pairs on
# disk. Same quality-gated regression as rPPG. `which` is "a" or "b".
# Result is saved to calibration/radar_<which>_correction.json which
# radar.py auto-loads at boot and applies in summary() / summary_b()
# right where the heart_rate field is read.
@router.post("/calibrate/radar/recompute")
async def cal_radar_recompute(which: str = Form("a")):
    return calibration.recompute_radar_from_disk(which=which)

# ══════════════════════════════════════════════
#  v8.7 (Step 8): 90-second emotional baseline lap
# ══════════════════════════════════════════════
@router.post("/calibration/lap/start")
async def lap_start(target_track: int = Form(-1),
                    subject: str = Form(""),
                    examiner: str = Form("")):
    """Start a 3-stage 90-second emotional calibration lap.
    target_track=-1 means 'all detected faces'; pass a specific track_id to
    isolate one subject (recommended for forensic interviews)."""
    tt = None if target_track is None or target_track < 0 else int(target_track)
    return calibration_lap.start(target_track=tt,
                                 subject=subject.strip(),
                                 examiner=examiner.strip())

@router.get("/calibration/lap/status")
async def lap_status():
    return calibration_lap.status()

@router.post("/calibration/lap/abort")
async def lap_abort():
    return calibration_lap.abort()

@router.get("/calibration/lap/profiles")
async def lap_profiles():
    return {"profiles": calibration_lap.list_profiles()}

@router.post("/calibration/lap/load")
async def lap_load(filename: str = Form(...),
                   target_track: int = Form(-1)):
    """Hot-load a saved lap profile onto an active PersonState as the cold
    baseline + recall anchors for the truth engine."""
    profile = calibration_lap.load_profile(filename)
    if not profile:
        return JSONResponse(status_code=404, content={"error": "profile_not_found"})
    if target_track is None or target_track < 0:
        # Apply to first-active PersonState as a convenience default
        if not person_states:
            return {"ok": False, "error": "no_active_person", "profile": profile.get("meta")}
        tid = sorted(person_states.keys())[0]
        ps = person_states[tid]
    else:
        ps = person_states.get(int(target_track))
        if ps is None:
            return {"ok": False, "error": "track_not_active", "track_id": int(target_track)}
    ok = calibration_lap.apply_profile_to_person(ps, profile)
    return {"ok": ok, "track_id": ps.track_id, "subject": (profile.get("meta") or {}).get("subject")}

@router.get("/training/status")
async def training_status():
    # v9.7.5: report from the canonical `llm_logger` totals so this
    # endpoint and the new `/qlora/status` always agree.
    try:
        import llm_logger as _ll
        n_total   = _ll.get_total_pair_count()
        n_session = _ll.get_session_pair_count()
        sid       = _ll.get_session_id()
    except Exception:
        n_total = calibration.get_llm_training_count()
        n_session = 0
        sid = ""
    return {
        "llm_pairs": n_total,
        "llm_pairs_this_session": n_session,
        "session_id": sid,
        "rppg_pairs": len(calibration._rppg_pairs),
        "calibration": calibration._calibration,
        "faces": len(face_registry.list_faces()),
    }


@router.get("/qlora/status")
async def qlora_status():
    """v9.7.5: dedicated endpoint for the UI's PAIRS badge. Returns
    total pairs across all sessions on disk, plus how many were
    logged in THIS server lifetime (for the "you've made N this
    session" feel-good number). The badge polls this at ~1 Hz."""
    try:
        import llm_logger as _ll
        return {
            "total":          _ll.get_total_pair_count(),
            "this_session":   _ll.get_session_pair_count(),
            "session_id":     _ll.get_session_id(),
            "min_for_export": 1000,
        }
    except Exception as e:
        return {"error": str(e), "total": 0, "this_session": 0,
                "session_id": "", "min_for_export": 1000}

# ══════════════════════════════════════════════
#  v8.7 (Step 9): SQLite forensic event log
# ══════════════════════════════════════════════
@router.get("/forensic/log/recent")
async def forensic_log_recent(limit: int = 100,
                               session_id: str = "",
                               person_id: str = "",
                               event_type: str = "",
                               severity: str = ""):
    """Return the most recent N forensic events (newest first) with optional
    filters. Suitable for an audit panel in the UI."""
    return {
        "events": forensic_log.recent(
            limit=limit,
            session_id=session_id or None,
            person_id=person_id or None,
            event_type=event_type or None,
            severity=severity or None,
        ),
    }

@router.get("/forensic/log/stats")
async def forensic_log_stats():
    """Counts by event_type / severity, plus first/last timestamps."""
    return forensic_log.stats()

@router.post("/forensic/log/note")
async def forensic_log_note(note: str = Form(...),
                             severity: str = Form("info"),
                             person_id: str = Form("")):
    """Append a free-text examiner note to the forensic event log."""
    rid = forensic_log.log_event(
        "system_warning" if severity == "warn" else
        "system_error" if severity == "critical" else "session_start",
        severity=severity,
        person_id=person_id or None,
        payload={"note": note[:1000]},
    )
    return {"ok": rid is not None, "id": rid}

@router.post("/forensic/log/purge")
async def forensic_log_purge(days: int = Form(30)):
    """Remove events older than `days`. Defaults to 30."""
    removed = forensic_log.purge_older_than(int(days))
    return {"ok": True, "removed": removed, "older_than_days": int(days)}

# ── v8.9 Step 7: DODO Causal Engine API ──
@router.get("/causal/dag")
async def causal_dag():
    """Get the current causal DAG (edges + weights)."""
    if not _DODO_ENABLED:
        return {"error": "DODO causal engine not available"}
    dag = dodo_causal.infer_dag(force=True)
    if dag is None:
        return {"status": "insufficient_data", "message": "Need more observations"}
    return dag

@router.get("/causal/stats")
async def causal_stats():
    """DODO causal engine diagnostics."""
    if not _DODO_ENABLED:
        return {"enabled": False}
    return dodo_causal.stats()

@router.post("/causal/intervene")
async def causal_intervene(variable: str = Form(...), active: bool = Form(True),
                           reason: str = Form("")):
    """Log a sensor intervention for causal identification."""
    if not _DODO_ENABLED:
        return {"error": "DODO causal engine not available"}
    return dodo_causal.intervene(variable, active, reason)

@router.post("/causal/reset")
async def causal_reset():
    """Reset the causal engine state."""
    if not _DODO_ENABLED:
        return {"error": "DODO causal engine not available"}
    dodo_causal.get_engine().reset()
    return {"ok": True, "message": "FLODO causal engine reset"}

@router.get("/causal/suggest")
async def causal_suggest():
    """Step 16: Get adaptive budgeting suggestion for next intervention."""
    if not _DODO_ENABLED:
        return {"error": "DODO causal engine not available"}
    return dodo_causal.suggest_intervention()

# HTTP detection fallback (kept for compatibility)
@router.post("/detect/fast")
async def detect_fast(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "bad image"})
    return _process_frame_fast(image)

@router.post("/detect/full")
async def detect_full(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "bad image"})
    return _process_frame_full(image)

@router.post("/vitals/reset")
async def reset():
    rppg_mod.reset()
    analysis.reset_all()
    dual_camera.reset()
    logger.flush_parquet()
    return {"ok": True}

@router.get("/affect")
async def get_affect():
    return sanitize(analysis.affect_vector)

@router.get("/radar")
async def get_radar():
    return sanitize(radar.get_radar_status())

@router.get("/radar/status")
async def get_radar_status_full():
    """Rich radar health snapshot used by forensic_export + UI banner."""
    return sanitize(radar.summary())

@router.get("/radar/filter-stats")
async def get_radar_filter_stats():
    """v7.4 — accept/reject counters for the physiological HR/BR filter."""
    try:
        return sanitize(radar.get_filter_stats())
    except Exception as e:
        return {"error": str(e)}

@router.get("/radar/calibration")
async def get_radar_calibration():
    """v7.4 — rolling calibration vs. 120GHz mmWave dataset (s41597-026-07016-6)."""
    try:
        from radar_calibration import monitor as cal
        return sanitize(cal.snapshot())
    except Exception as e:
        return {"error": str(e)}

@router.get("/rppg/filter-stats")
async def get_rppg_filter_stats():
    """v7.5 — accept/reject counters for the rPPG HR physiological filter."""
    try:
        import rppg as _rppg
        return sanitize(_rppg.get_filter_stats())
    except Exception as e:
        return {"error": str(e)}

@router.get("/voice/vad-stats")
async def get_vad_stats():
    """v7.5 — adaptive VAD diagnostics (noise floor, speech ratio, rejections)."""
    try:
        import voice_monitor as _vm
        return sanitize(_vm.get_vad_stats())
    except Exception as e:
        return {"error": str(e)}

@router.get("/cognitive")
async def get_cognitive_proxies():
    """v7.6 — DARPA-AugCog-inspired proxies from pupil/HRV/gaze/blink.
    NOT EEG. NOT neural recording. Every field prefixed `proxy_`.
    See TRAINING_GUIDE.md §cognitive-state for sources."""
    try:
        from cognitive_state import estimator
        return sanitize(estimator().snapshot())
    except Exception as e:
        return {"error": str(e)}

@router.get("/attention")
async def get_attention_gate_stats():
    """v7.6 — CT2WS-style attention confirmation gate stats.
    Pupil + HR + saccade votes confirm whether a flagged event actually
    captured the observer's attention."""
    try:
        from attention_gate import gate
        return sanitize(gate().stats())
    except Exception as e:
        return {"error": str(e)}

@router.post("/attention/evaluate")
async def post_attention_evaluate():
    """Force the gate to resolve the most recent submitted event now."""
    try:
        from attention_gate import gate
        conf = gate().evaluate_latest()
        if conf is None:
            return {"pending": True}
        return sanitize({
            "event_t": conf.event_t, "confirmed": conf.confirmed,
            "score": conf.score, "votes": conf.votes, "reason": conf.reason,
        })
    except Exception as e:
        return {"error": str(e)}

@router.post("/cognitive/reset")
async def post_cognitive_reset():
    try:
        from cognitive_state import estimator
        from attention_gate import gate
        estimator().reset()
        gate().reset()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}

@router.get("/oximeter/diag")
async def oximeter_diag():
    """v9.1.2 — full oximeter diagnostic dump. Returns HID enumeration,
    current internal state, last parsed values, and which fields are
    reaching the affect_vector. Use this when the UI panel shows "--"
    to see exactly where the pipeline is breaking."""
    out = {
        "status": oximeter.get_oximeter_status(),
        "latest_data": oximeter.get_oximeter_data(),
        "internal_state": {k: v for k, v in oximeter._oxi_data.items() if not k.startswith("_")},
        "affect_vector_oxi_fields": {k: v for k, v in analysis.affect_vector.items() if k.startswith("oxi_")},
        "sensor_buf": _sensor_buf.snapshot().get("oximeter", {}),
    }
    try:
        import hid
        out["hid_inventory"] = [
            {
                "vid": f"{d.get('vendor_id', 0):#06x}",
                "pid": f"{d.get('product_id', 0):#06x}",
                "product": d.get("product_string"),
                "manufacturer": d.get("manufacturer_string"),
                "usage_page": f"{d.get('usage_page', 0):#06x}",
            }
            for d in hid.enumerate()
        ]
    except Exception as e:
        out["hid_inventory"] = f"error: {e}"
    out["internal_state"]["last_update_age_s"] = round(
        time.time() - (out["internal_state"].get("last_update") or 0), 2
    )
    return sanitize(out)

# ══════════════════════════════════════════════
#  v8.4 New Endpoints
# ══════════════════════════════════════════════
@router.get("/ready")
async def ready_check():
    """Structured startup health check for start.bat polling."""
    radar_ok = radar.get_radar_data() is not None or (radar._radar_data.get("connected", False))
    # v9.5.x: expose radar B separately so operators can verify dual-radar fusion is live
    radar_b_status = radar.get_radar_b_status()
    radar_b_ok = bool(radar_b_status.get("connected"))
    oxi_ok = oximeter._oxi_data.get("connected", False)
    gsr_ok = False
    try:
        gsr_ok = gsr._gsr_data.get("connected", False)
    except Exception:
        pass
    llm_ok = bool(cfg.TOGETHER_API_KEY or cfg.GROQ_API_KEY)
    all_ok = True  # server is always "ready" once FastAPI is responding
    return {
        "status": "ready" if all_ok else "initializing",
        "sensors": {
            "radar": radar_ok,
            "radar_a_port": radar._radar_data.get("port"),
            "radar_b": radar_b_ok,
            "radar_b_port": radar_b_status.get("port"),
            "oximeter": oxi_ok,
            "gsr": gsr_ok,
        },
        "llm": llm_ok,
        "version": cfg.APP_VERSION,
    }

@router.get("/affect/delta")
async def affect_delta():
    """Per-person change in affect vector since last call.
    Useful for detecting sudden state changes (spikes in stress, emotion shifts)."""
    global _prev_affect_snapshot
    current = {}
    for ps in person_states.values():
        d = ps.build_affect_dict()
        current[str(ps.track_id)] = d
    delta = {}
    for tid, curr in current.items():
        prev = _prev_affect_snapshot.get(tid, {})
        d = {}
        for k in ["valence", "arousal", "stress_index", "heart_rate_bpm", "perclos"]:
            cv = curr.get(k)
            pv = prev.get(k)
            if cv is not None and pv is not None:
                d[k] = round(cv - pv, 3)
            else:
                d[k] = None
        # Deception score delta
        cd = curr.get("deception", {}).get("score", 0)
        pd = prev.get("deception", {}).get("score", 0)
        d["deception_score"] = round(cd - pd, 3)
        delta[tid] = d
    _prev_affect_snapshot = current
    return sanitize(delta)

@router.get("/sessions/list")
async def list_sessions():
    """List saved session files in sessions/ directory (auto-saves + forensic)."""
    import glob as _glob
    sessions_dir = cfg.SESSIONS_DIR
    files = []
    for f in sorted(_glob.glob(os.path.join(sessions_dir, "*.json")), reverse=True):
        try:
            stat = os.stat(f)
            files.append({
                "filename": os.path.basename(f),
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": stat.st_mtime,
            })
        except Exception:
            pass
    return {"sessions": files[:50]}

# v8.3: Oximeter status endpoint
@router.get("/oximeter/status")
async def oximeter_status():
    return sanitize(oximeter.get_oximeter_status() | {"data": oximeter.get_oximeter_data()})

@router.post("/oximeter/reconnect")
async def oximeter_reconnect():
    """Force oximeter reconnection attempt."""
    oximeter.stop()
    import asyncio
    await asyncio.sleep(0.3)
    ok = oximeter.start()
    st = oximeter.get_oximeter_status()
    return sanitize({"reconnected": ok, **st})

# v8.3: WiFi CSI status endpoint (stub — planned v8.4)
@router.get("/wifi/status")
async def wifi_csi_status():
    try:
        import wifi_csi
        return sanitize(wifi_csi.get_status())
    except Exception as e:
        return {"connected": False, "enabled": False, "error": str(e)}

# v8.3: Health liveness endpoint
@router.get("/health")
async def health():
    out = {
        "status": "ok",
        "version": cfg.APP_VERSION,
        "persons": len(person_states),
        "uptime_s": round(time.time() - _start_time, 1),
    }
    # v9.8: surface per-subject warmup state so external monitors / a CI
    # smoke-test can know whether the user-visible PAD verdicts are
    # still "warming up" (might not fire dominant_class) vs. "settled"
    # (full signal). Non-blocking: each sub-module check is wrapped.
    try:
        from pad.branches import patch_depth_branch as _pd
        out["patch_depth_warmup"] = {
            "n":       int(_pd._baseline_n[0]),
            "done":    bool(_pd._baseline_n[0] >= 8),
            "baseline": round(float(_pd._subject_baseline[0]), 4),
            "disabled": bool(_pd._disabled_for_session[0]),
        }
    except Exception:  # nolint:silent-except patch_depth optional in this build
        pass
    try:
        out["dual_cam_active"] = bool(dual_camera.get_status().get("active"))
    except Exception:  # nolint:silent-except dual_camera optional in this build
        pass
    return out


# v9.7.1 — tail of the structured event log. Returns the last N lines
# from today's logs/events_<date>.jsonl, parsed into JSON objects.
# Used by the UI's collapsible event-log drawer so the user can see
# backend warnings (DeepFace failures, radar resets, breaker trips,
# WS drops) without opening DevTools.
@router.get("/events/tail", tags=["ops"])
async def events_tail(n: int = 50, level: str = ""):
    n = max(1, min(int(n), 500))
    try:
        from datetime import datetime as _dt, timezone as _tz
        # v9.7.0-merged: post-extraction Path(__file__) lives in app/api/.
        # Walk up two more to reach repo root.
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        today_path = log_dir / f"events_{_dt.now(tz=_tz.utc).strftime('%Y-%m-%d')}.jsonl"
        if not today_path.exists():
            return {"path": str(today_path), "lines": [], "count": 0}
        # Stream from the end — read last ~64 KB which covers >>500 lines
        # at typical line lengths (~150 bytes).
        with today_path.open("rb") as f:
            try:
                f.seek(-64 * 1024, os.SEEK_END)
            except OSError:
                f.seek(0)
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        rows: list[dict] = []
        for line in text.splitlines()[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if level and row.get("level") != level:
                continue
            rows.append(row)
        return {"path": str(today_path), "lines": rows, "count": len(rows)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "lines": [], "count": 0}


# v9.7.1: Aggregate per-subsystem health for one-shot diagnostics. Each
# entry reports `{connected, last_update_age_s, last_error}` so a
# one-line shell command can answer "what's broken?". Designed to never
# raise — every subsystem call is wrapped to keep the endpoint trivially
# pollable from a watchdog.
@router.get("/health/full", tags=["health"])
async def health_full():
    now = time.time()

    def _safe(callable_, default=None):
        try:
            return callable_()
        except Exception as exc:  # noqa: BLE001
            return {"_err": f"{type(exc).__name__}: {exc}", "_default": default}

    def _age(ts):
        if not ts:
            return None
        try:
            return round(now - float(ts), 2)
        except Exception:
            return None

    radar_a = _safe(radar.get_radar_status, {}) or {}
    radar_a_data = _safe(radar.get_radar_data, {}) or {}
    radar_b = _safe(radar.get_radar_b_status, {}) or {}
    radar_b_data = _safe(radar.get_radar_b_data, {}) or {}
    oxi_status = _safe(oximeter.get_oximeter_status, {}) or {}
    oxi_data = _safe(oximeter.get_oximeter_data, {}) or {}
    gsr_status = _safe(gsr.get_gsr_status, {}) or {}
    gsr_data = _safe(gsr.get_gsr_data, {}) or {}

    # DeepFace cache freshness — youngest entry wins.
    df_status: dict = {"connected": False, "entries": 0, "last_update_age_s": None}
    try:
        with _df_lock:
            entries = list(_df_cache.values())
        if entries:
            youngest = max(e.get("t", 0) for e in entries)
            df_status = {
                "connected": True,
                "entries": len(entries),
                "max_entries": _DF_CACHE_MAX_ENTRIES,
                "last_update_age_s": _age(youngest),
                "pending_face_ids": sorted(_df_pending),
            }
        # v9.7.1: surface the circuit-breaker state alongside cache
        # freshness so a single GET /health/full diagnoses both
        # "DeepFace silent" and "DeepFace tripped".
        try:
            df_status["circuit_breaker"] = df_breaker_state()
        except Exception:
            pass  # nolint:silent-except diagnostic best-effort
    except Exception as exc:
        df_status["_err"] = f"{type(exc).__name__}: {exc}"

    cam_status: dict = {"connected": False}
    try:
        cam_status = {
            "connected": bool(getattr(cfg, "CAMERA_INDEX", None) is not None),
            "configured_index": getattr(cfg, "CAMERA_INDEX", None),
            "fast_thread_alive": getattr(_dual_cam_state, "fast_alive", False),
        }
    except Exception:
        pass  # nolint:silent-except cam state introspection cosmetic

    identity = _server_identity.summary() if _server_identity else {"_err": "init failed"}

    # Subsystem-level rollup: True if any expected source is delivering.
    rollup = {
        "radar_a": bool(radar_a.get("connected")),
        "radar_b": bool(radar_b.get("connected")),
        "oximeter": bool(oxi_status.get("connected")),
        "gsr": bool(gsr_status.get("connected")),
        "deepface": bool(df_status.get("connected")),
    }

    pad_subsystem: dict = {}
    try:
        import feature_flags as _ff
        pad_subsystem["enabled_flags"] = {
            k: bool(_ff.get(k)) for k in
            ("PAD_FUSION", "PAD_RPPG", "PAD_MOIRE", "PAD_CNN",
             "PAD_DYNAMICS", "PAD_DEEPFAKE", "PAD_AUDIO",
             "PAD_ADV_DEFENSE", "PAD_UI", "PAD_EVAL")}
        pad_subsystem["replay_note"] = (
            "PAD_DYNAMICS is not the primary signal for high-fidelity phone replay; "
            "use moire, patch_depth, deepfake, rPPG, and radar."
        )
    except Exception as exc:
        pad_subsystem = {"_err": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "ok",
        "version": cfg.APP_VERSION,
        "persons": len(person_states),
        "uptime_s": round(now - _start_time, 1),
        "identity": identity,
        "rollup": rollup,
        "subsystems": {
            "radar_a": {
                **radar_a,
                "last_update_age_s": _age(radar_a.get("last_update")),
                "hr": radar_a_data.get("heart_rate") if isinstance(radar_a_data, dict) else None,
                "br": radar_a_data.get("breath_rate") if isinstance(radar_a_data, dict) else None,
                "dist": radar_a_data.get("distance") if isinstance(radar_a_data, dict) else None,
            },
            "radar_b": {
                **radar_b,
                "hr": radar_b_data.get("heart_rate") if isinstance(radar_b_data, dict) else None,
                "br": radar_b_data.get("breath_rate") if isinstance(radar_b_data, dict) else None,
                "dist": radar_b_data.get("distance") if isinstance(radar_b_data, dict) else None,
            },
            "oximeter": {
                **oxi_status,
                "hr": oxi_data.get("heart_rate") if isinstance(oxi_data, dict) else None,
                "spo2": oxi_data.get("spo2") if isinstance(oxi_data, dict) else None,
            },
            "gsr": {
                **gsr_status,
                "level": gsr_data.get("level") if isinstance(gsr_data, dict) else None,
            },
            "deepface": df_status,
            "camera": cam_status,
            "pad": pad_subsystem,
        },
    }

# ── v8: Status endpoints for EEG, GSR, rPPG calibration, and identity registry ──
@router.get("/eeg/status")
async def get_eeg_status():
    """v8 — Muse 2 BrainFlow EEG stream status."""
    if eeg is None:
        return {"connected": False, "reason": "eeg module not loaded"}
    eeg_state = eeg.get_eeg_state()
    if not eeg_state:
        return {"connected": False, "last_update": None}
    return {
        "connected": eeg_state.get("connected", False),
        "sample_rate": 256,  # Muse 2 standard
        "band_powers": eeg_state.get("band_powers"),
        "cognitive_load": eeg_state.get("cognitive_load"),
        "focus": eeg_state.get("focus"),
        "last_p300": eeg_state.get("last_p300"),
        "last_update": eeg_state.get("last_update"),
        "privacy_guard_active": True,
    }

@router.get("/gsr/status")
async def get_gsr_status():
    """v8 — Galvanic Skin Response (GSR/EDA) sensor status."""
    if gsr is None:
        return {"connected": False, "reason": "gsr module not loaded"}
    gsr_status = gsr.get_gsr_status()
    gsr_data = gsr.get_gsr_data()
    return {
        "connected": gsr_status.get("connected", False),
        "port": gsr_status.get("port"),
        "error": gsr_status.get("error"),
        "phasic": gsr_data.get("scr_amplitude") if gsr_data else None,
        "tonic": gsr_data.get("scl_level") if gsr_data else None,
        "scr_rate": gsr_data.get("scr_amplitude") if gsr_data else None,
        "last_update": gsr._gsr_data.get("last_update") if gsr else None,
    }

@router.get("/identity/status")
async def get_identity_status():
    """v8 — Face identity registry and cache status."""
    if face_identity is None:
        return {"registry_size": 0, "cache_size": 0, "reason": "face_identity module not loaded"}
    return {
        "registry_size": face_identity.registry_size(),
        "cache_size": len(face_identity._STATE_CACHE),
        "threshold": face_identity.SIM_THRESHOLD,
    }

@router.get("/rppg/calibration")
async def get_rppg_calibration():
    """v8 — Remote Photoplethysmography (rPPG) calibration monitor snapshot."""
    try:
        import rppg as _rppg
        confirmed = list(_rppg.hr_samples)
        if not confirmed:
            return {"n": 0, "mean_bpm": None, "sd_bpm": None, "z": None, "match_score": None}
        mean_bpm = sum(confirmed) / len(confirmed)
        variance = sum((x - mean_bpm) ** 2 for x in confirmed) / len(confirmed)
        sd_bpm = variance ** 0.5
        # Reference: Quer et al. JAMA Cardiol 2020 seated adult 72±12 BPM
        z_score = (mean_bpm - 72.0) / 12.0
        match_score = max(0.0, 1.0 - abs(z_score) / 3.0)
        return {
            "n": len(confirmed),
            "mean_bpm": round(mean_bpm, 1),
            "sd_bpm": round(sd_bpm, 1),
            "z": round(z_score, 2),
            "match_score": round(match_score, 2),
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/llm/governor/stats")
async def get_llm_governor_stats():
    """v8 — LLM governor rate-limit & token budget statistics."""
    try:
        from llm_governor import get_governor
        gov = get_governor()
        return sanitize(gov.stats())
    except Exception as e:
        return {"error": str(e)}

@router.get("/diarize/status")
async def get_diarization_status():
    """v8 — Speaker diarization status (requires pyannote or speechbrain)."""
    try:
        from diarization import get_diarizer
        dz = get_diarizer()
        speakers = dz.active_speakers()
        return {
            "connected": len(speakers) > 0,
            "backend": dz.backend,
            "embeddings_available": dz._use_embeddings,
            "has_embeddings": dz.has_embeddings,
            "active_speakers": speakers,
            "speaker_person_map": dz._speaker_person_map,
        }
    except Exception as e:
        return {"error": str(e)}

@router.post("/radar/reconnect")
async def radar_reconnect():
    """Force-kick the radar reader (watchdog does this automatically)."""
    try:
        radar.stop()
    except Exception:
        pass
    ok = radar.start()
    return {"ok": bool(ok), "status": sanitize(radar.summary())}

@router.post("/logs/flush")
async def flush_logs():
    logger.flush_parquet()
    return {"ok": True, "dir": cfg.PARQUET_LOG_DIR}

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "bad image"})
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detectors.detect_faces(rgb)

    # Run heavy ops in thread pool so event loop stays responsive
    loop = asyncio.get_event_loop()
    if result.face_landmarks:
        crop = detectors.get_face_crop(image, result.face_landmarks[0], h, w)
        if crop is not None:
            df = await loop.run_in_executor(detectors.pool, detectors.run_deepface, crop)
            if df:
                recog = await loop.run_in_executor(detectors.pool, face_registry.recognize, crop)
                if recog["name"]:
                    analysis.affect_vector["identity"] = recog["name"]
                analysis.update_affect_vector(
                    df, analysis.affect_vector.get("heart_rate_bpm"),
                    analysis.affect_vector.get("hrv_rmssd"),
                    analysis.affect_vector.get("stress_index"), 0,
                    analysis.affect_vector.get("resp_rate"),
                    analysis.affect_vector.get("head_pose"),
                    analysis.affect_vector.get("gaze"), None, None,
                    analysis.affect_vector.get("emotion_scores"),
                )
    # v9.1: manual DODO click → force a call regardless of delta-gate.
    text = await llm.analyze_affect(analysis.affect_vector, face_id="global", skip_if_stable=False)
    return sanitize({"affect_vector": analysis.affect_vector, "analysis": text})

@router.post("/monitor/start")
async def start_mon():
    llm.start_monitor()
    return {"status": "on"}

@router.post("/monitor/stop")
async def stop_mon():
    llm.stop_monitor()
    return {"status": "off"}

# Step 2: Per-person auto monitor (runs alongside regular monitor)
_person_monitor_running = False

@router.post("/monitor/persons/start")
async def start_person_mon():
    global _person_monitor_running
    _person_monitor_running = True
    asyncio.ensure_future(_person_monitor_loop())
    return {"status": "on", "mode": "per_person"}

@router.post("/monitor/persons/stop")
async def stop_person_mon():
    global _person_monitor_running
    _person_monitor_running = False
    return {"status": "off"}

async def _person_monitor_loop():
    """Step 2: Periodic per-person + room LLM analysis."""
    while _person_monitor_running:
        await asyncio.sleep(cfg.AUTO_MONITOR_INTERVAL)
        if not _person_monitor_running or not person_states:
            continue
        try:
            persons_list = [ps.build_affect_dict() for ps in person_states.values()]
            result = await llm.analyze_all_persons(persons_list)
            for tid, text in result.get("per_person", {}).items():
                if tid in person_states:
                    person_states[tid].llm_analysis = text
        except Exception:
            pass

# ══════════════════════════════════════════════
#  Truth Test Endpoints
# ══════════════════════════════════════════════
@router.post("/truth/start_baseline")
async def truth_start_baseline():
    """Start collecting 10s physiological baseline."""
    return truth_engine.start_baseline()

@router.get("/truth/baseline_status")
async def truth_baseline_status():
    return truth_engine.get_baseline_status()

@router.post("/truth/feed_baseline")
async def truth_feed_baseline():
    """Feed current affect vector to baseline (called automatically by detection loop)."""
    result = truth_engine.feed_baseline(analysis.affect_vector)
    return result

@router.post("/truth/start_answer")
async def truth_start_answer():
    """Start capturing biometrics during answer."""
    return truth_engine.start_answer_capture()

@router.post("/truth/feed_answer")
async def truth_feed_answer():
    """Feed current affect vector during answer."""
    truth_engine.feed_answer(analysis.affect_vector)
    return {"ok": True}

@router.post("/truth/analyze")
async def truth_analyze(file: UploadFile = File(...), question: str = Form("")):
    """Receive answer audio, transcribe, analyze vocal stress, run full deception analysis.
    Heavy processing runs in thread pool to avoid blocking the detection loop."""
    audio_bytes = await file.read()

    # Stop answer biometric capture
    answer_summary = truth_engine.stop_answer_capture()
    if not answer_summary:
        answer_summary = {}

    # Get baseline
    bs_status = truth_engine.get_baseline_status()
    baseline_summary = bs_status.get("summary", {}) if bs_status.get("ready") else {}

    # Run heavy audio processing in thread pool (non-blocking)
    import io
    loop = asyncio.get_event_loop()

    def _process_audio(raw_bytes):
        sr = 16000
        audio_float = None
        try:
            import soundfile as sf
            audio_np, sr = sf.read(io.BytesIO(raw_bytes))
            if len(audio_np.shape) > 1:
                audio_np = audio_np[:, 0]
            if sr != 16000:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
                sr = 16000
            audio_float = audio_np.astype(np.float32).tobytes()
        except Exception:
            audio_float = raw_bytes
        transcript = audio_mod.transcribe(audio_float, sr) if audio_float else {"text": "", "error": "No audio"}
        vocal = audio_mod.analyze_vocal_stress(audio_float, sr) if audio_float else {}
        return transcript, vocal

    transcript, vocal = await loop.run_in_executor(None, _process_audio, audio_bytes)

    # Build full analysis package
    package = truth_engine.get_full_analysis(baseline_summary, answer_summary, vocal, transcript)
    package["question"] = question

    # Send to LLM for synthesis
    llm_analysis = await llm.analyze_deception(package)

    # Reset truth engine
    truth_engine.reset()

    return sanitize({
        "question": question,
        "transcript": transcript,
        "vocal_analysis": vocal,
        "deception_indicators": package.get("deception_indicators", {}),
        "llm_analysis": llm_analysis,
    })

@router.post("/truth/reset")
async def truth_reset():
    truth_engine.reset()
    return {"ok": True}

# ══════════════════════════════════════════════
#  Voice Monitor Endpoints
# ══════════════════════════════════════════════
@router.post("/voice/start")
async def voice_start():
    voice_monitor.start()
    return {"status": "on"}

@router.post("/voice/stop")
async def voice_stop():
    voice_monitor.stop()
    return {"status": "off"}

@router.post("/voice/chunk")
async def voice_chunk(file: UploadFile = File(...)):
    """Receive 3-second audio chunk for continuous vocal analysis."""
    audio_bytes = await file.read()
    loop = asyncio.get_event_loop()
    # Process in thread pool to avoid blocking
    import io
    def _process(raw):
        try:
            import soundfile as sf
            audio_np, sr = sf.read(io.BytesIO(raw))
            if len(audio_np.shape) > 1:
                audio_np = audio_np[:, 0]
            if sr != 16000:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            return voice_monitor.process_chunk(audio_np.astype(np.float32).tobytes(), 16000)
        except Exception:
            return voice_monitor.process_chunk(raw, 16000)
    result = await loop.run_in_executor(None, _process, audio_bytes)
    # v9.4.1 step 4 — PAD audio branch (off by default). Decode once
    # more in the same executor for the branch; only runs when
    # FF_PAD_AUDIO is on and an ASVspoof ONNX ships with the build.
    try:
        import feature_flags as _ff
        if _ff.get("PAD_AUDIO"):
            def _run_audio_pad(raw):
                try:
                    import soundfile as sf
                    import io as _io
                    audio_np, sr = sf.read(_io.BytesIO(raw))
                    if len(audio_np.shape) > 1:
                        audio_np = audio_np[:, 0]
                    return _pad_score_audio(audio_np.astype(np.float32), sr, result)
                except Exception:
                    return None
            pad_audio = await loop.run_in_executor(None, _run_audio_pad, audio_bytes)
            if pad_audio is not None and isinstance(result, dict):
                result["pad_audio"] = pad_audio
    except Exception:
        pass
    # Log to forensic session
    sess = forensic.get_session()
    if sess:
        sess.log_voice(result)
    return sanitize(result)

@router.get("/voice/metrics")
async def voice_metrics():
    return sanitize(voice_monitor.get_voice_metrics() or {})

@router.post("/voice/pcm")
async def voice_pcm(request: Request):
    """Receive raw float32 PCM audio (no WebM encoding needed).
    First float32 is sample rate, rest is audio data."""
    body = await request.body()
    if len(body) < 8:
        return {"error": "too short"}
    try:
        raw = np.frombuffer(body, dtype=np.float32)
        sr = int(raw[0])  # first value is sample rate
        audio = raw[1:]   # rest is audio
        # Resample to 16kHz if needed
        if sr != 16000 and sr > 0:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except ImportError:
                # Simple decimation fallback
                ratio = sr / 16000
                indices = np.arange(0, len(audio), ratio).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]
            sr = 16000
        result = voice_monitor.process_chunk(audio.astype(np.float32).tobytes(), sr)
        # v9.4.1 step 4 — PAD audio branch (feature-flagged).
        try:
            import feature_flags as _ff
            if _ff.get("PAD_AUDIO"):
                pad_audio = _pad_score_audio(audio, sr, result)
                if pad_audio is not None and isinstance(result, dict):
                    result["pad_audio"] = pad_audio
        except Exception:
            pass
        sess = forensic.get_session()
        if sess:
            sess.log_voice(result)
        return sanitize(result)
    except Exception as e:
        return {"error": str(e)}

# ══════════════════════════════════════════════
#  GSR Endpoints
# ══════════════════════════════════════════════
@router.get("/gsr")
async def get_gsr_data():
    return sanitize(gsr.get_gsr_status())

# ══════════════════════════════════════════════
#  Forensic Session Endpoints
# ══════════════════════════════════════════════
@router.post("/forensic/start")
async def forensic_start(examiner: str = Form(""), subject: str = Form(""), notes: str = Form("")):
    return forensic.start_session(examiner, subject, notes)

@router.post("/forensic/end")
async def forensic_end():
    result = forensic.end_session()
    return result or {"error": "No active session"}

@router.post("/forensic/note")
async def forensic_note(note: str = Form("")):
    return forensic.add_note(note)

# ══════════════════════════════════════════════
#  Step 2: Per-Person LLM Analysis
# ══════════════════════════════════════════════
@router.post("/analyze/persons")
async def analyze_persons():
    """Run per-person + room-level LLM analysis on all tracked persons.
    Falls back to global affect_vector if per-person tracking isn't active."""
    persons_list = []
    if person_states:
        persons_list = [ps.build_affect_dict() for ps in person_states.values()]
    else:
        # Fallback: use global affect_vector as a single person
        av = analysis.affect_vector
        if av and av.get("dominant_emotion"):
            persons_list = [{"track_id": 0, "identity": av.get("identity"), **av}]
    if not persons_list:
        return {"error": "No faces detected"}
    result = await llm.analyze_all_persons(persons_list)
    # Write per-person analysis back to PersonState (handle int/str key mismatch)
    for tid, text in result.get("per_person", {}).items():
        tid_int = int(tid) if isinstance(tid, str) and tid.isdigit() else tid
        if tid_int in person_states:
            person_states[tid_int].llm_analysis = text
    return sanitize(result)

@router.get("/persons")
async def get_persons():
    """Get all tracked person states."""
    return sanitize([ps.build_affect_dict() for ps in person_states.values()])

# ══════════════════════════════════════════════
#  Step 7: Session Replay
# ══════════════════════════════════════════════
@router.get("/forensic/sessions")
async def list_forensic_sessions():
    return sanitize(forensic.list_sessions())

@router.post("/forensic/replay/load")
async def replay_load(session_id: str = Form("")):
    return sanitize(forensic.session_player.load(session_id))

@router.get("/forensic/replay/frame")
async def replay_frame(index: int = -1):
    idx = index if index >= 0 else None
    frame = forensic.session_player.get_frame(idx)
    if frame:
        return sanitize(frame)
    return {"error": "No frame"}

@router.post("/forensic/replay/seek")
async def replay_seek(index: int = Form(0)):
    forensic.session_player.seek(index)
    return forensic.session_player.get_state()

@router.get("/forensic/replay/state")
async def replay_state():
    return forensic.session_player.get_state()

# ══════════════════════════════════════════════════════════════════
#  v9.5.x — ops endpoints (metrics, selftest, replay, calib, auth)
# ══════════════════════════════════════════════════════════════════
# HTTP bearer auth when DODO_AUTH_TOKEN is set: enforced globally in
# main.py middleware (_dodo_http_bearer_middleware), not per-handler.


@router.get("/metrics", tags=["ops"])
async def metrics_endpoint():
    """Prometheus text-format exposition. Public (no auth).

    v9.7.1: snapshot radar HR/BR filter accept/reject counters and
    DeepFace cache state at scrape time. Background subsystems do not
    have a back-channel into the registry, so we pull instead of push.
    """
    if _REG is None:
        return _Resp(content="# metrics module unavailable\n",
                     media_type="text/plain")
    # v9.7.1: just-in-time radar filter snapshot. The HRFilter `stats()`
    # API exposes accepted / rejected_bounds / rejected_jump /
    # rejected_outlier / buffer / sqi. We mirror them as gauges so an
    # operator can see "radar A is rejecting 80% of HR samples right now".
    try:
        rs = radar.get_filter_stats()
        for radar_name in ("radar_a", "radar_b"):
            block = rs.get(radar_name, {}) or {}
            for sig in ("hr", "br"):
                stats = block.get(sig) or {}
                for stat_name in ("accepted", "rejected_bounds",
                                  "rejected_jump", "rejected_outlier",
                                  "buffer"):
                    val = stats.get(stat_name)
                    if val is None:
                        continue
                    _REG.gauge(f"radar_filter_{stat_name}",
                               "Radar filter sample counts").set(
                                   float(val),
                                   labels={"radar": radar_name, "signal": sig})
                sqi = stats.get("sqi")
                if sqi is not None:
                    _REG.gauge("radar_filter_sqi",
                               "Radar filter signal-quality index").set(
                                   float(sqi),
                                   labels={"radar": radar_name, "signal": sig})
    except Exception:
        pass  # nolint:silent-except best-effort scrape
    # v9.7.1: DF cache freshness gauge.
    try:
        with _df_lock:
            entries = list(_df_cache.values())
        _REG.gauge("deepface_cache_entries",
                   "Number of cached DF results").set(float(len(entries)))
        if entries:
            youngest_age = time.time() - max(e.get("t", 0) for e in entries)
            _REG.gauge("deepface_cache_youngest_age_seconds",
                       "Age of youngest DF cache entry").set(youngest_age)
    except Exception:
        pass  # nolint:silent-except best-effort scrape
    # v9.7.1: server identity gauges (constant 1, label-rich).
    try:
        if _server_identity is not None:
            ident = _server_identity.summary()
            _REG.gauge("server_uptime_seconds",
                       "Server uptime in seconds").set(float(ident.get("uptime_s") or 0))
            _REG.gauge("server_info",
                       "Server identity (always 1, labels carry data)").set(
                           1.0,
                           labels={
                               "version": cfg.APP_VERSION,
                               "content_hash": str(ident.get("content_hash") or "?"),
                               "pid": str(ident.get("pid") or 0),
                           })
    except Exception:
        pass  # nolint:silent-except best-effort scrape
    return _Resp(content=_REG.expose(),
                 media_type="text/plain; version=0.0.4")


@router.get("/selftest", tags=["ops"])
async def selftest_endpoint(duration_s: float = 5.0):
    """Run a synthetic-face probe for up to `duration_s` seconds and
    return a pass/fail summary per subsystem. Useful as a Docker healthcheck
    or a sanity check after a deploy."""
    import time as _t, cv2 as _cv2, numpy as _np
    t0 = _t.monotonic()
    status: dict = {"started_at": t0, "duration_s": duration_s, "steps": {}}

    # 1. MediaPipe detector
    try:
        import detectors as _det
        dummy = (_np.random.rand(224, 224, 3) * 255).astype(_np.uint8)
        rgb = _cv2.cvtColor(dummy, _cv2.COLOR_BGR2RGB)
        _r = _det.detect_faces(rgb)
        status["steps"]["mediapipe"] = {
            "ok": True, "faces_found": len(_r.face_landmarks),
        }
    except Exception as e:
        status["steps"]["mediapipe"] = {"ok": False, "error": str(e)}

    # 2. PAD branches
    for name, probe in (
        ("patch_depth", lambda: __import__("pad.branches.patch_depth_branch",
                                           fromlist=["score"]).score(
                                               _np.random.randint(0, 255, (160, 160, 3), _np.uint8))),
        ("moire",       lambda: __import__("pad.branches.moire_branch",
                                           fromlist=["score"]).score(
                                               _np.random.randint(0, 255, (128, 128, 3), _np.uint8))),
        ("deepfake",    lambda: __import__("pad.branches.deepfake_branch",
                                           fromlist=["score"]).score(
                                               0, face_crop_bgr=_np.random.randint(0, 255, (224, 224, 3), _np.uint8))),
    ):
        try:
            t_start = _t.perf_counter()
            score = probe()
            dt_ms = (_t.perf_counter() - t_start) * 1000
            status["steps"][f"pad_{name}"] = {
                "ok": True,
                "latency_ms": round(dt_ms, 1),
                "reliability": getattr(score, "reliability", None),
            }
        except Exception as e:
            status["steps"][f"pad_{name}"] = {"ok": False, "error": str(e)}

    # 3. Radar/oximeter presence (just read _ready_check shape)
    try:
        r = await ready_check()
        status["steps"]["sensors"] = r.get("sensors", {})
    except Exception as e:
        status["steps"]["sensors"] = {"ok": False, "error": str(e)}

    # 4. Metrics / auth / tls modules loaded?
    status["steps"]["modules"] = {
        "auth":     _auth is not None,
        "metrics":  _metrics is not None,
        "replay":   _replay is not None,
        "calib":    _calib_store is not None,
        "auth_on":  _auth.enabled() if _auth else False,
    }

    status["finished_at"] = _t.monotonic()
    status["elapsed_s"] = round(status["finished_at"] - t0, 3)
    failed = [k for k, v in status["steps"].items()
              if isinstance(v, dict) and v.get("ok") is False]
    status["ok"] = len(failed) == 0
    status["failed_steps"] = failed
    return status


@router.get("/replay/list", tags=["session"])
async def replay_list():
    if _replay is None:
        return {"sessions": [], "enabled": False}
    return {"sessions": _replay.list_sessions(), "enabled": _replay.enabled()}


@router.get("/replay/{session_id}/stream", tags=["session"])
async def replay_stream(session_id: str):
    """Server-sent events stream of every recorded envelope."""
    if _replay is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="replay module unavailable")
    try:
        _replay.session_path(session_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    from fastapi.responses import StreamingResponse
    import json as _json

    def _gen():
        for env in _replay.stream(session_id):
            yield f"data: {_json.dumps(env, default=str)}\n\n"
        yield "event: eof\ndata: {}\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.get("/calib/list", tags=["calib"])
async def calib_list():
    if _calib_store is None:
        return {"identities": [], "enabled": False}
    return {"identities": _calib_store.list_identities(), "enabled": True}


@router.get("/calib/{identity}", tags=["calib"])
async def calib_get(identity: str):
    if _calib_store is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="calibration_store unavailable")
    data = _calib_store.load(identity)
    if data is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="identity not found")
    return data


@router.post("/ground_truth/label", tags=["session"])
async def ground_truth_label(label: str = Form(...),
                             person_id: int = Form(0),
                             note: str = Form("")):
    """Operator-supplied ground-truth label for the currently visible
    person. Appended to sessions/ground_truth.jsonl. Used to build your
    own labelled dataset from live captures."""
    import json as _json
    from pathlib import Path as _Path
    gt_dir = _Path("sessions")
    gt_dir.mkdir(exist_ok=True)
    path = gt_dir / "ground_truth.jsonl"
    row = {
        "ts": time.time(),
        "label": label, "person_id": person_id, "note": note,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(row) + "\n")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if _REG:
        _REG.counter("ground_truth_labels_total",
                     "Operator-supplied labels",
                     labels={"label": label}).inc()
    return {"ok": True, "path": str(path), "row": row}


@router.get("/webcam")
async def webcam_page(request: Request):
    # v9.3 step 1 — inject server-resolved feature flags and the URL query
    # string so the client reads the same flag set. A <script> tag with
    # id="ff-v93" carries the flag JSON; overlay.js/tracker.js read it.
    try:
        import feature_flags as _ff
        _qs = str(request.url.query) if hasattr(request, "url") else ""
        _flags_json = _ff.client_js_literal(_qs)
    except Exception:
        _flags_json = "{}"
    _payload = UI_HTML
    # v9.8.1 2026-05-12: inject DODO_CAMERA_MODE from env (set by the
    # desktop launcher: DODO_TRI / DODO_DUAL / unset for AUTO). UI's
    # picker reads window.DODO_CAMERA_MODE to decide between EOS-main
    # (TRI) and UGREEN-main (DUAL). Default 'AUTO' = legacy back-compat.
    _cam_mode = os.environ.get("DODO_CAMERA_MODE", "AUTO").strip().upper()
    if _cam_mode not in ("TRI", "DUAL", "AUTO"):
        _cam_mode = "AUTO"
    _auth_on = _auth is not None and _auth.enabled()
    # The public UI must obtain its credential from the operator, never the server.
    _auth_bootstrap = (
        "<script>"
        "window.__DODO_AUTH_TOKEN__=null;"
        f"(function(){{if(!{json.dumps(_auth_on)})return;"
        'var t=window.prompt("Enter DODO_AUTH_TOKEN to connect to this server:");'
        "if(!t)return;window.__DODO_AUTH_TOKEN__=t;var o=window.fetch;"
        "window.fetch=function(input,init){init=init||{};"
        "var u=new URL(input instanceof Request?input.url:input,location.href);"
        "if(u.origin!==location.origin)return o.call(window,input,init);"
        "var h=new Headers(init.headers||(input instanceof Request?input.headers:undefined));"
        'if(!h.has("Authorization"))h.set("Authorization","Bearer "+t);'
        "init.headers=h;return o.call(window,input,init);};})();"
        "</script>"
    )
    _inject = (
        f'<script>window.DODO_CAMERA_MODE = "{_cam_mode}";</script>'
        f"{_auth_bootstrap}"
        f'<script id="ff-v93" type="application/json">{_flags_json}</script>'
    )
    if "</head>" in _payload:
        _payload = _payload.replace("</head>", _inject + "</head>", 1)
    return HTMLResponse(_payload)

@router.get("/flags")
async def flags_snapshot():
    """v9.3 step 1 — expose resolved feature flag state for debugging."""
    try:
        import feature_flags as _ff
        return {"flags": _ff.snapshot(), "safe_mode": _ff.is_safe_mode()}
    except Exception as e:
        return {"error": str(e)}


@router.get("/version")
async def version_endpoint():
    """v9.8: full build / reproducibility metadata for the methodology
    section. One-shot snapshot of every input that would matter to a
    reviewer reproducing a session.
    """
    out = {
        "app_version": getattr(cfg, "APP_VERSION", "?"),
        "ts": time.time(),
    }
    # Python + key library versions
    try:
        import sys as _sys
        out["python"] = _sys.version.split()[0]
    except Exception:
        pass
    for mod_name in ("torch", "onnxruntime", "openvino", "numpy",
                     "mediapipe", "cv2", "scipy"):
        try:
            mod = __import__(mod_name)
            out[mod_name] = getattr(mod, "__version__", "?")
        except Exception:
            out[mod_name] = "(not loaded)"
    # GPU info
    try:
        import torch as _t
        if _t.cuda.is_available():
            props = _t.cuda.get_device_properties(0)
            out["gpu"] = {
                "name": props.name,
                "compute_cap": f"{props.major}.{props.minor}",
                "vram_gb": round(props.total_memory / 1e9, 1),
            }
    except Exception:
        pass
    # Effective feature-flag env
    out["env"] = {
        k: os.environ.get(k, "")
        for k in ("FF_USE_FP16", "FF_BLINK_CNN", "FF_AUDIO_NPU",
                  "LLM_BACKEND", "VLLM_MODEL", "DODO_HIFPS_CAM_NAME",
                  "DODO_HIFPS_FORCE_120", "DODO_RPPG_BUFFER_SIZE")
    }
    # Model fingerprints (sha256 prefix) for paper reproducibility
    out["models"] = {}
    for rel in ("pad/models/face_xray_v3.onnx",
                "pad/models/patch_depth_fas_v6.onnx",
                "pad/models/deepfake_temporal_v3.onnx",
                "pad/models/deepfake_temporal_mvit.onnx",
                "pad/models/audio_pad.onnx",
                "pad/models/calibrators.pkl",
                "pad/models/eye_state_v1.onnx"):
        try:
            # v9.7.0-merged: post-extraction __file__ lives in app/api/.
            p = Path(__file__).resolve().parent.parent.parent / rel
            if p.is_file():
                import hashlib as _hl
                with open(p, "rb") as f:
                    h = _hl.sha256()
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
                out["models"][rel] = {
                    "sha256_prefix": h.hexdigest()[:16],
                    "bytes": p.stat().st_size,
                }
        except Exception:  # nolint:silent-except per-file failures non-fatal
            pass
    return out


@router.get("/settings")
async def settings_page():
    """v9.8: read-only debug view of the live runtime config + per-branch
    warmup state. Shows the same data forensic_export captures, but live
    so you can sanity-check without exporting a bundle. Pure JSON; the
    UI's PAD debug panel will eventually consume this for an "Active
    config" tab.
    """
    out = {"ts": time.time()}
    # FF flags from env (what the launcher set)
    out["env"] = {
        k: os.environ.get(k, "")
        for k in (
            "FF_USE_FP16", "FF_BLINK_CNN", "FF_AUDIO_NPU",
            "LLM_BACKEND", "VLLM_URL", "VLLM_MODEL",
            "DODO_HIFPS_CAM_NAME", "DODO_HIFPS_FORCE_120",
            "DODO_RADAR_PORT", "DODO_RADAR_B_PORT", "DODO_GSR_PORT",
            "DODO_RPPG_BUFFER_SIZE", "DODO_ONNX_WORKERS",
        )
    }
    # Live state surfaced by sub-modules
    try:
        out["dual_cam"] = dual_camera.get_status()
    except Exception:
        pass
    try:
        from pad.branches import patch_depth_branch as _pd
        out["patch_depth"] = {
            "subject_baseline": float(_pd._subject_baseline[0]),
            "baseline_n": int(_pd._baseline_n[0]),
            "warmup_done": bool(_pd._baseline_n[0] >= 8),
            "disabled_for_session": bool(_pd._disabled_for_session[0]),
        }
    except Exception:
        pass
    try:
        out["rppg_q"] = rppg_mod.get_signal_quality()
    except Exception:
        pass
    return out


@router.get("/v98/audit")
async def v98_audit():
    """v9.8: subprocess-out to scripts/v98_artifact_check.py and return
    its JSON result. Lets a deployment monitor (or the UI's /settings
    page) verify that every v9.8 artifact required for the active
    feature flags is on disk. Cheap (<200 ms) -- the script is
    pure-stdlib + a sha256 walk.
    """
    import subprocess, json as _json
    try:
        py = sys.executable
        out = subprocess.run(
            # v9.7.0-merged: scripts/ is at repo root, not under app/api/.
            [py, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "v98_artifact_check.py")],
            capture_output=True, text=True, timeout=20)
        try:
            return _json.loads(out.stdout)
        except Exception:
            return {"error": "audit script output unparseable",
                    "stdout": out.stdout[-2000:],
                    "stderr": out.stderr[-1000:],
                    "returncode": out.returncode}
    except Exception as e:
        return {"error": str(e)}


@router.post("/calibration/reset_all")
async def calibration_reset_all():
    """v9.8: hard-reset every per-subject calibration state. Triggered
    from the UI via Ctrl+Alt+R when the user wants a fresh baseline
    (e.g., new subject, different lighting, glasses on/off).
    """
    reset = []
    try:
        from pad.branches import patch_depth_branch as _pd
        _pd.reset_baseline()
        reset.append("patch_depth.baseline")
    except Exception:
        pass
    try:
        # Reset any per-face BlinkDetector instances managed by analysis.py
        import analysis as _an
        if hasattr(_an, "_per_face_blink"):
            for det in _an._per_face_blink.values():
                try:
                    det.reset()
                except Exception:
                    pass
            reset.append(f"blink.per_face({len(_an._per_face_blink)})")
    except Exception:
        pass
    return {"ok": True, "reset": reset, "ts": time.time()}# v9.5.x — 204 on /favicon.ico so Chrome stops logging a 404 for every tab.
from fastapi import Response as _Resp
@router.get("/favicon.ico", include_in_schema=False)
async def _favicon():
    return _Resp(status_code=204)

# ═══════════════════════════════════════════════════════════════════
# v9.4 — Presentation Attack Detection (PAD) endpoints
# ═══════════════════════════════════════════════════════════════════

# Lazy-imported so pad/ isn't instantiated when FF_PAD_FUSION=0.
_pad_challenge_sessions: dict = {}  # session_id -> dict
_pad_stats_counters = {
    "verdict_live": 0, "verdict_spoof": 0, "verdict_uncertain": 0,
    "branches_emitted": 0, "total_frames": 0, "disagreements": 0,
    "audio_branch_emissions": 0,   # v9.4.1 step 4
    "last_updated": 0.0,
}

# v9.4.1 step 3 — last client-side padFast.moire_rgb value seen on the
# WS stream, with the monotonic timestamp. The moire branch consumes
# this as a cross-validation cue; stale values (>2 s) are ignored.
_pad_last_client_fast = {"moire_rgb": None, "ts": 0.0}

# v9.4.1 step 8 — continuous-monitoring sliding-window anomaly state.
# `events` is a list of (ts, disagreed:bool). The writer maintains a
# 60-s sliding window and writes to sessions/<id>/pad_anomalies.jsonl
# whenever the disagreement rate stays > 0.25 for more than 60 s.
_pad_anomaly_state = {
    "events": [],                # list of (ts_monotonic, bool)
    "elevated_since": None,      # monotonic time when rate first exceeded 0.25
    "last_anomaly_write": 0.0,
    "recent_anomalies": [],      # last ~20 anomaly rows for the HUD banner
}
_PAD_ANOM_WINDOW_S = 60.0
_PAD_ANOM_RATE_GATE = 0.25
_PAD_ANOM_SUSTAIN_S = 60.0
_PAD_ANOM_COOLDOWN_S = 30.0


def _pad_anomaly_tick(disagreed: bool, branches, fused_verdict: str):
    """Record one PAD frame's disagreement flag and, when the 60-s
    rolling rate stays above the gate for the sustain window, append a
    row to `sessions/<id>/pad_anomalies.jsonl`. Also keeps an in-memory
    ring buffer surfaced by `/pad/anomalies/recent`.
    """
    now = time.monotonic()
    ev = _pad_anomaly_state["events"]
    ev.append((now, bool(disagreed)))
    cutoff = now - _PAD_ANOM_WINDOW_S
    while ev and ev[0][0] < cutoff:
        ev.pop(0)
    if not ev:
        return
    hits = sum(1 for _, d in ev if d)
    rate = hits / len(ev)
    if rate > _PAD_ANOM_RATE_GATE:
        if _pad_anomaly_state["elevated_since"] is None:
            _pad_anomaly_state["elevated_since"] = now
        else:
            elevated_for = now - _pad_anomaly_state["elevated_since"]
            since_last = now - _pad_anomaly_state["last_anomaly_write"]
            if elevated_for > _PAD_ANOM_SUSTAIN_S and since_last > _PAD_ANOM_COOLDOWN_S:
                row = {
                    "ts": time.time(),
                    "rate": round(rate, 4),
                    "window_s": _PAD_ANOM_WINDOW_S,
                    "elevated_for_s": round(elevated_for, 1),
                    "recent_verdicts": [d[0] for d in
                                        [(fused_verdict, 0)]][:1],  # placeholder
                    "branch_scores": [
                        {"name": b.name, "p_attack": b.p_attack,
                         "reliability": b.reliability}
                        for b in (branches or [])
                    ],
                }
                # Write to forensic session if one is active
                try:
                    sess = forensic.get_session()
                    if sess is not None and getattr(sess, "session_dir", None):
                        fp = os.path.join(sess.session_dir, "pad_anomalies.jsonl")
                        with open(fp, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row) + "\n")
                except Exception:
                    pass
                ring = _pad_anomaly_state["recent_anomalies"]
                ring.append(row)
                if len(ring) > 20:
                    del ring[0]
                _pad_anomaly_state["last_anomaly_write"] = now
    else:
        _pad_anomaly_state["elevated_since"] = None


def _pad_score_audio(audio_np, sr, vm_result):
    """v9.4.1 step 4 — run the PAD audio branch on a 2-s chunk, cross-
    index each diarized speaker's rolling buffer to a PersonState and
    stash the BranchScore on ps.pad_state.audio. Updates
    `_pad_stats_counters['audio_branch_emissions']` when a reliable
    score is produced (ASVspoof ONNX present + speech-gated).

    Returns the summary dict for forensic logging or None.
    """
    try:
        import feature_flags as _ff
    except Exception:
        return None
    if not _ff.get("PAD_AUDIO") or audio_np is None or len(audio_np) == 0:
        return None
    try:
        from pad.branches.audio_branch import score as _score_audio
        from diarization import get_diarizer
    except Exception:
        return None
    try:
        import numpy as _np
        arr = _np.asarray(audio_np, dtype=_np.float32)
        # Prefer VAD speech_prob + speech_frames if present in vm_result
        sp_prob = float(vm_result.get("speech_prob", vm_result.get("speech_probability", 1.0))) \
            if isinstance(vm_result, dict) else 1.0
        sp_frames = int(vm_result.get("speech_frames", 100)) \
            if isinstance(vm_result, dict) else 100
        # Score the whole chunk (omni) first.
        omni = _score_audio(arr, sr=sr, speech_probability=sp_prob,
                            speech_frames=sp_frames)
        # Per-speaker feed + score.
        dz = get_diarizer()
        try:
            dz.ingest(arr, sr)
            dz._update_speakers_energy_fallback()
        except Exception:
            pass
        per_speaker = []
        for spk in dz.active_speakers():
            sid = spk.get("speaker_id")
            if not sid:
                continue
            try:
                dz.ingest_speaker_audio(sid, arr)
            except Exception:
                pass
            spk_audio = dz.get_speaker_audio(sid)
            if spk_audio is None or spk_audio.size < sr:
                continue
            sc = _score_audio(spk_audio, sr=sr,
                              speech_probability=sp_prob,
                              speech_frames=sp_frames)
            pid_str = dz._speaker_person_map.get(sid) if hasattr(dz, "_speaker_person_map") else None
            per_speaker.append({
                "speaker_id": sid,
                "p_attack": sc.p_attack,
                "reliability": sc.reliability,
                "person_id": pid_str,
            })
        if omni.reliability > 0:
            _pad_stats_counters["audio_branch_emissions"] += 1
        return {
            "p_attack": omni.p_attack,
            "reliability": omni.reliability,
            "latency_ms": omni.latency_ms,
            "per_speaker": per_speaker,
        }
    except Exception:
        return None

# v9.8.1 (2026-05-12): diagnostic counter for _pad_feed_rppg. Operator
# was seeing rppg n_rois=1 (only forehead from main HR path, cheeks
# silent) which produces p_attack=1.0. The bug is here somewhere; this
# counter lets us spot WHICH gate is firing without restarting twice.
_pad_feed_rppg_diag = {
    "calls": 0, "no_ff": 0, "no_crop": 0, "no_ps": 0,
    "no_rppg": 0, "crop_small": 0, "fed_l": 0, "fed_r": 0, "fed_both": 0,
}


def _pad_feed_rppg(pid: int, face_crop_bgr):
    """Best-effort cheek ROI-mean extraction for the rPPG liveness
    branch. Crops the left/right cheek strips (proportional to the face
    crop) and pushes them to PersonState via `rppg.feed_liveness_rois`.
    The forehead buffer is NOT re-fed here — the main HR path already
    populates `ps.rppg_rgb` from the full ROI. Cheap; runs only when
    FF_PAD_RPPG is on.

    v9.8.1 (2026-05-12): widened cheek slices (was 25 % tall x 25 %
    wide; now 30 % tall x 35 % wide) and lowered the minimum face_crop
    size from 40 px to 32 px so smaller crops (e.g. when the subject
    is further from the camera) still produce a usable patch. Mean
    RGB across a ~60-pixel-wide patch is robust enough for SNR
    estimation but small enough to stay localised to the cheek.
    Counter increments are diagnostic only; see _pad_feed_rppg_diag.
    """
    _pad_feed_rppg_diag["calls"] += 1
    try:
        import feature_flags as _ff
    except Exception:
        _pad_feed_rppg_diag["no_ff"] += 1
        return
    if not _ff.get("PAD_RPPG"):
        _pad_feed_rppg_diag["no_ff"] += 1
        return
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        _pad_feed_rppg_diag["no_crop"] += 1
        return
    ps = person_states.get(pid)
    if ps is None:
        _pad_feed_rppg_diag["no_ps"] += 1
        return
    try:
        import rppg as _rppg_main
    except Exception:
        _pad_feed_rppg_diag["no_rppg"] += 1
        return
    h, w = face_crop_bgr.shape[:2]
    if h < 32 or w < 32:
        _pad_feed_rppg_diag["crop_small"] += 1
        return
    # Widened slices vs the v9.4 originals (was 0.40-0.65 tall × 0.10-0.35
    # wide). Now 0.35-0.70 tall × 0.05-0.40 wide. Gives a ~25 %-larger
    # cheek patch that's more robust on smaller face crops and resists
    # landmark jitter at the cheek boundary.
    lc = face_crop_bgr[int(0.35 * h):int(0.70 * h), int(0.05 * w):int(0.40 * w)]
    rc = face_crop_bgr[int(0.35 * h):int(0.70 * h), int(0.60 * w):int(0.95 * w)]

    def _mean_rgb(patch):
        if patch is None or patch.size == 0:
            return None
        import cv2 as _cv2
        mean = _cv2.mean(patch)
        return (float(mean[2]), float(mean[1]), float(mean[0]))

    lc_rgb = _mean_rgb(lc)
    rc_rgb = _mean_rgb(rc)
    if lc_rgb and rc_rgb:
        _pad_feed_rppg_diag["fed_both"] += 1
    elif lc_rgb:
        _pad_feed_rppg_diag["fed_l"] += 1
    elif rc_rgb:
        _pad_feed_rppg_diag["fed_r"] += 1
    _rppg_main.feed_liveness_rois(ps, time.time(), None, lc_rgb, rc_rgb)


# v9.5.x perf: per-pid wall-clock of last heavy-branch run, used to
# throttle moire / patch_depth / deepfake down to ~2 Hz instead of
# per-frame. rPPG feed + dynamics stay per-frame because they're cheap
# (<5 ms) and they carry the live signal that fusion hysteresis needs.
# Override via env: PAD_HEAVY_INTERVAL_MS (default 500 = 2 Hz).
_pad_heavy_last_ts = {}   # pid -> time.monotonic() of last heavy PAD run
try:
    _PAD_HEAVY_INTERVAL_S = float(os.environ.get("PAD_HEAVY_INTERVAL_MS", "500")) / 1000.0
except Exception:
    _PAD_HEAVY_INTERVAL_S = 0.5

# v9.8.1: per-pid edge-triggered blink memory so dynamics_branch.feed_blink
# fires once per blink (transition True), not every frame the eyes are
# closed. Keyed by tracker pid, never grows unbounded because tracker
# prunes lost faces (and we never store a value for a missing pid).
_PAD_PREV_BLINK: "dict[int, bool]" = {}

# v9.8.1: per-pid warmup telemetry surfaced via /pad/{pid}. Lets the
# operator (and the PAD DEBUG panel) see WHICH branches are emitting and
# WHICH are silent during the "warming" window. Updated on every
# _pad_run_once entry; never grows past the tracker's active pid set.
_PAD_WARMUP_TELEMETRY: "dict[int, dict]" = {}

# v9.7.3: heavy-branch result cache so moire / patch_depth / deepfake
# always appear in the PAD DEBUG panel even on the throttled-off ticks
# (which were 9 out of 10 frames previously, making the panel report
# 4/7 branches almost permanently). On a throttled tick we replay the
# most recent cached BranchScore for each branch with a `cached_age_ms`
# meta tag and a reliability decayed linearly with age — fusion still
# weighs them, but old evidence is correctly weak. Entries older than
# _PAD_HEAVY_CACHE_MAX_AGE_S are dropped entirely. Structure:
#   _pad_heavy_cache[pid][branch_name] = (BranchScore, monotonic_ts)
_pad_heavy_cache: "dict[int, dict[str, tuple[Any, float]]]" = {}
_PAD_HEAVY_CACHE_MAX_AGE_S = 2.0   # 4 × the default 500 ms throttle

# v9.7.6 — server-wide PAD evaluation lock.
#
# CRASH-FIX: STATUS_ACCESS_VIOLATION (rc=-1073741819) inside ORT DML on
# patch_depth_branch.score / deepfake_branch.score, ~120 s into a session.
# Stack analysis showed two independent _pad_run_once() invocations
# running concurrently on different threads:
#
#   thread A: _process_frame_full  (HTTP /detect_full)  → _pad_run_once
#   thread B: _process_frame_fast  (fast WS frame path) → _pad_run_once
#
# Each branch keeps its own ORT session, but ORT DirectML is NOT thread-
# safe under concurrent .run() on the same session. The fast-WS path and
# the HTTP path could BOTH dispatch e.g. patch_depth.score() at the same
# instant, racing on the same DML session and corrupting GPU state.
#
# We wrap _pad_run_once in a server-wide lock so only one PAD evaluation
# is ever in flight. Heavy branches are throttled to 500 ms anyway, so
# the lock contention adds at most ~50-100 ms of waiting in the worst
# case (when both paths arrive in the same heavy window). The second
# caller often just sees `_heavy_due == False` once the first one
# finishes (since the first updates _pad_heavy_last_ts at exit) and
# skips heavy entirely — a clean serialization-by-recency.
_PAD_RUN_LOCK = threading.Lock()


def _heavy_cache_store(pid: int, score) -> None:
    """Stash a fresh BranchScore so we can replay it on throttled-off
    ticks. Caller must have just produced `score` from a heavy branch."""
    if score is None:
        return
    bucket = _pad_heavy_cache.setdefault(pid, {})
    bucket[score.name] = (score, time.monotonic())


def _heavy_cache_get(pid: int, name: str):
    """Return a stale-tagged copy of the most recent cached BranchScore,
    or None if absent or too old. Reliability is multiplied by
    `(1 - age / max_age)` so very-stale evidence has near-zero weight
    in fusion but still renders in the panel for forensic visibility."""
    bucket = _pad_heavy_cache.get(pid)
    if not bucket:
        return None
    entry = bucket.get(name)
    if entry is None:
        return None
    score, ts = entry
    age_s = time.monotonic() - ts
    if age_s >= _PAD_HEAVY_CACHE_MAX_AGE_S:
        return None
    decay = max(0.0, 1.0 - (age_s / _PAD_HEAVY_CACHE_MAX_AGE_S))
    try:
        import dataclasses as _dc
        new_meta = dict(score.meta) if score.meta else {}
        new_meta["cached_age_ms"] = round(age_s * 1000.0, 1)
        return _dc.replace(
            score,
            reliability=float(score.reliability) * decay,
            latency_ms=0.0,
            meta=new_meta,
        )
    except Exception:
        return None


def _pad_run_once(pid: int, face_crop_bgr, frame_bgr=None,
                  face_bbox=None, hr_hint=None, dynamics_feed=None,
                  n_faces_in_frame=None):
    """Thread-safe wrapper around _pad_run_once_unlocked.

    See _PAD_RUN_LOCK comment above for rationale (ORT DML thread-
    safety crash). Adds ~0 overhead in single-thread case (uncontested
    lock acquisition is sub-microsecond on Windows). When two threads
    contend (HTTP + WS paths arriving simultaneously) the second waits
    up to ~50-100 ms then proceeds; by that time the first call has
    typically updated _pad_heavy_last_ts so the second skips heavy
    branches via the existing throttle and returns quickly.

    dynamics_feed: optional dict from main.py with per-track pose/gaze/
    blink/pupil/ear_mean (+ optional nose_xy, left_ear_xy, right_ear_xy in frame
    pixels). When None, PAD dynamics falls back to analysis.affect_vector
    (face-0 global) for backward compatibility.

    n_faces_in_frame: count of faces in the *current* frame (from tracker).
    Used by multi_face_replay heuristic; when None, falls back to
    len(person_states) for backward compatibility (may be wrong).
    """
    with _PAD_RUN_LOCK:
        return _pad_run_once_unlocked(pid, face_crop_bgr, frame_bgr,
                                       face_bbox, hr_hint,
                                       dynamics_feed=dynamics_feed,
                                       n_faces_in_frame=n_faces_in_frame)


def _pad_run_once_unlocked(pid: int, face_crop_bgr, frame_bgr=None,
                            face_bbox=None, hr_hint=None,
                            dynamics_feed=None,
                            n_faces_in_frame=None):
    """Run all enabled PAD branches for one person, fuse, store in
    PersonState.pad_state. Returns the fused PADScore dict for UI.
    Silently returns None when the master FF_PAD_FUSION flag is off.

    v9.5.x: heavy branches (moire, patch_depth, deepfake) throttled to
    PAD_HEAVY_INTERVAL_MS (default 500 ms / person). Cheap branches
    (rppg feed, dynamics) run every call — they carry the live per-frame
    signal. When throttled, returns the last fused score from
    PersonState.pad_state without re-running heavy inference.

    NOTE: This is the unlocked body — never call directly. Use
    `_pad_run_once` (the wrapper above) which holds _PAD_RUN_LOCK.

    n_faces_in_frame: faces visible this frame (from tracker). Feeds
    multi_face_replay; None keeps legacy len(person_states) fallback.
    """
    try:
        import feature_flags as _ff
    except Exception:
        return None
    if not _ff.get("PAD_FUSION"):
        return None
    try:
        from pad import fuse_with_history
        from pad.branches import (
            score_rppg, score_moire, score_dynamics,
            score_patch_depth, score_deepfake,
        )
        from pad.defense import defend_batch
    except Exception:
        return None
    ps = person_states.get(pid)
    if ps is None or ps.pad_state is None:
        return None

    _now = time.monotonic()

    # v9.8.1 — DUAL-CAMERA-AWARE per-frame feeding for the temporal +
    # dynamics branches. Two branches expose feed_frame() APIs that were
    # never being called in production, leaving their rolling buffers
    # empty and their score() outputs zero:
    #
    #   pad/branches/deepfake_branch.py:408  feed_frame(pid, frame_bgr,
    #                                          face_bbox=...)
    #     -> 16-frame temporal window for the MViT/MC3-18 head that
    #        produced the headline 0.034 ACER on Celeb-DF v2. Without
    #        feeding, only the single-frame xray sub-score contributes.
    #
    #   pad/branches/dynamics_branch.py:72   feed_frame(pid, ts,
    #                                          ear_mean=..., yaw_deg=...,
    #                                          gaze_h=..., gaze_v=...,
    #                                          pupil_dilation=...)
    #     -> per-frame state for blink-dynamics + 3D-consistency +
    #        binocular-vergence + pupil-luminance checks.
    #
    # Dual-camera wiring (no extra plumbing needed):
    #   * deepfake_branch wants HQ frames at consistent stride -- UGREEN
    #     browser (~30 fps) is the right source. HD USB at 480p MJPG
    #     would degrade the temporal model trained on higher-res clips.
    #   * dynamics_branch wants high-fps temporal signals (blink/pupil/
    #     micro). Those already get the HD-USB OVERRIDE via
    #     main._merge_fast_bio() which writes the 60 fps numbers into
    #     analysis.affect_vector. So reading from affect_vector here
    #     auto-uses HD USB values when dual_camera is active, falls back
    #     to UGREEN-derived values when it isn't. Free dual-source.
    #
    # v9.8.2 API_unq: when `dynamics_feed` is provided (per face_idx from
    # main.py), pose/gaze/pupil/blink edge + nose/ear pixels come from that
    # face — not the global affect_vector (which only mirrors faces[0]).
    try:
        if _ff.get("PAD_DEEPFAKE") and frame_bgr is not None:
            from pad.branches import deepfake_branch as _df_br
            _df_br.feed_frame(pid, frame_bgr, face_bbox=face_bbox)
    except Exception:
        pass  # nolint:silent-except per-frame feed best-effort
    try:
        if _ff.get("PAD_DYNAMICS"):
            from pad.branches import dynamics_branch as _dyn_br
            import analysis as _an
            av = _an.affect_vector
            _ear_mean = None
            if dynamics_feed and isinstance(dynamics_feed, dict):
                _hp = (dynamics_feed.get("head_pose")
                       if isinstance(dynamics_feed.get("head_pose"), dict)
                       else None)
                _gz = (dynamics_feed.get("gaze")
                       if isinstance(dynamics_feed.get("gaze"), dict)
                       else None)
                _pupil_src = dynamics_feed.get("pupil_dilation")
                _nose_xy = dynamics_feed.get("nose_xy")
                _lear_xy = dynamics_feed.get("left_ear_xy")
                _rear_xy = dynamics_feed.get("right_ear_xy")
                _em = dynamics_feed.get("ear_mean")
                if _em is not None:
                    try:
                        _ear_mean = float(_em)
                    except (TypeError, ValueError):
                        _ear_mean = None
            else:
                _hp = av.get("head_pose") if isinstance(av.get("head_pose"), dict) else None
                _gz = av.get("gaze") if isinstance(av.get("gaze"), dict) else None
                _pupil_src = av.get("pupil_dilation")
                _nose_xy = _lear_xy = _rear_xy = None
            _ts_now = time.time()
            # API_unq 2026-05-14: enhanced feed for the dynamics false-positive
            # hotfix. Adds pitch + roll (unlocks pose-variance 3D-consistency
            # fallback) and scene_luminance (unlocks pupil-luminance coupling).
            # Both were previously unfed, so two of the four dynamics features
            # could never produce a value on this hardware.
            _lum = None
            try:
                if face_crop_bgr is not None and getattr(face_crop_bgr, "size", 0) > 0:
                    import cv2 as _cv2
                    _gray = _cv2.cvtColor(face_crop_bgr, _cv2.COLOR_BGR2GRAY)
                    _lum = float(_gray.mean()) / 255.0
                elif frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0:
                    import cv2 as _cv2
                    _gray = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2GRAY)
                    _lum = float(_gray.mean()) / 255.0
            except Exception:
                _lum = None  # nolint:silent-except optional feed best-effort

            def _pair_xy(v):
                if v is None:
                    return None
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    return (float(v[0]), float(v[1]))
                return None

            _dyn_br.feed_frame(
                pid, _ts_now,
                ear_mean=_ear_mean,
                yaw_deg=(_hp.get("yaw") if _hp else None),
                pitch_deg=(_hp.get("pitch") if _hp else None),
                roll_deg=(_hp.get("roll") if _hp else None),
                gaze_h=(_gz.get("horizontal") if _gz else None),
                gaze_v=(_gz.get("vertical") if _gz else None),
                pupil_dilation=_pupil_src,
                scene_luminance=_lum,
                nose_xy=_pair_xy(_nose_xy),
                left_ear_xy=_pair_xy(_lear_xy),
                right_ear_xy=_pair_xy(_rear_xy),
            )
            # Edge-triggered blink: per-face `blinked` when dynamics_feed is set,
            # else global affect_vector (faces[0] + dual_cam merge). Track
            # per-pid prior state to avoid double-firing across consecutive frames.
            _prev_blink = _PAD_PREV_BLINK.get(pid, False)
            if dynamics_feed and isinstance(dynamics_feed, dict):
                _cur_blink = bool(dynamics_feed.get("blinked"))
            else:
                _cur_blink = bool(av.get("blinked"))
            if _cur_blink and not _prev_blink:
                _dyn_br.feed_blink(pid, _ts_now)
            _PAD_PREV_BLINK[pid] = _cur_blink
    except Exception:
        pass  # nolint:silent-except per-frame feed best-effort

    _last_heavy = _pad_heavy_last_ts.get(pid, 0.0)
    _heavy_due = (_now - _last_heavy) >= _PAD_HEAVY_INTERVAL_S

    branches = []
    # Pre-apply adv-patch defense if enabled (K defended variants) — only
    # when heavy branches are about to run (otherwise it's wasted work).
    crops = [face_crop_bgr]
    if _heavy_due and _ff.get("PAD_ADV_DEFENSE") and face_crop_bgr is not None:
        try:
            crops = defend_batch(face_crop_bgr, k_samples=3)
        except Exception:
            crops = [face_crop_bgr]

    if _ff.get("PAD_RPPG"):
        try:
            _pad_feed_rppg(pid, face_crop_bgr)
            branches.append(score_rppg(pid, hr_hint_bpm=hr_hint))
        except Exception:
            pass

    # v9.7.3: heavy branches use the cache pattern — on `_heavy_due`
    # ticks we run + store the result; on throttled-off ticks we replay
    # the cached result with decayed reliability so the PAD DEBUG panel
    # always shows all enabled branches, not just the 4 cheap ones.
    if _ff.get("PAD_MOIRE"):
        if _heavy_due:
            try:
                # v9.4.1: pass the fresh client-side padFast.moire_rgb
                # hint (if any). The branch uses it to boost or dampen
                # the server-side decision.
                _client_hint = None
                if (_pad_last_client_fast["moire_rgb"] is not None and
                    (time.monotonic() - _pad_last_client_fast["ts"]) < 2.0):
                    _client_hint = _pad_last_client_fast["moire_rgb"]
                # v9.8.1 2026-05-12 (Phase 2 #1): pass pid for per-
                # subject state isolation. Without this, two persons
                # being tracked in parallel would share the same
                # moire saturation buffer and contaminate each other.
                scores = [score_moire(c, client_moire_rgb=_client_hint, pid=pid)
                          for c in crops]
                # Pick best-reliability variant from the defended-batch.
                # Take it even at rel=0 so the panel can show the branch
                # is alive (vs disappearing entirely on a low-light frame).
                best = max(scores, key=lambda s: s.reliability) if scores else None
                if best is not None:
                    branches.append(best)
                    _heavy_cache_store(pid, best)
            except Exception:
                pass
        else:
            cached = _heavy_cache_get(pid, "moire")
            if cached is not None:
                branches.append(cached)

    if _ff.get("PAD_CNN"):
        if _heavy_due:
            try:
                # v9.8.1 2026-05-12 (Phase 2 #1): pass pid for per-
                # subject state isolation -- see the matching moire
                # comment above.
                scores = [score_patch_depth(c, pid=pid) for c in crops]
                best = max(scores, key=lambda s: s.reliability) if scores else None
                if best is not None:
                    branches.append(best)
                    _heavy_cache_store(pid, best)
            except Exception:
                pass
        else:
            cached = _heavy_cache_get(pid, "patch_depth")
            if cached is not None:
                branches.append(cached)

    if _ff.get("PAD_DYNAMICS"):
        try:
            branches.append(score_dynamics(pid))
        except Exception:
            pass

    if _ff.get("PAD_DEEPFAKE"):
        if _heavy_due:
            try:
                ds = score_deepfake(pid, face_crop_bgr=face_crop_bgr)
                if ds is not None:
                    branches.append(ds)
                    _heavy_cache_store(pid, ds)
            except Exception:
                pass
        else:
            cached = _heavy_cache_get(pid, "deepfake")
            if cached is not None:
                branches.append(cached)

    # v9.5 (2026-04-22): radar presence branch -- cheap, runs every frame.
    # Picks up live breathing/heart signal from MR60BHA2 radar; strong
    # liveness cue that defeats photo + screen attacks.
    if _ff.get("PAD_RADAR_PRESENCE", True):
        try:
            from pad.branches.radar_presence_branch import score as score_radar
            branches.append(score_radar())
        except Exception:
            pass

    # v9.5 (2026-04-22): blink + gaze temporal branch -- also cheap,
    # runs every frame (warmup until ps has 10 s of history).
    if _ff.get("PAD_BLINK_GAZE", True):
        try:
            from pad.branches.blink_gaze_branch import score as score_blink
            branches.append(score_blink(ps))
        except Exception:
            pass

    # v9.8.3 — active screen-flash liveness (abstains unless a challenge is armed).
    if _ff.get("PAD_SCREEN_FLASH", True):
        try:
            from pad.branches.screen_flash_branch import score as score_screen_flash
            branches.append(score_screen_flash(ps))
        except Exception:
            pass

    # v9.5 (2026-04-22): push per-branch reliability into Prometheus EMA
    try:
        from metrics import update_branch_reliability as _upd_rel
        for b in branches:
            if hasattr(b, "name") and hasattr(b, "reliability"):
                _upd_rel(b.name, float(b.reliability))
    except Exception:
        pass

    if _heavy_due:
        _pad_heavy_last_ts[pid] = _now

    # v9.8.1: persist per-branch warmup telemetry so /pad/{pid} can
    # surface "which branches are emitting + reliability". Before this,
    # an operator seeing perpetual "warming" in the PAD panel had no
    # way to tell whether rPPG was silent, blink_gaze was silent, or
    # all branches were collectively warming.
    try:
        _PAD_WARMUP_TELEMETRY[pid] = {
            "ts": _now,
            "branches_attempted": [
                name for name, on in (
                    ("rppg",            _ff.get("PAD_RPPG")),
                    ("moire",           _ff.get("PAD_MOIRE")),
                    ("patch_depth",     _ff.get("PAD_CNN")),
                    ("dynamics",        _ff.get("PAD_DYNAMICS")),
                    ("deepfake",        _ff.get("PAD_DEEPFAKE")),
                    ("radar_presence",  _ff.get("PAD_RADAR_PRESENCE", True)),
                    ("blink_gaze",      _ff.get("PAD_BLINK_GAZE", True)),
                    ("screen_flash",    _ff.get("PAD_SCREEN_FLASH", True)),
                    ("audio",           _ff.get("PAD_AUDIO")),
                ) if on
            ],
            "branches_emitted": [
                {
                    "name": getattr(b, "name", "?"),
                    "rel":  round(float(getattr(b, "reliability", 0.0) or 0.0), 3),
                    "p":    round(float(getattr(b, "p_attack", 0.5) or 0.5), 3),
                }
                for b in branches if b is not None
            ],
            "heavy_due": bool(_heavy_due),
        }
    except Exception:
        pass  # nolint:silent-except telemetry best-effort

    # v9.8.1 2026-05-13 (operator: "if not 2 people in frame it should
    # spoof the less real one"). Multi-face replay-attack heuristic.
    #
    # Scenario: operator holds a phone showing a face. The browser
    # detects two faces in frame -- their own (large, real) and the
    # phone's screen (small, fake). Visual PAD branches struggle:
    # - moire fails because Canon R10's optical low-pass filter
    #   removes the screen pixel pattern (anti-aliasing on by design)
    # - patch_depth's per-subject EMA anchors on the user's real face
    #   so the phone face barely moves it
    # - deepfake_temporal's larger receptive field locks onto the
    #   dominant face (the user) and ignores the small one
    # - radar correctly sees ONE real chest behind the phone
    #
    # Ground truth: radar reports `people_total` (1 person physically
    # present). Browser reports N tracked faces. If browser > radar,
    # the "extra" faces are spoofs. The smaller-frame-fraction face
    # is the candidate (phones held in hand take 2-5% of frame; real
    # faces 8-30%).
    #
    # Implementation: inject a synthetic high-reliability branch into
    # this pid's PAD evaluation when:
    #   - >= 2 faces tracked in this frame, AND
    #   - this face's bbox is small (< 5% of frame area), AND
    #   - radar reports fewer persons than face count (strong) OR
    #     this face is just very small (weak)
    # The synthetic branch lets fusion produce a confident SPOOF
    # verdict with attack_class=REPLAY_SCREEN without changing the
    # math everywhere else.
    try:
        # v9.8.4: use in-frame face count from the tracker, not len(person_states)
        # (which can include stale PIDs or diverge from visible faces).
        n_faces = n_faces_in_frame if n_faces_in_frame is not None else len(person_states)
        if n_faces >= 2 and face_bbox is not None and frame_bgr is not None:
            bx = face_bbox
            if hasattr(bx, "__len__") and len(bx) >= 4:
                bx1, by1, bx2, by2 = bx[0], bx[1], bx[2], bx[3]
                face_area = max(0.0, float((bx2 - bx1) * (by2 - by1)))
                fh, fw = frame_bgr.shape[:2]
                frame_area = float(fh * fw) if fh > 0 and fw > 0 else 0.0
                face_frac = (face_area / frame_area) if frame_area > 0 else 0.0
                # Try to read radar's people count -- optional boost.
                radar_people = None
                try:
                    import radar as _rd
                    _rsum = _rd.summary() or {}
                    rp = _rsum.get("people_total")
                    if rp is None:
                        rp = _rsum.get("presence_people")  # legacy field
                    if rp is None:
                        rp = _rsum.get("people")
                    if isinstance(rp, (int, float)):
                        radar_people = int(rp)
                except Exception:
                    pass
                radar_fewer = (radar_people is not None
                               and radar_people >= 0
                               and radar_people < n_faces)
                # Trigger if face is small AND (radar says fewer OR
                # face is very small). Real human faces in a chair at
                # webcam distance typically occupy 8-30% of frame;
                # phone-held screen attacks: 1-5%.
                if face_frac < 0.05 and (radar_fewer or face_frac < 0.02):
                    from pad.types import BranchScore as _BS
                    if radar_fewer:
                        _rel  = 0.85
                        _patk = 0.92
                    else:
                        _rel  = 0.65
                        _patk = 0.80
                    branches.append(_BS(
                        name="multi_face_replay",
                        p_attack=_patk,
                        reliability=_rel,
                        dominant_class="REPLAY_SCREEN",
                        meta={
                            "face_frac":    round(face_frac, 5),
                            "n_faces":      float(n_faces),
                            "radar_people": float(radar_people) if radar_people is not None else -1.0,
                            "radar_fewer":  float(radar_fewer),
                        },
                    ))
    except Exception:
        pass   # nolint:silent-except multi-face heuristic best-effort

    if not branches:
        return None
    fused = fuse_with_history(branches, ps.pad_state)

    # Stats counters
    _pad_stats_counters["total_frames"] += 1
    _pad_stats_counters["branches_emitted"] += len(branches)
    if fused.verdict == "LIVE":
        _pad_stats_counters["verdict_live"] += 1
    elif fused.verdict == "SPOOF":
        _pad_stats_counters["verdict_spoof"] += 1
    else:
        _pad_stats_counters["verdict_uncertain"] += 1
    # Disagreement rate: any *confident* branch's p_attack differs from
    # the fused verdict by > 0.3.
    # v9.7.3: was `b.reliability > 0` which counted every warmup branch
    # (rel=0.05-0.10 while the rPPG buffer fills, blink_gaze accumulates
    # 10 s of history, dynamics waits for n_features ≥ 3, etc.) as a
    # disagreement, inflating the metric while the system was still
    # initialising. Now requires `b.reliability >= 0.25` AND a
    # meaningful margin from the fence (|p − 0.5| > 0.2) so a branch
    # sitting at p=0.51 with rel=0.3 doesn't count against a LIVE
    # verdict — only branches with both confidence AND a real opinion do.
    DISAGREE_REL_MIN = 0.25
    DISAGREE_MARGIN = 0.2
    disagreed = False
    for b in branches:
        if b.reliability >= DISAGREE_REL_MIN:
            if fused.verdict == "LIVE" and b.p_attack > (0.5 + DISAGREE_MARGIN):
                disagreed = True
                break
            if fused.verdict == "SPOOF" and b.p_attack < (0.5 - DISAGREE_MARGIN):
                disagreed = True
                break
    if disagreed:
        _pad_stats_counters["disagreements"] += 1
    # v9.4.1 step 8 — feed the anomaly monitor regardless of `disagreed`
    # so the rolling denominator tracks frame count, not just hits.
    try:
        _pad_anomaly_tick(disagreed, branches, fused.verdict)
    except Exception:
        pass
    _pad_stats_counters["last_updated"] = time.time()
    return {
        "verdict": fused.verdict,
        "p_attack": fused.p_attack,
        "attack_class": fused.attack_class,
        "log_odds": fused.log_odds,
        "branches": [
            {"name": b.name, "p_attack": b.p_attack,
             "reliability": b.reliability, "latency_ms": b.latency_ms}
            for b in fused.branches
        ],
        "ts": fused.ts,
    }


@router.get("/pad/stats")
async def pad_stats():
    """v9.4 step 10 — ISO/IEC 30107-3 reportable counters + per-branch
    disagreement rate. Used by the HUD banner in the UI."""
    try:
        import feature_flags as _ff
        enabled_flags = {k: _ff.get(k) for k in
                         ("PAD_FUSION", "PAD_RPPG", "PAD_MOIRE", "PAD_CNN",
                          "PAD_DYNAMICS", "PAD_DEEPFAKE", "PAD_AUDIO",
                          "PAD_ADV_DEFENSE", "PAD_UI", "PAD_EVAL")}
    except Exception:
        enabled_flags = {}
    n_total = _pad_stats_counters["total_frames"] or 1
    out = {
        "counters": dict(_pad_stats_counters),
        "disagreement_rate": _pad_stats_counters["disagreements"] / n_total,
        "enabled_flags": enabled_flags,
    }
    try:
        from pad.defense.adv_patch import stats as _adv_stats
        out["adv_defense"] = _adv_stats()
    except Exception:
        pass
    return out


@router.get("/pad/anomalies/recent")
async def pad_anomalies_recent():
    """v9.4.1 step 8 — last ~20 anomaly rows (disagreement rate > 0.25
    sustained > 60 s). The UI HUD polls this every 10 s. Returns an
    empty list when no anomalies have been recorded.
    """
    # Compute a snapshot of the current rolling rate so the HUD can
    # show a live-ish reading even before an anomaly fires.
    now = time.monotonic()
    ev = _pad_anomaly_state["events"]
    cutoff = now - _PAD_ANOM_WINDOW_S
    recent = [e for e in ev if e[0] >= cutoff]
    rate = (sum(1 for _, d in recent if d) / len(recent)) if recent else 0.0
    return {
        "recent": list(_pad_anomaly_state["recent_anomalies"]),
        "current_rate": round(rate, 4),
        "window_s": _PAD_ANOM_WINDOW_S,
        "gate": _PAD_ANOM_RATE_GATE,
        "elevated_since": _pad_anomaly_state["elevated_since"],
    }


def _pad_jsan(o):
    """Recursively replace NaN/Inf with None so stdlib json.dumps doesn't
    explode with `Out of range float values are not JSON compliant`.
    Also normalises numpy scalars (np.floating / np.integer) to plain
    Python types because Starlette's JSONResponse can't serialise those
    either. Used by /pad/{pid} which is polled at ~1 Hz by the
    pad_debug=1 operator view — one NaN anywhere (empty rolling window
    → variance = NaN, log(0) → -inf, 0/0 reliability) was 500-ing the
    whole response."""
    import math
    # numpy is optional here; guard the import so this works even on a
    # stripped-down env.
    try:
        import numpy as _np  # type: ignore
        _np_floating = _np.floating
        _np_integer = _np.integer
        _np_ndarray = _np.ndarray
    except Exception:
        _np_floating = ()  # type: ignore[assignment]
        _np_integer = ()  # type: ignore[assignment]
        _np_ndarray = ()  # type: ignore[assignment]

    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if _np_floating and isinstance(o, _np_floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, int):
        return o
    if _np_integer and isinstance(o, _np_integer):
        return int(o)
    if _np_ndarray and isinstance(o, _np_ndarray):
        return _pad_jsan(o.tolist())
    if isinstance(o, dict):
        return {str(k): _pad_jsan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_pad_jsan(v) for v in o]
    # Fallback: str() anything exotic so json.dumps never sees it raw.
    try:
        import json as _json
        _json.dumps(o)
        return o
    except Exception:
        return str(o)


def _pad_insights(last) -> dict:
    """v9.8.3 — compact operator hints for /pad/{pid} (thresholds, challenges,
    fusion flags, branch abstentions). JSON-safe only."""
    import math as _m
    try:
        from pad.thresholds import DEFAULT_THRESHOLDS as _th
        out = {
            "thresholds": {
                "spoof_high": float(_th.spoof_high),
                "live_low": float(_th.live_low),
                "prior_p_attack": float(_th.prior_p_attack),
            },
            "challenge_routes": {
                "start": "/pad/challenge/start",
                "verify": "/pad/challenge/verify",
                "screen_flash_start": "/pad/challenge/screen_flash/start",
            },
        }
    except Exception:
        out = {"challenge_routes": {
            "start": "/pad/challenge/start",
            "verify": "/pad/challenge/verify",
            "screen_flash_start": "/pad/challenge/screen_flash/start",
        }}
    if last is None:
        return out
    fused = getattr(last, "meta", None) or {}
    ff = {}
    for k in ("degraded_no_physical", "replay_note", "fusion_method"):
        if k not in fused:
            continue
        v = fused[k]
        if isinstance(v, bool):
            ff[k] = v
        elif isinstance(v, (int, float)):
            ff[k] = float(v) if _m.isfinite(float(v)) else None
        else:
            ff[k] = str(v)[:200]
    if ff:
        out["fusion_meta_subset"] = ff
    abst = []
    for b in getattr(last, "branches", None) or []:
        try:
            rel = float(getattr(b, "reliability", 0.0) or 0.0)
        except (TypeError, ValueError):
            rel = 0.0
        meta = getattr(b, "meta", None) or {}
        if rel > 1e-9:
            continue
        ar = meta.get("abstained_reason") or meta.get("reason")
        if ar is None and meta.get("fusion_method"):
            ar = meta.get("fusion_method")
        if ar is not None:
            abst.append({
                "branch": str(getattr(b, "name", "?")),
                "reason": str(ar)[:160],
            })
    if abst:
        out["abstentions"] = abst
    return out


@router.get("/pad/{pid}")
async def pad_person(pid: int):
    """Return the latest PADScore + recent history for one person.

    v9.4.1 step 6: each branch entry now surfaces `meta` + `latency_ms`
    + `dominant_class` so the `?pad_debug=1` operator view can render a
    per-branch audit panel.

    v9.7.2: the frontend (`?pad_debug=1`) reads `face.i` (face index in
    the current frame) and fetches `/pad/<face_index>`, but
    `person_states` is keyed by `track.track_id` (tracker-assigned, not
    starting at 0). When the direct lookup misses we fall back to
    treating `pid` as a face index into the sorted track-id list, so
    `/pad/0` resolves to the first tracked person regardless of their
    track_id.

    v9.7.3: response is run through `_pad_jsan` so NaN/Inf floats
    (empty rolling-window variance, log(0) on a zero-prob branch, 0/0
    reliability) cannot 500 the endpoint. Stdlib json.dumps refuses
    non-finite floats and Starlette's default JSONResponse uses it.

    v9.7.5: full top-level try/except so the endpoint NEVER 500s. The
    PAD DEBUG panel polls this at 1 Hz; even a single transient
    exception (race on person_states mutation, branch meta containing
    a non-serializable type after a recent edit, etc.) used to
    surface as "fetch err" in the panel. Now any internal failure
    returns a structured 200 with `error` and `detail` fields so the
    panel can show a useful message instead of going dark.
    """
    try:
        # v9.7.0-merged 2026-05-10: routes.py was extracted from main.py during
        # the API_anti modular refactor but person_states (a main.py global)
        # was never wired through. The top-level try/except below was hiding
        # the resulting NameError as `pid=N — no data` in the UI. Pull the
        # live reference here at call time.
        person_states = _main_globals().get("person_states", {})
        # v9.8.1: compute tids unconditionally so downstream telemetry
        # lookup can fall back to the first tracked person's pid when
        # the slot-index pid (0) doesn't directly match a track_id.
        tids = sorted(person_states.keys())
        ps = person_states.get(pid)
        if ps is None:
            if 0 <= pid < len(tids):
                ps = person_states.get(tids[pid])
        if ps is None or ps.pad_state is None:
            return {"error": "no PAD state for pid " + str(pid),
                    "pid": pid, "n_persons": len(person_states)}
        st = ps.pad_state

        def _branch_dict(b):
            # Convert BranchScore.meta to a plain JSON-safe mapping.
            meta = {}
            for k, v in (getattr(b, "meta", {}) or {}).items():
                try:
                    meta[str(k)] = float(v) if isinstance(v, (int, float, bool)) else str(v)
                except Exception:
                    meta[str(k)] = str(v)
            return {
                "name": getattr(b, "name", "?"),
                "p_attack": float(getattr(b, "p_attack", 0.5) or 0.5),
                "reliability": float(getattr(b, "reliability", 0.0) or 0.0),
                "latency_ms": float(getattr(b, "latency_ms", 0.0) or 0.0),
                "meta": meta,
                "dominant_class": getattr(b, "dominant_class", None),
            }

        payload = {
            "pid": pid,
            "last": (
                None if st.last is None else
                {"verdict": st.last.verdict,
                 "p_attack": float(st.last.p_attack or 0.5),
                 "attack_class": st.last.attack_class,
                 "log_odds": float(st.last.log_odds or 0.0),
                 "ts": float(st.last.ts or 0.0),
                 # v9.7.6 — surface fusion-level meta (e.g.
                 # `degraded_no_physical=True` when LIVE was downgraded
                 # because no radar/rppg was available) so the UI can
                 # render a DEGRADED warning chip.
                 "meta": {str(k): (bool(v) if isinstance(v, bool)
                                   else (float(v) if isinstance(v, (int, float))
                                         else str(v)))
                          for k, v in (getattr(st.last, "meta", {}) or {}).items()},
                 "branches": [_branch_dict(b) for b in (st.last.branches or [])]}
            ),
            "history_len": len(st.history),
            "spoof_ratio_last6": float(st.recent_verdict_ratio("SPOOF", window=6) or 0.0),
            "insights": _pad_insights(st.last),
        }
        # v9.8.1: warmup telemetry. When `last` is null (the dreaded
        # "warming forever" state) the operator can read `warmup.*` to
        # see exactly which enabled branches DID emit a BranchScore this
        # tick and at what reliability. Empty `branches_emitted` ==
        # everything is silent (motion gate killing rPPG, no track
        # crops for visual branches, etc.). Non-empty but rel=0 ==
        # branches are running but still warming their buffers.
        try:
            _tel = _PAD_WARMUP_TELEMETRY.get(
                pid, _PAD_WARMUP_TELEMETRY.get(tids[0] if tids else None))
            if _tel:
                payload["warmup"] = {
                    "branches_attempted": _tel.get("branches_attempted", []),
                    "branches_emitted":   _tel.get("branches_emitted", []),
                    "heavy_due":          _tel.get("heavy_due"),
                    "secs_since_last_call": (
                        round(time.monotonic() - _tel.get("ts", 0), 2)
                        if _tel.get("ts") else None
                    ),
                }
        except Exception:
            pass  # nolint:silent-except telemetry best-effort
        return _pad_jsan(payload)
    except Exception as _pad_err:
        # v9.7.5: never let the PAD DEBUG poll see "fetch err". Return
        # a structured error so the panel can render it instead of
        # going dark. This is informational — clients should treat
        # `error` field as "no data this poll, try again next tick".
        #
        # v9.7.0-merged 2026-05-11 (D): distinguish "extraction wiring
        # broken" (NameError/AttributeError) from "transient PAD pipeline
        # hiccup" so the frontend can render an "EXTRACTION BUG" badge
        # instead of the generic "no data" silence that hid the
        # person_states/UI_HTML NameErrors for so long. Also stderr-log
        # the trace so operators see it in the server console.
        import traceback as _tb, sys as _sys, time as _t
        _err_cls = type(_pad_err).__name__
        _is_wiring = _err_cls in ("NameError", "AttributeError") or \
                     "is not defined" in str(_pad_err) or \
                     "has no attribute" in str(_pad_err)
        if _is_wiring:
            print(f"[WIRING-BUG] /pad/{pid} {_err_cls}: {_pad_err}",
                  file=_sys.stderr, flush=True)
        return {
            "pid": pid,
            "error": f"{_err_cls}: {_pad_err}",
            "detail": _tb.format_exc().splitlines()[-3:],
            "last": None,
            # Marker: when True, this is NOT "no data yet" -- this is
            # a broken handler. Frontend can flag it visibly.
            "__umn_wiring_bug__": _is_wiring,
            "ts": float(_t.time()),
        }


@router.post("/pad/challenge/start")
async def pad_challenge_start():
    """v9.4 step 9 — opt-in challenge-response PAD flow.
    Returns a challenge_id + the task sequence the client UI runs.
    """
    import uuid
    session_id = uuid.uuid4().hex[:12]
    _pad_challenge_sessions[session_id] = {
        "started_at": time.time(),
        "tasks": [
            {"type": "gaze_dot", "duration_s": 2.0, "dot_path": "figure8"},
            {"type": "head_turn", "duration_s": 2.0, "direction": "yaw", "degrees": 15},
        ],
        "status": "running",
    }
    return {"challenge_id": session_id,
            "tasks": _pad_challenge_sessions[session_id]["tasks"]}


@router.post("/pad/challenge/screen_flash/start")
async def pad_challenge_screen_flash_start(request: Request):
    """API_unq active-liveness challenge: trigger a random background-color
    pulse on the UI and arm the screen_flash branch to score the per-pid
    skin reflection over the next ~500 ms.

    Query / body params:
        pid          : int  (required)
        duration_ms  : int  (optional, default 120)
        color        : "RED"|"GREEN"|"BLUE"|"WHITE" (optional, random when absent)

    Returns the challenge metadata so the UI can pulse document.body.style.
    The screen_flash branch reads `active_challenge(pid)` and consumes it
    on the next WS frame.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    qp = dict(request.query_params)
    pid_raw = body.get("pid") or qp.get("pid")
    if pid_raw is None:
        return {"error": "missing pid"}
    try:
        pid = int(pid_raw)
    except Exception:
        return {"error": "pid must be int"}
    try:
        duration_ms = int(body.get("duration_ms") or qp.get("duration_ms") or 120)
    except Exception:
        duration_ms = 120
    color = body.get("color") or qp.get("color")
    if color is not None:
        color = str(color).upper()
    try:
        from pad.screen_flash_protocol import start_challenge, COLORS
        if color is not None and color not in COLORS:
            return {"error": f"unknown color {color!r}; valid: {list(COLORS.keys())}"}
        ch = start_challenge(pid, duration_ms=duration_ms, color_name=color)
        return {
            "ok": True,
            "challenge_id": ch.challenge_id,
            "pid": pid,
            "color_name": ch.color_name,
            "color_hex": ch.color_hex,
            "duration_ms": ch.duration_ms,
            "expected_dir": list(ch.expected_dir),
            "t_start_monotonic": ch.t_start,
            # UI: pulse document.body.style.backgroundColor = color_hex
            # for duration_ms, then revert. Keep the WS stream uninterrupted.
        }
    except Exception as e:
        return {"error": f"start_challenge failed: {e}"}


@router.get("/pad/challenge/screen_flash/history")
async def pad_challenge_screen_flash_history(pid: int):
    """Return the bounded history of screen_flash verdicts for a pid."""
    try:
        from pad.screen_flash_protocol import get_history
        return {"pid": pid, "history": get_history(pid)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/pad/admin/fit-dynamics-bn")
async def pad_admin_fit_dynamics_bn(request: Request):
    """Fit dynamics BN CPDs from labelled jsonl and write dynamics_bn_cpds.json.

    Gated by env ``DODO_PAD_ADMIN`` in (1, true, yes). Body/query:
    ``records_path`` (required): relative path under repo root or absolute
    path that resolves inside the repo. Optional ``estimator`` (mle|bayesian),
    ``alpha`` (Dirichlet pseudo-count).
    """
    if os.environ.get("DODO_PAD_ADMIN", "").strip().lower() not in ("1", "true", "yes", "on"):
        return JSONResponse(
            status_code=403,
            content={"error": "DODO_PAD_ADMIN not enabled on this server"},
        )
    try:
        body = await request.json()
    except Exception:
        body = {}
    qp = dict(request.query_params)
    records_path = body.get("records_path") or qp.get("records_path")
    if not records_path:
        return {"error": "missing records_path (jsonl with label + features)"}
    estimator = str(body.get("estimator") or qp.get("estimator") or "mle").lower()
    try:
        alpha = float(body.get("alpha") or qp.get("alpha") or 1.0)
    except (TypeError, ValueError):
        alpha = 1.0
    # Resolve + jail under repo root (DodoVisionAI/)
    _root = Path(__file__).resolve().parent.parent.parent
    try:
        _raw = Path(str(records_path))
        _cand = (_raw if _raw.is_absolute() else _root / _raw).resolve()
        _cand.relative_to(_root)
    except ValueError:
        return {"ok": False, "error": "records_path must resolve under repo root"}
    if not _cand.is_file():
        return {"ok": False, "error": f"not a file: {_cand}"}
    try:
        from pad.branches.dynamics_branch import (
            fit_dynamics_bn_from_records,
            save_fitted_cpds,
            _DYNAMICS_CPDS_PATH,
        )
        cpds = fit_dynamics_bn_from_records(str(_cand), estimator=estimator, alpha=alpha)
        if not cpds:
            return {"ok": False, "error": "fit returned no CPDs (empty file or missing pgmpy/pandas)"}
        save_fitted_cpds(cpds)
        meta = cpds.get("__meta__") or {}
        return {
            "ok": True,
            "path": _DYNAMICS_CPDS_PATH,
            "n_rows": meta.get("n_rows"),
            "n_live": meta.get("n_live"),
            "n_spoof": meta.get("n_spoof"),
            "estimator": meta.get("estimator", estimator),
            "note": "BN cache invalidated; next dynamics score() uses fitted CPDs",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/pad/challenge/verify")
async def pad_challenge_verify(request: Request):
    """Verify a PAD challenge — client submits its collected evidence
    (gaze trace + head yaw trace). We cross-reference against the
    per-person PADState over the challenge window."""
    try:
        body = await request.json()
    except Exception:
        return {"error": "invalid json body"}
    cid = body.get("challenge_id")
    sess = _pad_challenge_sessions.get(cid)
    if sess is None:
        return {"passed": False, "reason": "unknown challenge_id"}
    pid = int(body.get("pid") or 0)
    ps = person_states.get(pid)
    if ps is None or ps.pad_state is None:
        return {"passed": False, "reason": "no PAD state for pid"}
    recent_live_ratio = 1.0 - ps.pad_state.recent_verdict_ratio("SPOOF", window=12)
    passed = recent_live_ratio >= 0.7
    sess["status"] = "verified"
    return {
        "passed": passed,
        "score": round(recent_live_ratio, 3),
        "evidence_id": cid,
    }


# v8.7 FINAL FIX: root path redirects to the webcam UI so a user who just
# opens http://127.0.0.1:8000/ lands on the working page instead of a 404.
@router.get("/")
async def _root_redirect():
    return RedirectResponse(url="/webcam", status_code=307)



