from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import time
import asyncio
from auth import browser_request_allowed

router = APIRouter()
active_websockets = set()

@router.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    if not browser_request_allowed(websocket):
        await websocket.close(code=1008, reason="untrusted host or browser origin")
        return
    ctx = websocket.app.state.ws_ctx
    _auth = ctx["_auth"]
    cfg = ctx["cfg"]
    _send_envelope = ctx["_send_envelope"]
    _onnx_pool = ctx["_onnx_pool"]
    _process_frame_fast = ctx["_process_frame_fast"]
    _process_frame_full = ctx["_process_frame_full"]
    _process_frame_fast_with_lmk = ctx["_process_frame_fast_with_lmk"]
    _frame_cache = ctx["_frame_cache"]
    _build_envelopes = ctx["_build_envelopes"]
    _REG = ctx["_REG"]
    _ws_active_delta = ctx["_ws_active_delta"]
    _start_time = ctx["_start_time"]
    _replay = ctx["_replay"]
    _ff_mod = ctx["_ff_mod"]
    _MSGPACK_OK = ctx["_MSGPACK_OK"]
    _pad_last_client_fast = ctx["_pad_last_client_fast"]
    _result_from_client_lmk = ctx["_result_from_client_lmk"]
    _runtime_profiler = ctx["_runtime_profiler"]
    _parse_v93_header = ctx["_parse_v93_header"]
    import cv2
    import numpy as np

    # v9.5.x — auth gate. No-op when DODO_AUTH_TOKEN is unset (local dev).
    token_for_rl = None
    if _auth is not None and _auth.enabled():
        token = _auth.extract_from_ws(websocket)
        if not _auth.check(token):
            # 1008 = policy violation
            await websocket.close(code=1008, reason="unauthorized")
            if _REG:
                _REG.counter("ws_auth_rejected_total",
                             "WS connections rejected by auth").inc()
            return
        token_for_rl = token
    # v9.5 (2026-04-22) — per-(token, ip) token-bucket rate limit on connect.
    # Prevents a runaway client from saturating the ONNX thread pool.
    try:
        from rate_limit import get_limiter, key_for_request
        client_ip = ""
        try:
            client_ip = websocket.client.host if websocket.client else ""
        except Exception:
            pass
        rl_key = key_for_request(client_ip, token_for_rl)
        if not await get_limiter().try_acquire(rl_key, cost=1.0):
            await websocket.close(code=4429, reason="rate limited")
            if _REG:
                _REG.counter("ws_rate_limited_total",
                             "WS connections rejected by rate limiter").inc()
            return
    except Exception:
        pass  # degrade open -- limiter not critical for correctness
    await websocket.accept()
    active_websockets.add(websocket)
    
    # v9.3: default to msgpack when the client asks (?format=msgpack) OR
    # when the multiplex flag is on — msgpack is ~3× smaller and critical
    # for the Worker+SAB ring path in step 7.
    _qs = str(websocket.query_params.get("format", ""))
    _multiplex_enabled = _ff_mod.get("WS_MULTIPLEX")
    _use_binary = _MSGPACK_OK and (_qs == "msgpack" or _multiplex_enabled)
    _fmt_label = "msgpack" if _use_binary else "json"
    _mux_label = "mux" if _multiplex_enabled else "legacy"
    print(f"[WS] Client connected (format={_fmt_label}, protocol={_mux_label})")
    # v9.5.x — session replay recorder (enabled via DODO_REPLAY_RECORD=1)
    _recorder = _replay.new_recorder() if _replay is not None else None
    if _REG:
        _REG.counter("ws_connections_total", "WS connections opened").inc()
        _ws_active_delta(+1)
    # v9.7.1: WS backpressure threshold. When CPU spikes, frame
    # processing falls behind realtime; we drop the oldest pending frame
    # rather than building unbounded queue depth that produces a
    # rubber-band UI. Threshold tuned to ~1.5 frame at 30 FPS so the
    # drop only triggers on genuine overload, not jitter.
    _BP_DROP_MS = float(getattr(cfg, "WS_BACKPRESSURE_DROP_MS", 200.0))
    # v9.7.1 — server-side heartbeat. A background task pushes a ping
    # envelope every HEARTBEAT_S seconds so the client liveness watchdog
    # can detect a half-open WS without waiting for the next frame.
    _hb_interval_s = float(getattr(cfg, "WS_HEARTBEAT_S", 5.0))
    _hb_stop = asyncio.Event()
    async def _heartbeat_task():
        try:
            while not _hb_stop.is_set():
                await asyncio.sleep(_hb_interval_s)
                if _hb_stop.is_set():
                    return
                try:
                    await _send_envelope(
                        websocket,
                        {"t": "ping", "ts_server_mono": time.monotonic(),
                         "uptime_s": round(time.time() - _start_time, 1)},
                        _use_binary,
                    )
                except Exception:
                    return  # client gone — let onclose handle it
        except asyncio.CancelledError:
            return
    _hb_handle = asyncio.create_task(_heartbeat_task())
    try:
        while True:
            data = await websocket.receive_bytes()
            _frame_recv_t = time.monotonic()
            if len(data) < 2:
                continue
            # v9.3 step 5: try the multiplex header first; fall through to legacy.
            hdr = _parse_v93_header(data) if _multiplex_enabled else None
            client_lmk = None
            if hdr is not None:
                mode = hdr["mode"]
                jpeg_bytes = hdr["jpeg"] or b""
                client_lmk = hdr["lmk"]
                # v9.4.1 step 3: cache the client-side moiré fast-guard
                # value for the moire_branch to cross-validate against.
                if hdr.get("pad_fast_moire") is not None:
                    _pad_last_client_fast["moire_rgb"] = hdr["pad_fast_moire"]
                    _pad_last_client_fast["ts"] = time.monotonic()
            else:
                mode = data[0]
                jpeg_bytes = data[1:]
            if not jpeg_bytes:
                # v9.3.1 step 4 consumer: mode=2 with client_lmk but no JPEG —
                # we still need the BGR image for rPPG ROI + DeepFace. The
                # client SHOULD send JPEG along with lmk in the same frame
                # (spec allows it); if they only send lmk, fall back to
                # the last cached frame's image so downstream consumers
                # don't fail. Send ack so the client has a receipt.
                await _send_envelope(websocket,
                    {"t": "ack", "mode": mode, "ts": time.monotonic()},
                    _use_binary)
                continue
            image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                await _send_envelope(websocket, {"t": "error", "error": "bad image"}, _use_binary)
                continue
            # v9.7.1: WS backpressure. If we're already > _BP_DROP_MS
            # behind the receive timestamp before processing even
            # starts, drop this frame. Send a tiny "drop" envelope so
            # the client knows to skip this frame index without timing
            # out. Counter exposed via /metrics for ops visibility.
            _frame_age_ms = (time.monotonic() - _frame_recv_t) * 1000.0
            if _frame_age_ms > _BP_DROP_MS:
                if _REG is not None:
                    try:
                        _REG.counter("ws_frames_dropped_total",
                                     "Frames dropped by backpressure",
                                     labels={"reason": "stale_pre_process"}).inc()
                    except Exception:
                        pass  # nolint:silent-except metrics best-effort
                await _send_envelope(websocket,
                    {"t": "drop", "reason": "stale_pre_process",
                     "age_ms": round(_frame_age_ms, 1)},
                    _use_binary)
                continue

            # v8.9: per-frame try/except — graceful degradation on sensor errors
            # v9.5.x: offload the synchronous per-frame pipeline to the ONNX
            # thread pool so MediaPipe / DeepFace / PAD inference don't block
            # the asyncio event loop (= don't block every other WebSocket
            # message for 50-200 ms).
            try:
                _loop = asyncio.get_running_loop()
                if mode == 2 and client_lmk is not None:
                    # v9.3.1 step 4: skip server-side MediaPipe, feed the
                    # client landmarks into the fast pipeline. Saves ~8-15 ms
                    # of server inference per frame for a setup where the
                    # client's on-device tracker is trusted.
                    _client_res = _result_from_client_lmk(client_lmk)
                    result = await _loop.run_in_executor(
                        _onnx_pool, _process_frame_fast_with_lmk,
                        image, _client_res)
                    result["_client_lmk"] = True
                elif mode == 1:
                    result = await _loop.run_in_executor(
                        _onnx_pool, _process_frame_full, image)
                else:
                    result = await _loop.run_in_executor(
                        _onnx_pool, _process_frame_fast, image)
            except Exception as _frame_err:
                print(f"[WS] Frame processing error: {_frame_err}")
                _cached = _frame_cache.get_or_stale(max_age_s=2.0)
                if _cached.get("error") != "stale_frame":
                    _cached["_fallback"] = True
                    _cached["_fallback_reason"] = str(_frame_err)
                    result = _cached
                else:
                    result = {"error": "processing_error", "message": str(_frame_err),
                              "faces": [], "av": {}, "mode": "full" if mode == 1 else "fast", "ms": 0}

            # v9.3 step 5: always send the legacy "frame" envelope (labelled
            # with type tag) so existing clients work unchanged.
            result_env = dict(result)
            result_env.setdefault("t", "frame")
            result_env["ts_server_mono"] = time.monotonic()
            
            # Strict Pydantic Serialization
            from app.models.schemas import WebSocketEnvelope
            result_env = WebSocketEnvelope.model_validate(result_env).model_dump(exclude_none=True)
            
            await _send_envelope(websocket, result_env, _use_binary)
            # v9.5.x — session replay hook + frame metric
            if _recorder is not None:
                _recorder.record(result_env)
            if _REG and isinstance(result.get("ms"), (int, float)):
                _REG.histogram("frame_latency_ms",
                               "Per-frame server processing time").observe(float(result["ms"]))
                _REG.counter("frames_total", "Frames processed",
                             labels={"mode": str(result.get("mode", "fast"))}).inc()
            # v9.7.1: per-frame timing CSV. Buffered + batch-flushed off
            # the hot path; the heavy I/O lands once every ~30 s. Stages
            # captured: total ms (always), plus any *_ms field the
            # processor stamped onto result (e.g. mediapipe_ms,
            # deepface_ms, rppg_ms, pad_ms, ws_ms). Missing fields are
            # skipped — schema grows when callers add timing keys.
            if _runtime_profiler is not None and isinstance(result.get("ms"), (int, float)):
                try:
                    _stages = {k: v for k, v in result.items()
                               if k.endswith("_ms") and isinstance(v, (int, float))}
                    _runtime_profiler.record(
                        mode=str(result.get("mode", "fast")),
                        total_ms=float(result["ms"]),
                        **_stages,
                    )
                except Exception:
                    pass  # nolint:silent-except telemetry best-effort
            if _multiplex_enabled:
                for env in _build_envelopes(result):
                    await _send_envelope(websocket, env, _use_binary)
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        # v9.7.1: stop the heartbeat task before tearing down so it
        # doesn't error on a closed socket.
        try:
            _hb_stop.set()
            _hb_handle.cancel()
        except Exception:
            pass  # nolint:silent-except shutdown best-effort
        if _recorder is not None:
            _recorder.close()
        _ws_active_delta(-1)

# ══════════════════════════════════════════════


