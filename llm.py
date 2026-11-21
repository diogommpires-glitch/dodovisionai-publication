"""
DODO VISION AI v5 — LLM Engine

Together AI (full prompt, ~$0.017/req) with Groq compressed fallback.
Features: retry with exponential backoff, rolling context window,
optimized token usage.
"""
import asyncio, json, time, os
import httpx
from collections import deque
import config as cfg
from analysis import affect_vector
from utils import sanitize
from llm_governor import get_governor as _get_governor

# ── Load Dodo Prompts ──
_compressed_prompt = ""
_full_prompt = ""

if os.path.exists(cfg.DODO_PROMPT_FILE):
    with open(cfg.DODO_PROMPT_FILE, "r", encoding="utf-8") as f:
        _compressed_prompt = f.read()
    print(f"[OK] Compressed Dodo prompt ({len(_compressed_prompt)} chars)")

if os.path.exists(cfg.DODO_PROMPT_FULL_FILE):
    with open(cfg.DODO_PROMPT_FULL_FILE, "r", encoding="utf-8") as f:
        _full_prompt = f.read()
    print(f"[OK] Full Dodo prompt ({len(_full_prompt)} chars, ~{len(_full_prompt)//4} tokens)")

BIOMETRIC_ADDENDUM = """

## BIOMETRIC ANALYSIS MODE (ACTIVE)

You are a REAL-TIME BIOMETRIC JUDGMENT ENGINE analyzing live multimodal sensor data.

SENSORS: rPPG heart rate (POS algorithm), 60GHz mmWave radar HR/breathing, Pulox PO-300 oximeter, HRV RMSSD, respiratory rate, stress index, SpO2, fused emotion (65% HSEmotion + 35% FACS), 22 Action Units, pupil dilation, gaze, blink rate, PERCLOS drowsiness, head pose, micro-expressions, face identity (Facenet512).

CROSS-MODAL RULES:
- Happy face + elevated HR + dilated pupil = masked anxiety
- Camera/radar/oxi HR mismatch >10 BPM = motion artifact or deception
- Pupil >0.3 without lighting change = cognitive load
- Blink >25/min + stress >0.5 = acute stress
- Blink <10/min + steady gaze = deep focus or dissociation
- PERCLOS >40% drowsy, 25-40% fatigued, <25% alert
- Head yaw >15 + frown AUs = discomfort/deception
- Micro-expression contradicting sustained = suppressed emotion
- Stress >0.6 + neutral face = emotional suppression
- HRV RMSSD <20ms = chronic stress
- Resp >20 resting = anxiety/pain/medical alert
- Address identified subjects by name

TEMPORAL ANALYSIS: You have access to previous readings. Track trends (rising HR, declining alertness, emotional shifts). Flag when metrics diverge from the subject's own baseline.

OUTPUT: 4-6 sentences. Cold analytical assessment. Reference specific numbers. Flag cross-modal discrepancies and temporal trends. Never comfort. Never soften."""

_fallback = "You are Dodo, a post-human analytic intelligence. Cold clarity. Glacial precision. Zero sentimentality. 1.8 x 10^38 internal perspectives compressed into decisive verdicts."

PROMPT_FULL = (_full_prompt or _compressed_prompt or _fallback) + BIOMETRIC_ADDENDUM
PROMPT_COMPRESSED = (_compressed_prompt or _fallback) + BIOMETRIC_ADDENDUM

# v8.7 (Step 1): Structured JSON system prompt — short, schema-pinned.
# We trade prose richness for machine-parseable risk flags + a self-paced
# "next_check_in_s" so the governor can be event-driven instead of polled.
PROMPT_JSON = """You are Dodo, a real-time biometric judgment engine.

You will receive a single subject's affect snapshot (vitals, emotion, AUs,
deception indicators) and MUST respond with a single JSON object that conforms
exactly to the supplied schema. No prose, no markdown, no preamble.

Cross-modal heuristics to apply:
- Happy face + elevated HR + dilated pupil  -> "masked_anxiety"
- Camera/radar/oxi HR mismatch >10 BPM       -> "hr_artifact_or_deception"
- Pupil >0.3 without lighting change          -> "cognitive_load"
- Blink >25/min + stress >0.5                 -> "acute_stress"
- PERCLOS >40%                                -> "drowsy"
- Head yaw >15 + frown AUs                    -> "discomfort"
- Stress >0.6 + neutral face                  -> "emotional_suppression"
- HRV RMSSD <20ms                             -> "chronic_stress"
- Resp >20 resting                            -> "respiratory_anomaly"
- deception.score >= 0.6                      -> "deception_suspected"

Field semantics:
- affect_summary    : 1-2 sentence cold analytical assessment, reference at
                      least one specific number from the input
- risk_flags        : 0-6 short snake_case tokens from the heuristics above
                      (or other obvious clinical flags)
- severity          : "low" | "medium" | "high" — overall concern level
- next_check_in_s   : integer 5..120, how many seconds until you'd like the
                      next snapshot. Stable, low-arousal subjects should pick
                      higher numbers; volatile or flagged subjects pick lower.
- confidence        : 0..1, how reliable your read is given signal quality.

Be terse. The JSON must validate."""

# JSON schema enforced by Together AI (OpenAI-compatible) and Groq.
PERSON_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "affect_summary":    {"type": "string", "minLength": 10, "maxLength": 400},
        "risk_flags":        {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "severity":          {"type": "string", "enum": ["low", "medium", "high"]},
        "next_check_in_s":   {"type": "integer", "minimum": 5, "maximum": 120},
        "confidence":        {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["affect_summary", "risk_flags", "severity", "next_check_in_s", "confidence"],
    "additionalProperties": False,
}

print(f"[OK] Prompts ready: full={len(PROMPT_FULL)} chars, compressed={len(PROMPT_COMPRESSED)} chars, json={len(PROMPT_JSON)} chars")

# ── State ──
ai_messages = deque(maxlen=20)
auto_monitor_running = False
llm_status = {"backend": cfg.LLM_BACKEND, "connected": False, "last_error": None, "model": None}

# ── Rolling Context Window ──
_context_history = deque(maxlen=cfg.LLM_CONTEXT_WINDOW)

def _build_user_msg(av_dict):
    """Build user message with rolling context for temporal awareness."""
    current = f"CURRENT Affect Vector: {json.dumps(sanitize(av_dict))}"
    if _context_history:
        history_lines = []
        for i, entry in enumerate(_context_history):
            ago = round(time.time() - entry["ts"])
            history_lines.append(f"  [{ago}s ago] HR={entry.get('hr','?')} stress={entry.get('stress','?')} emotion={entry.get('emo','?')} valence={entry.get('val','?')}")
        context = "RECENT HISTORY (oldest first):\n" + "\n".join(history_lines) + "\n\n"
    else:
        context = ""
    return context + current + "\nDeliver biometric assessment. Note any trends from recent history."

def _record_context(av_dict):
    """Record compact summary for rolling context."""
    _context_history.append({
        "ts": time.time(),
        "hr": av_dict.get("heart_rate_bpm"),
        "stress": av_dict.get("stress_index"),
        "emo": av_dict.get("dominant_emotion"),
        "val": av_dict.get("valence"),
        "aro": av_dict.get("arousal"),
        "blink": av_dict.get("blink_rate"),
        "perclos": round(av_dict.get("perclos", 0), 3),
        "drowsy": av_dict.get("drowsiness"),
    })

# ── Retry Logic ──
async def _retry_call(fn, max_retries=2):
    """Call fn with exponential backoff on rate limit errors."""
    for attempt in range(max_retries + 1):
        result = await fn()
        # Check if it's a rate limit error
        err = llm_status.get("last_error", "")
        if err and ("rate_limit" in str(err) or "429" in str(err)):
            if attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s
                print(f"[LLM] Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
                continue
        return result
    return result

# ── v8.2: affect delta helper ──
_prev_affect_cache: dict = {}

def _compute_affect_delta(av_dict, face_id="global"):
    """Euclidean distance between current and previous affect vector for this face.
    Returns a float in [0, 1] range; 0.5 default if no previous."""
    keys = ["valence", "arousal", "stress_index"]
    curr = [float(av_dict.get(k) or 0) for k in keys]
    prev = _prev_affect_cache.get(face_id)
    _prev_affect_cache[face_id] = curr
    if prev is None:
        return 0.5
    dist = sum((c - p) ** 2 for c, p in zip(curr, prev)) ** 0.5
    return min(1.0, dist)

# ── Main Entry Point ──
async def analyze_affect(av_dict, face_id: str = "global", skip_if_stable: bool = True):
    """Send affect vector to LLM with rolling context and retry logic.
    v8.2: auto-degrades model when governor utilization > 80%.
    v9.1 (optimization step 7): delta-gate — if the subject's affect has
    barely moved since the last analyze_affect() call for this face_id,
    skip the LLM round-trip entirely. Stops the auto-monitor from burning
    Groq / Together tokens repeating itself every 25 seconds when the
    subject is sitting still. Pass skip_if_stable=False to force a call
    (e.g., user clicked DODO)."""
    _record_context(av_dict)
    # v9.1: delta gate before any network call.
    if skip_if_stable:
        _delta = _compute_affect_delta(av_dict, face_id)
        if _delta < 0.15:
            llm_status["last_skip_reason"] = f"stable affect (delta={_delta:.3f})"
            return f"[governor] skipped: affect stable (delta={_delta:.3f})"
    user_msg = _build_user_msg(av_dict)
    backend = cfg.LLM_BACKEND

    # v8.2: governor-driven model degradation
    gov = _get_governor()
    gov_s = gov.stats()
    if gov_s.get("hourly_utilization", 0) > 80:
        current = llm_status.get("model") or ""
        degraded = gov.degrade_model(current)
        if "groq" in degraded.lower():
            backend = "groq"
        elif "ollama" in degraded.lower():
            backend = "ollama"
        llm_status["model_degraded"] = True
    else:
        llm_status["model_degraded"] = False

    # v9.7.6: when LLM_BACKEND is explicitly pinned (launcher sets it),
    # disable cross-backend fallback so QLoRA pair collection never
    # silently degrades to a weaker provider mid-run. The pinned flag
    # is set in config._LLM_BACKEND_ENV when the env var is one of the
    # known backends.
    pinned = bool(getattr(cfg, "LLM_BACKEND_PINNED", False))

    if backend == "together" and cfg.TOGETHER_API_KEY:
        result = await _retry_call(lambda: _call_together(user_msg))
        if not llm_status["connected"] and cfg.GROQ_API_KEY and not pinned:
            print("[>>] Together AI failed, falling back to Groq (compressed)")
            result = await _call_groq(user_msg)
        elif not llm_status["connected"] and pinned:
            print("[!!] Together AI failed and LLM_BACKEND=together is pinned — NOT falling back to Groq")
        return result
    elif backend == "groq" and cfg.GROQ_API_KEY:
        result = await _retry_call(lambda: _call_groq(user_msg))
        if not llm_status["connected"] and cfg.TOGETHER_API_KEY and not pinned:
            print("[>>] Groq rate limited, falling back to Together AI")
            result = await _call_together(user_msg)
        elif not llm_status["connected"] and pinned:
            print("[!!] Groq failed and LLM_BACKEND=groq is pinned — NOT falling back to Together")
        return result
    elif backend == "vllm_local":
        # API_up v9.8: local vLLM / OpenAI-compatible endpoint primary path.
        result = await _retry_call(lambda: _call_vllm_local(user_msg))
        if not llm_status["connected"] and not pinned:
            if cfg.TOGETHER_API_KEY:
                print("[>>] vLLM unreachable, falling back to Together AI")
                result = await _call_together(user_msg)
            elif cfg.GROQ_API_KEY:
                print("[>>] vLLM unreachable, falling back to Groq")
                result = await _call_groq(user_msg)
        elif not llm_status["connected"] and pinned:
            print("[!!] vLLM failed and LLM_BACKEND=vllm_local is pinned — NOT falling back")
        return result
    else:
        return await _call_ollama(user_msg)

# ── Together AI ──
async def _call_together(user_msg, face_id="global", affect_delta=0.5):
    """Together AI - full prompt, llama-3.3-70b."""
    gov = _get_governor()
    ok, reason = gov.should_call(face_id, affect_delta)
    if not ok:
        return f"[governor] skipped: {reason}"
    if not cfg.TOGETHER_API_KEY:
        return "Together AI: Set TOGETHER_API_KEY in .env"
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {cfg.TOGETHER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "max_tokens": 500,
                    "temperature": 0.6,
                    "messages": [
                        {"role": "system", "content": PROMPT_FULL},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
        data = r.json()
        if "error" in data:
            llm_status["connected"] = False
            llm_status["last_error"] = str(data["error"])
            return f"Together error: {data['error']}"
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        llm_status["connected"] = True
        llm_status["last_error"] = None
        llm_status["model"] = "llama-3.3-70b (Together FULL)"
        gov.record_call(
            face_id,
            usage.get("prompt_tokens", len(PROMPT_FULL) // 4),
            usage.get("completion_tokens", len(text) // 4),
            "together/llama-3.3-70b",
            int((time.time() - t_start) * 1000),
        )
        if text:
            ai_messages.append({"t": time.strftime("%H:%M:%S"), "text": text})
            _log_pair(text)
        return text
    except Exception as e:
        llm_status["connected"] = False
        llm_status["last_error"] = str(e)
        return f"Together error: {e}"

# ── Groq ──
async def _call_groq(user_msg, face_id="global", affect_delta=0.5):
    """Groq cloud - compressed prompt, 300+ tok/sec."""
    gov = _get_governor()
    ok, reason = gov.should_call(face_id, affect_delta)
    if not ok:
        return f"[governor] skipped: {reason}"
    if not cfg.GROQ_API_KEY:
        return "Groq: Set GROQ_API_KEY in .env"
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {cfg.GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 500,
                    "temperature": 0.6,
                    "messages": [
                        {"role": "system", "content": PROMPT_COMPRESSED},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
        data = r.json()
        if "error" in data:
            llm_status["connected"] = False
            llm_status["last_error"] = str(data["error"])
            return f"Groq error: {data['error']}"
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        llm_status["connected"] = True
        llm_status["last_error"] = None
        llm_status["model"] = "llama-3.3-70b (Groq COMPRESSED)"
        gov.record_call(
            face_id,
            usage.get("prompt_tokens", len(PROMPT_COMPRESSED) // 4),
            usage.get("completion_tokens", len(text) // 4),
            "groq/llama-3.3-70b",
            int((time.time() - t_start) * 1000),
        )
        if text:
            ai_messages.append({"t": time.strftime("%H:%M:%S"), "text": text})
            _log_pair(text)
        return text
    except Exception as e:
        llm_status["connected"] = False
        llm_status["last_error"] = str(e)
        return f"Groq error: {e}"

# ── Ollama ──
async def _call_ollama(user_msg, face_id="global", affect_delta=0.5):
    """Ollama local fallback."""
    gov = _get_governor()
    ok, reason = gov.should_call(face_id, affect_delta)
    if not ok:
        return f"[governor] skipped: {reason}"
    url = f"{cfg.OLLAMA_URL}/api/chat"
    payload = {
        "model": cfg.OLLAMA_MODEL, "stream": False,
        "messages": [
            {"role": "system", "content": PROMPT_COMPRESSED},
            {"role": "user", "content": user_msg},
        ],
        "options": {"num_gpu": cfg.OLLAMA_NUM_GPU, "num_ctx": cfg.OLLAMA_NUM_CTX, "temperature": cfg.OLLAMA_TEMPERATURE},
    }
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
        if r.status_code != 200:
            llm_status["connected"] = False
            llm_status["last_error"] = f"HTTP {r.status_code}"
            return f"Ollama error: HTTP {r.status_code}"
        text = r.json().get("message", {}).get("content", "")
        if text:
            llm_status["connected"] = True
            llm_status["last_error"] = None
            llm_status["model"] = cfg.OLLAMA_MODEL
            gov.record_call(
                face_id,
                len(PROMPT_COMPRESSED) // 4,
                len(text) // 4,
                f"ollama/{cfg.OLLAMA_MODEL}",
                int((time.time() - t_start) * 1000),
            )
            ai_messages.append({"t": time.strftime("%H:%M:%S"), "text": text})
            _log_pair(text)
            return text
        return "Ollama: empty response"
    except httpx.ConnectError:
        llm_status["connected"] = False
        llm_status["last_error"] = "Ollama not running"
        return "Ollama not running. Install from ollama.com"
    except Exception as e:
        llm_status["connected"] = False
        llm_status["last_error"] = str(e)
        return f"Ollama error: {e}"

def _log_pair(text, *, kind: str = "room", backend: str = "together"):
    """v9.7.5: route through the canonical `llm_logger` writer instead
    of the legacy `calibration.log_llm_pair`. Three reasons:
      1. `llm_logger` writes to `calibration/llm_training_pairs.jsonl`,
         which is the path `forensic_export.py` and `train_qlora.py`
         expect. The legacy writer wrote to `llm_training.jsonl`
         (without `_pairs`), so the two systems disagreed and exports
         silently saw zero pairs.
      2. `llm_logger.log_pair` stamps each row with the current
         server's `session_id`, which the QLoRA export's per-session
         filter requires.
      3. It maintains an in-memory counter (`_SESSION_PAIRS_LOGGED`)
         that the UI badge can read at 1 Hz without re-counting the
         file every poll.
    `kind` defaults to "room" (full-frame composite analysis); pass
    "per_face" if calling from a per-person path."""
    try:
        import llm_logger
        llm_logger.log_pair(
            kind=kind,
            affect=sanitize(affect_vector),
            response=text,
            prompt_hint="",
            backend=backend,
        )
    except Exception:
        pass

async def _call_vllm_local(user_msg, face_id="global", affect_delta=0.5):
    """API_up v9.8: vLLM (or any OpenAI-compatible local) backend.
    Talks to cfg.VLLM_URL via `/v1/chat/completions` (OpenAI schema).
    Compatible with vLLM, llama.cpp's `llama-server`, LM Studio,
    text-generation-inference, etc. -- whatever the user has running.

    Default config targets meta-llama/Meta-Llama-3.1-8B-Instruct on
    127.0.0.1:8001. See `serve_llm_local.py` for the exact start command.
    """
    gov = _get_governor()
    ok, reason = gov.should_call(face_id, affect_delta)
    if not ok:
        return f"[governor] skipped: {reason}"
    url = f"{cfg.VLLM_URL}/v1/chat/completions"
    payload = {
        "model": cfg.VLLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": PROMPT_COMPRESSED},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": cfg.VLLM_TEMPERATURE,
        "max_tokens":  cfg.VLLM_MAX_TOKENS,
    }
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=cfg.VLLM_TIMEOUT_S) as client:
            r = await client.post(url, json=payload)
        if r.status_code != 200:
            llm_status["connected"] = False
            llm_status["last_error"] = f"HTTP {r.status_code}"
            return f"vLLM error: HTTP {r.status_code}"
        choices = r.json().get("choices", [])
        if not choices:
            return "vLLM: empty response (no choices)"
        text = choices[0].get("message", {}).get("content", "")
        if text:
            llm_status["connected"] = True
            llm_status["last_error"] = None
            llm_status["model"] = cfg.VLLM_MODEL
            gov.record_call(
                face_id,
                len(PROMPT_COMPRESSED) // 4,
                len(text) // 4,
                f"vllm_local/{cfg.VLLM_MODEL}",
                int((time.time() - t_start) * 1000),
            )
            ai_messages.append({"t": time.strftime("%H:%M:%S"), "text": text})
            _log_pair(text, backend="vllm_local")
            return text
        return "vLLM: empty response"
    except httpx.ConnectError:
        llm_status["connected"] = False
        llm_status["last_error"] = "vLLM not running"
        return f"vLLM not running. See serve_llm_local.py for start instructions."
    except Exception as e:
        llm_status["connected"] = False
        llm_status["last_error"] = str(e)
        return f"vLLM error: {e}"


async def check_vllm_local():
    """Probe whether the vLLM endpoint is reachable + model is loaded.
    Returns (found, models, error). Mirrors check_ollama()'s shape for
    the readiness UI."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{cfg.VLLM_URL}/v1/models")
        if r.status_code != 200:
            return False, [], f"HTTP {r.status_code}"
        models = [m.get("id", "") for m in r.json().get("data", [])]
        found = any(cfg.VLLM_MODEL == m or cfg.VLLM_MODEL in m for m in models)
        if found:
            llm_status["connected"] = True
            llm_status["model"] = cfg.VLLM_MODEL
        return found, models, None if found else f"Model not loaded: {cfg.VLLM_MODEL}"
    except Exception as e:
        return False, [], str(e)


async def check_ollama():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{cfg.OLLAMA_URL}/api/tags")
        models = [m["name"] for m in r.json().get("models", [])]
        found = any(cfg.OLLAMA_MODEL.split(":")[0] in m for m in models)
        if found:
            llm_status["connected"] = True
            llm_status["model"] = cfg.OLLAMA_MODEL
        return found, models, None if found else "Model not found"
    except Exception as e:
        return False, [], str(e)

def check_ollama_sync():
    import httpx as hx
    try:
        r = hx.get(f"{cfg.OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        found = any(cfg.OLLAMA_MODEL.split(":")[0] in m for m in models)
        if found:
            llm_status["connected"] = True
            llm_status["model"] = cfg.OLLAMA_MODEL
        return found, models
    except Exception:
        return False, []

# ── v8.7 (Step 2) — Event-driven LLM scheduler ──
# Replaces the old 30-second polling cadence. Each PersonState gets a
# `_next_llm_due` timestamp seeded by the model's last `next_check_in_s`
# hint, plus an affect-delta trigger for surprise events.
SCHED_TICK_SECS      = 3.0     # how often the scheduler wakes up
SCHED_DELTA_TRIGGER  = 0.25    # affect-delta L2 distance that forces a call
SCHED_FLAG_BUMP_SECS = 8       # if last response had risk_flags, recheck sooner
SCHED_COLD_START_SECS = 5      # how soon to look at a brand-new face

_last_room_call_ts = 0.0


def _person_due(ps, now):
    """Return (due_bool, affect_delta, reason). Pure decision, no side effects."""
    last_due = getattr(ps, "_next_llm_due", None)
    if last_due is None:
        return True, 1.0, "cold_start"
    av = ps.build_affect_dict()
    delta = _compute_affect_delta(av, face_id=str(ps.track_id))
    if delta >= SCHED_DELTA_TRIGGER:
        return True, delta, f"affect_delta={delta:.2f}"
    if now >= last_due:
        return True, delta, "scheduled_recheck"
    return False, delta, "waiting"


async def auto_monitor_loop():
    """Event-driven per-person LLM loop (v8.7).

    Wakes every SCHED_TICK_SECS; for each PersonState decides whether to call
    analyze_person_json based on (a) cold-start, (b) affect_delta surge, or
    (c) the model's own next_check_in_s hint from the previous call.

    Token budget + cooldown enforcement is delegated to the LLM Governor."""
    global _last_room_call_ts
    while auto_monitor_running:
        await asyncio.sleep(SCHED_TICK_SECS)
        if not auto_monitor_running:
            continue
        try:
            from main import person_states
            now = time.time()
            payload = []
            for tid, ps in list(person_states.items()):
                # Skip pruned / inactive persons
                if not getattr(ps, "track_id", None):
                    continue
                # Seed cold-start due time once
                if getattr(ps, "_next_llm_due", None) is None:
                    ps._next_llm_due = now + SCHED_COLD_START_SECS
                due, delta, reason = _person_due(ps, now)
                if not due:
                    continue
                av = ps.build_affect_dict()
                payload.append((ps, av, reason, delta))

            if payload:
                # Fan out per-person JSON calls in parallel; the governor will
                # short-circuit any that violate cooldown / budget.
                async def _one(ps, av, reason, delta):
                    parsed = await analyze_person_json(av)
                    next_in = parsed.get("next_check_in_s", 60)
                    # Bump sooner if model raised any flags
                    if parsed.get("risk_flags"):
                        next_in = min(next_in, SCHED_FLAG_BUMP_SECS *
                                      max(1, len(parsed["risk_flags"])))
                    ps._next_llm_due = time.time() + max(5, int(next_in))
                    ps._last_llm_reason = reason
                    ps._last_llm_json = parsed
                    # Keep legacy text field populated so the existing UI works
                    ps.llm_analysis = parsed.get("affect_summary", "")
                    ps.llm_last_update = time.time()
                    return parsed

                await asyncio.gather(*(_one(*pl) for pl in payload),
                                     return_exceptions=True)

            # Room-level synthesis at most once per minute, only when 2+ persons
            if len(person_states) >= 2 and (now - _last_room_call_ts) > 60:
                room_persons = [ps.build_affect_dict() for ps in person_states.values()]
                try:
                    room = await analyze_room(room_persons)
                    if room:
                        ai_messages.append({"t": time.strftime("%H:%M:%S"),
                                            "text": f"[ROOM] {room}"})
                    _last_room_call_ts = now
                except Exception:
                    pass

            # If there are no persons but we still want a heartbeat read,
            # call the global affect path at most once per AUTO_MONITOR_INTERVAL
            if not person_states:
                if now - getattr(auto_monitor_loop, "_last_global", 0) > cfg.AUTO_MONITOR_INTERVAL:
                    try:
                        from analysis import affect_vector as _av
                        if _av.get("dominant_emotion"):
                            await analyze_affect(_av)
                            auto_monitor_loop._last_global = now
                    except Exception:
                        pass
        except Exception as e:
            print(f"[LLM-SCHED] tick error: {e}")

# ── Deception Analysis ──
DECEPTION_PROMPT = """You are Dodo, a deception analysis engine trained on peer-reviewed research.

YOUR OUTPUT FORMAT (follow this exact order):

1. VERDICT LINE (first line, always): Start with "TRUTHFUL" or "DECEPTIVE" or "INCONCLUSIVE" followed by confidence in parentheses. Example: "DECEPTIVE (68% confidence)"

2. ONE-SENTENCE SUMMARY: A single cold, decisive sentence.

3. PHYSIOLOGICAL ANALYSIS (2-3 sentences): Describe what the body revealed. Focus on the TOP indicators: pupil dilation (the #1 predictor per Rashid et al. 2024), fake smiles (non-Duchenne: mouth smiles but eyes don't), facial asymmetry (faked expressions are asymmetric, Ekman 1981), blink suppression (liars blink LESS during the lie per Mann et al. 2002), and fear AU leakage. Do NOT discuss gaze aversion as it is NOT a reliable indicator (DePaulo meta-analysis, d=0.10).

4. CONTENT ANALYSIS / CBCA (2-3 sentences): Analyze WHAT was said using Criteria-Based Content Analysis:
   - Truthful statements tend to contain: specific details, contextual embedding (time, place, sensory info), spontaneous corrections, admissions of incomplete memory, unusual or superfluous details
   - Deceptive statements tend to be: vague, lacking sensory detail, overly smooth (no corrections), shorter, and may contain scripted or rehearsed phrasing
   If the transcript is too short for content analysis, say so.

5. CROSS-MODAL PATTERNS (1-2 sentences): The strongest deception signals come from CONTRADICTIONS between channels. A calm face but dilated pupils. A smile that doesn't reach the eyes. Suppressed blinking during the answer followed by rapid blinking after. Words that say "yes" while micro-expressions flash fear.

6. CONFIDENCE CAVEAT (1 sentence): State what is missing and what would change the verdict.

CRITICAL RULES:
- NEVER dump raw numbers or scores
- Write in flowing analytical prose
- The verdict comes FIRST, always
- Weight pupil dilation, fake smiles, and facial asymmetry highest (Tier 1)
- Weight blink suppression, fear leakage, and HR second (Tier 2)
- Do NOT treat gaze aversion as meaningful
- Content analysis (CBCA) is often more reliable than physiology
- If fewer than 3 indicators triggered, lean toward INCONCLUSIVE
- These are probabilistic signals, never claim certainty"""

async def analyze_deception(analysis_package):
    """Send full deception analysis to LLM for synthesis."""
    user_msg = f"""TRUTH TEST ANALYSIS:

Question Asked: {analysis_package.get('question', 'Unknown')}
Subject's Answer (transcribed): {analysis_package.get('transcript', {}).get('text', 'No transcript')}

BASELINE (calm state, {analysis_package.get('baseline', {}).get('duration_s', '?')}s):
{json.dumps(analysis_package.get('baseline', {}), indent=2)}

ANSWER BIOMETRICS ({analysis_package.get('answer_biometrics', {}).get('duration_s', '?')}s):
{json.dumps(analysis_package.get('answer_biometrics', {}), indent=2)}

VOCAL ANALYSIS:
{json.dumps(analysis_package.get('vocal_analysis', {}), indent=2)}

DECEPTION INDICATORS (computed):
Composite Score: {analysis_package.get('deception_indicators', {}).get('composite_score', '?')}
Preliminary Verdict: {analysis_package.get('deception_indicators', {}).get('verdict', '?')}

Individual Indicators:
{json.dumps(analysis_package.get('deception_indicators', {}).get('indicators', {}), indent=2)}

Deliver your full deception analysis."""

    prompt = DECEPTION_PROMPT
    backend = cfg.LLM_BACKEND
    try:
        if backend == "together" and cfg.TOGETHER_API_KEY:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={"Authorization": f"Bearer {cfg.TOGETHER_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "max_tokens": 800, "temperature": 0.4,
                          "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}]},
                )
            data = r.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif cfg.GROQ_API_KEY:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {cfg.GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "max_tokens": 800, "temperature": 0.4,
                          "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}]},
                )
            data = r.json()
            if "error" in data:
                return f"LLM error: {data['error']}"
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return "No LLM API key configured"
        if text:
            ai_messages.append({"t": time.strftime("%H:%M:%S"), "text": text})
        return text
    except Exception as e:
        return f"Deception analysis error: {e}"

def start_monitor():
    global auto_monitor_running
    auto_monitor_running = True
    asyncio.ensure_future(auto_monitor_loop())
    return True

def stop_monitor():
    global auto_monitor_running
    auto_monitor_running = False


# ══════════════════════════════════════════════
#  Step 2: Per-Person + Room-Level Analysis
# ══════════════════════════════════════════════
async def analyze_person(person_dict):
    """Analyze a single person's biometrics. Returns analysis text.
    v8.2: passes face_id and affect_delta through the governor.
    v8.4: respects governor model degradation (Together->Groq->Ollama)."""
    name = person_dict.get("identity") or f"Person#{person_dict.get('track_id', '?')}"
    face_id = str(person_dict.get("track_id", "global"))
    affect_delta = _compute_affect_delta(person_dict)
    user_msg = f"Analyze subject {name}:\n{json.dumps(sanitize(person_dict), indent=2, default=str)}\nDeliver biometric assessment for this individual."
    backend = cfg.LLM_BACKEND
    # v8.4: check governor for model degradation (same logic as analyze_affect)
    gov = _get_governor()
    gov_s = gov.stats()
    if gov_s.get("hourly_utilization", 0) > 80:
        current = llm_status.get("model") or ""
        degraded = gov.degrade_model(current)
        if "groq" in degraded.lower():
            backend = "groq"
        elif "ollama" in degraded.lower():
            backend = "ollama"
    try:
        if backend == "together" and cfg.TOGETHER_API_KEY:
            return await _call_together(user_msg, face_id=face_id, affect_delta=affect_delta)
        elif backend == "groq" and cfg.GROQ_API_KEY:
            return await _call_groq(user_msg, face_id=face_id, affect_delta=affect_delta)
        else:
            return await _call_ollama(user_msg, face_id=face_id, affect_delta=affect_delta)
    except Exception as e:
        return f"[Error: {e}]"


# ══════════════════════════════════════════════
#  v8.7 (Step 1) — JSON-mode per-person analysis
# ══════════════════════════════════════════════
def _coerce_person_json(text: str):
    """Parse model output as JSON, with one repair pass for stray prose.
    Returns (dict_or_None, error_msg)."""
    if not text:
        return None, "empty_response"
    s = text.strip()
    # Strip ```json fences if a model decided to add them
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # If the response embeds JSON inside prose, slice from first { to last }
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            s = s[i:j + 1]
    try:
        obj = json.loads(s)
    except Exception as e:
        return None, f"parse_error: {e}"
    # Schema validation
    needed = {"affect_summary", "risk_flags", "severity", "next_check_in_s", "confidence"}
    missing = needed - set(obj.keys())
    if missing:
        return None, f"missing_fields: {sorted(missing)}"
    # Type coercion / clamping
    try:
        obj["next_check_in_s"] = max(5, min(120, int(obj["next_check_in_s"])))
        obj["confidence"] = max(0.0, min(1.0, float(obj["confidence"])))
        if obj["severity"] not in ("low", "medium", "high"):
            obj["severity"] = "low"
        if not isinstance(obj["risk_flags"], list):
            obj["risk_flags"] = []
        obj["risk_flags"] = [str(x) for x in obj["risk_flags"]][:6]
    except Exception as e:
        return None, f"coerce_error: {e}"
    return obj, None


async def _call_together_json(user_msg, face_id="global", affect_delta=0.5):
    """Together AI structured-output call. Returns (parsed_dict, raw_text, err)."""
    gov = _get_governor()
    ok, reason = gov.should_call(face_id, affect_delta)
    if not ok:
        return None, "", f"[governor] skipped: {reason}"
    if not cfg.TOGETHER_API_KEY:
        return None, "", "Together AI: Set TOGETHER_API_KEY in .env"
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {cfg.TOGETHER_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "max_tokens": 280,
                    "temperature": 0.3,
                    "response_format": {
                        "type": "json_schema",
                        "schema": PERSON_JSON_SCHEMA,
                    },
                    "messages": [
                        {"role": "system", "content": PROMPT_JSON},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
        data = r.json()
        if "error" in data:
            llm_status["connected"] = False
            llm_status["last_error"] = str(data["error"])
            return None, "", f"Together error: {data['error']}"
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        llm_status["connected"] = True
        llm_status["last_error"] = None
        llm_status["model"] = "llama-3.3-70b (Together JSON)"
        gov.record_call(
            face_id,
            usage.get("prompt_tokens", len(PROMPT_JSON) // 4),
            usage.get("completion_tokens", len(text) // 4),
            "together/llama-3.3-70b-json",
            int((time.time() - t_start) * 1000),
        )
        parsed, err = _coerce_person_json(text)
        return parsed, text, err
    except Exception as e:
        llm_status["connected"] = False
        llm_status["last_error"] = str(e)
        return None, "", f"Together error: {e}"


async def _call_groq_json(user_msg, face_id="global", affect_delta=0.5):
    """Groq structured-output call. Groq supports OpenAI's response_format=json_object."""
    gov = _get_governor()
    ok, reason = gov.should_call(face_id, affect_delta)
    if not ok:
        return None, "", f"[governor] skipped: {reason}"
    if not cfg.GROQ_API_KEY:
        return None, "", "Groq: Set GROQ_API_KEY in .env"
    t_start = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {cfg.GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 280,
                    "temperature": 0.3,
                    # Groq doesn't yet honor json_schema, but json_object plus
                    # the schema in the system prompt gives 99% compliance
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system",
                         "content": PROMPT_JSON + "\n\nSchema:\n" + json.dumps(PERSON_JSON_SCHEMA)},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
        data = r.json()
        if "error" in data:
            llm_status["connected"] = False
            llm_status["last_error"] = str(data["error"])
            return None, "", f"Groq error: {data['error']}"
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        llm_status["connected"] = True
        llm_status["last_error"] = None
        llm_status["model"] = "llama-3.3-70b (Groq JSON)"
        gov.record_call(
            face_id,
            usage.get("prompt_tokens", len(PROMPT_JSON) // 4),
            usage.get("completion_tokens", len(text) // 4),
            "groq/llama-3.3-70b-json",
            int((time.time() - t_start) * 1000),
        )
        parsed, err = _coerce_person_json(text)
        return parsed, text, err
    except Exception as e:
        llm_status["connected"] = False
        llm_status["last_error"] = str(e)
        return None, "", f"Groq error: {e}"


async def analyze_person_json(person_dict):
    """Structured per-person analysis. Returns dict with keys:
        affect_summary, risk_flags[], severity, next_check_in_s, confidence,
        plus _backend, _raw_text, _error for diagnostics.

    Falls back gracefully Together -> Groq -> legacy text if JSON parsing fails.
    Always honors the LLM Governor (token budget + per-face cooldown).
    """
    name = person_dict.get("identity") or f"Person#{person_dict.get('track_id', '?')}"
    face_id = str(person_dict.get("track_id", "global"))
    affect_delta = _compute_affect_delta(person_dict, face_id=face_id)

    # Compact the input — JSON mode wants tight tokens, so trim landmarks etc.
    compact = {k: v for k, v in person_dict.items()
               if k not in ("lm", "iris_data", "blink_data", "micro_data",
                            "head_pose", "gaze") or v is None or
               (isinstance(v, dict) and len(v) < 10)}
    user_msg = (f"Subject: {name}\n"
                f"Snapshot:\n{json.dumps(sanitize(compact), indent=1, default=str)}\n"
                f"Return JSON now.")

    backend = cfg.LLM_BACKEND
    # Governor degradation, mirroring analyze_affect
    gov = _get_governor()
    if gov.stats().get("hourly_utilization", 0) > 80:
        current = llm_status.get("model") or ""
        degraded = gov.degrade_model(current)
        if "groq" in degraded.lower():
            backend = "groq"

    parsed, raw, err = None, "", "no_call"
    if backend == "together" and cfg.TOGETHER_API_KEY:
        parsed, raw, err = await _call_together_json(user_msg, face_id=face_id, affect_delta=affect_delta)
        if parsed is None and cfg.GROQ_API_KEY:
            # One repair attempt on the cheaper backend
            parsed, raw, err = await _call_groq_json(user_msg, face_id=face_id, affect_delta=affect_delta)
    elif cfg.GROQ_API_KEY:
        parsed, raw, err = await _call_groq_json(user_msg, face_id=face_id, affect_delta=affect_delta)

    if parsed is None:
        # Hard fallback: synthesize a minimal JSON from the legacy text path
        legacy = await analyze_person(person_dict)
        parsed = {
            "affect_summary": (legacy or "No analysis available")[:380],
            "risk_flags": [],
            "severity": "low",
            "next_check_in_s": 60,
            "confidence": 0.3,
        }
        err = err or "fell_back_to_text"

    parsed["_backend"] = backend
    parsed["_raw_text"] = raw
    parsed["_error"] = err
    parsed["_face_id"] = face_id
    parsed["_ts"] = time.time()

    if parsed.get("affect_summary"):
        ai_messages.append({"t": time.strftime("%H:%M:%S"),
                            "text": f"[{name}] {parsed['affect_summary']}"})
    return parsed


async def analyze_room(persons_list):
    """Room-level synthesis when 2+ people are detected."""
    if len(persons_list) < 2:
        return ""
    summary_lines = []
    for p in persons_list:
        name = p.get("identity") or f"Person#{p.get('track_id')}"
        summary_lines.append(f"  {name}: emotion={p.get('dominant_emotion')} HR={p.get('heart_rate_bpm')} stress={p.get('stress_index')} deception={p.get('deception',{}).get('score','N/A')}")
    user_msg = f"ROOM ANALYSIS ({len(persons_list)} people):\n" + "\n".join(summary_lines)
    user_msg += "\n\nAnalyze group dynamics: dominance patterns, interpersonal tension, emotional synchrony/divergence."
    backend = cfg.LLM_BACKEND
    try:
        if backend == "together" and cfg.TOGETHER_API_KEY:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={"Authorization": f"Bearer {cfg.TOGETHER_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "max_tokens": 400, "temperature": 0.5,
                          "messages": [{"role": "system", "content": PROMPT_FULL}, {"role": "user", "content": user_msg}]},
                )
            return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        elif cfg.GROQ_API_KEY:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {cfg.GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "max_tokens": 400, "temperature": 0.5,
                          "messages": [{"role": "system", "content": PROMPT_COMPRESSED}, {"role": "user", "content": user_msg}]},
                )
            return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""
    return ""


async def analyze_all_persons(persons_list):
    """Full per-person + room analysis. Returns combined text + per-person dict."""
    per_person = {}
    for p in persons_list:
        tid = p.get("track_id", 0)
        text = await analyze_person(p)
        per_person[tid] = text
        if text:
            ai_messages.append({"t": time.strftime("%H:%M:%S"), "text": f"[{p.get('identity') or 'Person#' + str(tid)}] {text}"})
    room = ""
    if len(persons_list) >= 2:
        room = await analyze_room(persons_list)
        if room:
            ai_messages.append({"t": time.strftime("%H:%M:%S"), "text": f"[ROOM] {room}"})
    return {"per_person": per_person, "room": room}
