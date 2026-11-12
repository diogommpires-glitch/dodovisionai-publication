"""
DODO VISION AI v8 — LLM Governor

Thread-safe rate limiter for LLM calls with:
  - Per-face cooldown (widens when affect_delta is low)
  - Rolling 1-hour token budget
  - Model degradation chain (Together → Groq → Ollama)
  - Explicit consent tracking

Public API:
    governor = LLMGovernor(budget_tokens_per_hour=10000, per_face_min_cooldown_s=25)
    should_call, reason = governor.should_call(face_id, affect_delta)
    governor.record_call(face_id, prompt_tokens, response_tokens, model, latency_ms)
    fallback_model = governor.degrade_model(current_model)
    stats = governor.stats()
"""
from __future__ import annotations
import time
import threading
from collections import deque
from typing import Dict, Tuple, Optional, Deque

try:
    import config as cfg
except ImportError:
    cfg = None  # type: ignore


class LLMGovernor:
    def __init__(
        self,
        budget_tokens_per_hour: int = 10000,
        per_face_min_cooldown_s: float = 25,
        max_cooldown_s: float = 120,
    ) -> None:
        self.budget_tokens_per_hour = budget_tokens_per_hour
        self.per_face_min_cooldown_s = per_face_min_cooldown_s
        self.max_cooldown_s = max_cooldown_s

        # Rolling 1-hour token window
        self._token_history: Deque[Tuple[float, int]] = deque()
        self._lock = threading.Lock()

        # Per-face call tracking: {face_id -> (last_call_time, tokens_in_session)}
        self._last_call_time: Dict[str, float] = {}
        self._session_tokens: Dict[str, int] = {}

    def should_call(
        self, face_id: str, affect_delta: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Decide if we should call the LLM for this face.

        Parameters
        ----------
        face_id : str
            Unique face identifier.
        affect_delta : float
            Valence/arousal/stress change in [0, 1]. Low delta (stable state)
            widens cooldown; high delta (dynamic) allows faster re-queries.

        Returns
        -------
        (should_call, reason)
            If should_call=True, we have budget and cooldown has elapsed.
            Reason is a diagnostic string.
        """
        with self._lock:
            # 1. Check hourly budget
            now = time.time()
            cutoff = now - 3600
            self._token_history = deque(
                (t, tk) for t, tk in self._token_history if t > cutoff
            )
            hourly_tokens = sum(tk for _, tk in self._token_history)
            if hourly_tokens >= self.budget_tokens_per_hour:
                return False, f"hourly_budget_exhausted ({hourly_tokens}/{self.budget_tokens_per_hour})"

            # 2. Per-face cooldown (adaptive based on affect_delta)
            last_call = self._last_call_time.get(face_id, 0)
            elapsed = now - last_call
            # Low entropy (stable state) → wider cooldown
            base_cooldown = self.per_face_min_cooldown_s
            entropy_factor = max(0.5, affect_delta)  # [0.5, 1.0] range
            adaptive_cooldown = base_cooldown / entropy_factor
            adaptive_cooldown = min(adaptive_cooldown, self.max_cooldown_s)

            if elapsed < adaptive_cooldown:
                remaining = adaptive_cooldown - elapsed
                return (
                    False,
                    f"cooldown_active ({remaining:.1f}s remaining, entropy={entropy_factor:.2f})",
                )

        return True, "ok"

    def record_call(
        self,
        face_id: str,
        prompt_tokens: int,
        response_tokens: int,
        model: str,
        latency_ms: int,
    ) -> None:
        """
        Record an LLM call for budget tracking.
        """
        total_tokens = prompt_tokens + response_tokens
        with self._lock:
            now = time.time()
            self._token_history.append((now, total_tokens))
            self._last_call_time[face_id] = now
            self._session_tokens[face_id] = (
                self._session_tokens.get(face_id, 0) + total_tokens
            )

    # ── v9.5 — LLMLingua prompt compression ──────────────────────
    # Compresses `prompt` before it's sent to the LLM when the budget
    # is tight. Off by default (no-op) so we preserve exact current
    # behaviour; turn on with FF_LLM_COMPRESS or call directly.
    def maybe_compress(
        self, prompt: str, ratio: float = 4.0, force: bool = False,
        instruction: str = "",
    ):
        """Return (possibly-compressed prompt, stats dict).

        Compresses when `force=True`, or when the rolling 1-h budget
        is already past 70% used, so the cheap optimisation only kicks
        in when it actually matters.
        """
        try:
            import feature_flags as _ff
            flag_on = _ff.get("LLM_COMPRESS")
        except Exception:
            flag_on = False
        if not (force or flag_on):
            return prompt, {"backend": "disabled"}

        hourly = 0
        with self._lock:
            cutoff = time.time() - 3600
            hourly = sum(tk for t, tk in self._token_history if t > cutoff)
        budget_used = hourly / max(1, self.budget_tokens_per_hour)
        if not force and budget_used < 0.70:
            return prompt, {"backend": "bypass", "budget_used": budget_used}

        try:
            from llm_compress import compress as _compress
            return _compress(prompt, ratio=ratio, instruction=instruction)
        except Exception as e:
            return prompt, {"backend": "error", "err": str(e)}

    def degrade_model(self, current_model: str) -> str:
        """
        Return a cheaper fallback model.

        Chain: Together → Groq → Ollama

        Parameters
        ----------
        current_model : str
            The model name (e.g., "meta-llama/Llama-2-7b-chat-hf").

        Returns
        -------
        fallback : str
            Next model in degradation chain.
        """
        model_lower = current_model.lower()
        if "together" in model_lower or "llama" in model_lower:
            return "groq/mixtral-8x7b"
        elif "groq" in model_lower:
            return "ollama/neural-chat"
        else:
            return "ollama/mistral"

    def stats(self) -> Dict[str, object]:
        """
        Return snapshot of governor state.
        """
        with self._lock:
            now = time.time()
            cutoff = now - 3600
            self._token_history = deque(
                (t, tk) for t, tk in self._token_history if t > cutoff
            )
            hourly_tokens = sum(tk for _, tk in self._token_history)
            current_model = self.degrade_model("together/llama-3.3-70b") if hourly_tokens >= self.budget_tokens_per_hour * 0.8 else "together/llama-3.3-70b"
            return {
                "hourly_tokens_used": hourly_tokens,
                "hourly_tokens_budget": self.budget_tokens_per_hour,
                "hourly_utilization": round(
                    100 * hourly_tokens / self.budget_tokens_per_hour, 1
                ),
                "per_face_min_cooldown_s": self.per_face_min_cooldown_s,
                "max_cooldown_s": self.max_cooldown_s,
                "active_faces": len(self._last_call_time),
                "per_face_tokens": dict(self._session_tokens),
                "current_model": current_model,
            }


# Singleton governor (instantiated on first import)
_governor: Optional[LLMGovernor] = None


def get_governor() -> LLMGovernor:
    """Lazy singleton accessor."""
    global _governor
    if _governor is None:
        budget = 10000  # default
        try:
            if cfg and hasattr(cfg, "LLM_GOVERNOR_BUDGET_TOKENS_PER_HOUR"):
                budget = cfg.LLM_GOVERNOR_BUDGET_TOKENS_PER_HOUR
        except Exception:
            pass
        # v9.7.5: env override. The default 10 000 tokens/hour is a
        # cost-safety cap suitable for Together.ai's free tier, but
        # research workflows that hammer DODO/MON for hours blow
        # through it in 30 min and then start seeing
        # `hourly_budget_exhausted` errors. Set DODO_LLM_HOURLY_TOKENS
        # to a higher number (e.g. 200000 for casual research, or any
        # huge value to effectively disable) before launching the
        # server. Numbers larger than 999 999 999 are clamped to it
        # to avoid integer-overflow surprises in the rate-limit math.
        try:
            import os as _os
            env_v = _os.environ.get("DODO_LLM_HOURLY_TOKENS")
            if env_v:
                budget = max(1000, min(int(env_v), 999_999_999))
        except Exception:
            pass
        _governor = LLMGovernor(budget_tokens_per_hour=budget)
    return _governor
