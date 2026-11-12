// API_unq screen-flash client helper.
//
// Trigger a server-side active-liveness challenge, then briefly pulse
// document.body.style.backgroundColor while the WebSocket stream
// continues uninterrupted. The pad/branches/screen_flash_branch
// will score the per-pid skin reflection in CIELAB on the next WS
// frame and consume the challenge.
//
// USAGE (from devtools or wired to a button later):
//   await DodoScreenFlash.trigger(0);           // current pid=0, random color
//   await DodoScreenFlash.trigger(1, "RED");    // pin pid + color
//   DodoScreenFlash.autoLoop(0, 30);            // every 30 s on pid=0 (cancel via .cancel())
//
// To wire to the UI: add a button in ui.py that calls window.DodoScreenFlash.trigger(currentPid).
// This file is loaded via <script src="/static/js/screen_flash_client.js"></script>
// from a routes.py inject or by adding to static/index.html.

(function () {
  "use strict";

  const STATE = {
    inFlight: false,
    autoTimer: null,
    lastChallengeId: null,
    lastError: null,
  };

  async function trigger(pid, color) {
    if (STATE.inFlight) {
      return { ok: false, reason: "another flash in flight" };
    }
    STATE.inFlight = true;
    try {
      const body = { pid: Number(pid) };
      if (color) body.color = String(color).toUpperCase();
      const res = await fetch("/pad/challenge/screen_flash/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await res.json();
      if (!j || j.error) {
        STATE.lastError = j && j.error;
        return { ok: false, reason: STATE.lastError || "no response" };
      }
      STATE.lastChallengeId = j.challenge_id;
      _pulseBackground(j.color_hex, j.duration_ms);
      return j;
    } catch (e) {
      STATE.lastError = String(e);
      return { ok: false, reason: STATE.lastError };
    } finally {
      // Release the gate a bit after the pulse so back-to-back triggers don't
      // overlap their observation windows (branch wants 500 ms total).
      setTimeout(() => { STATE.inFlight = false; }, 550);
    }
  }

  function _pulseBackground(hex, durationMs) {
    const body = document.body;
    if (!body) return;
    const prev = body.style.backgroundColor || "";
    const prevTransition = body.style.transition || "";
    // No transition: we want a hard step into the color and a hard step out,
    // matching the assumed-rectangle the server uses to time the peak window.
    body.style.transition = "none";
    body.style.backgroundColor = hex;
    setTimeout(() => {
      body.style.backgroundColor = prev;
      body.style.transition = prevTransition;
    }, Math.max(60, Number(durationMs) || 120));
  }

  function autoLoop(pid, intervalSec) {
    cancel();
    const ms = Math.max(5000, Math.round(Number(intervalSec) * 1000) || 30000);
    STATE.autoTimer = setInterval(() => trigger(pid), ms);
  }

  function cancel() {
    if (STATE.autoTimer) {
      clearInterval(STATE.autoTimer);
      STATE.autoTimer = null;
    }
  }

  async function history(pid) {
    try {
      const res = await fetch(`/pad/challenge/screen_flash/history?pid=${Number(pid)}`);
      return await res.json();
    } catch (e) {
      return { error: String(e) };
    }
  }

  window.DodoScreenFlash = {
    trigger,
    autoLoop,
    cancel,
    history,
    _state: STATE,
  };
})();
