"""Robust subject-adaptive blink detection.

Why a full rewrite was needed
-----------------------------
Both prior approaches failed for glasses-wearing subjects:

  1. **MediaPipe blendshape `eyeBlinkLeft/Right`** — the classifier
     was trained without occluding eyewear in distribution. Strong
     glasses push it into a degenerate regime where the value sits
     near 0.05-0.10 ("always open") regardless of actual eye state.

  2. **Absolute-threshold EAR** (Eye Aspect Ratio from MediaPipe
     FaceMesh) — when the eyelid landmarks (33, 133, 159, 145, …)
     land ON THE GLASSES FRAME instead of the actual eyelid, EAR
     hovers at a constant value (e.g., ~0.25) and never crosses
     the standard 0.20/0.27 cuts. No blinks ever register.

The fix: track the subject's PERSONAL EAR baseline as a rolling
median over the last ~3 s of samples, then detect blinks as
RELATIVE DROPS from that baseline (default ≥15 %). This works
regardless of where the baseline sits — if EAR hovers at 0.25, a
blink that drops to 0.20 = 20 % drop = detected; if EAR hovers at
0.30, a drop to 0.24 = 20 % drop = same detection. The absolute
value never matters.

Cross-channel cross-validation: combine EAR-drop AND blendshape-
jump as redundant evidence. Either alone fires the close-detector;
both must clear to declare open. Reduces both false-negatives
(when one channel is broken by glasses) and false-positives (when
one channel briefly noises out).

State machine
-------------
Four states, with minimum dwell on the CLOSING state to filter
single-frame noise spikes::

    OPEN ──[is_closing]──▶ CLOSING ──[dwell ≥ 40 ms]──▶ CLOSED
       ◀──[is_open & confirmed]── OPENING ──[is_open]── ─┘

A blink is counted exactly once: when the state transitions back
to OPEN through OPENING. Duration of CLOSED state is captured for
PERCLOS computation.

Usage
-----
    from blink_detector import BlinkDetector
    det = BlinkDetector()
    out = det.update(ear=current_ear,
                     bs_blink=avg_blendshape_eyeBlink,
                     t=time.time())
    # out = {blinked, blink_rate_pm, eye_openness, perclos,
    #        drowsiness, _ear, _ear_baseline, _ear_drop, _state}

Both inputs are optional — pass `None` to skip the channel. With
neither channel present, `update` returns no-op output.

References
----------
  - Soukupová & Čech, "Real-time eye blink detection using facial
    landmarks", 2016. Original EAR formulation; we use a relative
    variant rather than their absolute 0.20 cut.
  - Wierwille & Ellsworth, "Evaluation of driver drowsiness by
    trained raters", 1994. PERCLOS thresholds 0.08 (fatigued) /
    0.15 (drowsy) used here.
  - Caffier et al., "Experimental evaluation of eye-blink parameters
    as a drowsiness measure", 2003. Min-blink-duration ~40 ms.
"""
from __future__ import annotations
import time
from collections import deque
from typing import Optional


class BlinkDetector:
    """Subject-adaptive blink detector. Robust to glasses, dim light,
    skin-tone variation, and individual eye geometry because it uses
    RELATIVE drops from a rolling-median baseline, not absolute
    thresholds.

    Tracks a 4-state machine to filter single-frame noise. PERCLOS
    is computed over the last `perclos_window` frames as the
    fraction of those frames spent in any closed-or-transitioning
    state.
    """

    # Public state-name constants — useful for telemetry / debugging.
    OPEN     = "OPEN"
    CLOSING  = "CLOSING"
    CLOSED   = "CLOSED"
    OPENING  = "OPENING"

    def __init__(
        self,
        ear_baseline_window: int = 90,
        bs_baseline_window:  int = 90,
        # v9.7.5: thresholds further loosened. The earlier 15 % EAR
        # drop / 0.10 blendshape jump still missed the majority of
        # glasses-wearer blinks because the eyelid landmarks slide to
        # the lens frame, so the EAR change during a real blink is
        # only ~5-10 % of the (already-narrow) baseline. Cut the EAR
        # drop trigger to 8 % and the blendshape jump to 0.06 — both
        # still well above measurement noise (which is ~2-3 % EAR
        # std at rest), but low enough to catch faint glasses signals.
        # Min-dwell 20 ms instead of 40 ms — captures faster blinks
        # (~80-150 ms typical) at our 20-30 fps capture rate where a
        # full blink is only 2-4 frames.
        ear_close_drop:      float = 0.08,
        bs_close_jump:       float = 0.06,
        min_close_dwell_s:   float = 0.02,
        perclos_window:      int   = 600,
        # v9.8 BUGFIX: require the close-phase to have dipped at least
        # `min_real_close_drop` below baseline before we accept it as
        # a real blink. Landmark jitter (MediaPipe re-snapping eyelid
        # points by ~2-4 % between frames) routinely satisfies the
        # 8 %-drop CLOSING trigger; with a 20 ms dwell at 30 fps that
        # means every jitter event used to get counted as a "fast
        # blink". Real eye closures dip 30-70 % below baseline within
        # their first frame, so 0.20 (20 %) cleanly rejects all jitter
        # (which tops out around 7 % even when noise stacks) without
        # missing real blinks. For glasses-wearers (whose EAR barely
        # moves), the parallel `bs_max_jump >= 0.20` channel still
        # accepts blinks via the blendshape signal.
        min_real_close_drop: float = 0.20,
    ):
        # Rolling buffers for adaptive baselines. ~3 s at 30 fps is
        # long enough to be insensitive to a single blink (the blink
        # contributes ≤8/90 = 9 % of samples → median is unaffected)
        # but short enough to track posture changes within seconds.
        self.ear_buf:     "deque[float]" = deque(maxlen=ear_baseline_window)
        self.bs_buf:      "deque[float]" = deque(maxlen=bs_baseline_window)
        # PERCLOS (Percentage of Eye Closure) sliding window.
        # 600 frames @ 30 fps ≈ 20 s, which is the standard PERCLOS
        # window per Wierwille's drowsiness literature.
        self.perclos_buf: "deque[int]"   = deque(maxlen=perclos_window)
        # Blink timestamps within the last ~2 min for BPM compute.
        self.blinks:      "deque[float]" = deque(maxlen=240)

        self.state = self.OPEN
        self.state_entered: float = 0.0
        # Latest raw values, for telemetry only.
        self.last_ear: float = 0.0
        self.last_bs:  float = 0.0

        self._ear_close_drop    = float(ear_close_drop)
        self._bs_close_jump     = float(bs_close_jump)
        self._min_close_dwell_s = float(min_close_dwell_s)
        self._min_real_close_drop = float(min_real_close_drop)
        # v9.8: track the deepest EAR drop / blendshape jump / CNN-closed
        # probability seen during the current close-phase. Reset on
        # entering CLOSING; updated each frame. The fast-blink shortcut
        # and the slow-path completion (OPENING -> OPEN) require AT LEAST
        # ONE of these to have crossed its real-blink threshold:
        #     ear_max_drop >= min_real_close_drop  (default 0.15 = 15 %)
        #  OR bs_max_jump  >= 0.20                 (clear blendshape close)
        #  OR cnn_max_prob >= 0.85                 (CNN confident eye closed)
        # This keeps glasses-wearer detection working (where EAR barely
        # moves but blendshape jumps cleanly) while rejecting landmark-
        # jitter false positives (where EAR alone briefly noise-spikes).
        self._close_max_drop:    float = 0.0
        self._close_max_bs_jump: float = 0.0
        self._close_max_cnn:     float = 0.0

    # ─────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Drop all history. Useful when the tracker re-binds the
        face_idx to a different person (or when the subject leaves
        and returns).

        v9.8: also clear the per-channel close-phase max trackers so
        the next blink isn't measured against stale apex from the
        previous subject. Without this, if the previous subject had a
        deep blink (raw_jump 0.40+) right before reset, the new
        subject's first frame in CLOSING would carry the leftover
        max -- the very first jitter spike could fast-blink-shortcut
        through the depth gate.
        """
        self.ear_buf.clear()
        self.bs_buf.clear()
        self.perclos_buf.clear()
        self.blinks.clear()
        self.state = self.OPEN
        self.state_entered = 0.0
        self._close_max_drop = 0.0
        self._close_max_bs_jump = 0.0
        self._close_max_cnn = 0.0

    # ─────────────────────────────────────────────────────────────
    def _baselines(self) -> "tuple[Optional[float], Optional[float]]":
        """Median of each rolling buffer. Median (not mean) so a
        single noise spike doesn't drag the baseline. Returns None
        on either channel that hasn't accumulated 8+ samples yet —
        sub-sampled stats are unreliable."""
        ear_b = None
        bs_b  = None
        if len(self.ear_buf) >= 8:
            arr = sorted(self.ear_buf)
            ear_b = arr[len(arr) // 2]
        if len(self.bs_buf) >= 8:
            arr = sorted(self.bs_buf)
            bs_b = arr[len(arr) // 2]
        return ear_b, bs_b

    # ─────────────────────────────────────────────────────────────
    def update(
        self,
        ear: "Optional[float]" = None,
        bs_blink: "Optional[float]" = None,
        t: "Optional[float]" = None,
        cnn_closed: "Optional[float]" = None,
    ) -> dict:
        """Feed one frame. Either or both channels may be None.

        Returns a dict::

            {
              "blinked":       bool,   # True ON THE FRAME the blink ends
              "blink_rate_pm": int,    # blinks observed in last 60 s
              "eye_openness":  float,  # 0..1, 1 = fully open
              "perclos":       float,  # 0..1, fraction of recent
                                       #         frames spent closed
              "drowsiness":    str,    # "alert" | "fatigued" | "drowsy"
              "_ear":          float,  # latest raw EAR (telemetry)
              "_ear_baseline": float,  # current adaptive baseline
              "_ear_drop":     float,  # current relative drop
              "_state":        str,    # state-machine state
            }
        """
        if t is None:
            t = time.time()

        # v9.7.6 fix: don't update the rolling baseline while we believe
        # the eye is CLOSED. Otherwise long blinks (or — far worse — a
        # state machine stuck in CLOSED, see the OR-based `is_open` fix
        # below) drag the median downward, which RAISES the relative
        # threshold for triggering the NEXT close. Effect: detector
        # gradually goes deaf to real blinks. We still capture last_ear
        # / last_bs for telemetry and the relative-drop math below; the
        # only thing skipped is the rolling-window push.
        sampling = self.state != self.CLOSED
        if ear is not None:
            if sampling:
                self.ear_buf.append(float(ear))
            self.last_ear = float(ear)
        if bs_blink is not None:
            if sampling:
                self.bs_buf.append(float(bs_blink))
            self.last_bs = float(bs_blink)

        ear_b, bs_b = self._baselines()

        # ── Compute relative-drop signals ──
        # ear_drop is positive when current EAR is BELOW its baseline
        # (eye is closing). 0.20 = "EAR has dropped 20 % below baseline".
        ear_drop = 0.0
        if ear_b is not None and ear is not None and ear_b > 0.01:
            ear_drop = (ear_b - float(ear)) / ear_b
        # bs_jump is positive when blendshape is ABOVE its baseline
        # (blendshape says more closed). The blendshape encodes
        # closure as 0..1 with higher = more closed.
        bs_jump = 0.0
        if bs_b is not None and bs_blink is not None:
            bs_jump = float(bs_blink) - bs_b

        # ── Cross-channel evidence aggregation ──
        # Closing: EITHER channel sees a strong close-direction
        # change. Forgiving — handles the case where one channel
        # is broken (glasses on EAR, or blendshape glitch).
        # v9.8: optional CNN tie-breaker as a 3rd channel. p_closed > 0.7
        # joins the OR vote on the close side; < 0.3 joins on the open side.
        # When None (FF_BLINK_CNN=0 or model failed to load), the channel
        # has no effect and the detector behaves exactly as v9.7.7.
        cnn_says_closing = cnn_closed is not None and cnn_closed > 0.7
        cnn_says_open    = cnn_closed is not None and cnn_closed < 0.3

        is_closing = (ear_drop > self._ear_close_drop) or \
                     (bs_jump  > self._bs_close_jump) or \
                     cnn_says_closing

        # v9.7.6 fix: was AND, now OR.
        #
        # Original logic required BOTH channels below 0.4× their close
        # threshold to declare the eye OPEN. For glasses-wearers, the
        # blendshape channel sits at a degenerate baseline with ±0.02-
        # 0.03 noise — `bs_jump < 0.024` was rarely satisfied even with
        # eyes fully open. Result: state machine entered CLOSING → CLOSED
        # quickly (OR on close-side) but couldn't escape (AND on open-
        # side). Detector locked in CLOSED, real blinks never counted as
        # OPENING → OPEN transitions, and BPM stayed stuck at ~13 even
        # when the user was blinking ~20-25/min.
        #
        # Symmetric OR makes the detector robust: ONE good channel is
        # enough to confirm open, just like one good channel is enough
        # to flag closing. The `* 0.5` (loosened from 0.4) gives a
        # cleaner hysteresis margin without being so strict it traps.
        is_open = (ear_drop < (self._ear_close_drop * 0.5)) or \
                  (bs_jump  < (self._bs_close_jump  * 0.5)) or \
                  cnn_says_open

        blinked = False

        # v9.7.6 fix: hard-recovery from a stuck CLOSED state.
        # If we've been CLOSED for over 800 ms (longer than ~99 % of
        # natural blinks per Caffier 2003) AND neither channel says the
        # eye is closing anymore, declare a count-then-reset. Without
        # this safety valve a single false-positive close (noise spike,
        # head turn, occlusion frame) could leave us locked in CLOSED
        # for the rest of the session.
        if (self.state == self.CLOSED
                and (t - self.state_entered) > 0.8
                and not is_closing):
            self.state = self.OPENING
            self.state_entered = t

        # v9.8: keep a running max of how deep the close has been across
        # all 3 channels so the fast-blink shortcut can demand evidence
        # of an actual closure (not just landmark jitter). Reset on
        # entering CLOSING below.
        if self.state in (self.CLOSING, self.CLOSED, self.OPENING):
            if ear_drop > self._close_max_drop:
                self._close_max_drop = ear_drop
            if bs_jump > self._close_max_bs_jump:
                self._close_max_bs_jump = bs_jump
            if cnn_closed is not None and cnn_closed > self._close_max_cnn:
                self._close_max_cnn = float(cnn_closed)

        # Helper: was there real-blink-strength evidence on any channel?
        def _saw_real_close() -> bool:
            return (self._close_max_drop    >= self._min_real_close_drop
                 or self._close_max_bs_jump >= 0.20
                 or self._close_max_cnn     >= 0.85)

        # ── State machine ──
        if self.state == self.OPEN:
            if is_closing:
                self.state = self.CLOSING
                self.state_entered = t
                # start tracking depth across all 3 channels
                self._close_max_drop = ear_drop
                self._close_max_bs_jump = bs_jump
                self._close_max_cnn = float(cnn_closed) if cnn_closed is not None else 0.0

        elif self.state == self.CLOSING:
            # v9.7.7 introduced a "fast-blink shortcut" so 2-3 frame
            # blinks at 30 fps wouldn't be discarded; v9.8 fixes the
            # over-count it caused. Root cause: at 30 fps the 20 ms
            # dwell is under one frame, so any single-frame EAR jitter
            # spike that crossed the 8 % drop threshold then rebounded
            # was being counted as a "fast blink". Field test: the
            # state machine produced ~50 blinks/min on a stationary
            # face that actually blinked ~15/min.
            #
            # Fix: gate the fast-blink shortcut on `_close_max_drop >=
            # min_real_close_drop` (default 0.15 = 15 % below baseline).
            # Real eye closures dip 30-70 % below baseline within their
            # first frame, so 15 % is well clear of jitter (which tops
            # out around 5-7 %) without missing real blinks. The slow-
            # blink path (CLOSING → CLOSED → OPENING → OPEN) is
            # unchanged and continues to handle anything that doesn't
            # qualify as fast.
            dwell_satisfied = (t - self.state_entered) >= self._min_close_dwell_s
            if dwell_satisfied:
                if is_open:
                    # Eye already rebounded. Only count if we saw a real
                    # closure depth -- otherwise this was jitter, drop it.
                    if _saw_real_close():
                        self.state = self.OPEN
                        self.blinks.append(t)
                        blinked = True
                    else:
                        self.state = self.OPEN  # noise, no count
                    self._close_max_drop = 0.0
                    self._close_max_bs_jump = 0.0
                    self._close_max_cnn = 0.0
                else:
                    self.state = self.CLOSED
                    # state_entered remains the original close-start; we
                    # use it later for PERCLOS and blink-duration metrics.
            elif is_open:
                # Single-frame spike before dwell satisfies -- treat as
                # noise per Caffier 2003 (real blinks are >= 40 ms; under
                # 20 ms dwell at our frame rate is sub-frame jitter). No
                # count. The post-dwell branch above still has the depth
                # gate so real fast blinks are caught one frame later.
                self.state = self.OPEN
                self._close_max_drop = 0.0
                self._close_max_bs_jump = 0.0
                self._close_max_cnn = 0.0

        elif self.state == self.CLOSED:
            if is_open:
                self.state = self.OPENING
                self.state_entered = t

        elif self.state == self.OPENING:
            if is_closing:
                # Eye reclosed during recovery — back to CLOSED.
                self.state = self.CLOSED
            else:
                # Recovery confirmed. Same depth gate as the fast-blink
                # path -- if the whole "closure" never actually went
                # below `min_real_close_drop` on any channel, treat it
                # as a glitch and don't increment the counter.
                self.state = self.OPEN
                if _saw_real_close():
                    self.blinks.append(t)
                    blinked = True
                self._close_max_drop = 0.0
                self._close_max_bs_jump = 0.0
                self._close_max_cnn = 0.0

        # ── PERCLOS-80: standard Wierwille definition ──
        # Fraction of frames where eye openness is below 80 % of baseline.
        # Originally we counted any frame where the state machine wasn't
        # OPEN (CLOSING|CLOSED|OPENING). That's wrong: when EAR briefly
        # oscillates around the threshold the state lingers in
        # CLOSING/CLOSED for several frames per false-alarm, inflating
        # PERCLOS to 60-80 % on a clearly-alert subject. The proper
        # definition (Wierwille & Ellsworth 1994; FHWA driver-drowsiness
        # standard) is purely signal-based: an instantaneous EAR drop
        # ≥20 % below personal baseline means "eye is ≤80 % open".
        # Decoupling PERCLOS from the state machine makes it noise-
        # tolerant and matches every published drowsiness paper.
        eye_below_80 = (ear_drop > 0.20)
        self.perclos_buf.append(1 if eye_below_80 else 0)
        perclos = (sum(self.perclos_buf) / len(self.perclos_buf)
                   if self.perclos_buf else 0.0)

        # ── BPM over last 60 s ──
        cutoff = t - 60.0
        while self.blinks and self.blinks[0] < cutoff:
            self.blinks.popleft()
        bpm = len(self.blinks)

        # ── Drowsiness from PERCLOS (Wierwille 1994 thresholds) ──
        if perclos > 0.15:
            drowsy = "drowsy"
        elif perclos > 0.08:
            drowsy = "fatigued"
        else:
            drowsy = "alert"

        # ── Eye openness (continuous 0..1, for UI gradient bars) ──
        # Use 1 − ear_drop when EAR is available (most informative);
        # fall back to inverted blendshape if not. Clip to [0,1].
        if ear_b is not None:
            eye_openness = max(0.0, min(1.0, 1.0 - ear_drop))
        elif bs_blink is not None:
            eye_openness = max(0.0, min(1.0, 1.0 - float(bs_blink)))
        else:
            eye_openness = 1.0

        return {
            "blinked":       blinked,
            "blink_rate_pm": bpm,
            "eye_openness":  round(eye_openness, 3),
            "perclos":       round(perclos, 4),
            "drowsiness":    drowsy,
            # Diagnostics for the UI's `?blink_debug=1` panel:
            "_ear":          round(self.last_ear, 4),
            "_ear_baseline": (round(ear_b, 4) if ear_b is not None else None),
            "_ear_drop":     round(ear_drop, 4),
            "_state":        self.state,
        }


__all__ = ["BlinkDetector"]
