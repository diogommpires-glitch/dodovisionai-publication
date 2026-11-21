"""v9.8.3 — dynamics_branch buffers stay isolated per logical pid."""
from __future__ import annotations


def test_dynamics_yaw_buffers_isolated_two_pids():
    """Per-pid deques in dynamics_branch must not cross-contaminate."""
    from pad.branches import dynamics_branch as dy

    dy.reset_all()
    t0 = 1_000_000.0
    for i in range(10):
        ts = t0 + i * 0.02
        dy.feed_frame(701, ts, yaw_deg=float(i))
        dy.feed_frame(702, ts, yaw_deg=float(-100 - i))
    ya = dy._get(701)["yaw_history"][-1][1]
    yb = dy._get(702)["yaw_history"][-1][1]
    assert ya != yb
    assert ya == 9.0 and yb == -109.0


def test_landmark_topology_nose_ear_pixels():
    from pad.landmark_topology import nose_ear_frame_pixels, MIN_LANDMARKS_FOR_NOSE_OVAL_TRIO

    class _P:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    short = [_P(0.1, 0.1)] * (MIN_LANDMARKS_FOR_NOSE_OVAL_TRIO - 1)
    assert nose_ear_frame_pixels(short, 640, 480) is None

    fl = [_P(0, 0)] * MIN_LANDMARKS_FOR_NOSE_OVAL_TRIO
    fl[4] = _P(0.5, 0.5)
    fl[234] = _P(0.2, 0.6)
    fl[454] = _P(0.8, 0.6)
    trio = nose_ear_frame_pixels(fl, 1000, 500)
    assert trio is not None
    nose, left, right = trio
    assert nose == (500.0, 250.0)
    assert left == (200.0, 300.0)
    assert right == (800.0, 300.0)
