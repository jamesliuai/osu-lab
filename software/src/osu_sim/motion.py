"""labelled (session 4) MOTION-COST v1: cursor-motion gate and path-length cost (training objective change, spec 9.2
action-cost class; runs/development/session4/cursor/workflow/DESIGN.md sections 2.1-2.2).

LABELLED OBJECTIVE CHANGE (session 4, spec 9.2 action-cost class; user request 2026-10-08: economical, human-like
cursor motion). Training reward per 60 Hz decision -= s * w(g) * |dp|, where dp = executed-cursor displacement in osu
gamefield px (both axes on the same scale) and g is computed from the pre-step state: w = idle_per_px when nothing is
on screen to aim at (no visible, not-fully-judged object; no slider or spinner in progress), aim_per_px otherwise, 0
while a spinner is in progress and on an episode's first step. s ramps 0 -> 1 over ramp_decisions from ramp_start and is
frozen per episode. The cost is never positive, depends only on the agent's own consecutive cursor positions (never on
object positions or proximity), and never rewards a click. Trainer-only judgement and visibility state is used only to
choose the price. rules.py (parity-critical) is not touched.
"""
from __future__ import annotations

import numpy as np

from . import ROOT  # noqa: F401  (sys.path for numba in .simpkgs)
from numba import njit, prange

GATE_AIM, GATE_IDLE, GATE_SPIN = 0, 1, 2
STILL_PX = 2.0          # a step with d < 2 osu px counts as "at rest" (evaluator idle_still, human reference 43-53%)
REV_DEADZONE_PX = 15.0  # labelled (session 4) MOTION-COST v2 REV2 default dead zone (cancelled osu px per step pair)
STEADY_PX = 4.0         # labelled (session 4) v2 metric: a judged press with d_k + d_{k-1} < 4 osu px is "steady"


@njit(cache=True, parallel=True)
def motion_gate(e_t, e_lo, e_win_lo, e_win_hi, e_map, m_preempt, o_kind, o_start, st_head_r, st_tail_r, out_gate):
    """Gate of every env from the state the policy just observed (call BEFORE rules.step_all).

    st_head_r / st_tail_r are SimEnv.st per-object arrays [N, omax] indexed by j relative to e_lo (the arrays that
    rules.step_all reads and writes; the K render slots in SimEnv.r are only a copy for the renderer). Every visible
    object (start - preempt <= t) is inside [e_win_lo, e_win_hi): the window admits objects admit_margin (800 ms) before
    they become visible and only releases objects 1 s after they are fully judged.

    Skip fully judged objects (circle head_r != 0; slider tail_r != 0, so a slider in progress or with a missed head is
    still unjudged; spinner head_r != 0, set at t >= end). SPIN if an unjudged spinner has start <= t (stop scanning);
    AIM if an unjudged object has start - preempt <= t (visible); IDLE otherwise."""
    N = e_t.shape[0]
    for n in prange(N):
        t = e_t[n]
        preempt = m_preempt[e_map[n]]
        gid0 = e_lo[n]
        g_out = GATE_IDLE
        for j in range(e_win_lo[n], e_win_hi[n]):
            g = gid0 + j
            kind = o_kind[g]
            if kind == 1:
                if st_tail_r[n, j] != 0:
                    continue
            elif st_head_r[n, j] != 0:
                continue
            start = o_start[g]
            if kind == 2 and start <= t:
                g_out = GATE_SPIN
                break
            if start - preempt <= t:
                g_out = GATE_AIM
        out_gate[n] = g_out


def move_cost(d, gate, scale, k_idle, k_aim, first0, done0):
    """Per-env cost (>= 0, float64) = scale * w(gate) * d with w = k_idle (IDLE), k_aim (AIM), 0 (SPIN); 0 on an
    episode's first step (first0, the previous cursor is the reset value) and on envs already done before the step."""
    d = np.asarray(d, np.float64)
    # labelled (session 5): k_aim may be a per-env array (per-episode frozen AIM price); a scalar gives the old values
    w = np.where(gate == GATE_IDLE, float(k_idle), np.where(gate == GATE_AIM, np.asarray(k_aim, np.float64), 0.0))
    live = ~(np.asarray(first0, bool) | np.asarray(done0, bool))
    return np.where(live, np.asarray(scale, np.float64) * w * d, 0.0)


def ramp_scale(decisions, ramp_start, ramp_decisions):
    """s = clip((decisions - ramp_start) / ramp_decisions, 0, 1); 1 when ramp_decisions == 0 (no ramp)."""
    if ramp_decisions <= 0:
        return 1.0
    return float(min(max((float(decisions) - float(ramp_start)) / float(ramp_decisions), 0.0), 1.0))


# ---------------------------------------------------------------------------------------------------------------------
# labelled (session 5) CURSOR PHYSICS (CURSOR_PHYSICS_PROMPT.md; spec 7.2 "physical motion filter", applied identically in
# training (SimEnv.step_np, before rules.step_all), evaluation and the real host (RealHost.step, before the cursor is sent)).
# The executed cursor moves toward the commanded cursor by at most vmax osu gamefield px per control step. Positions are
# in action units ([-1, 1]^2, the host maps a to client px 160 + (a + 1) * 959 / 2, (a + 1) * 719 / 2 and the playfield
# scale is 1.5 client px per osu px), so one action unit is 959 / 3 osu px in x and 719 / 3 in y.
ACT_PX_X = 959.0 / 3.0
ACT_PX_Y = 719.0 / 3.0
VMAX_OFF = float("inf")


def physics_step(prev, cmd, vmax):
    """Executed cursor (float64 action units [..., 2]) after one control step from `prev` toward the command `cmd`.

    Returns `cmd` bitwise wherever the commanded move is within the cap (|cmd - prev| <= vmax osu px, always when vmax is
    inf), else prev + (cmd - prev) * vmax / |cmd - prev|. vmax broadcasts against prev[..., 0] (per env). Own cursor only:
    no object position is read."""
    prev = np.asarray(prev, np.float64)
    cmd = np.asarray(cmd, np.float64)
    vmax = np.asarray(vmax, np.float64)
    dx = (cmd[..., 0] - prev[..., 0]) * ACT_PX_X
    dy = (cmd[..., 1] - prev[..., 1]) * ACT_PX_Y
    d = np.hypot(dx, dy)
    bind = d > vmax
    f = np.where(bind, vmax / np.where(bind, d, 1.0), 1.0)
    capped = prev + (cmd - prev) * f[..., None]
    return np.where(bind[..., None], capped, cmd), bind


def physics_step_torch(prev, cmd, vmax):
    """torch twin of physics_step for the policy's key head (float32; same formula). prev, cmd [..., 2]; vmax [...]
    (inf = off). Returns (executed [..., 2], bind [...] bool); executed is `cmd` itself wherever bind is False."""
    import torch
    sx = cmd.new_tensor((ACT_PX_X, ACT_PX_Y))
    delta = cmd - prev
    d = (delta * sx).norm(dim=-1)
    bind = d > vmax
    f = torch.where(bind, vmax / torch.where(bind, d, torch.ones_like(d)), torch.ones_like(d))
    return torch.where(bind.unsqueeze(-1), prev + delta * f.unsqueeze(-1), cmd), bind


def vmax_schedule(progress, start=800.0, mid=160.0, end=120.0, mid_decisions=30e6, end_decisions=10e6):
    """labelled (session 5) cap anneal: geometric start -> mid over mid_decisions of anneal progress, then geometric
    mid -> end over end_decisions; `end` afterwards. progress = un-paused decisions since the branch point."""
    p = max(float(progress), 0.0)
    if p < mid_decisions:
        return float(start * (mid / start) ** (p / mid_decisions))
    q = p - mid_decisions
    if q < end_decisions:
        return float(mid * (end / mid) ** (q / end_decisions))
    return float(end)


def reversal_excess(px, py, dx, dy, deadzone):
    """labelled (session 4) MOTION-COST v2 REV2 (runs/development/session4/cursor/v2/DESIGN_v2.md): cancelled motion of
    two consecutive executed-cursor steps p = (px, py) then d = (dx, dy) beyond a dead zone, osu px (>= 0, float64).

    kappa = (|p| + |d| - |p + d|) / 2 is 0 for collinear same-direction steps and whenever either step is 0, equals
    min(|p|, |d|) for an exact back-and-forth (A-B-A) pair and never exceeds min(|p|, |d|); for equal legs l turning by
    phi it is l * (1 - cos(phi / 2)). Returns max(0, kappa - deadzone). Own cursor positions only (no object positions)."""
    px, py, dx, dy = (np.asarray(v, np.float64) for v in (px, py, dx, dy))
    kappa = 0.5 * (np.hypot(px, py) + np.hypot(dx, dy) - np.hypot(px + dx, py + dy))
    return np.maximum(kappa - float(deadzone), 0.0)
