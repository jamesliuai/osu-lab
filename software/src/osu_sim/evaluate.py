"""Full-map evaluation in osu-sim (deterministic decoders), FC-centred metrics.

FC (lazer max-combo sense) = no Miss / LargeTickMiss / spinner Miss and every slider tail hit.
"Accuracy" here = lazer-style accuracy (see run_mode): circles/heads/spinners 300, ticks+repeats 30, tails 150.
Failures (labelled (session 3)) = breaks + missed slider tails; breaks already include circle/head misses,
LargeTickMiss and spinner misses, so failures counts every event that costs the FC above.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import ROOT
from . import rules
from .env import ClipConfig, SimEnv, UniformSampler
from .pool import MapPool


def make_renderer(kind, pool_gpu, device, breaks=None, hit_animations=True):
    # labelled (session 3): hit_animations False = lazer HitAnimations off (full renderer only)
    if kind == "simple":
        if not hit_animations:
            raise ValueError("hit_animations=False is only implemented by the full renderer")
        from .render_simple import SimpleRenderer
        return SimpleRenderer(pool_gpu, device)
    from .render import Renderer
    return Renderer(pool_gpu, device=device, breaks=breaks, hit_animations=hit_animations)


def summarize_maps(res):
    """One mode's summary from its per-map records. Pure (no torch/GPU): unit-testable and re-runnable on old
    eval.jsonl detail records. The first four keys are unchanged; the rest are labelled (session 3), plan item 3.

    failures/map = breaks + tail_miss; objects = sum of max_combo_possible (combo-bearing judgements: circles,
    slider heads/ticks/repeats/tails, spinners); fail_per_obj = failures / objects; expected_fc = mean over maps
    of exp(-failures) (smooth FC proxy); maps_leK = maps with at most K failures. Records written before
    session 3 have no 'spin_hit', so spin_hit/spin_nonmiss are None for them; spin_nonmiss is None without spinners.
    """
    fails = np.array([r["breaks"] + r["tail_miss"] for r in res], dtype=np.float64)
    objects = int(sum(r["max_combo_possible"] for r in res))
    spin_miss = int(sum(r["spin_miss"] for r in res))
    spin_hit = int(sum(r["spin_hit"] for r in res)) if all("spin_hit" in r for r in res) else None
    spin_n = None if spin_hit is None else spin_hit + spin_miss
    # labelled (session 4): pooled hit-guard counts for the motion-cost A/B (DESIGN.md 5.4 G3 / G6); records written
    # before session 4 have no early_miss / shake, so these are added only when every record has them
    extra = dict(tail_miss_sum=int(sum(r["tail_miss"] for r in res)),
                 tick_miss_sum=int(sum(r.get("tick_miss", 0) for r in res)))
    if all("early_miss" in r for r in res):
        extra.update(early_miss=int(sum(r["early_miss"] for r in res)), shake=int(sum(r["shake"] for r in res)))
    if all("bind_steps" in r for r in res):     # labelled (session 5) cursor physics
        extra.update(bind_steps=int(sum(r["bind_steps"] for r in res)),
                     unreached_press=int(sum(r["unreached_press"] for r in res)),
                     presses_sum=int(sum(r["presses"] for r in res)))
    return dict(fc_rate=round(float(np.mean([r["fc"] for r in res])), 4),
                mean_acc=round(float(np.mean([r["acc"] for r in res])), 4),
                mean_breaks=round(float(np.mean([r["breaks"] for r in res])), 3),
                mean_combo_frac=round(float(np.mean([r["max_combo"] / max(r["max_combo_possible"], 1)
                                                     for r in res])), 4),
                spin_hit=spin_hit, spin_miss=spin_miss,
                spin_nonmiss=round(spin_hit / spin_n, 4) if spin_n else None,
                objects=objects, failures=int(fails.sum()),
                fail_per_obj=round(float(fails.sum()) / max(objects, 1), 6),
                mean_tails=round(float(np.mean([r["tail_miss"] for r in res])), 3),
                maps_le0=int((fails <= 0).sum()), maps_le1=int((fails <= 1).sum()), maps_le2=int((fails <= 2).sum()),
                expected_fc=round(float(np.mean(np.exp(-fails))), 6), **extra)


IDLE_MS = 1000.0     # cursor metrics: "idle" = no slider/spinner in progress and the next object starts >= 1 s later


def cursor_metrics(t, x, y, objs, lo, cnt, gate=None, breaks=None, radius=None, head_r=None, head_t=None, rate=1.0):
    """labelled (session 4) cursor-motion metrics of one full-map play (user report: the cursor jumps around when idle).
    Pure numpy, unit-testable. t[k] = map time at the start of control step k, (x[k], y[k]) = cursor (osu px) applied
    from t[k] on; displacement d[k] = |c[k] - c[k-1]|. Steps inside a spinner are excluded everywhere (spinning needs
    motion). Returns per map:
      idle_steps, idle_px      steps that are idle (see IDLE_MS) and their summed displacement (idle px/step = ratio)
      move_px, ref_px          summed displacement and a minimal reference path over [first start, last end] minus
                               spinners: straight lines end(k-1) -> start(k) between consecutive non-spinner objects plus
                               each slider's path length (distance x spans); path efficiency = ref_px / move_px
      idle_still               idle steps with d < 2 px (cursor at rest)
    With gate (per-step motion gate of the pre-step state, osu_sim.motion: 0 AIM, 1 IDLE, 2 SPIN), breaks (pool.breaks
    of the map), radius and per-object head_r / head_t at map end, also the MOTION-COST v1 metrics of DESIGN.md section 4
    (labelled (session 4) metrics-only change; see motion_extra_metrics)."""
    t, x, y = np.asarray(t, np.float64), np.asarray(x, np.float64), np.asarray(y, np.float64)
    kind = objs["kind"][lo:lo + cnt]
    st = objs["start"][lo:lo + cnt].astype(np.float64)
    en = objs["end"][lo:lo + cnt].astype(np.float64)
    d = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    in_spin = np.zeros(len(t), bool)
    in_obj = np.zeros(len(t), bool)
    for k in np.nonzero(kind != 0)[0]:
        sel = (t >= st[k]) & (t < en[k])
        in_obj |= sel
        if kind[k] == 2:
            in_spin |= sel
    nxt = np.searchsorted(st, t, side="left")
    t_next = np.where(nxt < cnt, st[np.minimum(nxt, cnt - 1)] - t, np.inf)
    # labelled (session 6): t is gameplay (map) ms; time thresholds are real ms x rate (DT maps; rate 1 unchanged)
    idle = ~in_obj & (t_next >= IDLE_MS * rate)
    span = (t >= st[0]) & (t <= en[-1]) & ~in_spin
    ns = np.nonzero(kind != 2)[0]
    ref = 0.0
    for a, b in zip(ns[:-1], ns[1:]):
        if b == a + 1:      # no spinner in between
            ref += float(np.hypot(objs["x"][lo + b] - objs["ex"][lo + a], objs["y"][lo + b] - objs["ey"][lo + a]))
    sl = kind == 1
    ref += float((objs["distance"][lo:lo + cnt][sl] * objs["span_count"][lo:lo + cnt][sl]).sum())
    out = dict(idle_steps=int(idle.sum()), idle_px=round(float(d[idle].sum()), 1),
               idle_still=int((idle & (d < 2.0)).sum()), move_px=round(float(d[span].sum()), 1), ref_px=round(ref, 1))
    if gate is not None:
        out.update(motion_extra_metrics(t, x, y, d, objs, lo, cnt, np.asarray(gate), breaks, float(radius),
                                        head_r, head_t, in_obj, in_spin, t_next, rate=rate))
    return out


# labelled (session 4): fixed log-binned histograms pooled across maps (medians / p99 without storing per-step arrays):
# bin 0 = values < 0.1, bins 1..200 = log-spaced over [0.1, 2000), bin 201 = values >= 2000
HIST_EDGES = np.geomspace(0.1, 2000.0, 201)
HIST_N = len(HIST_EDGES) + 1
HIST_KEYS = ("step", "jerk", "trans", "landing", "gate_idle", "slider", "approach", "break")
HOME_X, HOME_Y = 512.0 / 3.0, 2.0 * 384.0 / 3.0       # bottom-left ninth of the playfield (descriptive only)


def log_hist(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return np.bincount(np.searchsorted(HIST_EDGES, v, side="right"), minlength=HIST_N).astype(np.int64)


def hist_pct(h, q):
    """q-th percentile of a log_hist histogram (log-linear interpolation inside the bin; 0 below 0.1, 2000 above)."""
    h = np.asarray(h, np.float64)
    tot = h.sum()
    if tot <= 0:
        return None
    c = np.cumsum(h)
    target = q / 100.0 * tot
    b = int(np.searchsorted(c, target, side="left"))
    if b <= 0:
        return 0.0
    if b >= HIST_N - 1:
        return float(HIST_EDGES[-1])
    lo_e, hi_e = HIST_EDGES[b - 1], HIST_EDGES[b]
    frac = (target - (c[b - 1] if b else 0.0)) / max(h[b], 1e-12)
    return float(lo_e * (hi_e / lo_e) ** min(max(frac, 0.0), 1.0))


def motion_extra_metrics(t, x, y, d, objs, lo, cnt, gate, breaks, radius, head_r, head_t, in_obj, in_spin, t_next,
                         rate=1.0):
    """labelled (session 4) MOTION-COST v1 evaluation metrics (DESIGN.md section 4), sums and histograms of one map.

    Window = [first start, last end] (as path_eff and runs/development/session4/cursor/workflow/
    human_vs_agent_metrics.py); spinner steps (gate SPIN or inside a spinner's [start, end)) are excluded except in the
    spinner metric. Gate sub-phases of IDLE steps: lead-in (before the first non-IDLE step), outro (after the last
    non-IDLE step), break (inside pool.breaks), mid-gap (the rest)."""
    n = len(t)
    kind = objs["kind"][lo:lo + cnt]
    st = objs["start"][lo:lo + cnt].astype(np.float64)
    en = objs["end"][lo:lo + cnt].astype(np.float64)
    ox = objs["x"][lo:lo + cnt].astype(np.float64)
    oy = objs["y"][lo:lo + cnt].astype(np.float64)
    oex = objs["ex"][lo:lo + cnt].astype(np.float64)
    oey = objs["ey"][lo:lo + cnt].astype(np.float64)
    k_ = np.arange(n)
    valid = k_ >= 1                                   # first step: previous cursor is the reset value (cost-exempt)
    spin = in_spin | (gate == 2)
    win = (t >= st[0]) & (t <= en[-1])
    nonspin = win & ~spin
    g_idle = valid & (gate == 1)
    g_aim = valid & (gate == 0)
    busy = np.nonzero(gate != 1)[0]
    if len(busy):
        lead = g_idle & (k_ < busy[0])
        outro = g_idle & (k_ > busy[-1])
    else:
        lead, outro = g_idle.copy(), np.zeros(n, bool)
    in_break = np.zeros(n, bool)
    for b0, b1 in np.asarray(breaks if breaks is not None else [], np.float64).reshape(-1, 2):
        in_break |= (t >= b0) & (t < b1)
    brk = g_idle & in_break & ~lead & ~outro
    mid = g_idle & ~lead & ~outro & ~brk
    # A-B-A: back within 5 px of the position 2 steps earlier after a > 20 px move
    back = np.full(n, np.inf)
    dprev = np.zeros(n)
    back[2:] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2])
    dprev[2:] = d[1:-1]
    aba = (back < 5.0) & (dprev > 20.0)
    slider = win & in_obj & ~spin
    approach = win & ~in_obj & ~spin & (t_next < 150.0 * rate)
    # reversals / circleguard snaps at the middle point k of moves (k-1 -> k, k -> k+1)
    mid_ok = np.zeros(n, bool)
    rev = np.zeros(n, bool)
    snap = np.zeros(n, bool)
    if n >= 3:
        vx, vy = np.diff(x), np.diff(y)
        l1, l2 = np.hypot(vx[:-1], vy[:-1]), np.hypot(vx[1:], vy[1:])
        cosv = (vx[:-1] * vx[1:] + vy[:-1] * vy[1:]) / np.maximum(l1 * l2, 1e-9)
        beta = np.degrees(np.arccos(np.clip(-cosv, -1.0, 1.0)))
        rev[1:-1] = (l1 > 20) & (l2 > 20) & (cosv < -0.5)
        snap[1:-1] = (np.minimum(l1, l2) > 8) & (beta < 10)
        mid_ok[1:-1] = True
    # labelled (session 4) MOTION-COST v2 metric (metrics only): REV2 excess px (dead zone 15) over consecutive step
    # pairs with both steps in the map window, outside spinners and outside slider bodies [start, end)
    rev2 = np.zeros(n)
    if n >= 3:
        leg = nonspin & ~in_obj
        kap = 0.5 * (d[1:-1] + d[2:] - np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]))
        rev2[2:] = np.where(leg[1:-1] & leg[2:], np.maximum(kap - 15.0, 0.0), 0.0)
    jerk = np.full(n, np.nan)
    if n >= 4:
        jerk[3:] = np.hypot(x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3], y[3:] - 3 * y[2:-1] + 3 * y[1:-2] - y[:-3])
    # transitions (gap <= 600 ms, jump >= 2R, consecutive non-spinner objects): path over (end_prev, start_next] / jump
    trans = []
    for a in range(cnt - 1):
        b = a + 1
        if kind[a] == 2 or kind[b] == 2:
            continue
        g = st[b] - en[a]
        if g <= 0 or g > 600.0 * rate:
            continue
        D = float(np.hypot(ox[b] - oex[a], oy[b] - oey[a]))
        if D < 2.0 * radius:
            continue
        i0, i1 = np.searchsorted(t, en[a], "right"), np.searchsorted(t, st[b], "right")
        if i1 - i0 < 2:
            continue
        trans.append(float(d[i0:i1].sum()) / D)
    # home-ninth visits in 100-200 ms gaps (previous object's judgement / slider end -> next start; neither in home)
    home = (x < HOME_X) & (y > HOME_Y)
    ns_idx = np.nonzero(kind != 2)[0]
    home_gaps = home_visits = 0
    hr = None if head_r is None else np.asarray(head_r)
    ht = None if head_t is None else np.asarray(head_t, np.float64)
    if ht is not None:
        for j0, j1 in zip(ns_idx[:-1], ns_idx[1:]):
            ta = en[j0] if kind[j0] == 1 else (ht[j0] if hr[j0] != 0 else st[j0])
            tb = st[j1]
            if not (100.0 * rate <= tb - ta < 200.0 * rate):
                continue
            if (oex[j0] < HOME_X and oey[j0] > HOME_Y) or (ox[j1] < HOME_X and oy[j1] > HOME_Y):
                continue
            a0, b0 = np.searchsorted(t, ta, "right"), np.searchsorted(t, tb, "left")
            if b0 - a0 < 1:
                continue
            home_gaps += 1
            home_visits += int(home[a0:b0].any())
    # hit companions from the per-object judgements at map end
    same = hits_inside = 0
    landing = []
    fast_miss = after_sl_miss = after_sl_late = 0
    # labelled (session 4) MOTION-COST v2 metrics: press-frame motion d_k + d_{k-1} of judged hits (steady < 4 px),
    # arrival lead (rows inside R before the press row, capped at 4; 0 = same-frame entry) of hits inside R, the
    # after-slider same-frame entries, and the head offset (judgement time - start) of judged hits.
    # st_lead_hist: arrival measured against the object's start time st, not the press row (invariant to a press-time
    # shift): rows inside R ending at the last row at or before st, capped at 4 (0 = outside R at st), judged hits
    lead_hist = np.zeros(5, np.int64)
    st_lead_hist = np.zeros(5, np.int64)
    judged = steady = as_in = as_same = 0
    press_sp2 = off_sum = 0.0
    if hr is not None and ht is not None:
        prev_ns = -1
        for j in range(cnt):
            if kind[j] == 2:
                continue
            if hr[j] == 4:
                if prev_ns >= 0 and st[j] - st[prev_ns] < 160.0 * rate:
                    fast_miss += 1
                if prev_ns >= 0 and kind[prev_ns] == 1:
                    after_sl_miss += 1
                    ks = int(np.searchsorted(t, st[j], "right")) - 1
                    if ks < 0 or np.hypot(x[ks] - ox[j], y[ks] - oy[j]) > radius:
                        after_sl_late += 1
            elif 1 <= hr[j] <= 3:
                kk = int(np.argmin(np.abs(t - ht[j])))
                dist = float(np.hypot(x[kk] - ox[j], y[kk] - oy[j]))
                landing.append(dist / radius)
                if dist <= radius:
                    hits_inside += 1
                    same += int(kk == 0 or np.hypot(x[kk - 1] - ox[j], y[kk - 1] - oy[j]) > radius)
                    ld = 0          # arrival lead (rows); not `lead`, which is the lead-in step mask
                    while ld < 4 and kk - 1 - ld >= 0 and np.hypot(x[kk - 1 - ld] - ox[j],
                                                                    y[kk - 1 - ld] - oy[j]) <= radius:
                        ld += 1
                    lead_hist[ld] += 1
                    if prev_ns >= 0 and kind[prev_ns] == 1:
                        as_in += 1
                        as_same += int(ld == 0)
                sp2 = float(d[kk] + (d[kk - 1] if kk >= 1 else 0.0))
                ks = int(np.searchsorted(t, st[j], "right")) - 1
                sl = 0
                while sl < 4 and ks - sl >= 0 and np.hypot(x[ks - sl] - ox[j], y[ks - sl] - oy[j]) <= radius:
                    sl += 1
                st_lead_hist[sl] += 1
                judged += 1
                press_sp2 += sp2
                steady += int(sp2 < 4.0)
                off_sum += float(ht[j] - st[j])
            prev_ns = j
    s = lambda m: round(float(d[m].sum()), 1)        # noqa: E731
    hist = dict(step=log_hist(d[nonspin]), jerk=log_hist(jerk[nonspin & (k_ >= 3)]), trans=log_hist(trans),
                landing=log_hist(landing), gate_idle=log_hist(d[g_idle]), slider=log_hist(d[slider]),
                approach=log_hist(d[approach]))
    hist["break"] = log_hist(d[brk])
    return dict(gate_idle_steps=int(g_idle.sum()), gate_idle_px=s(g_idle),
                gate_idle_still=int((g_idle & (d < 2.0)).sum()), gate_aim_steps=int(g_aim.sum()),
                gate_aim_px=s(g_aim), gate_spin_steps=int((valid & (gate == 2)).sum()), objects=int(cnt),
                leadin_steps=int(lead.sum()), leadin_px=s(lead), break_steps=int(brk.sum()), break_px=s(brk),
                break_aba=int((brk & aba).sum()), outro_steps=int(outro.sum()), outro_px=s(outro),
                midgap_steps=int(mid.sum()), midgap_px=s(mid), slider_steps=int(slider.sum()), slider_px=s(slider),
                approach_steps=int(approach.sum()), approach_px=s(approach), spinner_steps=int(in_spin.sum()),
                spinner_px=s(in_spin), nonspin_steps=int(nonspin.sum()), rev=int((rev & nonspin & mid_ok).sum()),
                snaps=int((snap & nonspin & mid_ok).sum()), gt100=int((nonspin & (d > 100.0)).sum()),
                home_gaps=home_gaps, home_visits=home_visits, hits_inside=hits_inside, same_frame_hits=same,
                fast_note_head_miss=fast_miss, after_slider_head_miss=after_sl_miss, after_slider_late=after_sl_late,
                rev2_px=round(float(rev2.sum()), 1), judged_hits=judged, steady_hits=steady,
                press_sp2_px=round(press_sp2, 1), head_offset_sum=round(off_sum, 2), lead_hist=lead_hist.tolist(),
                st_lead_hist=st_lead_hist.tolist(),
                after_slider_hits_inside=as_in, after_slider_same_frame=as_same,
                hist={k: v.tolist() for k, v in hist.items()})


def summarize_cursor(res):
    """Pooled cursor metrics over maps (None for records without them)."""
    cm = [r["cursor"] for r in res if "cursor" in r]
    if not cm:
        return {}
    idle_steps = sum(c["idle_steps"] for c in cm)
    move = sum(c["move_px"] for c in cm)
    out = dict(idle_px_per_step=round(sum(c["idle_px"] for c in cm) / max(idle_steps, 1), 2),
               idle_still_frac=round(sum(c["idle_still"] for c in cm) / max(idle_steps, 1), 4),
               path_eff=round(sum(c["ref_px"] for c in cm) / max(move, 1e-9), 4),
               move_px_per_map=round(move / len(cm), 1))
    if all("gate_idle_steps" in c for c in cm):
        out.update(summarize_motion(cm))
    return out


def pooled_hists(cm):
    """labelled (session 4): {key: summed log_hist list} over the maps' cursor dicts (None if no map has histograms)."""
    hs = [c["hist"] for c in cm if "hist" in c]
    if not hs:
        return None
    return {k: np.sum([np.asarray(h[k], np.int64) for h in hs], axis=0).tolist() for k in HIST_KEYS}


def summarize_motion(cm):
    """labelled (session 4): pool the motion_extra_metrics sums of several maps into DESIGN.md section 4 metrics."""
    S = lambda k: sum(c[k] for c in cm)               # noqa: E731

    def ratio(a, b, nd=2):
        return round(S(a) / S(b), nd) if S(b) else None
    H = {k: np.sum([np.asarray(c["hist"][k]) for c in cm if "hist" in c], axis=0) if any("hist" in c for c in cm)
         else None for k in HIST_KEYS}
    P = lambda k, q, nd=2: (None if H[k] is None or hist_pct(H[k], q) is None else round(hist_pct(H[k], q), nd))  # noqa: E731
    return dict(
        gate_idle_px_per_step=ratio("gate_idle_px", "gate_idle_steps"),
        gate_idle_still_frac=ratio("gate_idle_still", "gate_idle_steps", 4),
        gate_idle_p50=P("gate_idle", 50),
        gate_idle_share=round(S("gate_idle_steps") / max(S("gate_idle_steps") + S("gate_aim_steps")
                                                         + S("gate_spin_steps"), 1), 4),
        aim_px_per_obj=ratio("gate_aim_px", "objects", 1),
        leadin_px_per_step=ratio("leadin_px", "leadin_steps"),
        break_px_per_step=ratio("break_px", "break_steps"), break_p50=P("break", 50),
        break_aba=ratio("break_aba", "break_steps", 4), break_steps=S("break_steps"),
        outro_px_per_step=ratio("outro_px", "outro_steps"), midgap_px_per_step=ratio("midgap_px", "midgap_steps"),
        slider_px_per_step=ratio("slider_px", "slider_steps"), slider_p50=P("slider", 50),
        approach_px_per_step=ratio("approach_px", "approach_steps"), approach_p50=P("approach", 50),
        spinner_px_per_step=ratio("spinner_px", "spinner_steps"),
        rev_per_100=round(100.0 * S("rev") / max(S("nonspin_steps"), 1), 2),
        snaps_per_100=round(100.0 * S("snaps") / max(S("nonspin_steps"), 1), 2),
        frac_gt100=round(S("gt100") / max(S("nonspin_steps"), 1), 4),
        step_p50=P("step", 50), step_p99=P("step", 99), jerk_p50=P("jerk", 50),
        trans_path_ratio_p50=P("trans", 50, 3), trans_path_ratio_p90=P("trans", 90, 3),
        home_ninth_gap_frac=ratio("home_visits", "home_gaps", 4),
        same_frame_hit_share=ratio("same_frame_hits", "hits_inside", 4),
        landing_p50=P("landing", 50, 3), landing_p99=P("landing", 99, 3),
        fast_note_head_miss=S("fast_note_head_miss"), after_slider_head_miss=S("after_slider_head_miss"),
        after_slider_late=S("after_slider_late"), **summarize_motion_v2(cm))


def summarize_motion_v2(cm):
    """labelled (session 4) MOTION-COST v2 pooled metrics (DESIGN_v2.md); {} for maps evaluated without them."""
    if not all("steady_hits" in c for c in cm):
        return {}
    S = lambda k: sum(c[k] for c in cm)               # noqa: E731
    lead = np.sum([np.asarray(c["lead_hist"]) for c in cm], axis=0)
    J = max(S("judged_hits"), 1)
    return dict(rev2_px_per_obj=round(S("rev2_px") / max(S("objects"), 1), 2),
                steady_hit_share=round(S("steady_hits") / J, 4), press_sp2_per_hit=round(S("press_sp2_px") / J, 2),
                mean_head_offset_ms=round(S("head_offset_sum") / J, 3),
                arrival_lead_ge2_share=round(float(lead[2:].sum()) / max(float(lead.sum()), 1.0), 4),
                after_slider_same_frame_share=(round(S("after_slider_same_frame") / S("after_slider_hits_inside"), 4)
                                               if S("after_slider_hits_inside") else None),
                **summarize_st_lead(cm))


def summarize_st_lead(cm):
    """labelled (session 4) MOTION-COST v2 review: arrival relative to the object's start time (press-time invariant):
    share of judged hits with the cursor inside R at the last row at or before st (st_inside_share) and on that row
    and the one before it (st_arrival_ge2_share). {} if any map lacks st_lead_hist."""
    if not all("st_lead_hist" in c for c in cm):
        return {}
    h = np.sum([np.asarray(c["st_lead_hist"]) for c in cm], axis=0).astype(np.float64)
    tot = max(float(h.sum()), 1.0)
    return dict(st_inside_share=round(float(h[1:].sum()) / tot, 4),
                st_arrival_ge2_share=round(float(h[2:].sum()) / tot, 4))


def eval_score(summary):
    """best.pt score, labelled (session 3): max over modes of fc_rate - fail_per_obj (higher is better).
    Before any map FCs it ranks checkpoints by failures per object; one FC on a 45-map pool (+0.022) is worth
    about the whole 6* failure rate (~0.025), so FC dominates once maps start to FC."""
    return round(max(s["fc_rate"] - s["fail_per_obj"] for s in summary.values()), 6)


def legacy_score(summary):
    """Pre-session-3 best.pt score (fc_rate + 0.01 * mean_combo_frac), kept in the result for comparability."""
    return round(max(s["fc_rate"] + 0.01 * s["mean_combo_frac"] for s in summary.values()), 6)


class SimEvaluator:
    def __init__(self, cfg: dict, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.pool = MapPool.load(Path(ROOT) / cfg["pool"])
        self.pool_gpu = self.pool.to_torch(self.device)
        # labelled (session 3): hit_animations (default True) must match the training env's and the host profile
        self.renderer = make_renderer(cfg.get("renderer", "full"), self.pool_gpu, self.device, breaks=self.pool.breaks,
                                      hit_animations=bool(cfg.get("hit_animations", True)))
        self.modes = cfg.get("modes", ["deterministic", "any_key"])
        self.max_maps = cfg.get("max_maps", None)
        # labelled (session 3) robustness metric: evaluate with white pixel noise of this std (uint8 levels), fixed seed
        self.noise_sigma = float(cfg.get("noise_sigma", 0.0))
        # labelled (session 5) cursor physics: play with the cap vmax_px (osu px per step; absent = no physics, the old
        # evaluation); key_on_command (diagnostic) keeps the policy's key head on the commanded cursor instead of the
        # executed one
        self.vmax = cfg.get("vmax_px")
        if self.vmax == "train":          # set by train.py before each run (the controller's current cap)
            self.vmax = None
        self.key_on_command = bool(cfg.get("key_on_command", False))

    @torch.no_grad()
    def run_mode(self, agent, mode, maps):
        # labelled (session 4): zero motion prices (the evaluator never shapes rewards), gate logging on
        vm = getattr(self, "vmax", None)            # getattr: test doubles build the evaluator without __init__
        phys = None if vm is None else dict(vmax_px=float(vm))
        # labelled (session 6) DT: a pool with DT-tagged maps is played by osu_sim.rate.MapRateSimEnv (each map at its
        # m_clock_rate tag); a rate-one pool by SimEnv exactly as before
        from .rate import env_class_for_pool, pool_clock_rates
        tags = pool_clock_rates(self.pool)
        env = env_class_for_pool(self.pool)(self.pool, len(maps), "cpu", seed=0, sampler=UniformSampler(maps),
                                            clip=ClipConfig(full_map=True, random_initial_cursor=False), motion_cost=None,
                                            motion_metrics=True, cursor_physics=phys)
        vmax_t = None
        if phys is not None and not getattr(self, "key_on_command", False):
            vmax_t = torch.full((len(maps),), float(vm), device=self.device)
        for i, m in enumerate(maps):
            env._reset_one(i, int(m))
        n = len(maps)
        state = agent.initial_state(n, self.device)
        first = torch.ones(n, dtype=torch.bool, device=self.device)
        active = np.ones(n, bool)
        results = [None] * n
        agent.eval()
        steps = 0
        hist, end_step = [], np.zeros(n, np.int64)
        obj_state = [None] * n
        while active.any():
            rs = {k: torch.from_numpy(v).to(self.device) for k, v in env.render_state_np().items()}
            aux = torch.from_numpy(env.aux_np()).to(self.device)
            rgb = self.renderer.render(rs)
            if self.noise_sigma > 0:
                if steps == 0:
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(4321)
                noise = torch.randn(rgb.shape, generator=gen, device=self.device) * self.noise_sigma
                rgb = (rgb.float() + noise).round_().clamp_(0, 255).to(torch.uint8)
            action, _, state = agent.act(rgb, aux, state, first, mode=mode, vmax=vmax_t)
            first = torch.zeros_like(first)
            act = action.cursor.float().cpu().numpy()
            key = action.key.cpu().numpy()
            t_prev = env.e_t.copy()
            _, done, timeout = env.step_np(act, key)
            # labelled (session 4): cursor metrics; gate = motion gate of the pre-step state (osu_sim.motion)
            hist.append((t_prev, env.e_cx.copy(), env.e_cy.copy(), env.last_gate.copy()))
            steps += 1
            for i in np.nonzero(done & active)[0]:
                s = env.e_stats[i]
                poss = 0
                for g in range(env.e_lo[i], env.e_hi[i]):
                    poss += 1 + (int(self.pool.objs["nest_cnt"][g]) if self.pool.objs["kind"][g] == 1 else 0)
                G, O, M, X = s[rules.ST_GREAT], s[rules.ST_OK], s[rules.ST_MEH], s[rules.ST_MISS]
                SH, SM = s[rules.ST_SPIN_HIT], s[rules.ST_SPIN_MISS]
                judged = G + O + M + X + SH + SM
                TH, TM = s[rules.ST_TICK_HIT], s[rules.ST_TICK_MISS]
                LH, LM = s[rules.ST_TAIL_HIT], s[rules.ST_TAIL_MISS]
                # lazer-style accuracy: base score / max base score over accuracy-affecting results
                # (circles/heads/spinners 300, ticks+repeats 30, tails 150; spinner hits counted as 300)
                acc = (300 * G + 100 * O + 50 * M + 300 * SH + 30 * TH + 150 * LH) / max(
                    300 * judged + 30 * (TH + TM) + 150 * (LH + LM), 1)
                results[i] = dict(map=self.pool.keys[maps[i]], fc=bool(env.e_combo[i, 1] == poss and not timeout[i]),
                                  max_combo=int(env.e_combo[i, 1]), max_combo_possible=poss,
                                  breaks=int(s[rules.ST_BREAKS]), tail_miss=int(s[rules.ST_TAIL_MISS]),
                                  miss=int(X), tick_miss=int(s[rules.ST_TICK_MISS]), spin_miss=int(SM),
                                  spin_hit=int(SH),  # labelled (session 3): spinner non-miss rate needs hits too
                                  acc=round(float(acc), 5), presses=int(s[rules.ST_PRESS]), timeout=bool(timeout[i]),
                                  # labelled (session 4): counted by the rules, never recorded before (A/B guard G6)
                                  early_miss=int(s[rules.ST_EARLY_MISS]), shake=int(s[rules.ST_SHAKE]))
                if phys is not None:      # labelled (session 5) cursor physics
                    results[i].update(bind_steps=int(env.e_bind_steps[i]), unreached_press=int(env.e_unreached[i]))
                if tags[maps[i]] != 1.0:      # labelled (session 6) DT
                    results[i]["clock_rate"] = float(tags[maps[i]])
                cnt = int(env.e_hi[i] - env.e_lo[i])
                obj_state[i] = (env.st["head_r"][i, :cnt].copy(), env.st["head_t"][i, :cnt].copy())
                active[i] = False
                end_step[i] = steps
            # finished envs keep stepping harmlessly (e_done skips them)
        if hist:
            T = np.stack([h[0] for h in hist]); X = np.stack([h[1] for h in hist]); Y = np.stack([h[2] for h in hist])
            Gt = np.stack([h[3] for h in hist])
            for i, m in enumerate(maps):
                k = int(end_step[i])
                results[i]["cursor"] = cursor_metrics(T[:k, i], X[:k, i], Y[:k, i], self.pool.objs,
                                                      int(self.pool.maps["obj_off"][m]), int(self.pool.maps["obj_cnt"][m]),
                                                      gate=Gt[:k, i], breaks=self.pool.breaks[int(m)],
                                                      radius=float(self.pool.maps["radius"][m]),
                                                      head_r=obj_state[i][0], head_t=obj_state[i][1],
                                                      rate=float(tags[m]))
        return results, steps

    def run(self, agent):
        maps = np.arange(self.pool.n_maps)
        if self.max_maps:
            maps = maps[:self.max_maps]
        out = {}
        t0 = time.perf_counter()
        for mode in self.modes:
            res, steps = self.run_mode(agent, mode, maps)
            out[mode] = dict(summarize_maps(res), **summarize_cursor(res), maps=res)
            # labelled (session 4): the per-eval pooled histograms (HIST_KEYS x HIST_N ints) go into the detail record so
            # percentiles can be pooled exactly across evals (ab_motion_decide.py M3); per-map ones are not written
            ch = pooled_hists([r["cursor"] for r in res if "cursor" in r])
            if ch is not None:
                out[mode]["cursor_hist"] = ch
            for r in res:
                r.get("cursor", {}).pop("hist", None)
        summary = {m: {k: v for k, v in out[m].items() if k not in ("maps", "cursor_hist")} for m in self.modes}
        # labelled (session 3): score was max(fc_rate + 0.01 * mean_combo_frac), which stayed ~0.004 with no
        # FC and could not separate checkpoints; it is kept as score_legacy
        return dict(score=eval_score(summary), score_legacy=legacy_score(summary), summary=summary, detail=out,
                    eval_wall=round(time.perf_counter() - t0, 1))
