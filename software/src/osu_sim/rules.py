"""Vectorised (numba, CPU-parallel) port of pinned-lazer osu!standard judgement rules.

Reference: runs/development/sim_v1/understand/rules_spec.md, especially section 12 (per-update algorithm)
and section 10 (host STEP timing). One call of `step_all` advances every env by one decision:

  input frame at t_prev  : cursor move, releases (Z,X), presses (Z,X) dispatched to the earliest hovered
                           unjudged circle/slider-head (StartTimeOrderedHitPolicy), then UPDATE(t_prev)
  sub-updates            : UPDATE(t_prev+1) ... UPDATE(t_prev+16), UPDATE(t_k)   (60 Hz: dt = 16.667 ms)

All positions are stacked gamefield osu!px. Times are float64 map ms.
Result codes: 1 Great, 2 Ok, 3 Meh, 4 Miss (0 = not judged).
"""
from __future__ import annotations

import math

import numpy as np

from . import ROOT  # noqa: F401  (sys.path for numba in .simpkgs)
import numba as nb
from numba import njit, prange

GREAT, OK, MEH, MISS = 1, 2, 3, 4
K_SLOTS = 24
QUALITY = np.array([0.0, 1.0, 1.0 / 3.0, 1.0 / 6.0, -1.0])

# stats indices (per env accumulators, reset by caller)
ST_GREAT, ST_OK, ST_MEH, ST_MISS, ST_TICK_HIT, ST_TICK_MISS, ST_TAIL_HIT, ST_TAIL_MISS, \
    ST_SPIN_HIT, ST_SPIN_MISS, ST_PRESS, ST_SHAKE, ST_EARLY_MISS, ST_FORCE_MISS, ST_BREAKS, ST_REWARD = range(16)
N_STATS = 16


@njit(cache=True, inline="always")
def _result_for(off, great, ok, meh):
    a = abs(off)
    if a <= great:
        return GREAT
    if a <= ok:
        return OK
    if a <= meh:
        return MEH
    if a <= 400.0:
        return MISS
    return 0


@njit(cache=True, inline="always")
def _round_even(x):
    r = math.floor(x)
    d = x - r
    if d > 0.5:
        return r + 1.0
    if d < 0.5:
        return r
    return r + 1.0 if (r % 2.0) != 0.0 else r


@njit(cache=True, inline="always")
def _path_pos(px, py, pcum, off, cnt, distance, progress):
    """SliderPath.PositionAt(progress) on the stacked absolute path (float32 output)."""
    if cnt == 0:
        return np.float32(0.0), np.float32(0.0)
    p = min(max(progress, 0.0), 1.0)
    d = p * distance
    # lower_bound over cum[off:off+cnt]
    lo, hi = 0, cnt
    while lo < hi:
        mid = (lo + hi) >> 1
        if pcum[off + mid] < d:
            lo = mid + 1
        else:
            hi = mid
    i = lo
    if i <= 0:
        return px[off], py[off]
    if i >= cnt:
        return px[off + cnt - 1], py[off + cnt - 1]
    d0 = pcum[off + i - 1]
    d1 = pcum[off + i]
    if abs(d1 - d0) <= 1e-7:
        return px[off + i - 1], py[off + i - 1]
    w = np.float32((d - d0) / (d1 - d0))
    x0, y0 = px[off + i - 1], py[off + i - 1]
    return np.float32(x0 + (px[off + i] - x0) * w), np.float32(y0 + (py[off + i] - y0) * w)


@njit(cache=True, inline="always")
def _curve_pos(px, py, pcum, off, cnt, distance, span_count, t):
    """IHasPathWithRepeats.CurvePositionAt(t), t in [0,1] over the slider."""
    p = t * span_count
    span = math.floor(p)
    frac = p - span
    if span >= span_count:
        span = span_count - 1
        frac = 1.0
    prog = 1.0 - frac if (int(span) % 2) == 1 else frac
    return _path_pos(px, py, pcum, off, cnt, distance, prog)


@njit(cache=True, inline="always")
def _dist2(ax, ay, bx, by):
    dx = np.float32(ax - bx)
    dy = np.float32(ay - by)
    return np.float32(dx * dx + dy * dy)


@njit(cache=True)
def _judge_head(n, j, res, t, o_kind, o_w_head, st_head_r, st_head_t, stats, combo):
    """Apply a circle / slider-head result. Returns reward."""
    st_head_r[n, j] = res
    st_head_t[n, j] = t
    w = 1.0 if o_kind == 0 else o_w_head
    if res == GREAT:
        stats[n, ST_GREAT] += 1
    elif res == OK:
        stats[n, ST_OK] += 1
    elif res == MEH:
        stats[n, ST_MEH] += 1
    else:
        stats[n, ST_MISS] += 1
    if res == MISS:
        stats[n, ST_BREAKS] += 1
        combo[n, 0] = 0
        return -w
    combo[n, 0] += 1
    if combo[n, 0] > combo[n, 1]:
        combo[n, 1] = combo[n, 0]
    return w * (1.0 if res == GREAT else (1.0 / 3.0 if res == OK else 1.0 / 6.0))


@njit(cache=True)
def _force_miss_earlier(n, j, t, lo, hi, gid0, o_kind, o_start, o_w_head, st_head_r, st_head_t, stats, combo):
    """HandleHit: force-miss every unjudged circle/head with StartTime < this one."""
    r = 0.0
    s = o_start[gid0 + j]
    for k in range(lo, j):
        g = gid0 + k
        kind = o_kind[g]
        if kind == 2 or st_head_r[n, k] != 0:
            continue
        if o_start[g] < s:
            r += _judge_head(n, k, MISS, t, kind, o_w_head[g], st_head_r, st_head_t, stats, combo)
            stats[n, ST_FORCE_MISS] += 1
    return r


@njit(cache=True)
def _post_head(n, j, t, g, cx, cy, radius, o_start, o_end, o_x, o_y, o_path_off, o_path_cnt, o_distance,
               o_span_count, o_nest_off, o_nest_cnt, o_w_tick, o_w_tail, ne_type, ne_time, px, py, pcum,
               st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r, stats, combo, keys_now,
               st_hit_action, st_accept_any, st_last_pressed):
    """SliderInputManager.PostProcessHeadJudgement after a successful head hit at time t. Returns reward."""
    dur = o_end[g] - o_start[g]
    prog = min(max((t - o_start[g]) / dur, 0.0), 1.0) if dur > 0 else 1.0
    fx, fy = _curve_pos(px, py, pcum, o_path_off[g], o_path_cnt[g], o_distance[g], o_span_count[g], prog)
    r24 = np.float32(radius) * np.float32(2.4)
    if _dist2(cx, cy, fx, fy) > r24 * r24:
        return 0.0
    # 1. all already-passed unjudged nested within follow range?
    all_in = True
    nb0 = o_nest_off[g]
    for e in range(st_nest_n[n, j], o_nest_cnt[g]):
        et = ne_time[nb0 + e]
        if et > t:
            break
        pr = min(max((et - o_start[g]) / dur, 0.0), 1.0) if dur > 0 else 1.0
        qx, qy = _curve_pos(px, py, pcum, o_path_off[g], o_path_cnt[g], o_distance[g], o_span_count[g], pr)
        if _dist2(cx, cy, qx, qy) > r24 * r24:
            all_in = False
            break
    rew = 0.0
    e = st_nest_n[n, j]
    while e < o_nest_cnt[g]:
        et = ne_time[nb0 + e]
        if et > t:
            break
        typ = ne_type[nb0 + e]
        rew += _judge_nested(n, j, e, typ, all_in, t, o_w_tick[g], o_w_tail[g], st_nest_n, st_nest_miss,
                             st_nest_t, st_tail_r, stats, combo)
        e += 1
    # 3. updateTracking(allTicksInRange || IsMouseInFollowArea(false))
    rr = np.float32(radius)
    valid = all_in or (_dist2(cx, cy, fx, fy) <= rr * rr)
    _update_tracking(n, j, t, valid, keys_now, st_hit_action, st_accept_any, st_last_pressed, st_trk, st_trk_t,
                     st_tail_r, o_end[g])
    return rew


@njit(cache=True)
def _judge_nested(n, j, e, typ, hit, t, w_tick, w_tail, st_nest_n, st_nest_miss, st_nest_t, st_tail_r, stats, combo):
    st_nest_n[n, j] = e + 1
    st_nest_t[n, j] = t
    if typ == 2:  # tail
        st_tail_r[n, j] = 1 if hit else 2
        if hit:
            stats[n, ST_TAIL_HIT] += 1
            combo[n, 0] += 1
            if combo[n, 0] > combo[n, 1]:
                combo[n, 1] = combo[n, 0]
            return w_tail
        stats[n, ST_TAIL_MISS] += 1   # IgnoreMiss: no combo break, but not FC
        if e < 64:
            st_nest_miss[n, j] |= np.uint64(1) << np.uint64(e)
        return -w_tail
    if hit:
        stats[n, ST_TICK_HIT] += 1
        combo[n, 0] += 1
        if combo[n, 0] > combo[n, 1]:
            combo[n, 1] = combo[n, 0]
        return w_tick
    stats[n, ST_TICK_MISS] += 1
    stats[n, ST_BREAKS] += 1
    combo[n, 0] = 0
    if e < 64:
        st_nest_miss[n, j] |= np.uint64(1) << np.uint64(e)
    return -w_tick


@njit(cache=True)
def _update_tracking(n, j, t, valid_pos, keys_now, st_hit_action, st_accept_any, st_last_pressed, st_trk,
                     st_trk_t, st_tail_r, end):
    """SliderInputManager.updateTracking (st_hit_action: 0 none, 1 Left/Z, 2 Right/X)."""
    hit_action = st_hit_action[n, j]
    if hit_action == 0:
        st_accept_any[n, j] = np.nan
    elif math.isnan(st_accept_any[n, j]):
        other = 2 if hit_action == 1 else 1
        if (st_last_pressed[n, j] & other) == 0:
            st_accept_any[n, j] = t
    st_last_pressed[n, j] = keys_now
    valid_action = False
    if hit_action != 0 and (math.isnan(st_accept_any[n, j]) or t <= st_accept_any[n, j]):
        valid_action = (keys_now & hit_action) != 0
    else:
        valid_action = keys_now != 0
    all_judged = st_tail_r[n, j] != 0
    trk = ((not all_judged) or t <= end) and valid_pos and valid_action
    if trk != (st_trk[n, j] != 0):
        st_trk_t[n, j] = t
    st_trk[n, j] = 1 if trk else 0


@njit(cache=True)
def _update(n, t, lo, hi, gid0, cx, cy, keys_now, moved_input, radius, meh,
            o_kind, o_start, o_end, o_x, o_y, o_path_off, o_path_cnt, o_distance, o_span_count, o_nest_off,
            o_nest_cnt, o_w_head, o_w_tick, o_w_tail, o_spins_req, preempt, ne_type, ne_time, px, py, pcum,
            st_head_r, st_head_t, st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r, st_end_t,
            st_hit_action, st_accept_any, st_last_pressed,
            sp_has_pos, sp_last_angle, sp_total_acc, sp_acc_last, sp_cur_max, sp_completed, sp_rot, sp_trk,
            stats, combo, vis, spin_progress, spin_min_r):
    """One lazer update frame at time t for env n over objects [lo, hi) (relative indices).

    vis[n, j, :] visual state for the renderer: 0 slider visual tracking, 1 its change time (>= start),
    2 previous change time, 3 spinner damped rotation, 4 drawn rotation (one update behind), 5 tracking
    interpolation, 6 time started, 7 time completed, 8 last whole-spin change time, 9 whole spins, 10 last update.
    Returned reward includes the spinner progress reward (spin_progress, 0 = off).
    """
    rew = 0.0
    for j in range(lo, hi):
        g = gid0 + j
        kind = o_kind[g]
        start = o_start[g]
        if t < start - preempt:
            continue   # not alive yet
        if kind == 0:
            if st_head_r[n, j] == 0 and t - start > meh:
                rew += _force_miss_earlier(n, j, t, lo, hi, gid0, o_kind, o_start, o_w_head, st_head_r, st_head_t,
                                           stats, combo)
                rew += _judge_head(n, j, MISS, t, kind, o_w_head[g], st_head_r, st_head_t, stats, combo)
                st_end_t[n, j] = t
        elif kind == 1:
            if st_end_t[n, j] > -1e8 and st_tail_r[n, j] != 0:
                continue  # slider fully judged
            end = o_end[g]
            dur = end - start
            # DrawableSlider.Update copies Tracking (visual, one update behind) before the input manager updates
            if st_trk[n, j] != int(vis[n, j, 0]):
                vis[n, j, 2] = vis[n, j, 1]
                vis[n, j, 1] = max(t, start)
                vis[n, j, 0] = st_trk[n, j]
            # a. tracking update with expanded radius if tracking
            prog = min(max((t - start) / dur, 0.0), 1.0) if dur > 0 else 1.0
            fx, fy = _curve_pos(px, py, pcum, o_path_off[g], o_path_cnt[g], o_distance[g], o_span_count[g], prog)
            rr = np.float32(radius) * (np.float32(2.4) if st_trk[n, j] != 0 else np.float32(1.0))
            valid = _dist2(cx, cy, fx, fy) <= rr * rr
            _update_tracking(n, j, t, valid, keys_now, st_hit_action, st_accept_any, st_last_pressed, st_trk,
                             st_trk_t, st_tail_r, end)
            # b. nested (only after head judged)
            if st_head_r[n, j] != 0:
                nb0 = o_nest_off[g]
                e = st_nest_n[n, j]
                while e < o_nest_cnt[g]:
                    typ = ne_type[nb0 + e]
                    et = ne_time[nb0 + e]
                    if typ != 2:
                        if t < et:
                            break
                        rew += _judge_nested(n, j, e, typ, st_trk[n, j] != 0, t, o_w_tick[g], o_w_tail[g],
                                             st_nest_n, st_nest_miss, st_nest_t, st_tail_r, stats, combo)
                    else:
                        if t < et - 36.0:
                            break
                        if st_trk[n, j] != 0:
                            rew += _judge_nested(n, j, e, typ, True, t, o_w_tick[g], o_w_tail[g], st_nest_n,
                                                 st_nest_miss, st_nest_t, st_tail_r, stats, combo)
                        elif t >= et:
                            rew += _judge_nested(n, j, e, typ, False, t, o_w_tick[g], o_w_tail[g], st_nest_n,
                                                 st_nest_miss, st_nest_t, st_tail_r, stats, combo)
                        else:
                            break
                    e += 1
            # c. head auto-miss
            if st_head_r[n, j] == 0 and t - start > meh:
                rew += _force_miss_earlier(n, j, t, lo, hi, gid0, o_kind, o_start, o_w_head, st_head_r, st_head_t,
                                           stats, combo)
                rew += _judge_head(n, j, MISS, t, kind, o_w_head[g], st_head_r, st_head_t, stats, combo)
            # d. aggregate
            if st_tail_r[n, j] != 0 and t >= end and st_end_t[n, j] < -1e8:
                st_end_t[n, j] = t
        else:  # spinner
            judged = st_head_r[n, j] != 0
            end = o_end[g]
            spinnable = start <= t and t < end
            trk = spinnable and keys_now != 0 and not judged
            sp_trk[n, j] = 1 if trk else 0
            if moved_input:
                sp_has_pos[n, j] = 1
            if sp_has_pos[n, j] != 0:
                ang = np.float32(-math.degrees(math.atan2(np.float32(cx - np.float32(256.0)),
                                                          np.float32(cy - np.float32(192.0)))))
                if math.isnan(sp_last_angle[n, j]):
                    delta = np.float32(0.0)
                else:
                    delta = np.float32(ang - sp_last_angle[n, j])
                if delta > 180:
                    delta = np.float32(delta - 360)
                if delta < -180:
                    delta = np.float32(delta + 360)
                if trk and delta != 0:
                    tot0 = 360.0 * sp_completed[n, j] + sp_cur_max[n, j]
                    sp_rot[n, j] = np.float32(sp_rot[n, j] + delta)
                    tot = np.float32(sp_total_acc[n, j] + delta)
                    sp_total_acc[n, j] = tot
                    cur = np.float32(tot - sp_acc_last[n, j])
                    cm = max(sp_cur_max[n, j], abs(cur))
                    while cm >= 360:
                        d = np.float32(1.0) if cur > 0 else np.float32(-1.0)
                        sp_completed[n, j] += 1
                        sp_acc_last[n, j] = np.float32(sp_acc_last[n, j] + d * np.float32(360.0))
                        cur = np.float32(tot - sp_acc_last[n, j])
                        cm = abs(cur)
                    sp_cur_max[n, j] = cm
                    req = o_spins_req[g]
                    # no progress credit for steps faster than 150 deg (25 rps, far above any req): keeps the learned
                    # spin away from lazer's +-180 deg wrap, where float32 differences could flip the counted direction
                    # between sim and real (the judgement itself still counts every step exactly as lazer does)
                    # and (labelled, session 3) only while the cursor is >= spin_min_r osu px from the spinner centre:
                    # the minimum radius reduces angular effects from small positional changes
                    far = (cx - np.float32(256.0)) ** 2 + (cy - np.float32(192.0)) ** 2 >= spin_min_r * spin_min_r
                    if spin_progress != 0.0 and req != 0 and abs(delta) <= 150.0 and far:
                        # labelled (session 3) objective change, replaces the spec 11 progress potential (spin_beta):
                        # NON-telescoping capped spinner progress reward. tot = 360 * completed + cur_max (lazer
                        # SpinnerSpinHistory) only grows, and only here, so the increments sum to exactly
                        # spin_progress * min(final tot / 360 / req, 1) per spinner, with nothing taken back at
                        # judgement. Monotone and capped, so it cannot be farmed: back-and-forth oscillation through
                        # the centre earns its maximum excursion (< 1/req) once.
                        tot1 = 360.0 * sp_completed[n, j] + sp_cur_max[n, j]
                        rew += spin_progress * (min(tot1 / 360.0 / req, 1.0) - min(tot0 / 360.0 / req, 1.0))
                sp_last_angle[n, j] = ang
            # visual bookkeeping (ArgonSpinnerDisc / DrawableSpinner)
            el = abs(t - vis[n, j, 10]) if vis[n, j, 10] > -1e8 else 0.0
            if not (vis[n, j, 7] > -1e8 and t >= vis[n, j, 7]):
                vis[n, j, 5] += ((1.0 if trk else 0.0) - vis[n, j, 5]) * (1.0 - 0.985 ** el)
            if vis[n, j, 6] < -1e8 and trk:
                vis[n, j, 6] = t
            total = np.float32(360.0) * np.float32(sp_completed[n, j]) + sp_cur_max[n, j]
            req = o_spins_req[g]
            prog = np.float32(1.0) if req == 0 else min(max(np.float32(total / np.float32(360.0) / np.float32(req)),
                                                             np.float32(0.0)), np.float32(1.0))
            if prog >= 1 and vis[n, j, 7] < -1e8:
                vis[n, j, 7] = t
            whole = float(int(total / 360.0))
            if whole != vis[n, j, 9]:
                vis[n, j, 9] = whole
                vis[n, j, 8] = t
            vis[n, j, 4] = vis[n, j, 3]
            vis[n, j, 3] = vis[n, j, 3] + (sp_rot[n, j] - vis[n, j, 3]) * (1.0 - 0.99 ** el)
            vis[n, j, 10] = t
            if (not judged) and t >= end:
                if prog >= 1:
                    res = GREAT
                elif prog > 0.9:
                    res = OK
                elif prog > 0.75:
                    res = MEH
                else:
                    res = MISS
                st_head_r[n, j] = res
                st_head_t[n, j] = t
                st_end_t[n, j] = t
                if res == MISS:
                    stats[n, ST_SPIN_MISS] += 1
                    stats[n, ST_BREAKS] += 1
                    combo[n, 0] = 0
                    rew -= 1.0
                else:
                    stats[n, ST_SPIN_HIT] += 1
                    combo[n, 0] += 1
                    if combo[n, 0] > combo[n, 1]:
                        combo[n, 1] = combo[n, 0]
                    rew += 1.0 if res == GREAT else (1.0 / 3.0 if res == OK else 1.0 / 6.0)
    return rew


@njit(cache=True, parallel=True)
def step_all(act_x, act_y, act_key, frame_dt,
             e_map, e_t, e_lo, e_hi, e_win_lo, e_win_hi, e_settle, e_cx, e_cy, e_keys, e_press_t, e_release_t,
             e_done, e_reward, e_stats, e_combo, key_down_cost,
             m_radius, m_preempt, m_great, m_ok, m_meh,
             o_kind, o_start, o_end, o_x, o_y, o_path_off, o_path_cnt, o_distance, o_span_count, o_nest_off,
             o_nest_cnt, o_w_head, o_w_tick, o_w_tail, o_spins_req,
             ne_type, ne_time, px, py, pcum,
             st_head_r, st_head_t, st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r, st_end_t,
             st_hit_action, st_accept_any, st_last_pressed,
             sp_has_pos, sp_last_angle, sp_total_acc, sp_acc_last, sp_cur_max, sp_completed, sp_rot, sp_trk,
             origin_x, origin_y, client_scale, shaping_beta, shaping_gamma, e_phi, spin_beta, e_phi_spin, vis,
             admit_margin, break_penalty, tail_penalty, track_beta, e_phi_trk, e_sl_active, spin_progress, spin_min_r):
    N = e_t.shape[0]
    for n in prange(N):
        e_reward[n] = 0.0
        if e_done[n]:
            continue
        breaks0 = e_stats[n, ST_BREAKS]
        tails0 = e_stats[n, ST_TAIL_MISS]
        m = e_map[n]
        radius = m_radius[m]
        preempt = m_preempt[m]
        great, ok, meh = m_great[m], m_ok[m], m_meh[m]
        gid0 = e_lo[n]
        nobj = e_hi[n] - e_lo[n]
        t_prev = e_t[n]
        t_next = t_prev + frame_dt
        # --- cursor (host mapping: client pixel = 160 + round_even((a+1)*959/2), round_even((a+1)*719/2))
        cxp = 160.0 + _round_even((act_x[n] + 1.0) * 959.0 / 2.0)
        cyp = _round_even((act_y[n] + 1.0) * 719.0 / 2.0)
        cx = np.float32((cxp - origin_x) / client_scale)
        cy = np.float32((cyp - origin_y) / client_scale)
        moved = (cx != e_cx[n]) or (cy != e_cy[n])
        e_cx[n] = cx
        e_cy[n] = cy
        # --- window: objects alive by t_next
        hi = e_win_hi[n]
        while hi < nobj and o_start[gid0 + hi] - preempt - admit_margin <= t_next:
            hi += 1
        e_win_hi[n] = hi
        lo = e_win_lo[n]
        rew = 0.0
        # --- input frame at t_prev: releases then presses
        old = e_keys[n]
        new = act_key[n]
        held = old & new
        keys_now = held
        if new == 0 and old != 0:
            e_release_t[n] = t_prev
        for bit in (1, 2):
            if (new & bit) != 0 and (old & bit) == 0:
                keys_now |= bit
                e_press_t[n] = t_prev
                e_stats[n, ST_PRESS] += 1
                rew -= key_down_cost
                # dispatch the press to the earliest hovered unjudged circle / slider head
                r2 = np.float32(radius) * np.float32(radius)
                for j in range(lo, hi):
                    g = gid0 + j
                    kind = o_kind[g]
                    if kind == 2 or st_head_r[n, j] != 0:
                        continue
                    if t_prev < o_start[g] - preempt:
                        continue
                    if _dist2(cx, cy, o_x[g], o_y[g]) > r2:
                        continue
                    off = t_prev - o_start[g]
                    res = _result_for(off, great, ok, meh)
                    # latest blocking object with StartTime < this one
                    blocked = False
                    for k in range(j - 1, -1, -1):
                        gk = gid0 + k
                        if o_kind[gk] == 2:
                            continue
                        if o_start[gk] < o_start[g]:
                            if st_head_r[n, k] == 0 and t_prev < o_start[gk]:
                                blocked = True
                            break
                    if blocked or res == 0:
                        e_stats[n, ST_SHAKE] += 1
                    else:
                        rew += _force_miss_earlier(n, j, t_prev, lo, hi, gid0, o_kind, o_start, o_w_head, st_head_r,
                                                   st_head_t, e_stats, e_combo)
                        rew += _judge_head(n, j, res, t_prev, kind, o_w_head[g], st_head_r, st_head_t, e_stats,
                                           e_combo)
                        if res == MISS:
                            e_stats[n, ST_EARLY_MISS] += 1
                        if kind == 0:
                            st_end_t[n, j] = t_prev
                        elif res != MISS:
                            rew += _post_head(n, j, t_prev, g, cx, cy, radius, o_start, o_end, o_x, o_y, o_path_off,
                                              o_path_cnt, o_distance, o_span_count, o_nest_off, o_nest_cnt, o_w_tick,
                                              o_w_tail, ne_type, ne_time, px, py, pcum, st_trk, st_trk_t, st_nest_n,
                                              st_nest_miss, st_nest_t, st_tail_r, e_stats, e_combo, keys_now,
                                              st_hit_action, st_accept_any, st_last_pressed)
                    if kind == 1 and st_hit_action[n, j] == 0:
                        st_hit_action[n, j] = bit
                    break
        e_keys[n] = new
        # --- update at t_prev (input frame), then sub-steps
        rew += _update(n, t_prev, lo, hi, gid0, cx, cy, new, moved, radius, meh, o_kind, o_start, o_end, o_x, o_y,
                       o_path_off, o_path_cnt, o_distance, o_span_count, o_nest_off, o_nest_cnt, o_w_head, o_w_tick,
                       o_w_tail, o_spins_req, preempt, ne_type, ne_time, px, py, pcum,
                       st_head_r, st_head_t, st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r, st_end_t,
                       st_hit_action, st_accept_any, st_last_pressed,
                       sp_has_pos, sp_last_angle, sp_total_acc, sp_acc_last, sp_cur_max, sp_completed, sp_rot, sp_trk,
                       e_stats, e_combo, vis, spin_progress, spin_min_r)
        u = t_prev
        while u < t_next:
            u = min(t_next, u + 1.0)
            rew += _update(n, u, lo, hi, gid0, cx, cy, new, False, radius, meh, o_kind, o_start, o_end, o_x, o_y,
                           o_path_off, o_path_cnt, o_distance, o_span_count, o_nest_off, o_nest_cnt, o_w_head,
                           o_w_tick, o_w_tail, o_spins_req, preempt, ne_type, ne_time, px, py, pcum,
                           st_head_r, st_head_t, st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r,
                           st_end_t, st_hit_action, st_accept_any, st_last_pressed,
                           sp_has_pos, sp_last_angle, sp_total_acc, sp_acc_last, sp_cur_max, sp_completed, sp_rot,
                           sp_trk, e_stats, e_combo, vis, spin_progress, spin_min_r)
        e_t[n] = t_next
        # --- release finished objects from the front of the window (visuals end ~1 s after full judgement)
        while lo < hi:
            g = gid0 + lo
            fin = st_end_t[n, lo]
            if o_kind[g] == 0:
                fin = st_head_t[n, lo] if st_head_r[n, lo] != 0 else -1e9
            if fin > -1e8 and t_next - fin > 1000.0:
                lo += 1
            else:
                break
        e_win_lo[n] = lo
        # --- termination: everything judged and settlement time reached
        done = t_next >= e_settle[n]
        if done:
            for j in range(e_win_lo[n], nobj):
                g = gid0 + j
                if o_kind[g] == 1:
                    if st_tail_r[n, j] == 0 or st_head_r[n, j] == 0:
                        done = False
                        break
                elif st_head_r[n, j] == 0:
                    done = False
                    break
        e_done[n] = done
        # labelled FC objective: extra penalty per combo break (Miss / LargeTickMiss / spinner Miss) and per missed tail
        rew -= break_penalty * (e_stats[n, ST_BREAKS] - breaks0) + tail_penalty * (e_stats[n, ST_TAIL_MISS] - tails0)
        if shaping_beta != 0.0:
            # labelled spec 9.3 potential shaping: phi = 1 - clamp(d / (4 r), 0, 1) to the earliest unjudged
            # alive circle/slider head (or the active slider follow position); terminal phi = 0
            phi = 0.0
            if not done:
                for j in range(e_win_lo[n], e_win_hi[n]):
                    g = gid0 + j
                    kind = o_kind[g]
                    if kind == 2:
                        continue
                    if st_head_r[n, j] == 0:
                        tx, ty = o_x[g], o_y[g]
                    elif kind == 1 and st_tail_r[n, j] == 0:
                        dur = o_end[g] - o_start[g]
                        pr = min(max((t_next - o_start[g]) / dur, 0.0), 1.0) if dur > 0 else 1.0
                        tx, ty = _curve_pos(px, py, pcum, o_path_off[g], o_path_cnt[g], o_distance[g],
                                            o_span_count[g], pr)
                    else:
                        continue
                    d = math.sqrt(_dist2(cx, cy, tx, ty))
                    phi = 1.0 - min(d / (4.0 * radius), 1.0)
                    break
            rew += shaping_beta * (shaping_gamma * phi - e_phi[n])
            e_phi[n] = phi
        if spin_beta != 0.0:
            # labelled spec 11 spinner progress potential: sum of capped progress of started, unjudged spinners;
            # drops to 0 when the spinner is judged (and at episode end), so it telescopes per spinner.
            # Superseded (labelled, session 3) by the non-telescoping spin_progress reward in _update; set it to 0.
            phs = 0.0
            if not done:
                for j in range(e_win_lo[n], e_win_hi[n]):
                    g = gid0 + j
                    if o_kind[g] != 2 or st_head_r[n, j] != 0 or t_next < o_start[g]:
                        continue
                    req = o_spins_req[g]
                    tot = 360.0 * sp_completed[n, j] + sp_cur_max[n, j]
                    phs += 1.0 if req == 0 else min(tot / 360.0 / req, 1.0)
            rew += spin_beta * (shaping_gamma * phs - e_phi_spin[n])
            e_phi_spin[n] = phs
        if track_beta != 0.0:
            # labelled potential shaping (session 3): phi = number of started sliders whose tail is not yet judged
            # and which are currently tracked. Telescopes per slider (policy-invariant). Corrected (session 3): it
            # does NOT give a held frame real immediate credit. Under GAE(lambda=0.95) and a critic that has not
            # learned phi, a frame tracked for one step gains only gamma*(1-lambda)*beta extra advantage (~0.05 beta),
            # and exactly 0 once the critic absorbs phi (V' = V + beta*phi leaves every TD error unchanged). It is
            # kept only so that runs already using it do not get a critic transient when it is switched off.
            pht = 0.0
            if not done:
                for j in range(e_win_lo[n], e_win_hi[n]):
                    g = gid0 + j
                    if o_kind[g] != 1 or st_tail_r[n, j] != 0 or t_next < o_start[g]:
                        continue
                    if st_trk[n, j] != 0:
                        pht += 1.0
            rew += track_beta * (shaping_gamma * pht - e_phi_trk[n])
            e_phi_trk[n] = pht
        # state flag for the next decision (training-only, never shown to the actor): some started slider has a
        # judged head and an unjudged tail, i.e. holding a key is what the rules reward now
        act_sl = 0
        if not done:
            for j in range(e_win_lo[n], e_win_hi[n]):
                g = gid0 + j
                if o_kind[g] == 1 and st_head_r[n, j] != 0 and st_tail_r[n, j] == 0 and t_next >= o_start[g]:
                    act_sl = 1
                    break
        e_sl_active[n] = act_sl
        e_reward[n] = rew
        e_stats[n, ST_REWARD] += rew


@njit(cache=True, parallel=True)
def render_slots(e_t, e_lo, e_win_lo, e_win_hi, e_map, m_preempt, o_start,
                 st_head_r, st_head_t, st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r, st_end_t,
                 sp_rot, sp_total, sp_trk, vis, o_kind,
                 r_gid, r_head_r, r_head_t, r_trk, r_trk_t, r_nest_n, r_nest_miss, r_nest_t, r_tail_r, r_end_t,
                 r_spin_rot, r_spin_total, r_spin_trk, r_trk_tp, r_spin_ti, r_spin_ts, r_spin_tc, r_spin_last):
    """Fill the K render slots (RENDER_INTERFACE.md) from the rules state."""
    N = e_t.shape[0]
    K = r_gid.shape[1]
    for n in prange(N):
        lo, hi = e_win_lo[n], e_win_hi[n]
        if hi - lo > K:
            lo = hi - K
        k = 0
        for j in range(lo, hi):
            r_gid[n, k] = e_lo[n] + j
            r_head_r[n, k] = st_head_r[n, j]
            r_head_t[n, k] = st_head_t[n, j]
            r_trk[n, k] = int(vis[n, j, 0])
            r_trk_t[n, k] = vis[n, j, 1]
            r_trk_tp[n, k] = vis[n, j, 2]
            r_spin_ti[n, k] = vis[n, j, 5]
            r_spin_ts[n, k] = vis[n, j, 6]
            r_spin_tc[n, k] = vis[n, j, 7]
            r_spin_last[n, k] = vis[n, j, 8]
            r_nest_n[n, k] = st_nest_n[n, j]
            r_nest_miss[n, k] = st_nest_miss[n, j]
            r_nest_t[n, k] = st_nest_t[n, j]
            r_tail_r[n, k] = st_tail_r[n, j]
            r_end_t[n, k] = st_end_t[n, j]
            r_spin_rot[n, k] = vis[n, j, 4] if o_kind[e_lo[n] + j] == 2 else sp_rot[n, j]
            r_spin_total[n, k] = sp_total[n, j]
            r_spin_trk[n, k] = sp_trk[n, j]
            k += 1
        while k < K:
            r_gid[n, k] = -1
            k += 1
