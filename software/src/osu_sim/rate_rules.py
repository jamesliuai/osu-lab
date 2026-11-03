"""Experimental CPU kernels for explicitly labelled rate conditions (per-env rates).

Only _update and step_all are forked from osu_sim.rules (2026-10-10); unchanged helpers/constants are imported from
rules. The ONLY differences are the declared RATE_EDITS below:
  * step_all takes frame_dt and clock_rate as float64[N] arrays (one gameplay step / one game rate per env slot, so
    NM, DT1.25 and DT slots can share one batch); slot n advances t_next = t_prev + frame_dt[n] gameplay ms;
  * _update takes that slot's scalar clock_rate; a spinner's wrapped PHYSICAL angle delta is credited as
    float32(float64(delta) * clock_rate) and never wrapped again (pinned SpinnerRotationTracker.AddRotation);
  * the training-only spinner progress-reward guard (|step| <= 150 deg) stays on the PHYSICAL delta.
Sub-updates stay <= 1 gameplay ms. With every rate 1.0 and frame_dt[n] == rules' frame_dt the kernels are bitwise
identical to rules.step_all (tests/test_sim_rate.py). No monkeypatches; no protected module is changed.

Historical source provenance: BASE_RULES_SHA256 records the original repository's rules.py bytes used when this
fork was synchronized. It is retained as that historical identifier, not as the hash of this release's rules.py.
The original project's tests/test_sim_rate.py used that hash and the declared RATE_EDITS as drift guards. The release
contains documented explanatory-text edits; its current artifact identities are in PUBLIC_ARTIFACT_MANIFEST.json.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from .rules import (
    GREAT, OK, MEH, MISS, ST_GREAT, ST_OK, ST_MEH, ST_MISS,
    ST_TICK_HIT, ST_TICK_MISS, ST_TAIL_HIT, ST_TAIL_MISS,
    ST_SPIN_HIT, ST_SPIN_MISS, ST_PRESS, ST_SHAKE, ST_EARLY_MISS,
    ST_FORCE_MISS, ST_BREAKS, ST_REWARD, _round_even, _result_for,
    _curve_pos, _dist2, _force_miss_earlier, _judge_head, _post_head,
    _judge_nested, _update_tracking,
)

# Historical original-repository base hash; not the hash of the exported rules.py derivative.
BASE_RULES_SHA256 = "5ece747e6388153eb2e301de08e11583049bdbf87513e1bcc542f840ba598ebb"

_SPIN_BASE = "                if trk and delta != 0:\n"
_SPIN_FORK = _SPIN_BASE + "".join(" " * 20 + line + "\n" for line in (
    "# Pinned SpinnerRotationTracker.AddRotation: wrap the physical delta first, then multiply by",
    "# abs(true gameplay rate) of this env (clock_rate) in double, cast to float, never re-wrapped.",
    "physical_delta = delta",
    "delta = np.float32(float(delta) * clock_rate)"))
# Declared rate-specific edits: (function, base text, fork text, occurrences in the base function). Applying them to the
# base function source (from its `def` line) must give this file's function source exactly; nothing else may differ.
RATE_EDITS = (
    ("_update", "stats, combo, vis, spin_progress, spin_min_r):\n",
     "stats, combo, vis, spin_progress, spin_min_r, clock_rate):\n", 1),
    ("_update", _SPIN_BASE, _SPIN_FORK, 1),
    ("_update", "abs(delta) <= 150.0", "abs(physical_delta) <= 150.0", 1),
    ("step_all", "e_sl_active, spin_progress, spin_min_r):\n",
     "e_sl_active, spin_progress, spin_min_r, clock_rate):\n", 1),
    ("step_all", "t_next = t_prev + frame_dt\n", "t_next = t_prev + frame_dt[n]\n", 1),
    ("step_all", "vis, spin_progress, spin_min_r)\n", "vis, spin_progress, spin_min_r, clock_rate[n])\n", 2),
)

@njit(cache=True)
def _update(n, t, lo, hi, gid0, cx, cy, keys_now, moved_input, radius, meh,
            o_kind, o_start, o_end, o_x, o_y, o_path_off, o_path_cnt, o_distance, o_span_count, o_nest_off,
            o_nest_cnt, o_w_head, o_w_tick, o_w_tail, o_spins_req, preempt, ne_type, ne_time, px, py, pcum,
            st_head_r, st_head_t, st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r, st_end_t,
            st_hit_action, st_accept_any, st_last_pressed,
            sp_has_pos, sp_last_angle, sp_total_acc, sp_acc_last, sp_cur_max, sp_completed, sp_rot, sp_trk,
            stats, combo, vis, spin_progress, spin_min_r, clock_rate):
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
                    # Pinned SpinnerRotationTracker.AddRotation: wrap the physical delta first, then multiply by
                    # abs(true gameplay rate) of this env (clock_rate) in double, cast to float, never re-wrapped.
                    physical_delta = delta
                    delta = np.float32(float(delta) * clock_rate)
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
                    if spin_progress != 0.0 and req != 0 and abs(physical_delta) <= 150.0 and far:
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
             admit_margin, break_penalty, tail_penalty, track_beta, e_phi_trk, e_sl_active, spin_progress, spin_min_r, clock_rate):
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
        t_next = t_prev + frame_dt[n]
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
                       e_stats, e_combo, vis, spin_progress, spin_min_r, clock_rate[n])
        u = t_prev
        while u < t_next:
            u = min(t_next, u + 1.0)
            rew += _update(n, u, lo, hi, gid0, cx, cy, new, False, radius, meh, o_kind, o_start, o_end, o_x, o_y,
                           o_path_off, o_path_cnt, o_distance, o_span_count, o_nest_off, o_nest_cnt, o_w_head,
                           o_w_tick, o_w_tail, o_spins_req, preempt, ne_type, ne_time, px, py, pcum,
                           st_head_r, st_head_t, st_trk, st_trk_t, st_nest_n, st_nest_miss, st_nest_t, st_tail_r,
                           st_end_t, st_hit_action, st_accept_any, st_last_pressed,
                           sp_has_pos, sp_last_angle, sp_total_acc, sp_acc_last, sp_cur_max, sp_completed, sp_rot,
                           sp_trk, e_stats, e_combo, vis, spin_progress, spin_min_r, clock_rate[n])
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


