"""Opt-in CPU reference environment for labelled DT/rate conditions, one game rate per env slot.

Original maps, native SR metadata and existing training factories are untouched. Disabled or all-rate-one calls of
make_rate_env return the original SimEnv. Accelerated wall-clock collection is separate from this change of game
condition. Real-host parity has NOT been established; see runs/development/session5_data/RATE_DESIGN.md.

Pool tags: a pool may carry maps["clock_rate"] (npz array m_clock_rate, float64 [M]; written by
scripts/sim_data_export.py) naming the game rate its metadata (native stars, condition key) was computed for. Pools
without it are legacy rate-one pools. A map is never played at a rate other than its tag (assert_pool_rate, and a
per-reset guard in RateSimEnv).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .env import CLIENT_SCALE, ORIGIN_X, ORIGIN_Y, SimEnv, UniformSampler
from . import motion, rate_rules

POOL_RATE_FIELD = "clock_rate"      # MapPool.maps key (saved as m_clock_rate)

# Declared differences of the two SimEnv methods RateSimEnv re-implements: (base source text, fork source text,
# occurrences). tests/test_sim_rate.py applies them to the CURRENT SimEnv source and requires the result to parse to
# the same AST as RateSimEnv's method (comments/formatting ignored); anything else is drift.
SIMENV_OVERRIDE_EDITS = {
    "step_np": (
        ("rules.step_all(act[:, 0].copy(), act[:, 1].copy(), key, self.dt,",
         "rate_rules.step_all(act[:, 0].copy(), act[:, 1].copy(), key, self.e_dt,", 1),
        ("self.e_sl_active, self.spin_progress, self.spin_min_r)",
         "self.e_sl_active, self.spin_progress, self.spin_min_r, self.e_rate)", 1),
    ),
    "reset": (
        ("        maps = self.sampler.sample(self.rng, len(idx))\n",
         "        self.e_rate[idx] = self.e_rate_next[idx]\n"
         "        self.e_dt[idx] = self.real_step_ms * self.e_rate[idx]\n"
         "        maps = self.sampler.sample_slots(self.rng, self.e_rate[idx])\n", 1),
    ),
}


def canonical_rate(rate):
    """Validated game rate as the canonical double of its 0.01 step: 1.0, or a DT speed in [1.01, 2.00]."""
    r = float(rate)
    if not math.isfinite(r) or not (r == 1.0 or 1.01 <= r <= 2.0):
        raise ValueError(f"rate must be 1.0 or a DT speed in [1.01, 2.0], got {rate!r}")
    if abs(r * 100 - round(r * 100)) > 1e-9:
        raise ValueError(f"DT speed precision is 0.01 in the pinned game, got {rate!r}")
    return round(r * 100) / 100


@dataclass(frozen=True)
class RateClock:
    """Units are gameplay/map milliseconds and real/control milliseconds (one rate; RateSimEnv has one per slot)."""

    rate: float = 1.0
    control_hz: int = 60

    def __post_init__(self):
        object.__setattr__(self, "rate", canonical_rate(self.rate))
        if self.control_hz not in (60, 120):
            raise ValueError("control_hz must be 60 or 120")

    @property
    def real_step_ms(self):
        return 1000.0 / self.control_hz

    @property
    def gameplay_step_ms(self):
        return self.real_step_ms * self.rate

    def to_real_ms(self, gameplay_ms):
        return gameplay_ms / self.rate

    def to_gameplay_ms(self, real_ms):
        return real_ms * self.rate

    def metadata(self):
        return {
            "game_condition": "NM" if self.rate == 1.0 else "DT",
            "clock_rate": self.rate,
            "control_hz": self.control_hz,
            "decision_interval_real_ms": self.real_step_ms,
            "decision_interval_gameplay_ms": self.gameplay_step_ms,
            "mod_settings": {} if self.rate == 1.0 else {"speed_change": self.rate},
            "is_default_dt": self.rate == 1.5,
            "real_host_parity": "not_validated",
        }


def slot_rates(clock_rate, n):
    """Per-slot rates float64[n] from a scalar (every slot) or a length-n sequence; each validated and canonical."""
    a = np.asarray(clock_rate, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(n, float(a))
    elif a.shape != (n,):
        raise ValueError(f"clock_rate must be a scalar or one rate per env slot (shape ({n},)), got shape {a.shape}")
    values, inverse = np.unique(a, return_inverse=True)
    return np.array([canonical_rate(v) for v in values], dtype=np.float64)[inverse.reshape(-1)]


def pool_clock_rates(pool):
    """float64[M] per-map game-rate tags (maps['clock_rate'] / npz m_clock_rate). A legacy pool without the array is
    rate 1.0 throughout. Each tag must be a canonical game rate."""
    tags = pool.maps.get(POOL_RATE_FIELD)
    if tags is None:
        return np.ones(pool.n_maps, dtype=np.float64)
    tags = np.asarray(tags)
    if tags.dtype != np.float64 or tags.shape != (pool.n_maps,):
        raise ValueError(f"pool m_clock_rate must be float64[{pool.n_maps}], got {tags.dtype}{list(tags.shape)}")
    for t in np.unique(tags).tolist():
        try:
            ok = canonical_rate(t) == t
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"pool m_clock_rate tag {t!r} is not a canonical game rate")
    return tags.copy()


def assert_pool_rate(pool, rates, *, rate_matched=False):
    """Refuse a pool whose map tags (m_clock_rate; legacy pools = 1.0) disagree with the env's per-slot rates.

    rates: scalar or one rate per slot (rate disabled / ordinary SimEnv = 1.0). Always: every slot rate must be the tag
    of at least one map. rate_matched=False (one sampler that may hand any map to any slot): every map tag must equal
    every slot rate, i.e. one uniform rate, so a rate-one env refuses any DT-tagged map and a DT env refuses a legacy
    or other-rate pool. rate_matched=True (per-rate samplers: slot n draws only maps tagged rates[n], RateSimEnv's
    default): maps of other tags are never drawn and are allowed. Returns the per-map tags.
    """
    tags = pool_clock_rates(pool)
    a = np.asarray(rates, dtype=np.float64)
    want = sorted(set(slot_rates(a, a.size if a.ndim == 1 else 1).tolist()))
    if not want:
        raise ValueError("no slot rates given")
    have = sorted(set(tags.tolist()))
    if not rate_matched:
        if len(want) > 1:
            raise ValueError(f"mixed per-slot rates {want} need rate-matched sampling (RateSimEnv's default or "
                             "sampler={rate: sampler}); one sampler could play any map at any slot's rate")
        extra = [t for t in have if t != want[0]]
        if extra:
            hint = ("a DT-tagged pool needs make_rate_env(rate_enabled=True, clock_rate=<its tag>)" if want[0] == 1.0
                    else "use a pool exported for this condition only, or rate-matched sampling")
            raise ValueError(f"pool maps tagged clock_rate {extra} would be played at rate {want[0]}: {hint}")
    missing = [r for r in want if r not in have]
    if missing:
        raise ValueError(f"pool has no map tagged clock_rate {missing} (pool tags {have}; a pool without m_clock_rate "
                         "is rate 1.0): export the pool for that condition (scripts/sim_data_export.py --condition)")
    return tags


def make_rate_env(pool, n=1, device="cpu", *, rate_enabled=False, clock_rate=1.5, control_hz=60,
                  allow_unvalidated_gpu=False, **kwargs):
    """Build the CPU rate reference; opt-out (or every slot at rate 1.0) returns an ordinary SimEnv.

    clock_rate: scalar or one rate per slot (mixed NM/DT1.25/DT in one batch). Explicit example:
    make_rate_env(pool, rate_enabled=True, clock_rate=1.5, control_hz=120). The pool's m_clock_rate tags must agree
    (assert_pool_rate): a DT-tagged pool is refused on the rate-one path. No original .osu file or pool array is
    rewritten. GPU devices/renderers are refused on the non-unit path unless allow_unvalidated_gpu=True, which passes
    device/renderer/pool_gpu to SimEnv unchanged (real-host and pixel parity NOT validated).
    """
    if rate_enabled:
        rates = slot_rates(clock_rate, n)
        RateClock(1.0, control_hz)                                   # validates control_hz
    if not rate_enabled or (rates == 1.0).all():
        if isinstance(kwargs.get("sampler"), dict):
            raise ValueError("a {rate: sampler} dict is for RateSimEnv; the rate-one SimEnv path takes one sampler")
        assert_pool_rate(pool, 1.0)
        return SimEnv(pool, n, device, control_hz=control_hz, **kwargs)
    return RateSimEnv(pool, n, device, clock_rate=rates, control_hz=control_hz,
                      allow_unvalidated_gpu=allow_unvalidated_gpu, **kwargs)


CURRICULUM_HOOKS = ("state", "full_state", "load")    # sampler methods SimEnv probes with hasattr (env.py)


class RateMatchedSampler:
    """Per-rate map samplers: a slot running rate r only draws maps tagged r. sample_slots draws the rate groups in
    ascending rate order with one call per group, so a single-rate env consumes the generator exactly as SimEnv.reset
    does with that sampler. report() is routed by the map's tag.

    Curriculum hooks (CURRICULUM_HOOKS; SimEnv.curriculum_state/load_curriculum/full_state/load_full_state call them
    only if hasattr): with exactly ONE wrapped sampler they are that sampler's own, so a single-rate env keeps e.g. a
    StarCurriculum frontier or StarMixture map priorities across logging and checkpoint resume exactly as SimEnv does.
    SEVERAL per-rate samplers of which any carries a hook are refused: aggregating per-rate curriculum state is a
    trainer integration step (RATE_DESIGN.md), and hiding the hooks would silently reset a resumed curriculum."""

    def __init__(self, map_rates, samplers):
        self.map_rates = np.asarray(map_rates, dtype=np.float64)
        self.samplers = dict(samplers)
        if len(self.samplers) > 1:
            stateful = sorted(r for r, s in self.samplers.items() if any(hasattr(s, h) for h in CURRICULUM_HOOKS))
            if stateful:
                raise ValueError(f"per-rate samplers for rate(s) {stateful} carry curriculum state "
                                 f"({'/'.join(CURRICULUM_HOOKS)}) that RateSimEnv cannot yet aggregate across "
                                 f"{len(self.samplers)} per-rate samplers (a resumed run would silently reset it): "
                                 "use stateless per-rate samplers, or one curriculum sampler on a single-rate env "
                                 "(RATE_DESIGN.md, integration steps)")

    def __getattr__(self, name):
        # Only reached for names not found normally. A sole wrapped sampler lends its curriculum hooks (absent hook:
        # AttributeError, so SimEnv's hasattr probes see exactly what they would see on that sampler).
        samplers = self.__dict__.get("samplers", {})
        if name in CURRICULUM_HOOKS and len(samplers) == 1:
            return getattr(next(iter(samplers.values())), name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def sample_slots(self, rng, rates):
        rates = np.asarray(rates, dtype=np.float64)
        out = np.empty(len(rates), dtype=np.int64)
        for r in sorted(set(rates.tolist())):
            sel = rates == r
            out[sel] = self.samplers[r].sample(rng, int(sel.sum()))
        return out

    def report(self, map_ids, stats):
        for mid in map_ids:
            self.samplers[float(self.map_rates[mid])].report([mid], stats)


class RateSimEnv(SimEnv):
    """Isolated CPU reference with one game rate per env slot (clock_rate scalar = every slot).

    Slot n advances e_dt[n] = (1000 / control_hz) * e_rate[n] gameplay ms per real decision; e_t/render state stay in
    gameplay ms, and each slot's e_cap is SimEnv's formula for its own e_dt. Rates are frozen per episode: set_rates()
    applies at each slot's next reset. Map sampling is rate-matched (sampler=None: uniform over the maps tagged with
    the slot's rate; or sampler={rate: sampler}; a single sampler only for a single-rate env whose pool has that one
    tag, and it keeps its curriculum state()/full_state()/load() hooks: RateMatchedSampler). Cursor physics stays in
    osu px per REAL decision, as in the base env: a DT slot needs `rate` times the per-decision travel for the same
    gameplay pattern; for equal physical speed at 120 Hz halve vmax_px.
    """

    def __init__(self, pool, n, device="cpu", *, clock: RateClock | None = None, clock_rate=None, control_hz=None,
                 sampler=None, allow_unvalidated_gpu=False, **kwargs):
        if clock is not None:
            if clock_rate is not None or control_hz is not None:
                raise ValueError("pass either clock=RateClock(...) or clock_rate/control_hz, not both")
            clock_rate, control_hz = clock.rate, clock.control_hz
        if clock_rate is None:
            raise ValueError("RateSimEnv needs an explicit clock_rate (scalar or one per slot) or clock=RateClock(...)")
        control_hz = 60 if control_hz is None else control_hz
        RateClock(1.0, control_hz)                                   # validates control_hz
        rates = slot_rates(clock_rate, n)
        gpu = str(device) != "cpu" or kwargs.get("renderer") is not None or kwargs.get("pool_gpu") is not None
        if gpu and not allow_unvalidated_gpu:
            raise ValueError("unvalidated rate path is CPU reference only; GPU/renderer use requires parity work "
                             "(allow_unvalidated_gpu=True passes them through: real-host parity NOT validated)")
        map_rates = pool_clock_rates(pool)
        if sampler is None:
            samplers = {r: UniformSampler(np.flatnonzero(map_rates == r)) for r in sorted(set(map_rates.tolist()))}
        elif isinstance(sampler, dict):
            samplers = {canonical_rate(r): s for r, s in sampler.items()}
        else:
            samplers = {float(rates[0]): sampler}
        matched = RateMatchedSampler(map_rates, samplers)          # refuses several stateful per-rate samplers
        self.rate_matched = sampler is None or isinstance(sampler, dict)
        self.control_hz = int(control_hz)
        self.real_step_ms = 1000.0 / control_hz
        self.map_rates = map_rates
        self._check_rates(pool, rates, samplers)
        self.e_rate = rates.copy()                          # rate of each slot's current episode
        self.e_rate_next = rates.copy()                     # set_rates: applied when the slot next resets
        self.e_dt = self.real_step_ms * self.e_rate         # gameplay ms per decision of each slot (kernel frame_dt)
        self._cap_slot = None
        super().__init__(pool, n, device, control_hz=control_hz, sampler=matched, **kwargs)

    def _check_rates(self, pool, rates, samplers):
        assert_pool_rate(pool, rates, rate_matched=self.rate_matched)
        missing = sorted(set(rates.tolist()) - set(samplers))
        if missing:
            raise ValueError(f"no sampler for slot rate(s) {missing}; sampler dict keys are {sorted(samplers)}")

    # ------------------------------------------------------------------ rates
    @property
    def dt(self):
        """SimEnv-compatible scalar gameplay ms per decision (Autopilot, scripts, SimEnv's own e_cap formula).
        Inside _reset_one(n) it is slot n's step, so SimEnv computes that slot's cap unchanged; otherwise it exists
        only while every slot runs one rate (a mixed-rate env raises: per-slot steps are e_dt)."""
        if self._cap_slot is not None:
            return float(self.e_dt[self._cap_slot])
        if not (self.e_dt == self.e_dt[0]).all():
            raise AttributeError("mixed-rate RateSimEnv has no scalar dt; e_dt[n] is slot n's gameplay ms per decision")
        return float(self.e_dt[0])

    @dt.setter
    def dt(self, value):
        # SimEnv.__init__ assigns its rate-one step 1000 / control_hz; gameplay steps derive from the slot rates
        if value != self.real_step_ms:
            raise AttributeError("RateSimEnv.dt is derived from the slot rates (e_dt); change rates with set_rates")

    @property
    def rate_clock(self):
        """RateClock of the current episodes while every slot runs one rate (mixed-rate env: slot_clock(n))."""
        if not (self.e_rate == self.e_rate[0]).all():
            raise AttributeError("mixed-rate RateSimEnv has no single rate_clock; use slot_clock(n)")
        return self.slot_clock(0)

    def slot_clock(self, n):
        return RateClock(float(self.e_rate[n]), self.control_hz)

    def set_rates(self, rates):
        """Per-slot rates (scalar or one per slot) for episodes that start from now on: slot n switches at its next
        reset (finish_episodes / reset), never mid-episode, like set_vmax. Validated before anything changes."""
        r = slot_rates(rates, self.n)
        self._check_rates(self.pool, r, self.sampler.samplers)
        self.e_rate_next[:] = r

    # ------------------------------------------------------------------ episodes
    def reset(self, idx):
        # Local fork of SimEnv.reset (SIMENV_OVERRIDE_EDITS["reset"]): slots starting now take their pending rate and
        # draw a map tagged with it.
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return
        self.e_rate[idx] = self.e_rate_next[idx]
        self.e_dt[idx] = self.real_step_ms * self.e_rate[idx]
        maps = self.sampler.sample_slots(self.rng, self.e_rate[idx])
        for n, mid in zip(idx, maps):
            self._reset_one(int(n), int(mid))

    def _reset_one(self, n, mid, start_index=None):
        # Refuses a map tagged with another rate (also guards direct calls, e.g. evaluate.py). SimEnv._reset_one reads
        # self.dt for e_cap; dt is slot n's own gameplay step while it runs.
        if self.map_rates[mid] != self.e_rate[n]:
            raise ValueError(f"slot {n} runs clock_rate {self.e_rate[n]} but map {mid} ({self.pool.keys[mid]}) is "
                             f"tagged {self.map_rates[mid]}")
        self._cap_slot = n
        try:
            super()._reset_one(n, mid, start_index)
        finally:
            self._cap_slot = None

    # ------------------------------------------------------------------ protocol
    def step_np(self, act, key):
        """act float [N,2] in [-1,1], key int [N] in 0..3 (numpy). Returns reward, done (numpy)."""
        # Local fork of SimEnv.step_np: only the kernel, its per-slot frame_dt (e_dt) and the appended per-slot rate
        # differ (SIMENV_OVERRIDE_EDITS["step_np"], AST-checked against env.py by tests/test_sim_rate.py).
        o, m, ne, p, st, sp = self.o, self.m, self.ne, self.p, self.st, self.sp
        act = np.clip(np.asarray(act, dtype=np.float64), -1.0, 1.0)
        key = np.asarray(key, dtype=np.int64)
        if self.phys_on:
            act_cmd = act
            act, bind = motion.physics_step(self.e_prev_act.astype(np.float64), act_cmd, self.e_vmax)
            ph_live = ~self.e_done
            ph_press = ph_live & (((key & ~self.e_keys) & 3) != 0)
        mm = self.motion_metrics
        if mm:
            cx0, cy0 = self.e_cx.astype(np.float64), self.e_cy.astype(np.float64)
            first0, done0 = self.e_first.copy(), self.e_done.copy()
            sl0, keys0 = self.e_sl_active != 0, self.e_keys.copy()
            motion.motion_gate(self.e_t, self.e_lo, self.e_win_lo, self.e_win_hi, self.e_map, m["preempt"], o["kind"],
                               o["start"], st["head_r"], st["tail_r"], self.last_gate)
        rate_rules.step_all(act[:, 0].copy(), act[:, 1].copy(), key, self.e_dt,
                            self.e_map, self.e_t, self.e_lo, self.e_hi, self.e_win_lo, self.e_win_hi, self.e_settle,
                            self.e_cx, self.e_cy, self.e_keys, self.e_press_t, self.e_release_t, self.e_done,
                            self.e_reward, self.e_stats, self.e_combo, self.key_down_cost,
                            m["radius"], m["preempt"], m["great"], m["ok"], m["meh"],
                            o["kind"], o["start"], o["end"], o["x"], o["y"], o["path_off"], o["path_cnt"],
                            o["distance"], o["span_count"], o["nest_off"], o["nest_cnt"], o["w_head"], o["w_tick"],
                            o["w_tail"], o["spins_req"], ne["type"], ne["time"], p["x"], p["y"], p["cum"],
                            st["head_r"], st["head_t"], st["trk"], st["trk_t"], st["nest_n"], st["nest_miss"],
                            st["nest_t"], st["tail_r"], st["end_t"], st["hit_action"], st["accept_any"],
                            st["last_pressed"], sp["has_pos"], sp["last_angle"], sp["total_acc"], sp["acc_last"],
                            sp["cur_max"], sp["completed"], sp["rot"], sp["trk"],
                            ORIGIN_X, ORIGIN_Y, CLIENT_SCALE, self.shaping_beta, self.shaping_gamma, self.e_phi,
                            self.spin_beta, self.e_phi_spin, self.vis, self.admit_margin,
                            self.break_penalty, self.tail_penalty, self.track_beta, self.e_phi_trk,
                            self.e_sl_active, self.spin_progress, self.spin_min_r, self.e_rate)
        self.e_prev_act[:] = act.astype(np.float32)
        self.e_steps += 1
        self.e_first[:] = False
        reward = self.e_reward.copy()
        if self.phys_on:
            b_live = bind & ph_live
            unreached = ph_press & bind
            self.e_bind_steps += b_live
            self.e_unreached += unreached
            gap = np.hypot((act_cmd[:, 0] - act[:, 0]) * motion.ACT_PX_X, (act_cmd[:, 1] - act[:, 1]) * motion.ACT_PX_Y)
            self._ph = dict(ph_live_n=float(ph_live.sum()), ph_bind_n=float(b_live.sum()),
                            ph_press_n=float(ph_press.sum()), ph_unreached_n=float(unreached.sum()),
                            ph_gap_px=float(np.where(b_live, gap, 0.0).sum()))
        if mm:
            self._motion_step(cx0, cy0, first0, done0, reward, sl0, keys0, key)
        timeout = self.e_steps >= self.e_cap
        done = self.e_done | timeout
        return reward, done, timeout

    def finish_episodes(self, done):
        idx = np.nonzero(done)[0]
        rates = self.e_rate[idx].copy()          # SimEnv.finish_episodes resets these slots (pending rates apply)
        start = len(self.episode_log)
        super().finish_episodes(done)
        for rate, record in zip(rates.tolist(), self.episode_log[start:], strict=True):
            record.update(RateClock(rate, self.control_hz).metadata())
            record["duration_real_ms"] = record["steps"] * self.real_step_ms


class MapRateSimEnv(RateSimEnv):
    """labelled (session 6) MAP-FIRST rates (FC7_NM_DT_PROMPT.md B1, the recommended design): ONE sampler over the whole
    pool (e.g. a StarMixture over a mixed NM + DT pool, DT rows on offset pseudo-stars) draws maps exactly as
    SimEnv.reset does, and each slot then plays its episode at the sampled map's m_clock_rate tag (frozen per episode).
    The sampler's state (map priorities, checkpoint format, prefix restore when DT rows are appended after the NM rows)
    is SimEnv's, unchanged. Everything else is RateSimEnv: per-slot gameplay step e_dt = (1000 / control_hz) * rate,
    rate_rules kernels, per-slot timeout caps, rate metadata on episode records. set_rates is refused (the rate is the
    map's). With an all-1.0 pool every slot runs rate one."""

    def __init__(self, pool, n, device="cpu", *, sampler=None, control_hz=60, allow_unvalidated_gpu=False, **kwargs):
        if isinstance(sampler, dict):
            raise ValueError("MapRateSimEnv takes ONE sampler over the whole pool (map-first rates), not {rate: sampler}")
        tags = pool_clock_rates(pool)
        self.mf_sampler = sampler or UniformSampler(np.arange(pool.n_maps))
        rates = sorted(set(tags.tolist()))
        # RateSimEnv's construction machinery with stateless per-tag samplers (never sampled from: reset() draws through
        # mf_sampler) and every tag allowed (rate-matched check); slot rates are replaced by each map's tag at reset
        super().__init__(pool, n, device, clock_rate=rates[0], control_hz=control_hz,
                         sampler={r: UniformSampler(np.flatnonzero(tags == r)) for r in rates},
                         allow_unvalidated_gpu=allow_unvalidated_gpu, **kwargs)
        self.sampler = self.mf_sampler     # SimEnv's report / curriculum_state / full_state / load hooks use it

    def set_rates(self, rates):
        raise ValueError("MapRateSimEnv plays each map at its m_clock_rate tag; per-slot rates cannot be set")

    def reset(self, idx):
        # = SimEnv.reset with the map-first sampler (the generator is consumed exactly as SimEnv.reset does)
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return
        maps = self.mf_sampler.sample(self.rng, len(idx))
        for n, mid in zip(idx, maps):
            self._reset_one(int(n), int(mid))

    def _reset_one(self, n, mid, start_index=None):
        # the slot adopts the map's tag (also for direct calls such as evaluate.py's), then RateSimEnv's guarded reset
        r = float(self.map_rates[mid])
        self.e_rate[n] = self.e_rate_next[n] = r
        self.e_dt[n] = self.real_step_ms * r
        super()._reset_one(n, mid, start_index)


def env_class_for_pool(pool):
    """labelled (session 6): SimEnv for a rate-one pool (legacy or all tags 1.0), MapRateSimEnv when any map is tagged
    with another rate (evaluation and analysis tools: each map plays at its tag)."""
    tags = pool_clock_rates(pool)
    return SimEnv if (tags == 1.0).all() else MapRateSimEnv
