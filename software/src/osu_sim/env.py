"""SimEnv: N parallel osu! episodes (numba rules on CPU, renderer on GPU) implementing the ppo.py env protocol."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import motion, rules
from .pool import MapPool

# native_default_v1 / sim_v1 playfield transform (client px = origin + scale * gamefield)
ORIGIN_X, ORIGIN_Y, CLIENT_SCALE = 256.0, 84.0, 1.5
NS = rules.N_STATS


def action_to_gamefield(ax, ay):
    cxp = 160.0 + np.round((np.asarray(ax, np.float64) + 1.0) * 959.0 / 2.0)
    cyp = np.round((np.asarray(ay, np.float64) + 1.0) * 719.0 / 2.0)
    return ((cxp - ORIGIN_X) / CLIENT_SCALE).astype(np.float32), ((cyp - ORIGIN_Y) / CLIENT_SCALE).astype(np.float32)


@dataclass
class ClipConfig:
    min_objects: int = 12
    max_objects: int = 48
    lead_ms: tuple = (200.0, 1200.0)     # extra pre-roll before the first object's fade-in
    full_map: bool = False               # evaluation: whole map from time 0 (like the real host)
    random_initial_cursor: bool = True
    prioritized: float = 0.0             # share of clips whose start bucket is sampled by recent failure rate
    bucket: int = 16                     # objects per prioritisation bucket


class UniformSampler:
    """Uniform over a fixed list of map indices (curriculum samplers live in curriculum.py)."""

    def __init__(self, maps):
        self.maps = np.asarray(maps, dtype=np.int64)

    def sample(self, rng, k):
        return self.maps[rng.integers(0, len(self.maps), size=k)]

    def report(self, map_ids, stats):
        pass


# labelled (session 9, spec v3 App. D g1-5) exposure counters: pool maps["source"] codes (make_env: 1 = generated)
EXPOSURE_SOURCES = {0: "human", 1: "procedural", 2: "drill"}


class ExposureCounters:
    """labelled (session 9, spec v3 App. D g1-5; spec 7.2 / 8.3 "Exposure"): cumulative realized exposure of a training
    env, by map source (pool maps["source"]: 0 human, 1 procedural / generated, 2 drill; other codes "source_<k>"; a
    pool without the array is all human, as make_env's generated flag treats it) and by mixture range (the sampler's
    StarMixture range of each map, first match as StarMixture._range_of; samplers without ranges: "none").

    decisions = every env step of every env slot, credited to the map that slot played in that step, so the decision
    total equals the trainer's decisions since counting began (origin_decisions); clips = finished episodes; notes =
    objects of each finished clip (e_hi - e_lo). Counting only: no RNG, no effect on sampling, rewards or env state.
    state() / load() carry the counts across checkpoint resume; labels a resumed config no longer has keep their saved
    counts (reported, never incremented), so cumulative totals stay continuous."""

    FIELDS = ("decisions", "clips", "notes")

    def __init__(self, source, sampler=None):
        src = np.asarray(source).astype(np.int64)
        codes = sorted(set(src.tolist()) | set(EXPOSURE_SOURCES))
        self.src_labels = [EXPOSURE_SOURCES.get(c, f"source_{c}") for c in codes]
        self.map_src = np.searchsorted(np.asarray(codes, dtype=np.int64), src).astype(np.int64)
        ranges = list(getattr(sampler, "ranges", None) or []) if hasattr(sampler, "d") else []
        self.has_ranges = bool(ranges)
        self.map_rng = np.full(len(src), len(ranges), dtype=np.int64)          # index len(ranges) = "none"
        if ranges:
            d = np.asarray(sampler.d, dtype=np.float64)
            for i in range(len(ranges) - 1, -1, -1):        # descending, so the FIRST matching range wins
                lo, hi = float(ranges[i][0]), float(ranges[i][1])
                self.map_rng[(d >= lo) & (d < hi)] = i
        labels = []
        for i, r in enumerate(ranges):
            lab = f"{float(r[0]):g}-{float(r[1]):g}"
            labels.append(lab if lab not in labels else f"{lab}#{i}")
        self.rng_labels = labels + ["none"]
        self.src = np.zeros((3, len(self.src_labels)), dtype=np.int64)       # rows: FIELDS
        self.rng = np.zeros((3, len(self.rng_labels)), dtype=np.int64)
        self.carried = {"by_source": {}, "by_range": {}}
        self.origin_decisions = None      # trainer decisions when counting began (set by train.py or restored)

    def count_step(self, e_map, done, e_lo, e_hi):
        """One env step of every slot: e_map = the maps just played (before finished slots are reset), done = [N] bool
        finished-episode mask, e_lo / e_hi = the finished clips' object range."""
        s = self.map_src[e_map]
        r = self.map_rng[e_map]
        self.src[0] += np.bincount(s, minlength=self.src.shape[1])
        self.rng[0] += np.bincount(r, minlength=self.rng.shape[1])
        idx = np.flatnonzero(done)
        if len(idx):
            notes = (np.asarray(e_hi)[idx] - np.asarray(e_lo)[idx]).astype(np.int64)
            np.add.at(self.src[1], s[idx], 1)
            np.add.at(self.src[2], s[idx], notes)
            np.add.at(self.rng[1], r[idx], 1)
            np.add.at(self.rng[2], r[idx], notes)

    def _table(self, counts, labels, carried, keep):
        out = {}
        for j, lab in enumerate(labels):
            if keep(lab, counts[:, j]):
                out[lab] = {f: int(counts[i, j]) for i, f in enumerate(self.FIELDS)}
        for lab, v in carried.items():
            out[lab] = {f: int(v.get(f, 0)) for f in self.FIELDS}
        return out

    def record(self):
        """updates.jsonl rec["exposure"]: cumulative {"by_source": {label: {decisions, clips, notes}}, "by_range": {...},
        "origin_decisions": ...}. by_range lists "none" only for a sampler without ranges or when it was credited."""
        return {"by_source": self._table(self.src, self.src_labels, self.carried["by_source"], lambda lab, c: True),
                "by_range": self._table(self.rng, self.rng_labels, self.carried["by_range"],
                                        lambda lab, c: lab != "none" or not self.has_ranges or bool(c.any())),
                "origin_decisions": self.origin_decisions}

    def state(self):
        st = self.record()
        st["by_range"] = self._table(self.rng, self.rng_labels, self.carried["by_range"], lambda lab, c: True)
        return st

    def load(self, st):
        for key, labels, counts in (("by_source", self.src_labels, self.src), ("by_range", self.rng_labels, self.rng)):
            for lab, v in (st.get(key) or {}).items():
                if lab in labels:
                    j = labels.index(lab)
                    for i, f in enumerate(self.FIELDS):
                        counts[i, j] = int(v.get(f, 0))
                else:
                    self.carried[key][lab] = {f: int(v.get(f, 0)) for f in self.FIELDS}
        if st.get("origin_decisions") is not None:
            self.origin_decisions = int(st["origin_decisions"])


class SimEnv:
    def __init__(self, pool: MapPool, n: int, device="cuda", seed=0, *, sampler=None, clip: ClipConfig = None,
                 omax: int = 256, control_hz: int = 60, key_down_cost: float = 0.0, renderer=None,
                 pool_gpu=None, shaping_beta: float = 0.0, shaping_gamma: float = 0.99, spin_beta: float = 0.0,
                 break_penalty: float = 0.0, tail_penalty: float = 0.0, track_beta: float = 0.0,
                 spin_progress: float = 0.0, spin_min_r: float = 0.0, aug_prob: float = 0.0, aug_sigma_max: float = 0.0, aug_bank: int = 256,
                 motion_cost: dict | None = None, motion_metrics: bool = True, cursor_physics: dict | None = None):
        self.pool, self.n, self.device = pool, n, torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.clip = clip or ClipConfig()
        self.sampler = sampler or UniformSampler(np.arange(pool.n_maps))
        self.omax = omax if not self.clip.full_map else int(pool.maps["obj_cnt"].max())
        self.dt = 1000.0 / control_hz
        self.key_down_cost = float(key_down_cost)
        self.shaping_beta, self.shaping_gamma = float(shaping_beta), float(shaping_gamma)
        self.spin_beta = float(spin_beta)
        self.break_penalty, self.tail_penalty = float(break_penalty), float(tail_penalty)
        self.track_beta = float(track_beta)
        # labelled (session 3): non-telescoping capped spinner progress reward coefficient (rules._update); 0 = off
        self.spin_progress = float(spin_progress)
        self.spin_min_r = float(spin_min_r)
        # labelled (session 3) training-only frame augmentation: with probability aug_prob an episode gets pixel noise of
        # std U(0, aug_sigma_max) (uint8 levels).
        # Noise comes from a fixed bank indexed per step (stored in the render state), so PPO's re-render of a stored
        # step reproduces exactly the frame the action was sampled on.
        self.aug_prob, self.aug_sigma_max, self.aug_bank_n = float(aug_prob), float(aug_sigma_max), int(aug_bank)
        self.e_aug_sigma = np.zeros(n, dtype=np.float32)
        self._aug_bank = None
        # labelled (session 4) MOTION-COST v1 (objective change, spec 9.2 action-cost class; DESIGN.md section 2 in
        # runs/development/session4/cursor/workflow; osu_sim.motion). reward -= s * w(gate) * |executed cursor step| in osu
        # px, w = idle_per_px (nothing visible left to aim at) / aim_per_px (otherwise) / 0 (spinner in progress, first
        # step of an episode). Absent or both prices 0 -> the reward path is byte-identical to the pre-session-4 code.
        mc = dict(motion_cost or {})
        unknown = {k for k in mc if not k.startswith("_")} - {"idle_per_px", "aim_per_px", "ramp_start",
                                                              "ramp_decisions", "exempt_spinners", "press_per_px",
                                                              "rev_per_px", "rev_deadzone_px", "v2_ramp_start",
                                                              "v2_ramp_decisions"}
        if unknown:
            raise ValueError(f"unknown motion_cost keys {sorted(unknown)}")
        self.mc_idle = float(mc.get("idle_per_px", 0.0))
        self.mc_aim = float(mc.get("aim_per_px", 0.0))
        self.mc_ramp_start = float(mc.get("ramp_start", 0.0))
        self.mc_ramp_decisions = float(mc.get("ramp_decisions", 0.0))
        if not bool(mc.get("exempt_spinners", True)):
            raise ValueError("motion_cost.exempt_spinners=false is not implemented (DESIGN.md 2.3 item 6)")
        if self.mc_idle < 0 or self.mc_aim < 0:
            raise ValueError("motion_cost prices must be >= 0 (the cost is never a reward)")
        self.mc_on = self.mc_idle != 0.0 or self.mc_aim != 0.0
        # labelled (session 4) MOTION-COST v2 (runs/development/session4/cursor/v2/DESIGN_v2.md; spec 9.2 action-cost
        # class): reward -= s2 * (press_per_px * SP + rev_per_px * REV2), osu px. SP = d_k + d_{k-1} on a step with a newly
        # pressed key (STEADY-PRESS); REV2 = max(0, kappa - rev_deadzone_px), kappa = cancelled motion of the step pair
        # (k-1, k). A step counts only if live (not first, not done), gate != SPIN and no slider was in progress before
        # it; d_{k-1} counts if step k-1 was live and gate != SPIN. s2: own ramp (v2_ramp_start / v2_ramp_decisions),
        # frozen per episode, shared by both terms. Both prices 0 or absent -> rewards bitwise those of the v1 code.
        self.mc_press = float(mc.get("press_per_px", 0.0))
        self.mc_rev = float(mc.get("rev_per_px", 0.0))
        self.mc_rev_dz = float(mc.get("rev_deadzone_px", motion.REV_DEADZONE_PX))
        self.v2_ramp_start = float(mc.get("v2_ramp_start", 0.0))
        self.v2_ramp_decisions = float(mc.get("v2_ramp_decisions", 0.0))
        if self.mc_press < 0 or self.mc_rev < 0 or self.mc_rev_dz < 0:
            raise ValueError("motion_cost press_per_px, rev_per_px, rev_deadzone_px must be >= 0 (never a reward)")
        # DESIGN_v2.md 5.1: negative v2 ramp values are rejected (a negative v2_ramp_decisions would silently switch the
        # ramp off, i.e. full price from the branch point)
        if not (self.v2_ramp_start >= 0 and self.v2_ramp_decisions >= 0):
            raise ValueError("motion_cost v2_ramp_start, v2_ramp_decisions must be >= 0")
        self.mc_v2 = self.mc_press != 0.0 or self.mc_rev != 0.0
        # gate + displacement logging (never changes rewards); forced on when a price is non-zero
        self.motion_metrics = bool(motion_metrics) or self.mc_on or self.mc_v2
        self.decisions = 0.0          # trainer decision count, set by train.py through set_decisions (ramp clock)
        # labelled (session 5) CURSOR PHYSICS (osu_sim.motion.physics_step; CURSOR_PHYSICS_PROMPT.md): the executed cursor
        # moves toward the command by at most vmax osu px per step, applied before rules.step_all, so judgements, the
        # rendered cursor, the motion cost and the cursor metrics all see the executed position; aux[0:2] carries the
        # previous EXECUTED position (equal to the previous command whenever the cap did not bind). vmax is frozen per
        # episode (set_vmax changes it for episodes that start later). Absent / None = off: byte-identical old behaviour.
        cp = dict(cursor_physics or {})
        unknown = {k for k in cp if not k.startswith("_")} - {"vmax_px"}
        if unknown:
            raise ValueError(f"unknown cursor_physics keys {sorted(unknown)}")
        self.phys_on = cursor_physics is not None
        self.vmax = float(cp.get("vmax_px", motion.VMAX_OFF))
        if not self.vmax > 0:
            raise ValueError("cursor_physics.vmax_px must be > 0 (inf = no cap)")
        self.e_vmax = np.full(n, self.vmax, dtype=np.float64)
        self._ph = {}
        o, m, ne, p = pool.objs, pool.maps, pool.nest, pool.path
        c = np.ascontiguousarray
        self.m = dict(radius=c(m["radius"], np.float32), preempt=c(m["preempt"], np.float64),
                      great=c(m["great"], np.float64), ok=c(m["ok"], np.float64), meh=c(m["meh"], np.float64),
                      obj_off=m["obj_off"].astype(np.int64), obj_cnt=m["obj_cnt"].astype(np.int64))
        self.o = dict(kind=c(o["kind"], np.int64), start=c(o["start"], np.float64), end=c(o["end"], np.float64),
                      x=c(o["x"], np.float32), y=c(o["y"], np.float32), path_off=c(o["path_off"], np.int64),
                      path_cnt=c(o["path_cnt"], np.int64), distance=c(o["distance"], np.float64),
                      span_count=c(o["span_count"], np.int64), nest_off=c(o["nest_off"], np.int64),
                      nest_cnt=c(o["nest_cnt"], np.int64), w_head=c(o["w_head"], np.float64),
                      w_tick=c(o["w_tick"], np.float64), w_tail=c(o["w_tail"], np.float64),
                      spins_req=c(o["spins_req"], np.int64))
        self.ne = dict(type=c(ne["type"], np.int64), time=c(ne["time"], np.float64))
        self.p = dict(x=c(p["x"], np.float32), y=c(p["y"], np.float32), cum=c(p["cum"], np.float64))
        N, O = n, self.omax
        z = lambda *s, dt=np.float64, v=0: np.full(s, v, dtype=dt)
        # env scalars
        self.e_map, self.e_t = z(N, dt=np.int64), z(N)
        self.e_lo, self.e_hi = z(N, dt=np.int64), z(N, dt=np.int64)
        self.e_win_lo, self.e_win_hi = z(N, dt=np.int64), z(N, dt=np.int64)
        self.e_settle, self.e_t0 = z(N), z(N)
        self.e_cx, self.e_cy = z(N, dt=np.float32), z(N, dt=np.float32)
        self.e_keys = z(N, dt=np.int64)
        self.e_press_t, self.e_release_t = z(N, v=-1e9), z(N, v=-1e9)
        self.e_done = np.zeros(N, dtype=np.bool_)
        self.e_reward = z(N)
        self.e_stats = z(N, NS)
        self.e_combo = z(N, 2, dt=np.int64)
        self.e_steps = z(N, dt=np.int64)
        self.e_cap = z(N, dt=np.int64)
        self.e_prev_act = z(N, 2, dt=np.float32)
        self.e_first = np.ones(N, dtype=np.bool_)
        self.e_phi = z(N)
        self.e_phi_spin = z(N)
        self.e_phi_trk = z(N)
        self.e_sl_active = np.zeros(N, dtype=np.int8)
        # labelled (session 4) motion cost: per-episode frozen ramp scale, last pre-step gate, per-episode accumulators
        # over charged-eligible steps (not first, not done, gate != SPIN): moved px, cost, idle px / steps / still
        # steps, aim px. Per-step values of the last step: last_move_px, last_move_cost.
        self.e_move_s = z(N)
        self.last_gate = np.full(N, motion.GATE_IDLE, dtype=np.int8)
        self.last_move_px = z(N)
        self.last_move_cost = z(N)
        self.e_move_px, self.e_move_cost, self.e_idle_px = z(N), z(N), z(N)
        self.e_idle_steps, self.e_idle_still, self.e_aim_px = z(N), z(N), z(N)
        # labelled (session 4) MOTION-COST v2 state: frozen ramp scale; previous step's displacement (float64 osu px),
        # whether it was a REV2 leg (live, non-SPIN, no slider in progress) and whether it was live and non-SPIN (its d
        # counts in SP); per-episode REV2 px, SP px, eligible new presses, presses with SP >= STEADY_PX, v2 cost
        self.e_v2_s = z(N)
        self.e_prev_dx, self.e_prev_dy, self.e_prev_d = z(N), z(N), z(N)
        self.e_prev_leg = np.zeros(N, dtype=np.bool_)
        self.e_prev_ok = np.zeros(N, dtype=np.bool_)
        self.e_rev_px, self.e_sp_px, self.e_press_n, self.e_press_moving, self.e_v2_cost = z(N), z(N), z(N), z(N), z(N)
        # labelled (session 5): AIM price frozen per episode (the in-run escalation / back-off changes it for episodes that
        # start later); cursor-physics per-episode counters (steps where the cap bound, presses fired while it bound)
        self.e_mc_aim = np.full(N, self.mc_aim, dtype=np.float64)
        self.e_bind_steps, self.e_unreached = z(N), z(N)
        self._mv = {}
        # per-object state
        self.st = dict(head_r=z(N, O, dt=np.int64), head_t=z(N, O, v=-1e9), trk=z(N, O, dt=np.int64),
                       trk_t=z(N, O, v=-1e9), nest_n=z(N, O, dt=np.int64), nest_miss=z(N, O, dt=np.uint64),
                       nest_t=z(N, O, v=-1e9), tail_r=z(N, O, dt=np.int64), end_t=z(N, O, v=-1e9),
                       hit_action=z(N, O, dt=np.int64), accept_any=z(N, O, v=np.nan),
                       last_pressed=z(N, O, dt=np.int64))
        self.sp = dict(has_pos=z(N, O, dt=np.int64), last_angle=z(N, O, dt=np.float32, v=np.nan),
                       total_acc=z(N, O, dt=np.float32), acc_last=z(N, O, dt=np.float32),
                       cur_max=z(N, O, dt=np.float32), completed=z(N, O, dt=np.int64), rot=z(N, O, dt=np.float32),
                       trk=z(N, O, dt=np.int64))
        self.vis = z(N, O, 11)
        self.vis[:, :, 1:3] = -1e9
        self.vis[:, :, 6:9] = -1e9
        self.vis[:, :, 10] = -1e9
        self.admit_margin = 800.0
        self.episode_log = []    # finished-episode records (drained by caller)
        self.exposure = None     # labelled (session 9, spec v3 App. D g1-5): ExposureCounters once enabled (make_env)
        # prioritized clip starts (PLR-style): failure-rate EMA per (map, 16-object bucket); unseen buckets = 1
        B = self.clip.bucket
        nb = (self.m["obj_cnt"] + B - 1) // B
        self.bucket_off = np.concatenate(([0], np.cumsum(nb)[:-1])).astype(np.int64)
        self.bucket_fail = np.ones(int(nb.sum()), dtype=np.float64)
        self.e_i0 = np.zeros(N, dtype=np.int64)
        self.renderer = renderer
        self.pool_gpu = pool_gpu
        K = rules.K_SLOTS
        self.K = K
        self.r = dict(gid=np.zeros((N, K), np.int32), head_r=np.zeros((N, K), np.int64), head_t=np.zeros((N, K)),
                      trk=np.zeros((N, K), np.int64), trk_t=np.zeros((N, K)), nest_n=np.zeros((N, K), np.int64),
                      nest_miss=np.zeros((N, K), np.uint64), nest_t=np.zeros((N, K)), tail_r=np.zeros((N, K), np.int64),
                      end_t=np.zeros((N, K)), spin_rot=np.zeros((N, K), np.float32),
                      spin_total=np.zeros((N, K), np.float32), spin_trk=np.zeros((N, K), np.int64),
                      trk_tp=np.zeros((N, K)), spin_ti=np.zeros((N, K)), spin_ts=np.zeros((N, K)),
                      spin_tc=np.zeros((N, K)), spin_last=np.zeros((N, K)))
        self.reset(np.arange(N))

    # ------------------------------------------------------------------ episodes
    def reset(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return
        maps = self.sampler.sample(self.rng, len(idx))
        for n, mid in zip(idx, maps):
            self._reset_one(int(n), int(mid))

    def _reset_one(self, n, mid, start_index=None):
        cnt = int(self.m["obj_cnt"][mid])
        off = int(self.m["obj_off"][mid])
        preempt = float(self.m["preempt"][mid])
        if self.clip.full_map:
            i0, i1 = 0, cnt
            t0 = 0.0
        else:
            length = int(self.rng.integers(self.clip.min_objects, self.clip.max_objects + 1))
            length = min(length, cnt, self.omax)
            if start_index is not None:
                i0 = start_index
            elif self.clip.prioritized > 0 and self.rng.random() < self.clip.prioritized and cnt > length:
                B = self.clip.bucket
                nb = (cnt + B - 1) // B
                w = self.bucket_fail[self.bucket_off[mid]:self.bucket_off[mid] + nb] + 0.02
                b = int(self.rng.choice(nb, p=w / w.sum()))
                i0 = int(min(max(b * B + self.rng.integers(-B // 2, B // 2 + 1), 0), cnt - length))
            else:
                i0 = int(self.rng.integers(0, cnt - length + 1))
            i1 = i0 + length
            first = self.o["start"][off + i0]
            t0 = first - preempt - float(self.rng.uniform(*self.clip.lead_ms))
        lo, hi = off + i0, off + i1
        kinds = self.o["kind"][lo:hi]
        ends = self.o["end"][lo:hi] + np.where(kinds == 0, 400.0, 0.0)
        self.e_map[n] = mid
        self.e_i0[n] = 0 if self.clip.full_map else i0
        self.e_t[n] = t0
        self.e_t0[n] = t0
        self.e_lo[n], self.e_hi[n] = lo, hi
        self.e_win_lo[n] = 0
        # objects already alive at t0
        w = 0
        while w < hi - lo and self.o["start"][lo + w] - preempt <= t0:
            w += 1
        self.e_win_hi[n] = w
        self.e_settle[n] = float(ends.max()) + 250.0
        self.e_cap[n] = int(math.ceil((self.e_settle[n] - t0) / self.dt)) + 30
        if self.clip.random_initial_cursor:
            a = self.rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
        else:
            a = np.zeros(2, np.float32)
        gx, gy = action_to_gamefield(a[0], a[1])
        self.e_cx[n], self.e_cy[n] = gx, gy
        self.e_prev_act[n] = a
        self.e_keys[n] = 0
        self.e_press_t[n] = self.e_release_t[n] = -1e9
        self.e_done[n] = False
        self.e_stats[n] = 0
        self.e_combo[n] = 0
        self.e_steps[n] = 0
        self.e_first[n] = True
        self.e_phi[n] = 0.0
        self.e_phi_spin[n] = 0.0
        self.e_phi_trk[n] = 0.0
        self.e_sl_active[n] = 0
        # labelled (session 4) motion cost: ramp scale frozen for the whole episode; accumulators restart
        self.e_move_s[n] = self.motion_scale()
        self.e_move_px[n] = self.e_move_cost[n] = self.e_idle_px[n] = 0.0
        self.e_idle_steps[n] = self.e_idle_still[n] = self.e_aim_px[n] = 0.0
        self.e_v2_s[n] = self.v2_scale()
        self.e_prev_dx[n] = self.e_prev_dy[n] = self.e_prev_d[n] = 0.0
        self.e_prev_leg[n] = self.e_prev_ok[n] = False
        self.e_rev_px[n] = self.e_sp_px[n] = self.e_press_n[n] = self.e_press_moving[n] = self.e_v2_cost[n] = 0.0
        self.e_mc_aim[n] = self.mc_aim                     # labelled (session 5): per-episode AIM price
        self.e_vmax[n] = self.vmax                         # labelled (session 5): per-episode cursor cap
        self.e_bind_steps[n] = self.e_unreached[n] = 0.0
        if self.aug_prob > 0:
            self.e_aug_sigma[n] = (self.rng.uniform(0.0, self.aug_sigma_max) if self.rng.random() < self.aug_prob
                                   else 0.0)
        for k, v in self.st.items():
            v[n] = {"head_t": -1e9, "trk_t": -1e9, "nest_t": -1e9, "end_t": -1e9, "accept_any": np.nan}.get(k, 0)
        for k, v in self.sp.items():
            v[n] = np.nan if k == "last_angle" else 0
        self.vis[n] = 0.0
        self.vis[n, :, 1:3] = -1e9
        self.vis[n, :, 6:9] = -1e9
        self.vis[n, :, 10] = -1e9

    # ------------------------------------------------------------------ motion cost (labelled (session 4))
    def motion_scale(self):
        """Ramp scale s for an episode that starts now (DESIGN.md 2.1); 1 when ramp_decisions == 0."""
        return motion.ramp_scale(self.decisions, self.mc_ramp_start, self.mc_ramp_decisions)

    def v2_scale(self):
        """labelled (session 4) MOTION-COST v2 ramp scale for an episode that starts now (own clock; 1 without a ramp)."""
        return motion.ramp_scale(self.decisions, self.v2_ramp_start, self.v2_ramp_decisions)

    def set_decisions(self, d, refresh=False):
        """Store the trainer's decision count (ramp clock). refresh=True re-freezes every live episode's scale: needed
        once after a checkpoint load, because __init__ resets every env before train.py knows the decision count."""
        self.decisions = float(d)
        if refresh:
            self.e_move_s[:] = self.motion_scale()
            self.e_v2_s[:] = self.v2_scale()
            self.e_vmax[:] = self.vmax                     # labelled (session 5)
            self.e_mc_aim[:] = self.mc_aim

    # ------------------------------------------------------------------ cursor physics (labelled (session 5))
    def set_vmax(self, vmax):
        """Cursor cap (osu px per step) for episodes that start from now on (frozen per episode)."""
        if not self.phys_on:
            raise ValueError("set_vmax needs an env built with cursor_physics")
        v = float(vmax)
        if not v > 0:
            raise ValueError("vmax must be > 0")
        self.vmax = v

    def vmax_np(self):
        """[N] float32 per-env cap of the current episodes (inf = off); the policy's key head needs it to place its
        click at the executed position (osu_sim.motion.physics_step_torch)."""
        return self.e_vmax.astype(np.float32)

    # ------------------------------------------------------------------ protocol
    def step_np(self, act, key):
        """act float [N,2] in [-1,1], key int [N] in 0..3 (numpy). Returns reward, done (numpy)."""
        o, m, ne, p, st, sp = self.o, self.m, self.ne, self.p, self.st, self.sp
        act = np.clip(np.asarray(act, dtype=np.float64), -1.0, 1.0)
        key = np.asarray(key, dtype=np.int64)
        if self.phys_on:
            # labelled (session 5) CURSOR PHYSICS: executed = filter(previous executed, command); act is replaced by the
            # executed position from here on (rules, aux and every metric use it). Bitwise the command where unbound.
            act_cmd = act
            act, bind = motion.physics_step(self.e_prev_act.astype(np.float64), act_cmd, self.e_vmax)
            ph_live = ~self.e_done
            ph_press = ph_live & (((key & ~self.e_keys) & 3) != 0)
        mm = self.motion_metrics
        if mm:
            # labelled (session 4): gate from the state the policy just observed, before the kernel advances it
            cx0, cy0 = self.e_cx.astype(np.float64), self.e_cy.astype(np.float64)
            first0, done0 = self.e_first.copy(), self.e_done.copy()
            # labelled (session 4) MOTION-COST v2: slider in progress and keys held before this step
            sl0, keys0 = self.e_sl_active != 0, self.e_keys.copy()
            motion.motion_gate(self.e_t, self.e_lo, self.e_win_lo, self.e_win_hi, self.e_map, m["preempt"], o["kind"],
                               o["start"], st["head_r"], st["tail_r"], self.last_gate)
        rules.step_all(act[:, 0].copy(), act[:, 1].copy(), key, self.dt,
                       self.e_map, self.e_t, self.e_lo, self.e_hi, self.e_win_lo, self.e_win_hi, self.e_settle,
                       self.e_cx, self.e_cy, self.e_keys, self.e_press_t, self.e_release_t, self.e_done,
                       self.e_reward, self.e_stats, self.e_combo, self.key_down_cost,
                       m["radius"], m["preempt"], m["great"], m["ok"], m["meh"],
                       o["kind"], o["start"], o["end"], o["x"], o["y"], o["path_off"], o["path_cnt"], o["distance"],
                       o["span_count"], o["nest_off"], o["nest_cnt"], o["w_head"], o["w_tick"], o["w_tail"],
                       o["spins_req"], ne["type"], ne["time"], p["x"], p["y"], p["cum"],
                       st["head_r"], st["head_t"], st["trk"], st["trk_t"], st["nest_n"], st["nest_miss"],
                       st["nest_t"], st["tail_r"], st["end_t"], st["hit_action"], st["accept_any"],
                       st["last_pressed"], sp["has_pos"], sp["last_angle"], sp["total_acc"], sp["acc_last"],
                       sp["cur_max"], sp["completed"], sp["rot"], sp["trk"],
                       ORIGIN_X, ORIGIN_Y, CLIENT_SCALE, self.shaping_beta, self.shaping_gamma, self.e_phi,
                       self.spin_beta, self.e_phi_spin, self.vis, self.admit_margin,
                       self.break_penalty, self.tail_penalty, self.track_beta, self.e_phi_trk,
                       self.e_sl_active, self.spin_progress, self.spin_min_r)
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

    def _motion_step(self, cx0, cy0, first0, done0, reward, sl0, keys0, key):
        """labelled (session 4): executed-cursor displacement (osu px) and motion cost of the step just taken. Subtracts
        the cost from `reward` in place (the PPO reward only; e_reward / e_stats[:, ST_REWARD] stay the game objective)
        when a price is non-zero; accumulates per-episode stats and the per-step sums returned by _step_info.
        Cost = v1 path cost (idle / aim prices) + v2 STEADY-PRESS + REV2 (DESIGN_v2.md)."""
        # float64 before subtracting (a float32 difference would carry ~1e-5 px rounding)
        dx = self.e_cx.astype(np.float64) - cx0
        dy = self.e_cy.astype(np.float64) - cy0
        d = np.hypot(dx, dy)
        gate = self.last_gate
        live = ~(first0 | done0)
        idle = live & (gate == motion.GATE_IDLE)
        aim = live & (gate == motion.GATE_AIM)
        spin = live & (gate == motion.GATE_SPIN)
        if self.mc_on:
            # labelled (session 5): the AIM price is the per-episode frozen e_mc_aim (equal to mc_aim unless changed)
            cost = motion.move_cost(d, gate, self.e_move_s, self.mc_idle, self.e_mc_aim, first0, done0)
        else:
            cost = np.zeros(self.n)
        # labelled (session 4) MOTION-COST v2. ok = live, non-spinner step; leg = ok and no slider in progress before it
        # (following the ball, its repeats and the departure right after the tail are never REV2-charged; a press while
        # a slider is in progress is not SP-charged). REV2 on the second leg of each consecutive leg pair; SP on a leg
        # step with a newly pressed key (rules.py: a bit down now and up before), d_{k-1} counted iff step k-1 was ok.
        ok = live & (gate != motion.GATE_SPIN)
        leg = ok & ~sl0
        pair = leg & self.e_prev_leg
        rev_px = np.where(pair, motion.reversal_excess(self.e_prev_dx, self.e_prev_dy, dx, dy, self.mc_rev_dz), 0.0)
        newpress = leg & (((np.asarray(key, np.int64) & ~keys0) & 3) != 0)
        sp_px = np.where(newpress, d + np.where(self.e_prev_ok, self.e_prev_d, 0.0), 0.0)
        if self.mc_v2:
            cost2 = self.e_v2_s * (self.mc_press * sp_px + self.mc_rev * rev_px)
            cost = cost + cost2
        else:
            cost2 = np.zeros(self.n)
        if self.mc_on or self.mc_v2:
            reward -= cost
        moving = newpress & (sp_px >= motion.STEADY_PX)
        self.e_prev_dx[:] = np.where(leg, dx, 0.0)
        self.e_prev_dy[:] = np.where(leg, dy, 0.0)
        self.e_prev_leg[:] = leg
        self.e_prev_d[:] = np.where(ok, d, 0.0)
        self.e_prev_ok[:] = ok
        self.e_rev_px += rev_px
        self.e_sp_px += sp_px
        self.e_press_n += newpress
        self.e_press_moving += moving
        self.e_v2_cost += cost2
        self.last_move_px[:] = d
        self.last_move_cost[:] = cost
        mv = idle | aim
        dm = np.where(mv, d, 0.0)
        di = np.where(idle, d, 0.0)
        da = np.where(aim, d, 0.0)
        still = idle & (d < motion.STILL_PX)
        self.e_move_px += dm
        self.e_move_cost += cost
        self.e_idle_px += di
        self.e_idle_steps += idle
        self.e_idle_still += still
        self.e_aim_px += da
        self._mv = dict(mc_cost=float(cost.sum()), mc_cost_sq=float((cost * cost).sum()), mc_px=float(dm.sum()),
                        mc_idle_px=float(di.sum()), mc_idle_n=float(idle.sum()), mc_idle_still=float(still.sum()),
                        mc_aim_px=float(da.sum()), mc_aim_n=float(aim.sum()), mc_spin_n=float(spin.sum()),
                        mc_charged_n=float(mv.sum()), mc_rev_px=float(rev_px.sum()), mc_rev_n=float(pair.sum()),
                        mc_rev_hit_n=float((rev_px > 0.0).sum()), mc_sp_px=float(sp_px.sum()),
                        mc_press_n=float(newpress.sum()), mc_press_moving=float(moving.sum()),
                        mc_v2_cost=float(cost2.sum()))

    def finish_episodes(self, done):
        idx = np.nonzero(done)[0]
        for n in idx:
            mid = int(self.e_map[n])
            nobj = int(self.e_hi[n] - self.e_lo[n])
            max_combo_possible = 0
            for g in range(self.e_lo[n], self.e_hi[n]):
                max_combo_possible += 1 + (int(self.o["nest_cnt"][g]) if self.o["kind"][g] == 1 else 0)
            s = self.e_stats[n]
            rec = dict(map=mid, objects=nobj, steps=int(self.e_steps[n]), reward=float(s[rules.ST_REWARD]),
                       max_combo=int(self.e_combo[n, 1]), max_combo_possible=max_combo_possible,
                       breaks=int(s[rules.ST_BREAKS]), tail_miss=int(s[rules.ST_TAIL_MISS]),
                       great=int(s[rules.ST_GREAT]), ok=int(s[rules.ST_OK]), meh=int(s[rules.ST_MEH]),
                       miss=int(s[rules.ST_MISS]), tick_hit=int(s[rules.ST_TICK_HIT]),
                       tick_miss=int(s[rules.ST_TICK_MISS]), spin_hit=int(s[rules.ST_SPIN_HIT]),
                       spin_miss=int(s[rules.ST_SPIN_MISS]), presses=int(s[rules.ST_PRESS]),
                       timeout=bool(not self.e_done[n]))
            # labelled (session 4): early misses and shakes are counted by the rules (rules.py press dispatch) but were
            # never recorded; motion stats (osu px, cost, idle steps) for the motion-cost A/B
            rec.update(early_miss=int(s[rules.ST_EARLY_MISS]), shake=int(s[rules.ST_SHAKE]))
            if self.motion_metrics:
                rec.update(move_px=round(float(self.e_move_px[n]), 2), move_cost=float(self.e_move_cost[n]),
                           idle_px=round(float(self.e_idle_px[n]), 2), idle_steps=int(self.e_idle_steps[n]),
                           idle_still=int(self.e_idle_still[n]), aim_px=round(float(self.e_aim_px[n]), 2),
                           move_s=float(self.e_move_s[n]), rev_px=round(float(self.e_rev_px[n]), 2),
                           sp_px=round(float(self.e_sp_px[n]), 2), press_n=int(self.e_press_n[n]),
                           press_moving=int(self.e_press_moving[n]), v2_cost=float(self.e_v2_cost[n]),
                           v2_s=float(self.e_v2_s[n]))
            if self.phys_on:      # labelled (session 5) cursor physics
                rec.update(vmax=float(self.e_vmax[n]), bind_steps=int(self.e_bind_steps[n]),
                           unreached_press=int(self.e_unreached[n]), mc_aim=float(self.e_mc_aim[n]))
            rec["fc"] = (rec["breaks"] == 0 and rec["tail_miss"] == 0 and not rec["timeout"])
            if self.clip.prioritized > 0 and not self.clip.full_map:
                B = self.clip.bucket
                rate = min((rec["breaks"] + rec["tail_miss"]) / max(nobj, 1), 1.0)
                i0 = int(self.e_i0[n])
                for b in range(i0 // B, (i0 + nobj - 1) // B + 1):
                    k = self.bucket_off[mid] + b
                    self.bucket_fail[k] = 0.7 * self.bucket_fail[k] + 0.3 * rate
            self.episode_log.append(rec)
            self.sampler.report([mid], rec)
        self.reset(idx)

    def render_state_np(self):
        st, sp, r = self.st, self.sp, self.r
        rules.render_slots(self.e_t, self.e_lo, self.e_win_lo, self.e_win_hi, self.e_map, self.m["preempt"],
                           self.o["start"], st["head_r"], st["head_t"], st["trk"], st["trk_t"], st["nest_n"],
                           st["nest_miss"], st["nest_t"], st["tail_r"], st["end_t"], sp["rot"],
                           (np.float32(360.0) * sp["completed"].astype(np.float32) + sp["cur_max"]), sp["trk"],
                           self.vis, self.o["kind"],
                           r["gid"], r["head_r"], r["head_t"], r["trk"], r["trk_t"], r["nest_n"], r["nest_miss"],
                           r["nest_t"], r["tail_r"], r["end_t"], r["spin_rot"], r["spin_total"], r["spin_trk"],
                           r["trk_tp"], r["spin_ti"], r["spin_ts"], r["spin_tc"], r["spin_last"])
        N, K = self.n, self.K
        rs = dict(
            t=self.e_t.astype(np.float32), map=self.e_map.astype(np.int32),
            cursor=np.stack((self.e_cx, self.e_cy), 1).astype(np.float32), keys=self.e_keys.astype(np.int32),
            press_t=self.e_press_t.astype(np.float32), release_t=self.e_release_t.astype(np.float32),
            gid=r["gid"].copy(), head_r=r["head_r"].astype(np.int8), head_t=r["head_t"].astype(np.float32),
            trk=r["trk"].astype(np.int8), trk_t=r["trk_t"].astype(np.float32),
            nest_n=r["nest_n"].astype(np.int16), nest_miss=r["nest_miss"].view(np.int32).reshape(N, K, 2).copy(),
            nest_t=r["nest_t"].astype(np.float32), tail_r=r["tail_r"].astype(np.int8),
            end_t=r["end_t"].astype(np.float32), spin_rot=r["spin_rot"].copy(), spin_total=r["spin_total"].copy(),
            spin_trk=r["spin_trk"].astype(np.int8), trk_tp=r["trk_tp"].astype(np.float32),
            spin_ti=r["spin_ti"].astype(np.float32), spin_ts=r["spin_ts"].astype(np.float32),
            spin_tc=r["spin_tc"].astype(np.float32), spin_last=r["spin_last"].astype(np.float32))
        return rs

    def aux_np(self):
        a = np.zeros((self.n, 8), np.float32)
        a[:, :2] = self.e_prev_act
        a[np.arange(self.n), 2 + self.e_keys] = 1.0
        a[:, 6] = np.where(self.e_first, 0.0, 1.0)
        return a

    # ------------------------------------------------------------------ torch-facing protocol (ppo.py)
    def observe(self):
        rs = self.render_state_np()
        if self.aug_prob > 0:
            rs["aug_sigma"] = self.e_aug_sigma.copy()
            rs["aug_idx"] = self.rng.integers(0, self.aug_bank_n, size=self.n).astype(np.int32)
        rs_t = {k: torch.from_numpy(v).to(self.device, non_blocking=True) for k, v in rs.items()}
        aux = torch.from_numpy(self.aux_np()).to(self.device, non_blocking=True)
        return rs_t, aux

    def render(self, rs):
        if "aug_sigma" not in rs:
            return self.renderer.render(rs)
        rgb = self.renderer.render({k: v for k, v in rs.items() if k not in ("aug_sigma", "aug_idx")})
        sig = rs["aug_sigma"]
        if not bool((sig > 0).any()):
            return rgb
        if self._aug_bank is None:
            g = torch.Generator(device=rgb.device)
            g.manual_seed(20261007)
            self._aug_bank = torch.randn((self.aug_bank_n,) + tuple(rgb.shape[1:]), generator=g, device=rgb.device,
                                         dtype=torch.float16)
        out = torch.empty_like(rgb)
        for a in range(0, rgb.shape[0], 512):           # chunks keep the fp16 temporaries ~150 MB
            b = min(a + 512, rgb.shape[0])
            noise = self._aug_bank[rs["aug_idx"][a:b].long()] * sig[a:b].view(-1, 1, 1, 1).half()
            out[a:b] = (rgb[a:b].half() + noise).round_().clamp_(0, 255).to(torch.uint8)
        return out

    def slider_active(self):
        """[N] bool on device: a started slider has a judged head and an unjudged tail (training-only signal)."""
        return torch.from_numpy(self.e_sl_active.astype(np.bool_)).to(self.device, non_blocking=True)

    def step(self, cursor, key):
        act = cursor.detach().float().cpu().numpy()
        k = key.detach().cpu().numpy()
        reward, done, timeout = self.step_np(act, k)
        info = self._step_info()
        if getattr(self, "exposure", None) is not None:
            # labelled (session 9, spec v3 App. D g1-5): counted before finish_episodes resets the finished slots
            self.exposure.count_step(self.e_map, done, self.e_lo, self.e_hi)
        if done.any():
            self.finish_episodes(done)
        return (torch.from_numpy(reward.astype(np.float32)).to(self.device),
                torch.from_numpy(done).to(self.device), info)

    def _step_info(self):
        # labelled (session 4): per-step motion sums over envs; PPOTrainer.collect adds them up over the rollout
        out = dict(self._mv) if self.motion_metrics else {}
        if self.phys_on:
            out.update(self._ph)       # labelled (session 5) cursor-physics sums (ph_*)
        return out

    def curriculum_state(self):
        st = self.sampler.state() if hasattr(self.sampler, "state") else {}
        log = self.episode_log[-512:]
        if log:
            objs = sum(r["objects"] for r in log)
            st.update(ep_fc=round(float(np.mean([r["fc"] for r in log])), 4),
                      ep_fail_per_obj=round(sum(r["breaks"] + r["tail_miss"] for r in log) / max(objs, 1), 4),
                      ep_reward_per_obj=round(sum(r["reward"] for r in log) / max(objs, 1), 4),
                      ep_presses_per_obj=round(sum(r["presses"] for r in log) / max(objs, 1), 3),
                      episodes=len(self.episode_log))
            # labelled (session 3): spinner outcomes over the same window (spec 11 C5 asks for them separately)
            sh = sum(r.get("spin_hit", 0) for r in log)
            sm = sum(r.get("spin_miss", 0) for r in log)
            st.update(ep_spin_hit=int(sh), ep_spin_miss=int(sm),
                      ep_spin_nonmiss=round(sh / (sh + sm), 4) if sh + sm else None)
            # labelled (session 4): motion-cost monitoring (DESIGN.md 2.5 / 7) over the same window
            o1 = max(objs, 1)
            st.update(ep_early_miss_per_obj=round(sum(r.get("early_miss", 0) for r in log) / o1, 5),
                      ep_shake_per_obj=round(sum(r.get("shake", 0) for r in log) / o1, 5))
            mlog = [r for r in log if "move_px" in r]
            if mlog:
                mo = max(sum(r["objects"] for r in mlog), 1)
                isteps = sum(r["idle_steps"] for r in mlog)
                st.update(ep_move_px_per_obj=round(sum(r["move_px"] for r in mlog) / mo, 2),
                          ep_move_cost_per_obj=round(sum(r["move_cost"] for r in mlog) / mo, 6),
                          ep_idle_px_per_step=round(sum(r["idle_px"] for r in mlog) / isteps, 3) if isteps else None,
                          ep_idle_still_frac=round(sum(r["idle_still"] for r in mlog) / isteps, 4) if isteps else None)
                # labelled (session 4) MOTION-COST v2 monitoring
                v2log = [r for r in mlog if "sp_px" in r]
                if v2log:
                    vo = max(sum(r["objects"] for r in v2log), 1)
                    pn = sum(r["press_n"] for r in v2log)
                    st.update(ep_rev_px_per_obj=round(sum(r["rev_px"] for r in v2log) / vo, 3),
                              ep_sp_px_per_press=round(sum(r["sp_px"] for r in v2log) / pn, 3) if pn else None,
                              ep_steady_press_share=round(1.0 - sum(r["press_moving"] for r in v2log) / pn, 4)
                              if pn else None,
                              ep_v2_cost_per_obj=round(sum(r["v2_cost"] for r in v2log) / vo, 6))
            # labelled (session 6) DT: failures per object and episodes per game rate (records of rate envs only)
            if any("clock_rate" in r for r in log):
                for rate in sorted({r.get("clock_rate", 1.0) for r in log}):
                    rl = [r for r in log if r.get("clock_rate", 1.0) == rate]
                    ro = max(sum(r["objects"] for r in rl), 1)
                    st[f"ep_fail_per_obj_r{rate:g}"] = round(sum(r["breaks"] + r["tail_miss"] for r in rl) / ro, 4)
                    st[f"ep_n_r{rate:g}"] = len(rl)
            # labelled (session 5) cursor physics over the same window
            plog = [r for r in log if "bind_steps" in r]
            if plog:
                po = max(sum(r["objects"] for r in plog), 1)
                ps = max(sum(r["steps"] for r in plog), 1)
                st.update(ep_vmax_mean=round(float(np.mean([r["vmax"] for r in plog])), 2),
                          ep_bind_frac=round(sum(r["bind_steps"] for r in plog) / ps, 5),
                          ep_unreached_press_per_obj=round(sum(r["unreached_press"] for r in plog) / po, 5))
        if len(self.episode_log) > 4096:
            self.episode_log = self.episode_log[-1024:]
        return st

    def load_curriculum(self, state):
        if hasattr(self.sampler, "load"):
            self.sampler.load(state)

    def full_state(self):
        """Sampler statistics worth keeping across restarts (clip-start bucket failure EMA, map priorities)."""
        st = {"bucket_fail": self.bucket_fail.copy()}
        if hasattr(self.sampler, "full_state"):
            st.update(self.sampler.full_state())
        if getattr(self, "exposure", None) is not None:
            st["exposure"] = self.exposure.state()        # labelled (session 9, spec v3 App. D g1-5)
        return st

    def enable_exposure_counters(self):
        """labelled (session 9, spec v3 App. D g1-5): start cumulative exposure counting (ExposureCounters). make_env
        calls it after construction (env.exposure_counters true), when env.sampler is final (MapRateSimEnv replaces
        the sampler after SimEnv.__init__). Off (never called): no state, no RNG use, no record key."""
        src = self.pool.maps.get("source")
        if src is None:
            print("[env] exposure counters: the pool has no maps['source'] (m_source); every map counts as human",
                  flush=True)
        self.exposure = ExposureCounters(np.zeros(self.pool.n_maps, np.int64) if src is None else src, self.sampler)
        r = self.exposure.record()
        print(f"[env] exposure counters on: sources {list(r['by_source'])}, ranges {self.exposure.rng_labels}",
              flush=True)
        return self.exposure

    def load_full_state(self, st):
        if "bucket_fail" in st:
            bf = np.asarray(st["bucket_fail"], dtype=np.float64)
            if len(bf) == len(self.bucket_fail):
                self.bucket_fail = bf.copy()
            elif len(bf) < len(self.bucket_fail):
                # labelled (session 3): a pool that appends maps to the old one (train_mix3 = train_mix2 + drills)
                # keeps the old maps' buckets as a prefix; restore it, the new tail keeps the unseen default
                self.bucket_fail[:len(bf)] = bf
                print(f"[env] bucket_fail: restored {len(bf)} saved buckets as a prefix of {len(self.bucket_fail)} "
                      f"(pool grew; {len(self.bucket_fail) - len(bf)} new buckets keep the default)", flush=True)
        if hasattr(self.sampler, "load"):
            self.sampler.load(st)
        if getattr(self, "exposure", None) is not None:
            # labelled (session 9, spec v3 App. D g1-5): resume continues the cumulative counts; a checkpoint without
            # them (older code, counters off, a branch point) starts at 0 and train.py records origin_decisions
            if "exposure" in st:
                self.exposure.load(st["exposure"])
                print(f"[env] exposure counters restored (origin_decisions {self.exposure.origin_decisions})",
                      flush=True)
            else:
                print("[env] exposure counters: the checkpoint has none; counting starts at 0 here", flush=True)


def make_env(cfg: dict, device="cuda", seed=0):
    """Factory used by train.py. cfg keys: pool, n, renderer (full|simple), clip{...}, curriculum{...},
    key_down_cost, omax, shaping_beta, spin_beta, break_penalty, tail_penalty, track_beta, spin_progress,
    hit_animations (labelled (session 3), default True; False = lazer HitAnimations off, see make_renderer),
    motion_cost {idle_per_px, aim_per_px, ramp_start, ramp_decisions, exempt_spinners; v2: press_per_px, rev_per_px,
    rev_deadzone_px, v2_ramp_start, v2_ramp_decisions} and motion_metrics (labelled (session 4), MOTION-COST v1/v2;
    absent / prices 0 = no cost), exposure_counters (labelled (session 9, spec v3 App. D g1-5), default false)."""
    from pathlib import Path
    from . import ROOT
    from .curriculum import StarCurriculum, StarMixture
    from .evaluate import make_renderer
    pool = MapPool.load(Path(ROOT) / cfg["pool"])
    pool_gpu = pool.to_torch(device)
    renderer = make_renderer(cfg.get("renderer", "full"), pool_gpu, device, breaks=pool.breaks,
                             hit_animations=bool(cfg.get("hit_animations", True)))   # labelled (session 3)
    cur_cfg = dict(cfg.get("curriculum", {}))
    stars = pool.maps.get("stars", np.ones(pool.n_maps))
    gen = pool.maps.get("source", np.zeros(pool.n_maps)) == 1
    if "mixture" in cfg:
        sampler = StarMixture(stars, generated=gen, **cfg["mixture"])
    elif cfg.get("use_curriculum", True):
        sampler = StarCurriculum(stars, generated=gen, **cur_cfg)
    else:
        sampler = UniformSampler(np.arange(pool.n_maps))
    # labelled (session 6) DT: env.rate = {"mode": "map_tags", "control_hz": 60, "allow_unvalidated_gpu": bool} builds
    # osu_sim.rate.MapRateSimEnv (each episode plays at its sampled map's m_clock_rate tag; one sampler over the mixed
    # pool). Without it the rate-one SimEnv is built exactly as before, and a pool with any DT-tagged map is refused
    # (never played at 1.0x). allow_unvalidated_gpu (renderer on DT episodes) is set only after the real-host DT parity
    # gate (FC7_NM_DT_PROMPT.md B4) has passed.
    from .rate import MapRateSimEnv, assert_pool_rate
    rate_cfg = dict(cfg.get("rate") or {})
    if rate_cfg:
        unknown = {k for k in rate_cfg if not k.startswith("_")} - {"mode", "control_hz", "allow_unvalidated_gpu"}
        if unknown or rate_cfg.get("mode") != "map_tags":
            raise ValueError(f"env.rate needs mode 'map_tags' (unknown keys {sorted(unknown)})")
        env_cls, extra = MapRateSimEnv, dict(control_hz=int(rate_cfg.get("control_hz", 60)),
                                             allow_unvalidated_gpu=bool(rate_cfg.get("allow_unvalidated_gpu", False)))
    else:
        assert_pool_rate(pool, 1.0)
        env_cls, extra = SimEnv, {}
    env = env_cls(pool, int(cfg.get("n", 1024)), device, seed=seed, sampler=sampler, **extra,
                 clip=ClipConfig(**cfg.get("clip", {})), omax=int(cfg.get("omax", 256)),
                 key_down_cost=float(cfg.get("key_down_cost", 0.0)), renderer=renderer, pool_gpu=pool_gpu,
                 shaping_beta=float(cfg.get("shaping_beta", 0.0)), spin_beta=float(cfg.get("spin_beta", 0.0)),
                 break_penalty=float(cfg.get("break_penalty", 0.0)), tail_penalty=float(cfg.get("tail_penalty", 0.0)),
                 track_beta=float(cfg.get("track_beta", 0.0)), spin_progress=float(cfg.get("spin_progress", 0.0)),
                 spin_min_r=float(cfg.get("spin_min_r", 0.0)),
                 aug_prob=float(cfg.get("aug_prob", 0.0)), aug_sigma_max=float(cfg.get("aug_sigma_max", 0.0)),
                 motion_cost=cfg.get("motion_cost"),                                   # labelled (session 4)
                 motion_metrics=bool(cfg.get("motion_metrics", True)),
                 cursor_physics=cfg.get("cursor_physics"))                             # labelled (session 5)
    if cfg.get("exposure_counters", False):
        env.enable_exposure_counters()             # labelled (session 9, spec v3 App. D g1-5); default off
    return env
