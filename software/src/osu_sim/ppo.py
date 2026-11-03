"""High-throughput recurrent PPO for osu-sim vector environments.

Observations are NOT stored as images. The environment exposes a compact,
deterministic render state per step; the rollout stores those and minibatches
re-render their frames on demand.

Env protocol (all tensors on one CUDA device, leading dim N):
  env.n                         number of environments
  env.observe() -> (rs, aux)    rs: dict[str, Tensor[N,...]] render state, aux: [N,8] float32
  env.render(rs) -> uint8 [M,3,192,256]   pure function of a (possibly gathered) render state
  env.step(cursor[N,2], key[N]) -> (reward[N] f32, terminated[N] bool, info dict)
                                 auto-resets terminated envs; next observe() is the new episode
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch
from torch.nn import functional as F

from .policy import Agent


@dataclass
class PPOConfig:
    rollout_steps: int = 128          # T per update
    sequence_length: int = 64         # L (T % L == 0); stored LSTM state at each chunk start
    sequences_per_minibatch: int = 64  # S
    epochs: int = 2
    learning_rate: float = 2.5e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99               # per decision
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    cursor_entropy_coefficient: float = 0.001
    key_entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    normalize_returns: bool = True
    adaptive_lr: bool = False          # rl_games-style: lr /= 1.5 if kl > kl_band*kl_adapt, *= 1.5 if < kl_adapt/kl_band
    kl_adapt: float = 0.016
    kl_band: float = 2.0               # labelled (session 3): adaptive-LR dead band (2.0 = the old hard-coded rule)
    lr_min: float = 2e-5
    lr_max: float = 6e-4
    refresh_old_logp: bool = True      # recompute rollout log-probs with the update's numerics before epoch 0
    # labelled training-only exploration floor (session 3): in states where a started slider has a judged head and an
    # unjudged tail (env.slider_active), add coef * relu(log(hold_floor) - log P(any key down)). Only acts where the
    # policy gives holding < hold_floor probability, so "hold" is sampled.
    hold_floor_coef: float = 0.0
    hold_floor: float = 0.1
    # labelled (session 4) evaluation/deployment-side: exponential moving averages (Polyak averaging) of the agent's
    # parameters, one per decay, updated after every optimizer step. Training (rollouts, PPO loss, optimizer) is unchanged;
    # an EMA is only evaluated / exported.
    ema_decays: tuple = ()
    # labelled (session 9, spec v3 App. D g1-2) warm-up EMA (spec v3 section 3 "EMA"): EMA update number t + 1 of decay d
    # uses decay_t = min(d, (1 + t) / (10 + t)), t = the EMA steps that EMA has already taken (0 for its first update), so
    # an EMA started at random init forgets the init weights. t is saved in checkpoints (payload["ema_t"]) and restored
    # by load(). False = the plain EMA, bitwise as before (same torch._foreach_lerp_ call, no counter in the payload).
    ema_warmup: bool = False
    # labelled (session 5) skill protection for the cursor-physics fine-tune (CURSOR_PHYSICS_PROMPT.md step 4): forward
    # KL(ref || policy) to a frozen pre-fine-tune actor (self-distillation, no human data) on the key head on every step
    # (same key input: the taken cursor's executed position and cell) and on the S1 cell distribution on steps where a key
    # is newly pressed. The reference distributions are computed at collection time from the same rendered frame, aux and
    # its own LSTM state stepped alongside the policy, so the update needs no second network pass. 0 = off.
    anchor_key_coef: float = 0.0
    anchor_cell_coef: float = 0.0


class RunningScale:
    """Running standard deviation of returns (scale-only normalisation of the value target)."""

    def __init__(self, device):
        self.var = torch.ones((), device=device)
        self.count = 0

    def update(self, x: torch.Tensor):
        v = x.float().var()
        self.var = v if self.count == 0 else 0.99 * self.var + 0.01 * v
        self.count += 1

    @property
    def std(self):
        return self.var.clamp(min=1e-4).sqrt()


def ema_warmup_decay(d, t):
    """labelled (session 9, spec v3 App. D g1-2) warm-up EMA: the decay of EMA update number t + 1 of decay d, where t is
    the number of EMA steps already taken (t = 0 for the first update): min(d, (1 + t) / (10 + t))."""
    return min(float(d), (1.0 + t) / (10.0 + t))


class PPOTrainer:
    def __init__(self, env, agent: Agent, cfg: PPOConfig, out_dir: str, device="cuda", ref_agent: Agent = None):
        assert cfg.rollout_steps % cfg.sequence_length == 0
        self.env, self.agent, self.cfg, self.device = env, agent, cfg, torch.device(device)
        # labelled (session 5): cursor physics (per-step cap stored for the key head's executed-position input) and the
        # KL-anchor reference actor (frozen; its own LSTM state, reset with the policy's episodes)
        self.use_phys = bool(getattr(env, "phys_on", False))
        self.ref = ref_agent
        if (cfg.anchor_key_coef > 0 or cfg.anchor_cell_coef > 0) and ref_agent is None:
            raise ValueError("anchor_*_coef > 0 needs a reference agent")
        if self.ref is not None:
            self.ref.eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
            self.ref_state = self.ref.initial_state(env.n, self.device)[:, :2]
        self.opt = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
        self.ema = {}
        self.reset_ema()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        n = env.n
        self.state = agent.initial_state(n, self.device)
        self.episode_start = torch.ones(n, dtype=torch.bool, device=self.device)
        self.buf = None
        self.decisions = 0
        self.updates = 0
        self.ret_scale = RunningScale(self.device)
        self.log = open(os.path.join(out_dir, "updates.jsonl"), "a", encoding="utf-8")
        self.use_floor = cfg.hold_floor_coef > 0 and hasattr(env, "slider_active")
        # labelled (session 4) metrics only (MOTION-COST v1 monitoring): the env's pre-step motion gate per stored step
        # (0 AIM, 1 IDLE, 2 SPIN) splits the logged cursor entropy into IDLE / AIM parts; loss, ratios and optimiser
        # are unchanged (tests/test_sim_motion_cost.py T10: bitwise identical parameters after an update)
        self.use_gate = bool(getattr(env, "motion_metrics", False)) and hasattr(env, "last_gate")
        self.adv_stats = {}

    # ------------------------------------------------------------------ rollout
    def _alloc(self, rs, aux):
        T, n, d = self.cfg.rollout_steps, self.env.n, self.device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=d)
        self.buf = dict(
            rs={k: torch.zeros((T,) + tuple(v.shape), dtype=v.dtype, device=d) for k, v in rs.items()},
            aux=z(T, n, aux.shape[-1]), reset=z(T, n, dt=torch.bool), latent=z(T, n, 2), cursor=z(T, n, 2),
            key=z(T, n, dt=torch.long), cell=z(T, n, dt=torch.long), log_prob=z(T, n), value=z(T, n), reward=z(T, n),
            terminated=z(T, n, dt=torch.bool), slider=z(T, n, dt=torch.bool), gate=z(T, n, dt=torch.int8),
            state=z(T // self.cfg.sequence_length, n, self.agent.state_slots, self.agent.actor_core.hidden))
        if self.use_phys:      # labelled (session 5)
            self.buf["vmax"] = z(T, n)
        if self.ref is not None:
            from .policy import NCELL
            self.buf["ref_key_lp"] = z(T, n, 4)
            self.buf["ref_cell_lp"] = z(T, n, NCELL)

    @torch.no_grad()
    def collect(self):
        self.agent.eval()
        cfg, env, T, L = self.cfg, self.env, self.cfg.rollout_steps, self.cfg.sequence_length
        stats = {}
        t_render = t_policy = t_env = 0.0
        for t in range(T):
            rs, aux = env.observe()
            if self.buf is None:
                self._alloc(rs, aux)
            b = self.buf
            for k, v in rs.items():
                b["rs"][k][t] = v
            b["aux"][t], b["reset"][t] = aux, self.episode_start
            if self.use_floor:
                b["slider"][t] = env.slider_active()
            if t % L == 0:
                b["state"][t // L] = self.state
            c0 = time.perf_counter()
            rgb = env.render(rs)
            c1 = time.perf_counter()
            vmax = None
            if self.use_phys:      # labelled (session 5): per-env cap of the current episodes
                vmax = torch.from_numpy(env.vmax_np()).to(self.device, non_blocking=True)
                b["vmax"][t] = vmax
            action, value, self.state = self.agent.act(rgb, aux, self.state, self.episode_start, vmax=vmax)
            if self.ref is not None:
                # labelled (session 5) KL anchor: the frozen reference on the same frame / aux, given the taken action
                b["ref_key_lp"][t], b["ref_cell_lp"][t], self.ref_state = self.ref.anchor_step(
                    rgb, aux, self.ref_state, self.episode_start, action.cursor, action.cell, vmax)
            c2 = time.perf_counter()
            reward, terminated, info = env.step(action.cursor, action.key)
            c3 = time.perf_counter()
            if self.use_gate:
                # computed inside env.step from the pre-step state (before the rules kernel), i.e. the state of step t
                b["gate"][t] = torch.from_numpy(env.last_gate.copy()).to(self.device)
            t_render += c1 - c0
            t_policy += c2 - c1
            t_env += c3 - c2
            b["latent"][t], b["cursor"][t], b["key"][t], b["cell"][t] = action.latent, action.cursor, action.key, action.cell
            b["log_prob"][t], b["value"][t], b["reward"][t], b["terminated"][t] = (
                action.log_prob, value, reward, terminated)
            self.episode_start = terminated
            for k, v in info.items():
                stats[k] = stats.get(k, 0) + (float(v) if not torch.is_tensor(v) else float(v.sum()))
        rs, aux = env.observe()
        rgb = env.render(rs)
        vmax = torch.from_numpy(env.vmax_np()).to(self.device) if self.use_phys else None
        _, boot, _ = self.agent.act(rgb, aux, self.state, self.episode_start, vmax=vmax)
        self.decisions += T * env.n
        stats.update(t_render=t_render, t_policy=t_policy, t_env=t_env)
        return boot, stats

    @torch.no_grad()
    def advantages(self, boot):
        b, cfg = self.buf, self.cfg
        T = cfg.rollout_steps
        scale = self.ret_scale.std if cfg.normalize_returns else 1.0
        values = b["value"] * scale        # value head predicts normalised returns
        boot = boot * scale
        adv = torch.zeros_like(values)
        last = torch.zeros_like(boot)
        for t in range(T - 1, -1, -1):
            nonterminal = (~b["terminated"][t]).float()
            next_v = boot if t == T - 1 else values[t + 1]
            delta = b["reward"][t] + cfg.gamma * next_v * nonterminal - values[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
            adv[t] = last
        returns = adv + values
        if cfg.normalize_returns:
            self.ret_scale.update(returns)
            returns = returns / self.ret_scale.std
        b["returns"] = returns
        # labelled (session 4) metrics only: raw advantage / reward scale before the batch normalisation (the motion
        # cost's pull relative to the advantages, DESIGN.md section 7 risk 6)
        self.adv_stats = dict(adv_raw_std=round(float(adv.std()), 5), adv_raw_mean=round(float(adv.mean()), 5),
                              reward_mean=round(float(b["reward"].mean()), 6),
                              reward_std=round(float(b["reward"].std()), 5))
        b["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        ev = 1 - (returns - b["value"]).var() / returns.var().clamp(min=1e-8)
        return float(ev)

    # ------------------------------------------------------------------ update
    def update(self):
        self.agent.train()
        cfg, b, n = self.cfg, self.buf, self.env.n
        T, L, S = cfg.rollout_steps, cfg.sequence_length, cfg.sequences_per_minibatch
        chunks = T // L
        total = chunks * n
        metrics = []
        steps = 0
        kl_stopped = False
        ar = torch.arange(L, device=self.device)
        perms = [torch.randperm(total, device=self.device) for _ in range(cfg.epochs)]
        refresh_err = 0.0
        floor_stats = []
        gate_stats = []
        anchor_stats = []
        if cfg.refresh_old_logp:
            # bf16 conv / batched-vs-stepwise LSTM numerics differ between act() and evaluate(); with a sharp cursor
            # distribution that noise alone produced |ratio-1| up to 0.6 before any update. Re-score the stored
            # actions with the exact evaluate() path and epoch-0 minibatch layout so ratios start at exactly 1.
            with torch.no_grad():
                for start in range(0, total, S):
                    ids = perms[0][start:start + S]
                    c, e = ids // n, ids % n
                    ti = (c * L).view(1, -1) + ar.view(-1, 1)
                    ei = e.view(1, -1).expand(L, -1)
                    rs = {k: v[ti, ei].flatten(0, 1) for k, v in b["rs"].items()}
                    rgb = self.env.render(rs).view(L, len(ids), 3, 192, 256)
                    g = lambda key: b[key][ti, ei]
                    logp = self.agent.evaluate(
                        rgb, g("aux"), b["state"][c, e], g("reset"), g("latent"), g("cursor"), g("key"), g("cell"),
                        vmax=g("vmax") if self.use_phys else None)[0]
                    refresh_err = max(refresh_err, float((logp - g("log_prob")).abs().max()))
                    b["log_prob"][ti, ei] = logp
        for epoch in range(cfg.epochs):
            perm = perms[epoch]
            epoch_kl = []
            for start in range(0, total, S):
                ids = perm[start:start + S]
                c, e = ids // n, ids % n
                ti = (c * L).view(1, -1) + ar.view(-1, 1)          # [L,S]
                ei = e.view(1, -1).expand(L, -1)
                rs = {k: v[ti, ei].flatten(0, 1) for k, v in b["rs"].items()}
                rgb = self.env.render(rs).view(L, len(ids), 3, 192, 256)
                g = lambda key: b[key][ti, ei]
                ev_out = self.agent.evaluate(rgb, g("aux"), b["state"][c, e], g("reset"), g("latent"), g("cursor"),
                                          g("key"), g("cell"), return_any=self.use_floor,
                                          vmax=g("vmax") if self.use_phys else None,
                                          return_dists=self.ref is not None)
                logp, cur_ent, key_ent, value = ev_out[:4]
                old = g("log_prob")
                log_ratio = logp - old
                ratio = log_ratio.exp()
                adv = g("advantages")
                pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)).mean()
                vloss = 0.5 * (value - g("returns")).square().mean()
                loss = (pg + cfg.value_coefficient * vloss - cfg.cursor_entropy_coefficient * cur_ent.mean()
                        - cfg.key_entropy_coefficient * key_ent.mean())
                if self.use_floor:
                    sl = g("slider").float()
                    below = F.relu(math.log(cfg.hold_floor) - ev_out[4]) * sl
                    loss = loss + cfg.hold_floor_coef * below.sum() / sl.sum().clamp(min=1.0)
                    with torch.no_grad():
                        nsl = sl.sum()
                        floor_stats.append(torch.stack((nsl, ((ev_out[4] < math.log(cfg.hold_floor)).float() * sl).sum(),
                                                        (ev_out[4].exp() * sl).sum())))
                if self.ref is not None:
                    # labelled (session 5) KL anchor: forward KL(ref || policy), key head on every step, S1 cell
                    # distribution on steps with a newly pressed key (key bit down now, up in the previous step's keys)
                    dists = ev_out[-1]
                    rk, rc = g("ref_key_lp"), g("ref_cell_lp")
                    pk = rk.exp()
                    kl_key = torch.where(pk > 0, pk * (rk - dists["key_lp"]), torch.zeros_like(pk)).sum(-1)
                    pc = rc.exp()
                    kl_cell = torch.where(pc > 0, pc * (rc - dists["cell_lp"]), torch.zeros_like(pc)).sum(-1)
                    prev_key = g("aux")[..., 2:6].argmax(-1)
                    newpress = (((g("key") & ~prev_key) & 3) != 0).float()
                    npress = newpress.sum()
                    kl_cell_m = (kl_cell * newpress).sum() / npress.clamp(min=1.0)
                    kl_key_m = kl_key.mean()
                    loss = loss + cfg.anchor_key_coef * kl_key_m + cfg.anchor_cell_coef * kl_cell_m
                    with torch.no_grad():
                        anchor_stats.append(torch.stack((kl_key_m.detach(), (kl_cell * newpress).sum().detach(), npress,
                                                         kl_key.detach().max())))
                if self.use_gate:
                    with torch.no_grad():       # labelled (session 4) metrics only
                        gt = g("gate")
                        ce = cur_ent.detach().float()
                        gi, ga = (gt == 1).float(), (gt == 0).float()
                        gate_stats.append(torch.stack(((ce * gi).sum(), gi.sum(), (ce * ga).sum(), ga.sum())))
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.opt.step()
                self._ema_step()
                steps += 1
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    m = torch.stack((loss, pg, vloss, cur_ent.mean(), key_ent.mean(), kl,
                                     ((ratio - 1).abs() > cfg.clip_range).float().mean(), gn,
                                     (ratio - 1).abs().max() if (epoch == 0 and start == 0) else torch.zeros((), device=self.device)))
                metrics.append(m)
                epoch_kl.append(kl)
            if cfg.target_kl and float(torch.stack(epoch_kl).mean()) > cfg.target_kl:
                kl_stopped = True
                break
        m = torch.stack(metrics).float()
        mean_kl = float(m[:, 5].mean())
        if cfg.adaptive_lr:
            lr = self.opt.param_groups[0]["lr"]
            if mean_kl > cfg.kl_band * cfg.kl_adapt:
                lr = max(cfg.lr_min, lr / 1.5)
            elif mean_kl < cfg.kl_adapt / cfg.kl_band:
                lr = min(cfg.lr_max, lr * 1.5)
            for gp in self.opt.param_groups:
                gp["lr"] = lr
        names = ("loss", "policy_loss", "value_loss", "cursor_entropy", "key_entropy", "approx_kl",
                 "clip_fraction", "grad_norm")
        out = {k: round(float(m[:, i].mean()), 6) for i, k in enumerate(names)}
        out["first_ratio_err"] = round(float(m[0, -1]), 6)
        out["refresh_logp_maxdiff"] = round(refresh_err, 5)
        out.update(optimizer_steps=steps, kl_stopped=kl_stopped, lr=self.opt.param_groups[0]["lr"])
        if gate_stats:
            gs = torch.stack(gate_stats).sum(0)
            out.update(cursor_entropy_idle=round(float(gs[0] / gs[1]), 6) if float(gs[1]) > 0 else None,
                       cursor_entropy_aim=round(float(gs[2] / gs[3]), 6) if float(gs[3]) > 0 else None)
        if anchor_stats:       # labelled (session 5) KL anchor monitoring
            a = torch.stack(anchor_stats)
            out.update(anchor_kl_key=round(float(a[:, 0].mean()), 7),
                       anchor_kl_cell_press=round(float(a[:, 1].sum() / a[:, 2].sum().clamp(min=1.0)), 6),
                       anchor_press_frac=round(float(a[:, 2].sum()) / max(len(anchor_stats) * S * L, 1), 5),
                       anchor_kl_key_max=round(float(a[:, 3].max()), 5))
        if floor_stats:
            fs = torch.stack(floor_stats).sum(0)
            out.update(slider_step_frac=round(float(fs[0]) / max(len(floor_stats) * S * L, 1), 4),
                       slider_below_floor=round(float(fs[1] / fs[0].clamp(min=1)), 4),
                       slider_p_any=round(float(fs[2] / fs[0].clamp(min=1)), 4))
        self.updates += 1
        return out

    # ------------------------------------------------------------------ io
    # ------------------------------------------------------------------ parameter EMA (labelled (session 4))
    def reset_ema(self):
        """(Re)start every EMA at the current parameters."""
        self.ema = {float(d): [p.detach().clone().float() for p in self.agent.parameters()]
                    for d in self.cfg.ema_decays}
        # labelled (session 9, spec v3 App. D g1-2): warm-up counter t per EMA (EMA steps taken); a (re)started EMA is at 0
        self.ema_t = {d: 0 for d in self.ema}

    @torch.no_grad()
    def _ema_step(self):
        if self.ema:
            params = [p.detach().float() for p in self.agent.parameters()]
            for d, tensors in self.ema.items():
                if self.cfg.ema_warmup:
                    # labelled (session 9, spec v3 App. D g1-2) warm-up EMA: decay_t = min(d, (1 + t) / (10 + t)); once
                    # (1 + t) / (10 + t) >= d the weight is the plain path's 1.0 - d exactly
                    t = self.ema_t[d]
                    torch._foreach_lerp_(tensors, params, 1.0 - ema_warmup_decay(d, t))
                    self.ema_t[d] = t + 1
                else:
                    torch._foreach_lerp_(tensors, params, 1.0 - d)      # ema += (1 - d) * (p - ema)

    def ema_state_dict(self, decay):
        """Agent state_dict with the parameters replaced by EMA `decay` (buffers from the live agent)."""
        sd = {k: v.detach().clone() for k, v in self.agent.state_dict().items()}
        for (name, p), e in zip(self.agent.named_parameters(), self.ema[float(decay)]):
            sd[name] = e.to(p.dtype).clone()
        return sd

    def save(self, path, extra=None):
        payload = dict(agent=self.agent.state_dict(), agent_config=self.agent.config, opt=self.opt.state_dict(),
                       cfg=asdict(self.cfg), decisions=self.decisions, updates=self.updates,
                       ret_var=float(self.ret_scale.var), ret_count=self.ret_scale.count, extra=extra or {})
        if self.ema:   # labelled (session 4): {decay: {param name: tensor}}; load() / apply_ema_payload() read it
            names = [n for n, _ in self.agent.named_parameters()]
            payload["ema"] = {str(d): dict(zip(names, (t.detach() for t in ts))) for d, ts in self.ema.items()}
            if self.cfg.ema_warmup:
                # labelled (session 9, spec v3 App. D g1-2): warm-up counter per EMA {decay: EMA steps taken}
                payload["ema_t"] = {str(d): int(t) for d, t in self.ema_t.items()}
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load(self, path, optimizer=True, grow=False):
        """grow=True: the agent may have extra zero-initialised blocks (encoder .res.*) that the checkpoint lacks;
        existing weights and their Adam moments are carried over by parameter name."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not grow:
            self.agent.load_state_dict(payload["agent"])
            if optimizer and "opt" in payload:
                self.opt.load_state_dict(payload["opt"])
        else:
            from .policy import Agent, grow_moment, grow_state_dict
            sd, resized = grow_state_dict(payload["agent"], payload["agent_config"], self.agent)
            self.agent.load_state_dict(sd)
            if optimizer and "opt" in payload:
                # Adam moments carry over by name; resized tensors keep old moments in their old slice.
                # Brand-new tensors (extra residual blocks) start fresh.
                old_names = [n for n, _ in Agent(**payload["agent_config"]).named_parameters()]
                new_index = {n: i for i, (n, _) in enumerate(self.agent.named_parameters())}
                params = list(self.agent.parameters())
                oc, nc = payload["agent_config"], self.agent.config
                for i, st in payload["opt"]["state"].items():
                    name = old_names[int(i)]
                    prm = params[new_index[name]]
                    st = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in st.items()}
                    if name in resized:
                        # New entries inherit the saved step count. New exp_avg_sq entries start
                        # at the old tensor's mean; exp_avg stays zero.
                        st["exp_avg"] = grow_moment(name, st["exp_avg"], prm.shape, oc, nc)
                        st["exp_avg_sq"] = grow_moment(name, st["exp_avg_sq"], prm.shape, oc, nc,
                                                       fill=float(st["exp_avg_sq"].mean()))
                    self.opt.state[prm] = st
                for gp in self.opt.param_groups:
                    gp["lr"] = payload["opt"]["param_groups"][0]["lr"]
        if optimizer and "opt" in payload:
            self._fix_loaded_opt()
        self.decisions, self.updates = payload.get("decisions", 0), payload.get("updates", 0)
        # labelled (session 4): restore parameter EMAs saved with the checkpoint (same decay, same parameter names
        # and shapes); any EMA without a saved match (new decay, growth, old checkpoint) restarts at the loaded weights
        self.reset_ema()
        saved = payload.get("ema", {}) if not grow else {}
        for d in list(self.ema):
            src = saved.get(str(d))
            named = list(self.agent.named_parameters())
            if src and all(n in src and src[n].shape == p.shape for n, p in named):
                self.ema[d] = [src[n].to(self.device).float().clone() for n, _ in named]
                print(f"[ema] restored decay {d} from checkpoint", flush=True)
                if self.cfg.ema_warmup:
                    # labelled (session 9, spec v3 App. D g1-2): the warm-up counter travels with its EMA. A restored
                    # EMA without a saved counter (checkpoint of a run without ema_warmup, or older code) restarts the
                    # warm-up at t = 0: the restored tensors keep weight 9! k! / (k + 9)! after k more steps (< 1e-12
                    # after 100), i.e. the warm-up EMA restarts at the resumed weights. Restarted EMAs (no saved match,
                    # growth) are at t = 0 like the fresh EMA they are.
                    saved_t = (payload.get("ema_t") or {}).get(str(d))
                    if saved_t is None:
                        print(f"[ema] WARNING decay {d}: ema_warmup is on but the checkpoint has no warm-up counter "
                              f"(ema_t); the warm-up restarts at t = 0 on the restored EMA", flush=True)
                    else:
                        self.ema_t[d] = int(saved_t)
                        print(f"[ema] decay {d}: restored warm-up counter t = {self.ema_t[d]}", flush=True)
            elif self.cfg.ema_decays:
                print(f"[ema] decay {d}: no saved EMA, starting at the loaded weights", flush=True)
        self.ret_scale.var = torch.tensor(payload.get("ret_var", 1.0), device=self.device)
        self.ret_scale.count = payload.get("ret_count", 0)
        return payload

    def _fix_loaded_opt(self):
        """labelled (session 3), after either optimiser-load branch of load().

        A1: a loaded Adam 'step' stays wherever torch.load mapped it (cuda here); with neither capturable nor fused
        Adam reads it with .item() twice per parameter per optimizer step (84 host syncs). Fresh Adam hosts it on CPU
        as float32, so do the same; bias correction is computed in Python floats either way (numerics unchanged).
        A2: Optimizer.load_state_dict restores the checkpoint's param-group hyperparameters, so cfg.adam_eps was
        silently ignored on --resume. Re-apply it; betas stay as loaded, lr is the checkpoint's (or the caller's)."""
        for st in self.opt.state.values():
            if torch.is_tensor(st.get("step")):
                st["step"] = st["step"].detach().to("cpu", torch.float32)
        for gp in self.opt.param_groups:
            gp["eps"] = self.cfg.adam_eps

    def write(self, record):
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()
