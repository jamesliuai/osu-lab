"""osu-sim training entry point.

    python -m osu_sim.train --config <yaml/json> --out runs/sim/<name> [--resume <ckpt>] [--init <ckpt>]

Writes updates.jsonl (one record per PPO update), eval.jsonl, latest.pt / best.pt, and honours
a STOP file in the run directory (orderly stop: final eval + checkpoint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

from . import ROOT
from .policy import Agent
from .ppo import PPOConfig, PPOTrainer

# labelled (session 9, spec v3 App. D g1-4 / R6) sealed registry (written by runs/development/paper/draw_tests.py),
# relative to ROOT; the environment variable OSU_SEALED_REGISTRY adds a registry (tests; never disables this one). The
# Training and evaluation entries must explicitly name their pools.
SEALED_REGISTRY = "sealed_registry.json"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def build_ppo_config(raw: dict) -> PPOConfig:
    names = {f.name for f in fields(PPOConfig)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown ppo keys {sorted(unknown)}")
    return PPOConfig(**raw)


# ---------------------------------------------------------------- evaluation schedule (labelled (session 9))
def parse_eval_at(cfg):
    """labelled (session 9, spec v3 App. D g1-3; spec 12.5) config "eval_at": absolute decision counts of the scheduled
    evaluation rows, or None when the key is absent (eval_every_decisions then applies exactly as before). Entries must
    be finite numbers >= 0. Unsorted or duplicated lists are sorted and de-duplicated with a printed note (labelled: the
    schedule only asks for the first entry above a decision count, so order and repeats carry no meaning). An empty
    list schedules no periodic row (eval_at_start and the final evaluation still apply)."""
    raw = cfg.get("eval_at")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"eval_at must be a list of absolute decision counts, got {raw!r}")
    vals = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"eval_at entries must be finite decision counts >= 0, got {v!r}")
        vals.append(float(v))
    out = sorted(set(vals))
    if out != vals:
        print(f"[eval_at] entries sorted / de-duplicated: {vals} -> {out}", flush=True)
    return out


def next_eval_at(eval_at, decisions):
    """labelled (session 9, spec v3 App. D g1-3): the first eval_at entry strictly above `decisions` (inf: none left)."""
    above = [e for e in eval_at if e > decisions]
    return min(above) if above else math.inf


def eval_schedule_next(cfg, decisions, eval_at, start=False):
    """labelled (session 9, spec v3 App. D g1-3; spec 12.5) decision count at which main() runs the next scheduled
    evaluation (the first update boundary with trainer.decisions >= it). start=True: at a (re)start, after any
    checkpoint load; start=False: right after an evaluation at `decisions`.

    Without eval_at (eval_at None): the pre-session-9 rule, same expressions: start -> decisions if eval_at_start
    (default true) else decisions + eval_every_decisions; after an evaluation -> decisions + eval_every_decisions.
    With eval_at: start with eval_at_start -> decisions (the extra start evaluation); otherwise, and after every
    evaluation, the first entry strictly above `decisions` (resume row alignment: a run resumed at decisions D
    continues with the first entry > D, so rows stay on schedule and the STOP row of a stopped run is never repeated);
    eval_every_decisions is ignored. Entries at or below the start point are never evaluated (labelled; with
    eval_at_start only the start row). A checkpoint written exactly at an entry precedes that entry's row (main()
    evaluates at the top of the next loop), so a crash in that window loses the row rather than duplicating it."""
    if eval_at is None:
        if start and cfg.get("eval_at_start", True):
            return decisions
        return decisions + float(cfg.get("eval_every_decisions", 5e6))
    if start and cfg.get("eval_at_start", True):
        return decisions
    return next_eval_at(eval_at, decisions)


# ---------------------------------------------------------------- sealed-registry refusal (labelled (session 9))
def _norm_pool_path(p, root):
    """Lexical normal form of a pool path (no filesystem access): ROOT-relative unless absolute (as make_env's
    Path(ROOT) / pool), normalised, case-folded on Windows, forward slashes."""
    q = os.path.abspath(os.path.join(str(root), str(p)))
    return os.path.normcase(os.path.normpath(q)).replace("\\", "/")


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def configured_pool_paths(cfg):
    """labelled (session 9, spec v3 App. D g1-4): every pool path a training config can make train.py load: env.pool
    (required), each evaluation entry's required pool ("evals", else the legacy "eval"), plus, anywhere in the
    config outside "_" note keys, every string value of a key named pool,
    pools, *_pool or *_pools and every string ending in .npz (e.g. a legacy "eval" pool next to "evals"). Anchor
    references, --resume / --init / --grow are checkpoints, not pools."""
    paths = [cfg["env"]["pool"]]
    for e in (cfg.get("evals") or [dict(cfg.get("eval") or {}, name="main")]):
        paths.append(e["pool"])

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                if not (isinstance(k, str) and k.startswith("_")):
                    walk(v, k)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and ((isinstance(key, str) and re.search(r"(^|_)pools?$", key))
                                     or x.lower().endswith(".npz")):
            paths.append(x)
    walk(cfg)
    return list(dict.fromkeys(p for p in paths if isinstance(p, str)))


def sealed_registry_entries(path):
    """labelled (session 9, spec v3 App. D g1-4): the entries of a sealed registry {"entries": [{"path": <ROOT-relative
    or absolute>, "sha256": <hex>, "ledger_event": ..., ...}, ...]}; also accepted: a top-level list, or "pools" /
    "sealed" holding a list or {name: entry}. Every entry needs a path or a sha256. Unreadable or unrecognised content
    refuses the run (fail closed)."""
    try:
        with open(path, encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[sealed-registry] REFUSED: cannot read {path}: {exc}")
    entries = reg
    if isinstance(reg, dict):
        key = next((k for k in ("entries", "pools", "sealed") if k in reg), None)
        if key is None:
            raise SystemExit(f"[sealed-registry] REFUSED: {path} has none of the keys entries / pools / sealed")
        entries = reg[key]
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not isinstance(entries, list) or not all(isinstance(e, dict) and (e.get("path") or e.get("sha256"))
                                                for e in entries):
        raise SystemExit(f"[sealed-registry] REFUSED: {path}: every entry must be an object with a path or a sha256")
    return entries


def check_sealed_registry(cfg, root=None):
    """labelled (session 9, spec v3 App. D g1-4 / R6) sealed-registry refusal. main() calls it on EVERY start (fresh,
    --resume, --init, --grow; so a STOP + resume cannot evaluate a registered pool) before the run directory, any pool,
    the env, the evaluators or CUDA are touched. It raises SystemExit (exit status 1) when a configured pool
    (configured_pool_paths) is in the registry: by path (lexical normal form; then, for existing files, the resolved
    path, which catches links), else by the sha256 of the file. Paths that matched are refused before anything is
    hashed, and a registry entry's own path is never opened, stat-ed or hashed. A configured path that does not exist is
    not hashed (it cannot be loaded either). No registry file: nothing to check. It only refuses; it changes no
    numerics. The registries checked are ROOT/SEALED_REGISTRY (always, when it exists) PLUS the ROOT-relative / absolute
    OSU_SEALED_REGISTRY when that environment variable is set (tests; announced): the variable can add entries but never
    disable the project registry (labelled executive fix, session 9: a missing override file only prints a warning).
    Returns the checked [(path, sha256)]."""
    root = str(ROOT if root is None else root)
    override = os.environ.get("OSU_SEALED_REGISTRY")
    reg_paths = [os.path.join(root, SEALED_REGISTRY)]
    if override:
        reg_paths.append(os.path.join(root, override))
        print(f"[sealed-registry] OSU_SEALED_REGISTRY set: also checking {reg_paths[-1]}", flush=True)
        if not os.path.isfile(reg_paths[-1]):
            print(f"[sealed-registry] WARNING: {reg_paths[-1]} does not exist (only the project registry is checked)",
                  flush=True)
    reg_paths = [r for r in reg_paths if os.path.isfile(r)]
    if not reg_paths:
        return []
    reg_path = " + ".join(reg_paths)
    t0 = time.perf_counter()
    entries = [e for r in reg_paths for e in sealed_registry_entries(r)]
    by_path = {_norm_pool_path(e["path"], root): e for e in entries if e.get("path")}
    by_sha = {str(e["sha256"]).strip().lower(): e for e in entries if e.get("sha256")}
    pools = [(p, _norm_pool_path(p, root)) for p in configured_pool_paths(cfg)]

    def refuse(p, how, e):
        shown = {k: e[k] for k in ("name", "path", "sha256", "ledger_event") if k in e}
        raise SystemExit(f"[sealed-registry] REFUSED: configured pool {p!r} is a registered sealed pool ({how}; entry "
                         f"{shown}; registry {reg_path}). Registered pools are evaluated only by "
                         "an independently authorized, recorded evaluation procedure. Nothing was loaded or written.")

    for p, n in pools:
        if n in by_path:
            refuse(p, "path", by_path[n])
    for p, n in pools:
        if os.path.exists(n):
            r = _norm_pool_path(os.path.realpath(n), root)
            if r in by_path:
                refuse(p, f"resolved path {r}", by_path[r])
    checked = []
    for p, n in pools:
        if not os.path.isfile(n):
            print(f"[sealed-registry] {p}: no such file, not hashed", flush=True)
            continue
        h = _sha256_file(n)
        if h in by_sha:
            refuse(p, f"sha256 {h}", by_sha[h])
        checked.append((p, h))
    print(f"[sealed-registry] {reg_path}: {len(entries)} entries; {len(checked)} configured pool(s) checked by path and "
          f"sha256 in {time.perf_counter() - t0:.1f} s: none registered", flush=True)
    return checked


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--init", default=None, help="warm start weights only")
    p.add_argument("--grow", default=None, help="resume a checkpoint into an agent with extra zero-init blocks")
    p.add_argument("--max-decisions", type=float, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    check_motion_v2(cfg)                                  # labelled (session 4) MOTION-COST v2 price condition
    # labelled (session 9, spec v3 App. D g1-4 / R6): refuse a registered sealed pool on every start (fresh or resumed)
    # before the run directory, any pool, the env, the evaluators or CUDA are touched
    check_sealed_registry(cfg)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "args": vars(args)}, f, indent=2)
    seed = int(cfg.get("seed", 1))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    from .env import make_env  # late import: env pulls in triton kernels
    env = make_env(cfg["env"], device=device, seed=seed)
    agent = Agent(**cfg.get("agent", {})).to(device)
    if cfg.get("compile_encoder", False):
        # Use one dynamic-shape graph for different actor, PPO and evaluation batch sizes.
        # Patching forward (not wrapping the module) keeps state_dict keys unchanged.
        for core in (agent.actor_core, agent.critic_core):
            if core is not None:
                core.encoder.forward = torch.compile(core.encoder.forward, dynamic=True)
    # labelled (session 5) KL-anchor reference: a frozen copy of the RAW weights of cfg["anchor"]["ref"] (not its EMA:
    # the anchor must start at KL ~0 against the weights the fine-tune resumes from)
    ref_agent = None
    if cfg.get("anchor"):
        rp = torch.load(cfg["anchor"]["ref"], map_location=device, weights_only=False)
        ref_agent = Agent(**rp["agent_config"]).to(device)
        ref_agent.load_state_dict(rp["agent"])
        if cfg.get("compile_encoder", False):
            ref_agent.actor_core.encoder.forward = torch.compile(ref_agent.actor_core.encoder.forward, dynamic=True)
        print(f"[anchor] reference = raw weights of {cfg['anchor']['ref']} ({rp.get('decisions')} decisions)", flush=True)
        del rp
    trainer = PPOTrainer(env, agent, build_ppo_config(cfg.get("ppo", {})), args.out, device=device,
                         ref_agent=ref_agent)
    if args.resume or args.grow:
        payload = trainer.load(args.resume or args.grow, grow=bool(args.grow))
        if cfg.get("reset_lr_on_resume", False):
            for gp in trainer.opt.param_groups:
                gp["lr"] = trainer.cfg.learning_rate
        if args.grow and cfg.get("grow_lr"):
            # start low after growth; the KL-adaptive rule raises it 1.5x per quiet update
            for gp in trainer.opt.param_groups:
                gp["lr"] = float(cfg["grow_lr"])
            # labelled (session 3): else a high-KL update's max(lr_min, lr / 1.5) would jump lr up to lr_min
            trainer.cfg.lr_min = min(trainer.cfg.lr_min, float(cfg["grow_lr"]))
        if "curriculum" in payload.get("extra", {}) and hasattr(env, "load_curriculum"):
            env.load_curriculum(payload["extra"]["curriculum"])
        if "env_state" in payload.get("extra", {}) and hasattr(env, "load_full_state"):
            env.load_full_state(payload["extra"]["env_state"])
    elif args.init:
        trainer.load(args.init, optimizer=False)
        trainer.decisions = trainer.updates = 0
    # labelled (session 5) cursor-physics controller (cap anneal with pause, motion-price ramp start, AIM escalation and
    # back-off); restored from the checkpoint when the run is itself resumed from a run that had it
    ctl = None
    if cfg.get("physics_schedule"):
        saved = (payload.get("extra", {}).get("phys_ctl") if (args.resume or args.grow) else None)
        ctl = PhysicsController(cfg["physics_schedule"], env, trainer.decisions, saved)
        ctl.before_rollout(trainer.decisions)
    if hasattr(env, "set_decisions"):
        # labelled (session 4) motion-cost ramp clock: __init__ reset every env at decisions 0, so re-freeze the live
        # episodes' ramp scale at the resumed decision count
        env.set_decisions(trainer.decisions, refresh=True)
    if getattr(env, "exposure", None) is not None and env.exposure.origin_decisions is None:
        # labelled (session 9, spec v3 App. D g1-5): counting starts here (fresh run, or a checkpoint without counters)
        env.exposure.origin_decisions = int(trainer.decisions)

    from .evaluate import SimEvaluator
    # "evals": [{name, pool, modes, ...}, ...] evaluates several pools; the FIRST one is primary (best.pt score).
    # Legacy "eval": {...} is a single pool named "main".
    eval_cfgs = cfg.get("evals") or [dict(cfg.get("eval", {}), name="main")]
    evaluators = [(e.get("name", Path(e.get("pool", "main")).stem), SimEvaluator(e, device=device)) for e in eval_cfgs]
    eval_round = 0         # labelled (session 5): an entry with "every": k runs on every k-th evaluation round (and final)
    ckpt_every = int(cfg.get("checkpoint_every_updates", 25))
    max_decisions = args.max_decisions or float(cfg.get("max_decisions", 1e10))
    max_seconds = 3600 * (args.max_hours or float(cfg.get("max_hours", 1e6)))
    start = time.perf_counter()
    # labelled (session 9, spec v3 App. D g1-3; spec 12.5): "eval_at" (absolute decision list) replaces
    # eval_every_decisions and keeps resumed runs' rows on schedule; absent = the old schedule (eval_schedule_next)
    eval_at = parse_eval_at(cfg)
    if eval_at is not None:
        print(f"[eval_at] rows at the first update boundary >= each of {eval_at}"
              + ("; eval_every_decisions ignored" if "eval_every_decisions" in cfg else ""), flush=True)
    next_eval = eval_schedule_next(cfg, trainer.decisions, eval_at, start=True)
    # labelled (session 9, spec v3 App. D g1-1 / R9): "save_best": false never writes best.pt (default true)
    save_best = bool(cfg.get("save_best", True))
    best = -1e9
    stop_file = os.path.join(args.out, "STOP")
    eval_log = open(os.path.join(args.out, "eval.jsonl"), "a", encoding="utf-8")

    ema_agent = None

    def eval_agent(ecfg):
        """labelled (session 4): an eval entry with "ema": <decay> evaluates that parameter EMA (eager copy of the agent);
        without it, the live (raw) agent."""
        nonlocal ema_agent
        d = ecfg.get("ema")
        if d is None:
            return agent
        if float(d) not in trainer.ema:
            raise ValueError(f"eval asks for EMA {d} but ppo.ema_decays = {list(trainer.ema)}")
        if ema_agent is None:
            ema_agent = Agent(**cfg.get("agent", {})).to(device)
        ema_agent.load_state_dict(trainer.ema_state_dict(d))
        return ema_agent

    def run_eval(final=False):
        nonlocal best, eval_round
        primary = None
        results = {}
        for name, evaluator in evaluators:
            if not final and (eval_round + 1) % int(evaluator.cfg.get("every", 1)) != 0:
                continue
            if evaluator.cfg.get("vmax_px") == "train":
                # labelled (session 5): evaluate at the CURRENT training cap, only while the cap anneal is running
                if ctl is None or ctl.st["anneal_done_at"] is not None:
                    continue
                evaluator.vmax = ctl.vmax()
            t0 = time.perf_counter()
            result = evaluator.run(eval_agent(evaluator.cfg))
            if evaluator.vmax is not None:
                result["vmax_eval"] = float(evaluator.vmax)
            if evaluator.cfg.get("ema") is not None:
                result["ema"] = float(evaluator.cfg["ema"])
            result.update(name=name, decisions=trainer.decisions, updates=trainer.updates, final=final,
                          wall_s=round(time.perf_counter() - start, 1), eval_s=round(time.perf_counter() - t0, 1))
            eval_log.write(json.dumps(result) + "\n")
            eval_log.flush()
            print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}), flush=True)
            results[name] = result
            if primary is None:
                primary = result
        eval_round += 1
        if ctl is not None:
            ctl.on_eval(trainer.decisions, results)
        score = primary.get("score", 0.0)
        if score > best:
            best = score
            if save_best:                 # labelled (session 9, spec v3 App. D g1-1): the score bookkeeping still runs
                trainer.save(os.path.join(args.out, "best.pt"), extra=_extra(env, primary))
        # labelled (session 4) infrastructure: keep the exact checkpoint behind every evaluation so model selection and
        # the overfitting guard can return to any evaluated point (latest.pt is overwritten every few updates)
        if cfg.get("snapshot_at_eval", False):
            snap = os.path.join(args.out, f"ckpt_{trainer.decisions}.pt")
            if not os.path.exists(snap):
                trainer.save(snap, extra=_extra(env, primary))
        if hasattr(env, "on_eval"):
            env.on_eval(primary)

    while True:
        # labelled (session 5): stop check first, so a run that ends on an evaluation boundary is evaluated once (by the
        # final evaluation below) instead of twice on the same weights
        if os.path.exists(stop_file) or trainer.decisions >= max_decisions or time.perf_counter() - start > max_seconds:
            break
        if trainer.decisions >= next_eval:
            run_eval()
            next_eval = eval_schedule_next(cfg, trainer.decisions, eval_at)      # labelled (session 9, g1-3)
        t0 = time.perf_counter()
        if ctl is not None:
            ctl.before_rollout(trainer.decisions)     # labelled (session 5): cap / prices for episodes reset now
        if hasattr(env, "set_decisions"):
            env.set_decisions(trainer.decisions)      # labelled (session 4): ramp clock for episodes reset in this rollout
        d0 = trainer.decisions
        boot, stats = trainer.collect()
        t1 = time.perf_counter()
        ev = trainer.advantages(boot)
        upd = trainer.update()
        t2 = time.perf_counter()
        rec = dict(update=trainer.updates, decisions=trainer.decisions, wall_s=round(t2 - start, 1),
                   collect_s=round(t1 - t0, 3), update_s=round(t2 - t1, 3),
                   dps=round(trainer.cfg.rollout_steps * env.n / (t2 - t0), 1), explained_var=round(ev, 4))
        rec.update(upd)
        rec.update(motion_record(env, stats, getattr(trainer, "adv_stats", {})))     # labelled (session 4)
        rec.update(physics_record(stats))                                            # labelled (session 5)
        rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()})
        if hasattr(env, "curriculum_state"):
            rec["curriculum"] = env.curriculum_state()
        if getattr(env, "exposure", None) is not None:
            rec["exposure"] = env.exposure.record()        # labelled (session 9, spec v3 App. D g1-5)
        if ctl is not None:
            rec["phys_ctl"] = ctl.after_update(d0, trainer.decisions, rec.get("curriculum", {}))
        trainer.write(rec)
        if trainer.updates % ckpt_every == 0:
            trainer.save(os.path.join(args.out, "latest.pt"), extra=_extra(env, ctl=ctl))
    run_eval(final=True)
    trainer.save(os.path.join(args.out, "latest.pt"), extra=_extra(env, ctl=ctl))
    print("done", trainer.decisions, flush=True)


class PhysicsController:
    """labelled (session 5) cursor-physics fine-tune schedule (CURSOR_PHYSICS_PROMPT.md steps 2-3; predeclared in
    runs/development/session5_cursor/EVIDENCE.md). All clocks are absolute trainer decision counts.

    1. Cap anneal: vmax = motion.vmax_schedule(progress) (geometric start_px -> mid_px over mid_decisions, then -> end_px
       over end_decisions), progress = decisions trained while NOT paused since the branch. Paused while the mean of the
       last pause_window training ep_fail_per_obj values exceeds pause_ratio x pre_branch_fail.
    2. When the anneal is complete, the motion-cost ramp starts (env.mc_ramp_start = that decision count; env prices and
       ramp length come from env.motion_cost).
    3. AIM escalation (at evaluation time): once decisions >= ramp end + escalate_after, the readout eval's metric (EMA-5M
       fc6_val home_ninth_gap_frac) > threshold and the current AIM level has been held >= hold decisions -> AIM price x
       factor (at most max_escalations times, capped at max_aim).
    4. Back-off (armed after the first escalation): when the mean of the last backoff_window ep_fail_per_obj values
       exceeds backoff_ratio x pre_branch_fail and the last price change is >= backoff_cooldown decisions old -> AIM
       price / 2 (not below the base price).
    Prices and the cap apply to episodes that start after a change (frozen per episode)."""

    DEFAULTS = dict(start_px=800.0, mid_px=160.0, end_px=120.0, mid_decisions=30e6, end_decisions=10e6,
                    pause_ratio=1.2, pause_window=5, pre_branch_fail=None, escalate_after=20e6,
                    readout="fc6_val_ema5m", readout_metric="home_ninth_gap_frac", threshold=0.8, factor=2.0,
                    max_escalations=2, max_aim=1e-4, hold=15e6, backoff_ratio=1.2, backoff_window=20,
                    backoff_cooldown=5e6)

    def __init__(self, cfg, env, decisions, saved=None):
        from collections import deque
        unknown = {k for k in cfg if not k.startswith("_")} - set(self.DEFAULTS) - {"branch_decisions"}
        if unknown:
            raise ValueError(f"unknown physics_schedule keys {sorted(unknown)}")
        self.c = dict(self.DEFAULTS, **{k: v for k, v in cfg.items() if not k.startswith("_")})
        if self.c["pre_branch_fail"] is None:
            raise ValueError("physics_schedule.pre_branch_fail is required (training ep_fail_per_obj before the branch)")
        if not getattr(env, "phys_on", False):
            raise ValueError("physics_schedule needs env.cursor_physics")
        self.env = env
        self.base_aim = float(env.mc_aim)
        self.st = dict(branch=float(cfg.get("branch_decisions", decisions)), progress=0.0, paused=False,
                       anneal_done_at=None, aim=self.base_aim, escalations=0, backoffs=0, last_change=None,
                       events=[])
        if saved:
            self.st.update(saved)
            print(f"[phys_ctl] restored {saved}", flush=True)
        self.fails = deque(maxlen=max(int(self.c["pause_window"]), int(self.c["backoff_window"])))
        # the env prices: AIM level from the state; the motion ramp starts only when the anneal is done
        env.mc_aim = float(self.st["aim"])
        env.mc_ramp_start = float(self.st["anneal_done_at"]) if self.st["anneal_done_at"] is not None else 1e18

    def vmax(self):
        from .motion import vmax_schedule
        c = self.c
        return vmax_schedule(self.st["progress"], c["start_px"], c["mid_px"], c["end_px"], c["mid_decisions"],
                             c["end_decisions"])

    def before_rollout(self, decisions):
        self.env.set_vmax(self.vmax())
        self.env.mc_aim = float(self.st["aim"])

    def _mean(self, k):
        v = list(self.fails)[-int(k):]
        return sum(v) / len(v) if v else None

    def after_update(self, d0, d1, cur):
        c, st = self.c, self.st
        f = cur.get("ep_fail_per_obj")
        # right after a (re)start the 512-episode window holds only the few short episodes that finished first (the
        # smoke runs logged 0.22 on 1 episode): ignore readings until the window is full
        if f is not None and int(cur.get("episodes", 0)) >= 512:
            self.fails.append(float(f))
        pre = float(c["pre_branch_fail"])
        m_pause = self._mean(c["pause_window"])
        anneal_total = c["mid_decisions"] + c["end_decisions"]
        if st["anneal_done_at"] is None:
            st["paused"] = bool(m_pause is not None and len(self.fails) >= int(c["pause_window"])
                                and m_pause > c["pause_ratio"] * pre)
            if not st["paused"]:
                st["progress"] = min(st["progress"] + (d1 - d0), anneal_total)
            if st["progress"] >= anneal_total:
                st["anneal_done_at"] = float(d1)
                self.env.mc_ramp_start = float(d1)
                st["events"].append(dict(at=d1, event="anneal_done_motion_ramp_start"))
                print(f"[phys_ctl] anneal complete at {d1}: vmax {self.vmax()}; motion-cost ramp starts", flush=True)
        else:
            st["paused"] = False
        # mandatory back-off once escalated
        m_bo = self._mean(c["backoff_window"])
        if (st["escalations"] > 0 and m_bo is not None and len(self.fails) >= int(c["backoff_window"])
                and m_bo > c["backoff_ratio"] * pre and st["aim"] > self.base_aim
                and (st["last_change"] is None or d1 - st["last_change"] >= c["backoff_cooldown"])):
            old = st["aim"]
            st["aim"] = max(self.base_aim, st["aim"] / 2.0)
            st["backoffs"] += 1
            st["last_change"] = float(d1)
            st["events"].append(dict(at=d1, event="backoff", aim_from=old, aim_to=st["aim"], fail_mean=m_bo))
            print(f"[phys_ctl] BACK-OFF at {d1}: AIM {old} -> {st['aim']} (ep_fail mean {m_bo:.5f})", flush=True)
        return dict(vmax=round(self.vmax(), 3), progress=st["progress"], paused=st["paused"],
                    anneal_done_at=st["anneal_done_at"], aim=st["aim"], escalations=st["escalations"],
                    backoffs=st["backoffs"], fail_mean_pause=None if m_pause is None else round(m_pause, 5),
                    fail_mean_backoff=None if m_bo is None else round(m_bo, 5))

    def on_eval(self, decisions, results):
        c, st = self.c, self.st
        if st["anneal_done_at"] is None:
            return
        ramp_end = st["anneal_done_at"] + float(self.env.mc_ramp_decisions)
        r = results.get(c["readout"])
        if r is None or decisions < ramp_end + c["escalate_after"]:
            return
        held_since = st["last_change"] if st["last_change"] is not None else ramp_end
        if decisions - held_since < c["hold"] and st["last_change"] is not None:
            return
        s = r.get("summary", {}).get("deterministic", {})
        val = s.get(c["readout_metric"])
        if val is None:
            return
        if val > c["threshold"] and st["escalations"] < int(c["max_escalations"]) and st["aim"] < c["max_aim"]:
            old = st["aim"]
            st["aim"] = min(float(c["max_aim"]), st["aim"] * float(c["factor"]))
            st["escalations"] += 1
            st["last_change"] = float(decisions)
            st["events"].append(dict(at=decisions, event="escalate", aim_from=old, aim_to=st["aim"], readout=val))
            print(f"[phys_ctl] ESCALATE at {decisions}: {c['readout']} {c['readout_metric']} {val} > "
                  f"{c['threshold']} -> AIM {old} -> {st['aim']}", flush=True)
        else:
            st["events"].append(dict(at=decisions, event="readout", value=val, aim=st["aim"]))

    def state(self):
        return dict(self.st)


def physics_record(stats):
    """labelled (session 5) cursor-physics monitoring of one update: pops the raw ph_* sums (SimEnv._step_info summed over
    the rollout) and returns the share of live steps where the cap bound, unreached-target presses (a key newly pressed
    while the cap bound) per press, and the mean commanded-minus-executed gap (osu px) on bound steps."""
    raw = {k: stats.pop(k) for k in list(stats) if k.startswith("ph_")}
    if not raw:
        return {}
    live, bind, press = raw.get("ph_live_n", 0.0), raw.get("ph_bind_n", 0.0), raw.get("ph_press_n", 0.0)
    return dict(ph_bind_frac=round(bind / live, 5) if live else None,
                ph_unreached_per_press=round(raw.get("ph_unreached_n", 0.0) / press, 5) if press else None,
                ph_gap_px=round(raw.get("ph_gap_px", 0.0) / bind, 2) if bind else None)


def check_motion_v2(cfg):
    """labelled (session 4) MOTION-COST v2 (runs/development/session4/cursor/v2/DESIGN_v2.md): with STEADY-PRESS on,
    the ADVANCE lemma (completing the final approach 2 frames before the press never raises the discounted cost) needs
    rev_per_px <= gamma^2 * press_per_px; refuse configs that break it. Also applied when press_per_px is 0 (a REV2-only
    price breaks the lemma: a last-frame flick would be cheaper than arriving 2 frames early). PPO credits the 2-step
    advance through GAE, which weighs the SP saving 2 rows later by (gamma * gae_lambda)^2 until the critic predicts it,
    so the check uses that tighter factor ((0.99 * 0.95)^2 = 0.8845 at the defaults; V: 5e-5 / 6e-5 = 0.833 passes)."""
    mc = (cfg.get("env") or {}).get("motion_cost") or {}
    wp, wr = float(mc.get("press_per_px", 0.0)), float(mc.get("rev_per_px", 0.0))
    ppo = cfg.get("ppo") or {}
    g = float(ppo.get("gamma", PPOConfig.gamma))
    lam = float(ppo.get("gae_lambda", PPOConfig.gae_lambda))
    f = (g * lam) ** 2
    if wr > f * wp:
        raise ValueError(f"motion_cost: rev_per_px {wr} > (gamma * gae_lambda)^2 * press_per_px {f * wp} "
                         f"(DESIGN_v2.md ADVANCE lemma; REV2 needs a large enough STEADY-PRESS price)")


def motion_record(env, stats, adv_stats):
    """labelled (session 4) MOTION-COST v1 monitoring fields of one update (DESIGN.md section 7). Pops the raw per-step
    sums mc_* (SimEnv._step_info, summed over the rollout by PPOTrainer.collect) from `stats` and returns per-step means
    over charged-eligible steps (not first, not done, gate != SPIN), plus the raw advantage / reward scale."""
    raw = {k: stats.pop(k) for k in list(stats) if k.startswith("mc_")}
    out = dict(adv_stats)
    if hasattr(env, "motion_scale"):
        # ramp scale of episodes starting now; 0 when both prices are 0 (no cost is applied, e.g. control arm A)
        out["motion_scale"] = round(float(env.motion_scale()), 6) if getattr(env, "mc_on", False) else 0.0
    if hasattr(env, "v2_scale"):
        # labelled (session 4) MOTION-COST v2 ramp scale of episodes starting now; 0 when both v2 prices are 0
        out["v2_scale"] = round(float(env.v2_scale()), 6) if getattr(env, "mc_v2", False) else 0.0
    if raw:
        n = raw.get("mc_charged_n", 0.0)
        idle_n, aim_n, spin_n = raw.get("mc_idle_n", 0.0), raw.get("mc_aim_n", 0.0), raw.get("mc_spin_n", 0.0)
        steps = idle_n + aim_n + spin_n
        mean = raw.get("mc_cost", 0.0) / n if n else None
        std = (max(raw.get("mc_cost_sq", 0.0) / n - mean * mean, 0.0) ** 0.5) if n else None
        out.update(mc_cost_mean=round(mean, 8) if n else None, mc_cost_std=round(std, 8) if n else None,
                   mc_px_mean=round(raw.get("mc_px", 0.0) / n, 3) if n else None,
                   mc_idle_px_mean=round(raw.get("mc_idle_px", 0.0) / idle_n, 3) if idle_n else None,
                   mc_idle_still_frac=round(raw.get("mc_idle_still", 0.0) / idle_n, 4) if idle_n else None,
                   mc_aim_px_mean=round(raw.get("mc_aim_px", 0.0) / aim_n, 3) if aim_n else None,
                   mc_idle_share=round(idle_n / steps, 4) if steps else None,
                   mc_spin_share=round(spin_n / steps, 4) if steps else None)
        a = adv_stats.get("adv_raw_std")
        out["mc_cost_to_adv"] = round(std / a, 5) if (std is not None and a) else None
        # labelled (session 4) MOTION-COST v2: REV2 excess px per eligible pair and share charged; SP px per eligible
        # new press and share of steady presses (SP < 4 px); v2 cost per charged-eligible step (mc_cost includes it)
        rn, pn = raw.get("mc_rev_n", 0.0), raw.get("mc_press_n", 0.0)
        out.update(mc_rev_px_mean=round(raw.get("mc_rev_px", 0.0) / rn, 4) if rn else None,
                   mc_rev_hit_share=round(raw.get("mc_rev_hit_n", 0.0) / rn, 4) if rn else None,
                   mc_sp_px_per_press=round(raw.get("mc_sp_px", 0.0) / pn, 3) if pn else None,
                   mc_steady_press_share=round(1.0 - raw.get("mc_press_moving", 0.0) / pn, 4) if pn else None,
                   mc_v2_cost_mean=round(raw.get("mc_v2_cost", 0.0) / n, 8) if n else None)
    return out


def _extra(env, result=None, ctl=None):
    extra = {}
    if hasattr(env, "curriculum_state"):
        extra["curriculum"] = env.curriculum_state()
    if hasattr(env, "full_state"):
        extra["env_state"] = env.full_state()
    if result is not None:
        extra["eval"] = {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
    if ctl is not None:
        extra["phys_ctl"] = ctl.state()       # labelled (session 5)
    return extra


if __name__ == "__main__":
    main()
