"""MapPool: many Beatmaps compiled into flat, immutable arrays (CPU numpy for rules, GPU torch for rendering).

Global ids index every table so that a stored render state (which only holds ids, times and
judgement results) can be re-rendered at any later time (render-on-demand PPO).

Tables (all positions are STACKED gamefield osu!px; times are map ms, float64 on CPU / float32 on GPU):
  maps[m]   : radius, scale, preempt, fade_in, great, ok, meh, od, hp, obj_off, obj_cnt, duration
  objs[g]   : kind(0 circle,1 slider,2 spinner), map, start, end, x, y, ex, ey (stacked end pos),
              combo_index, number(index_in_combo+1), new_combo, last_in_combo,
              slider: path_off, path_cnt, distance, span_count, span_dur, nest_off, nest_cnt,
                      w_head, w_tick, w_tail (reward weights), first_repeat
              spinner: spins_req, max_bonus
  nest[e]   : type(0 tick,1 repeat,2 tail), time, progress, x, y (stacked), span_index, span_start
  path[v]   : x, y (stacked absolute vertex), cum (cumulative length)
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .beatmap import CIRCLE, SLIDER, SPINNER, Beatmap, build_beatmap, stacked
from .osufile import parse_osu

OBJ_FIELDS = ("kind", "map", "start", "end", "x", "y", "ex", "ey", "combo_index", "number", "new_combo",
              "last_in_combo", "path_off", "path_cnt", "distance", "span_count", "span_dur", "nest_off",
              "nest_cnt", "w_head", "w_tick", "w_tail", "spins_req", "max_bonus", "velocity")
MAP_FIELDS = ("radius", "scale", "preempt", "fade_in", "great", "ok", "meh", "od", "hp", "obj_off", "obj_cnt",
              "first_start", "last_end", "cs", "ar")
NEST_FIELDS = ("type", "time", "progress", "x", "y", "span_index", "span_start", "obj")


@dataclass
class MapPool:
    maps: dict          # field -> np.ndarray [M]
    objs: dict          # field -> np.ndarray [G]
    nest: dict          # field -> np.ndarray [E]
    path: dict          # x, y, cum -> np.ndarray [V]
    keys: list          # map identifiers (e.g. relative path or generator seed)
    breaks: list        # per map list of (start, end)

    @property
    def n_maps(self):
        return len(self.keys)

    def save(self, path):
        arrays = {}
        for name, table in (("m", self.maps), ("o", self.objs), ("n", self.nest), ("p", self.path)):
            for k, v in table.items():
                arrays[f"{name}_{k}"] = v
        arrays["keys"] = np.array(self.keys, dtype=object)
        arrays["breaks"] = np.array([np.array(b, dtype=np.float64).reshape(-1, 2) for b in self.breaks] +
                                    [np.zeros((0, 2))], dtype=object)[:-1]
        np.savez(path, **arrays, allow_pickle=True)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        tables = {"m": {}, "o": {}, "n": {}, "p": {}}
        for k in z.files:
            if k in ("keys", "breaks"):
                continue
            prefix, field = k.split("_", 1)
            tables[prefix][field] = z[k]
        return MapPool(tables["m"], tables["o"], tables["n"], tables["p"], list(z["keys"]), list(z["breaks"]))

    def to_torch(self, device):
        """GPU copies for the renderer: float32 for floats, int32 for ints."""
        import torch

        def conv(table):
            out = {}
            for k, v in table.items():
                if v.dtype.kind == "f":
                    out[k] = torch.as_tensor(v.astype(np.float32), device=device)
                else:
                    out[k] = torch.as_tensor(v.astype(np.int32), device=device)
            return out
        return {"maps": conv(self.maps), "objs": conv(self.objs), "nest": conv(self.nest), "path": conv(self.path)}


def _weights(bm: Beatmap, o):
    """Spec 9.2 per-parent reward weights for a slider: head 300, tick/repeat 30, tail 150 (sum normalised)."""
    n_ticks = sum(1 for e in o.events if e[0] != 2)
    total = 300 + 30 * n_ticks + 150
    return 300 / total, 30 / total, 150 / total


def compile_pool(beatmaps: list[tuple[str, Beatmap]]) -> MapPool:
    maps = {k: [] for k in MAP_FIELDS}
    objs = {k: [] for k in OBJ_FIELDS}
    nest = {k: [] for k in NEST_FIELDS}
    path = {"x": [], "y": [], "cum": []}
    keys, breaks = [], []
    for m, (key, bm) in enumerate(beatmaps):
        keys.append(key)
        breaks.append(list(bm.breaks))
        g0 = len(objs["kind"])
        maps["radius"].append(bm.radius)
        maps["scale"].append(bm.scale)
        maps["preempt"].append(bm.preempt)
        maps["fade_in"].append(bm.fade_in)
        maps["great"].append(bm.windows[0])
        maps["ok"].append(bm.windows[1])
        maps["meh"].append(bm.windows[2])
        maps["od"].append(bm.od)
        maps["hp"].append(bm.hp)
        maps["cs"].append(bm.cs)
        maps["ar"].append(bm.ar)
        maps["obj_off"].append(g0)
        maps["obj_cnt"].append(len(bm.objects))
        maps["first_start"].append(bm.objects[0].start if bm.objects else 0.0)
        maps["last_end"].append(max(o.end for o in bm.objects) if bm.objects else 0.0)
        for o in bm.objects:
            sx, sy = stacked(bm, o)
            if o.kind == SLIDER:
                off = np.float32(o.stack_height) * np.float32(bm.scale) * np.float32(-6.4)
                ex, ey = float(np.float32(o.end_x) + off), float(np.float32(o.end_y) + off)
            else:
                ex, ey = sx, sy
            objs["kind"].append(o.kind)
            objs["map"].append(m)
            objs["start"].append(o.start)
            objs["end"].append(o.end)
            objs["x"].append(sx)
            objs["y"].append(sy)
            objs["ex"].append(ex)
            objs["ey"].append(ey)
            objs["combo_index"].append(o.combo_index)
            objs["number"].append(o.index_in_combo + 1)
            objs["new_combo"].append(int(o.new_combo))
            objs["last_in_combo"].append(int(o.last_in_combo))
            objs["spins_req"].append(o.spins_required)
            objs["max_bonus"].append(o.max_bonus_spins)
            objs["velocity"].append(o.velocity)
            if o.kind == SLIDER:
                p0 = len(path["x"])
                verts = o.path.vertices
                path["x"].extend((verts[:, 0] + np.float32(sx)).astype(np.float32).tolist())
                path["y"].extend((verts[:, 1] + np.float32(sy)).astype(np.float32).tolist())
                path["cum"].extend(o.path.cumulative.tolist())
                objs["path_off"].append(p0)
                objs["path_cnt"].append(len(verts))
                objs["distance"].append(o.path.distance)
                objs["span_count"].append(o.span_count)
                objs["span_dur"].append(o.span_duration)
                n0 = len(nest["type"])
                for (typ, time, prog, span, span_start) in o.events:
                    pos = o.path.position_at(prog)
                    nest["type"].append(typ)
                    nest["time"].append(time)
                    nest["progress"].append(prog)
                    nest["x"].append(float(np.float32(sx) + pos[0]))
                    nest["y"].append(float(np.float32(sy) + pos[1]))
                    nest["span_index"].append(span)
                    nest["span_start"].append(span_start)
                    nest["obj"].append(len(objs["kind"]) - 1)
                objs["nest_off"].append(n0)
                objs["nest_cnt"].append(len(o.events))
                wh, wt, wl = _weights(bm, o)
            else:
                objs["path_off"].append(0)
                objs["path_cnt"].append(0)
                objs["distance"].append(0.0)
                objs["span_count"].append(1)
                objs["span_dur"].append(0.0)
                objs["nest_off"].append(0)
                objs["nest_cnt"].append(0)
                wh, wt, wl = 1.0, 0.0, 0.0
            objs["w_head"].append(wh)
            objs["w_tick"].append(wt)
            objs["w_tail"].append(wl)
    ints = {"kind", "map", "combo_index", "number", "new_combo", "last_in_combo", "path_off", "path_cnt",
            "span_count", "nest_off", "nest_cnt", "spins_req", "max_bonus", "type", "span_index", "obj",
            "obj_off", "obj_cnt"}
    f64 = {"start", "end", "time", "span_start", "cum", "span_dur", "distance", "velocity", "first_start",
           "last_end", "preempt", "fade_in", "great", "ok", "meh"}

    def arr(table):
        out = {}
        for k, v in table.items():
            if k in ints:
                out[k] = np.asarray(v, dtype=np.int64)
            elif k in f64:
                out[k] = np.asarray(v, dtype=np.float64)
            else:
                out[k] = np.asarray(v, dtype=np.float32)
        return out
    return MapPool(arr(maps), arr(objs), arr(nest), arr(path), keys, breaks)


def _build_one(args):
    key, source = args
    try:
        return key, build_beatmap(parse_osu(source))
    except Exception as e:  # noqa
        return key, e


def build_pool_from_files(items: list[tuple[str, object]], workers: int = 16) -> tuple[MapPool, list]:
    """items: (key, path-or-osu-text). Returns pool and list of (key, error) failures."""
    from concurrent.futures import ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_build_one, items, chunksize=4))
    good = [(k, b) for k, b in results if isinstance(b, Beatmap) and b.objects]
    bad = [(k, repr(b)) for k, b in results if not isinstance(b, Beatmap)]
    return compile_pool(good), bad
