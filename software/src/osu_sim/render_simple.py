"""Cheap placeholder renderer (pipeline testing only; NOT the calibrated renderer in render.py).

Stamps each visible element into a 48x48 local window and composites with max. Geometry (positions,
radii, combo colours, approach timing) follows the real host so learning dynamics are roughly right.
"""
from __future__ import annotations

import math

import torch

H, W = 192, 256
WIN = 48
COLOURS = torch.tensor([[239, 115, 0], [0, 233, 51], [0, 92, 255], [249, 0, 0], [220, 223, 0], [106, 0, 255]],
                       dtype=torch.float32)


class SimpleRenderer:
    def __init__(self, pool_gpu, device="cuda", slider_stamps=12):
        self.p = pool_gpu
        self.device = torch.device(device)
        self.colours = COLOURS.to(self.device)
        o = torch.arange(WIN, device=self.device, dtype=torch.float32) - WIN // 2
        self.oy, self.ox = torch.meshgrid(o, o, indexing="ij")
        self.S = slider_stamps

    def _stamp(self, img, m_idx, cx, cy, val):
        """img [M*H*W*3] flat float; cx,cy [P] actor px centres; val(dist [P,WIN,WIN]) -> [P,WIN,WIN,3]."""
        bx = torch.floor(cx)
        by = torch.floor(cy)
        px = bx[:, None, None] + self.ox
        py = by[:, None, None] + self.oy
        d = torch.sqrt((px + 0.5 - cx[:, None, None]) ** 2 + (py + 0.5 - cy[:, None, None]) ** 2)
        v = val(d)  # [P,WIN,WIN,3]
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = ((m_idx[:, None, None] * H + py.clamp(0, H - 1).long()) * W + px.clamp(0, W - 1).long())
        idx = idx[ok]
        v = v[ok]
        img.view(-1, 3).index_reduce_(0, idx, v, "amax")

    @torch.no_grad()
    def render(self, rs, aug=None):
        dev = self.device
        t = rs["t"].float()
        M = t.shape[0]
        img = torch.zeros(M * H * W, 3, device=dev)
        gid = rs["gid"].long()
        valid = gid >= 0
        g = gid.clamp(min=0)
        P = self.p
        o, mp = P["objs"], P["maps"]
        mi = rs["map"].long()
        radius = mp["radius"][mi][:, None].expand_as(g) * 0.4
        preempt = mp["preempt"][mi][:, None].expand_as(g)
        fade_in = mp["fade_in"][mi][:, None].expand_as(g)
        start, end = o["start"][g], o["end"][g]
        kind = o["kind"][g]
        x, y = o["x"][g] * 0.4 + 25.6, o["y"][g] * 0.4 + 22.4
        col = self.colours[o["combo_index"][g] % 6]
        tt = t[:, None].expand_as(g)
        appear = start - preempt
        alpha = ((tt - appear) / fade_in).clamp(0, 1)
        head_r, head_t = rs["head_r"].long(), rs["head_t"].float()
        judged = head_r > 0
        hit_fade = (1 - (tt - head_t) / 240.0).clamp(0, 1)
        miss_fade = (1 - (tt - head_t) / 100.0).clamp(0, 1)
        alpha = torch.where(judged, alpha * torch.where(head_r == 4, miss_fade, hit_fade), alpha)
        alpha = alpha * valid
        mrow = torch.arange(M, device=dev)[:, None].expand_as(g)
        # --- slider bodies (stamped discs along the path) and ball / follow circle
        is_sl = (kind == 1) & valid & (alpha > 0) | ((kind == 1) & valid & (rs["tail_r"] == 0) & (tt >= start))
        if is_sl.any():
            sm = mrow[is_sl]
            sg = g[is_sl]
            rad = radius[is_sl]
            c = col[is_sl]
            snake = ((tt[is_sl] - appear[is_sl]) / (preempt[is_sl] / 3)).clamp(0, 1)
            body_alpha = ((tt[is_sl] - appear[is_sl]) / fade_in[is_sl]).clamp(0, 1) * \
                (rs["tail_r"][is_sl] == 0).float()
            off, cnt = o["path_off"][sg], o["path_cnt"][sg]
            fr = torch.linspace(0, 1, self.S, device=dev)[None] * snake[:, None]
            vi = (off[:, None] + (fr * (cnt[:, None] - 1).float()).round().long())
            vx = P["path"]["x"][vi] * 0.4 + 25.6
            vy = P["path"]["y"][vi] * 0.4 + 22.4
            rr = (rad * 0.86)[:, None].expand_as(vx).reshape(-1)
            cc = (c[:, None, :] * body_alpha[:, None, None] * 0.25).expand(-1, self.S, -1).reshape(-1, 3)
            bc = (c[:, None, :] * body_alpha[:, None, None]).expand(-1, self.S, -1).reshape(-1, 3)
            self._stamp(img, sm[:, None].expand_as(vx).reshape(-1), vx.reshape(-1), vy.reshape(-1),
                        lambda d: torch.where((d < rr[:, None, None] - 1.5)[..., None], cc[:, None, None, :],
                                              torch.where((d < rr[:, None, None])[..., None], bc[:, None, None, :],
                                                          torch.zeros_like(cc[:, None, None, :]))))
            # ball
            active = (tt[is_sl] >= start[is_sl]) & (tt[is_sl] <= end[is_sl])
            if active.any():
                dur = (end[is_sl] - start[is_sl]).clamp(min=1e-3)
                prog = ((tt[is_sl] - start[is_sl]) / dur).clamp(0, 1)
                spans = o["span_count"][sg].float()
                q = prog * spans
                span = torch.floor(q).clamp(max=spans - 1)
                frac = torch.where(q >= spans, torch.ones_like(q), q - span)
                pr = torch.where(span.remainder(2) == 1, 1 - frac, frac)
                bi = off + (pr * (cnt - 1).float()).round().long()
                bx = P["path"]["x"][bi] * 0.4 + 25.6
                by = P["path"]["y"][bi] * 0.4 + 22.4
                trk = rs["trk"][is_sl].float()
                a = active.float()
                self._stamp(img, sm, bx, by, lambda d: (
                    (((d < rad[:, None, None]) & (d > rad[:, None, None] - 1.5)).float() * a[:, None, None] * 255)[..., None]
                    .expand(-1, -1, -1, 3)
                    + ((((d - rad[:, None, None] * 2.4).abs() < 1.0).float() * trk[:, None, None] * a[:, None, None])[..., None]
                       * c[:, None, None, :])))
        # --- circles and slider heads (disc + border + approach ring)
        heads = (kind != 2) & valid & (alpha > 0)
        if heads.any():
            hm = mrow[heads]
            r = radius[heads]
            a = alpha[heads]
            c = col[heads]
            prog = ((tt[heads] - appear[heads]) / preempt[heads]).clamp(0, 1)
            ar = r * (4 - 3 * prog)
            ap_a = ((tt[heads] - appear[heads]) / (2 * fade_in[heads])).clamp(0, 0.9) * \
                (tt[heads] < start[heads]).float() * (~judged[heads]).float()
            dim = torch.where(tt[heads] < start[heads] - 400, 0.55, 1.0)

            def head_val(d):
                rr, aa = r[:, None, None], a[:, None, None]
                disc = (d < rr - 1.0).float() * aa
                border = ((d >= rr - 1.0) & (d < rr)).float() * aa
                ring = ((d - ar[:, None, None]).abs() < 0.7).float() * ap_a[:, None, None]
                out = disc[..., None] * c[:, None, None, :] * 0.6 * dim[:, None, None, None] + \
                    border[..., None] * 255 * dim[:, None, None, None]
                return torch.maximum(out, ring[..., None] * c[:, None, None, :])
            self._stamp(img, hm, x[heads], y[heads], head_val)
        # --- spinners: ring + centre
        sp = (kind == 2) & valid & (tt >= start - preempt) & (head_r == 0)
        if sp.any():
            prog = rs["spin_total"][sp].float() / 360.0 / o["spins_req"][g[sp]].float().clamp(min=1)
            self._stamp(img, mrow[sp], x[sp], y[sp], lambda d: (
                ((d - 22).abs() < 1.0).float()[..., None] * 200 + ((d < 4).float() * prog.clamp(0, 1)[:, None, None]
                                                                   * 255)[..., None]).expand(-1, -1, -1, 3))
        # --- cursor
        cx = rs["cursor"][:, 0] * 0.4 + 25.6
        cy = rs["cursor"][:, 1] * 0.4 + 22.4
        pink = torch.tensor([252, 97, 143], device=dev, dtype=torch.float32)
        self._stamp(img, torch.arange(M, device=dev), cx, cy,
                    lambda d: (((d < 4.5) & (d > 2.5)).float()[..., None] * pink + (d < 1.0).float()[..., None] * 255))
        out = img.view(M, H, W, 3).clamp(0, 255).round().to(torch.uint8).permute(0, 3, 1, 2).contiguous()
        return out
