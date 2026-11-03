"""osu-sim renderer: 256x192 actor frames of the real host's sim_v1 presentation (Argon skin).

Contract: runs/development/sim_v1/RENDER_INTERFACE.md.  ``Renderer(pool_gpu).render(rs) -> uint8 [M,3,192,256]``.

Design
  * ``_prepare`` (torch, vectorised over [M,K] slots) turns the render state into per-slot animation
    parameters (alphas, radii, colours already tinted in linear light, glyph layouts, culling boxes) following
    the pinned lazer transforms (visual_spec.md; sources cited there).
  * one Triton kernel rasterises tiles of TH x TW actor pixels.  Every actor pixel integrates the scene over its
    exact 3.75 x 3.75 client-pixel footprint (the INTER_AREA box of the real pipeline): shape edges use the
    exact box coverage of a straight edge (trapezoid CDF of the projected square), radial/linear colour
    gradients are evaluated at the footprint centre (exact for linear fields), and glyph/texture sprites are
    box-integrated with summed-area tables.  ``SS`` > 1 additionally splits the footprint into SS x SS
    sub-footprints (non-linear compositing of overlapping partial layers).
  * Lazer draw order (OsuPlayfield): spinners -> follow points -> judgement layer (ring explosions) -> hit
    objects (later objects below earlier ones) -> judgement text / tick-miss proxies -> approach circles ->
    cursor.  Blending in sRGB space, tint products in linear light (osu-framework SRGBColour), additive layers
    clamp at 1 (UNORM target).
Pure function of ``rs`` + static tables (no hidden state; bit-identical re-renders).
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np
import torch

import triton
import triton.language as tl       # noqa: E402
from triton.language.extra import libdevice  # noqa: E402

ASSET_DIR = Path(os.environ["OSU_SIM_ASSET_DIR"]).expanduser().resolve() if os.environ.get("OSU_SIM_ASSET_DIR") else None
FA_DIR = Path(os.environ["OSU_SIM_FONT_AWESOME_DIR"]).expanduser().resolve() if os.environ.get("OSU_SIM_FONT_AWESOME_DIR") else None

H_ACT, W_ACT = 192, 256
# gamefield -> actor pixel: a = (25.6, 22.4) + 0.4 g   (sim_v1: client = (256,84) + 1.5 g, crop x>=160, /3.75)
AX0, AY0, APX = 25.6, 22.4, 0.4
CLIENT_PER_OSU = 1.5
STEP_MS = 1000.0 / 60.0           # control step: rotation is credited once per STEP (heuristic ages only)
AA = 1.0 / CLIENT_PER_OSU        # masking blend range: 1 screen pixel, in osu px
AAc = tl.constexpr(AA)

# ------------------------------------------------------------------------------------------- colours
ARGON_COMBO = [(241, 116, 0), (0, 241, 53), (0, 82, 241), (241, 0, 0), (232, 235, 0), (92, 0, 241)]
RESULT_COL = {1: (0x66, 0xcc, 0xff), 2: (0x88, 0xb3, 0x00), 3: (0xff, 0xcc, 0x22), 4: (0xed, 0x11, 0x21)}
PINK, PINK_DARK = (0xFC, 0x61, 0x8F), (0xBB, 0x1A, 0x41)


def _to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.asarray(c, np.float64)
    return np.where(c < 0.0031308, 12.92 * c, 1.055 * np.abs(c) ** (1 / 2.4) - 0.055)


def _hspa_normalise(rgb255, amount=0.2):
    """DrawableHitObject.UpdateComboColour: lerp_linear(c, HSPA(c){P=0.6}, 0.2) (HSPAColour.cs, float32)."""
    f = np.float32
    r, g, b = (f(v / 255.0) for v in rgb255)
    pr, pg, pb = f(0.299), f(0.587), f(0.114)
    P = f(math.sqrt(r * r * pr + g * g * pg + b * b * pb))
    H = S = f(0)
    if not (r == g == b):
        if r >= g and r >= b:
            if b >= g:
                H, S = f(1) - f(1 / 6) * (b - g) / (r - g), f(1) - g / r
            else:
                H, S = f(0) + f(1 / 6) * (g - b) / (r - b), f(1) - b / r
        elif g >= r and g >= b:
            if r >= b:
                H, S = f(2 / 6) - f(1 / 6) * (r - b) / (g - b), f(1) - b / g
            else:
                H, S = f(2 / 6) + f(1 / 6) * (b - r) / (g - r), f(1) - r / g
        else:
            if g >= r:
                H, S = f(4 / 6) - f(1 / 6) * (g - r) / (b - r), f(1) - r / b
            else:
                H, S = f(4 / 6) + f(1 / 6) * (r - g) / (b - g), f(1) - g / b
    P = f(0.6)
    mom = f(1) - S
    h = H
    sq = lambda v: f(math.sqrt(v))
    if mom > 0:
        if h < 1 / 6:
            h = f(6) * h; part = f(1) + h * (f(1) / mom - f(1))
            B = P / sq(pr / mom / mom + pg * part * part + pb); R = B / mom; G = B + h * (R - B)
        elif h < 2 / 6:
            h = f(6) * (-h + f(2 / 6)); part = f(1) + h * (f(1) / mom - f(1))
            B = P / sq(pg / mom / mom + pr * part * part + pb); G = B / mom; R = B + h * (G - B)
        elif h < 3 / 6:
            h = f(6) * (h - f(2 / 6)); part = f(1) + h * (f(1) / mom - f(1))
            R = P / sq(pg / mom / mom + pb * part * part + pr); G = R / mom; B = R + h * (G - R)
        elif h < 4 / 6:
            h = f(6) * (-h + f(4 / 6)); part = f(1) + h * (f(1) / mom - f(1))
            R = P / sq(pb / mom / mom + pg * part * part + pr); B = R / mom; G = R + h * (B - R)
        elif h < 5 / 6:
            h = f(6) * (h - f(4 / 6)); part = f(1) + h * (f(1) / mom - f(1))
            G = P / sq(pb / mom / mom + pr * part * part + pg); B = G / mom; R = G + h * (B - G)
        else:
            h = f(6) * (-h + f(1)); part = f(1) + h * (f(1) / mom - f(1))
            G = P / sq(pr / mom / mom + pb * part * part + pg); R = G / mom; B = G + h * (R - G)
    else:
        if h < 1 / 6:
            h = f(6) * h; R = sq(P * P / (pr + pg * h * h)); G = R * h; B = f(0)
        elif h < 2 / 6:
            h = f(6) * (-h + f(2 / 6)); G = sq(P * P / (pg + pr * h * h)); R = G * h; B = f(0)
        elif h < 3 / 6:
            h = f(6) * (h - f(2 / 6)); G = sq(P * P / (pg + pb * h * h)); B = G * h; R = f(0)
        elif h < 4 / 6:
            h = f(6) * (-h + f(4 / 6)); B = sq(P * P / (pb + pg * h * h)); G = B * h; R = f(0)
        elif h < 5 / 6:
            h = f(6) * (h - f(4 / 6)); B = sq(P * P / (pb + pr * h * h)); R = B * h; G = f(0)
        else:
            h = f(6) * (-h + f(1)); R = sq(P * P / (pr + pb * h * h)); B = R * h; G = f(0)
    src = np.array([r, g, b], np.float64)
    dst = np.array([R, G, B], np.float64)
    return _to_srgb(_to_lin(src) + (_to_lin(dst) - _to_lin(src)) * amount)


ACCENTS = np.stack([_hspa_normalise(c) for c in ARGON_COMBO]).astype(np.float32)     # sRGB, unclamped


# ------------------------------------------------------------------------------------------- assets
def _parse_bmf(path):
    d = Path(path).read_bytes()
    assert d[:3] == b"BMF", path
    i, out = 4, {"chars": {}, "kern": {}}
    while i < len(d):
        t = d[i]
        n = struct.unpack_from("<I", d, i + 1)[0]
        b = d[i + 5:i + 5 + n]
        i += 5 + n
        if t == 1:
            out["size"] = struct.unpack_from("<h", b, 0)[0]
        elif t == 2:
            out["lineHeight"], out["base"], out["scaleW"], out["scaleH"], out["pages"] = struct.unpack_from("<HHHHH", b, 0)
        elif t == 3:
            out["page_names"] = [s.decode() for s in b.split(b"\0") if s]
        elif t == 4:
            for j in range(0, n, 20):
                cid, x, y, w, h, xo, yo, xa, pg, _ = struct.unpack_from("<IHHHHhhhBB", b, j)
                out["chars"][cid] = (x, y, w, h, xo, yo, xa, pg)
        elif t == 5:
            for j in range(0, n, 10):
                a, bb, amt = struct.unpack_from("<IIh", b, j)
                out["kern"][(a, bb)] = amt
    return out


def _read_rgba(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = np.dstack([im, im, im, np.full_like(im, 255)])
    return im[..., [2, 1, 0, 3]]          # RGBA


class Assets:
    """Glyph/texture atlas of summed-area tables + text layouts (font units)."""

    TORUS_BOLD = "osu.Game.Resources.Fonts.Torus.Torus-Bold"
    TORUS_SEMI = "osu.Game.Resources.Fonts.Torus.Torus-SemiBold"
    TORUS_REG = "osu.Game.Resources.Fonts.Torus.Torus-Regular"
    VENERA_BOLD = "osu.Game.Resources.Fonts.Venera.Venera-Bold"

    def __init__(self):
        if ASSET_DIR is None or FA_DIR is None:
            raise RuntimeError("The full renderer needs separately cleared assets. Set OSU_SIM_ASSET_DIR and "
                               "OSU_SIM_FONT_AWESOME_DIR; this package does not include textures or fonts. "
                               "Use renderer='simple' only for the documented placeholder demonstration.")
        self.glyphs = []        # (sat_off, w, h, kind) kind: 0 alpha-glyph
        self.sat = []
        self.n_sat = 0
        self.index = {}
        self.fonts = {}
        for name in (self.TORUS_BOLD, self.TORUS_SEMI):
            self.fonts[name] = _parse_bmf(ASSET_DIR / f"{name}.bin")
            self._pages(name, [ASSET_DIR / f"{name}_0.png"])
        fa = _parse_bmf(ASSET_DIR / "FontAwesome-Solid.fnt")
        self.fonts["fa"] = fa
        self._pages("fa", [FA_DIR / p for p in fa["page_names"]])
        bold = "0123456789GREATOKMHISX"
        semi = "0123456789SPINERMUT"
        for c in bold:
            self._add_char(self.TORUS_BOLD, ord(c))
        for c in semi:
            self._add_char(self.TORUS_SEMI, ord(c))
        for cp in (0xF054, 0xF105, 0xF101):        # ChevronRight, AngleRight, AngleDoubleRight
            self._add_char("fa", cp)
        edge = _read_rgba(ASSET_DIR / "osu.Game.Resources.Textures.Gameplay.osu.repeat-edge-piece.png")
        self._add_bitmap(("tex", "edge"), edge[..., 3].astype(np.float64) / 255.0 * (edge[..., :3].mean(-1) / 255.0))
        assert self.index[("fa", 0xF054)] == 41 and self.index[("tex", "edge")] == 44      # kernel constants
        # break overlay: ChevronLeft, Venera Bold digits, Torus Regular labels, extra Torus Bold characters
        self._add_char("fa", 0xF053)
        for name in (self.VENERA_BOLD, self.TORUS_REG):
            self.fonts[name] = _parse_bmf(ASSET_DIR / f"{name}.bin")
            self._pages(name, [ASSET_DIR / f"{name}_0.png"])
        for c in "0123456789":
            self._add_char(self.VENERA_BOLD, ord(c))
        for c in "AcuraygGde":
            self._add_char(self.TORUS_REG, ord(c))
        for c in "CUNPBD.%":
            self._add_char(self.TORUS_BOLD, ord(c))
        self.sat = torch.as_tensor(np.concatenate(self.sat).astype(np.float32))
        self.glyph_tab = torch.as_tensor(np.array(self.glyphs, np.float32).reshape(-1, 4))

    def _pages(self, name, paths):
        self.fonts[name]["page_img"] = [None] * len(paths)
        self.fonts[name]["page_paths"] = paths

    def _page(self, name, p):
        f = self.fonts[name]
        if f["page_img"][p] is None:
            f["page_img"][p] = _read_rgba(f["page_paths"][p])
        return f["page_img"][p]

    def _add_bitmap(self, key, a):
        h, w = a.shape
        s = np.zeros((h + 1, w + 1), np.float64)
        s[1:, 1:] = a.cumsum(0).cumsum(1)
        self.index[key] = len(self.glyphs)
        self.glyphs.append((float(self.n_sat), float(w), float(h), 0.0))
        self.sat.append(s.reshape(-1))
        self.n_sat += s.size

    def _add_char(self, font, cp):
        x, y, w, h, xo, yo, xa, pg = self.fonts[font]["chars"][cp]
        img = self._page(font, pg)
        a = img[y:y + h, x:x + w, 3].astype(np.float64) / 255.0
        self._add_bitmap((font, cp), a)

    def char(self, font, cp):
        return self.fonts[font]["chars"][cp]

    def layout(self, font, text, size, spacing=0.0):
        """osu-framework TextBuilder (UseFullGlyphHeight): glyph (index, x0, y0, w, h) in units, text box (W, H)."""
        f = self.fonts[font]
        s = size / 100.0                          # FontStore ScaleAdjust 100
        x = 0.0
        out = []
        prev = None
        for i, ch in enumerate(text):
            cp = ord(ch)
            if ch == " ":
                x += f["chars"][cp][6] * s + (spacing if i > 0 else 0.0)
                prev = cp
                continue
            gx, gy, w, h, xo, yo, xa, pg = f["chars"][cp]
            if i > 0:
                x += f["kern"].get((prev, cp), 0) * s + spacing
            out.append((self.index[(font, cp)], x + xo * s, yo * s, w * s, h * s))
            x += xa * s
            prev = cp
        return out, (x, float(size))


# ------------------------------------------------------------------------------------------- easing
def _clamp01(x):
    return x.clamp(0.0, 1.0)


def e_out_quint(x):
    return 1 + (x - 1) ** 5


def e_out_quad(x):
    return x * (2 - x)


def e_in_quad(x):
    return x * x


def e_in_quint(x):
    return x ** 5


_EH = 2.0 ** -10 * math.sin((0.5 - 0.075) * (2 * math.pi) / 0.3)


def e_out_elastic_half(x):
    return torch.pow(2.0, -10 * x) * torch.sin((0.5 * x - 0.075) * (2 * math.pi) / 0.3) + 1 - _EH * x


def lin(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def srgb(c):
    return torch.where(c < 0.0031308, 12.92 * c, 1.055 * c.clamp(min=0) ** (1 / 2.4) - 0.055)


# ------------------------------------------------------------------------------------------- param layout
P_FIELDS = """
cx cy q R kind
ob_x0 ob_y0 ob_x1 ob_y1 ab_x0 ab_y0 ab_x1 ab_y1 jb_x0 jb_y0 jb_x1 jb_y1 fb_x0 fb_y0 fb_x1 fb_y1
a_ofill of_r of_g of_b
a_ograd r_ograd og_tr og_tg og_tb og_br og_bg og_bb
a_igrad ig_tr ig_tg ig_tb ig_br ig_bg ig_bb a_ifill
a_num num_c num_s num_x0 num_x1 num_x2 num_y a_shadow
a_flash fl_r fl_g fl_b
r_border t_border bd_tr bd_tg bd_tb bd_ta bd_br bd_bg bd_bb bd_ba
a_app r_app app_r app_g app_b
a_body bo_r bo_g bo_b in_r in_g in_b in_a r_out r_bin d0 d1
sl_a sl_start sl_pre span_dur acc_r acc_g acc_b rp_white rp_ir rp_ig rp_ib
a_ball bx by b_cos b_sin ic_sx ic_sy ba_tr ba_tg ba_tb ba_br ba_bg ba_bb
a_fol r_fol fo_tr fo_tg fo_tb fo_br fo_bg fo_bb
j_x j_y j_a j_s j_c j_sn j_dy j_r j_g j_b ex_a ex_p ex_n ex_tr j_s0
fp_t0 fp_t1 fp_sx fp_sy fp_ex fp_ey fp_d fpo_t0 fpo_t1 fpo_sx fpo_sy fpo_ex fpo_ey fpo_d fp_pre
tm_x0 tm_y0
sp_a sp_disc sp_cen sp_csize sp_tick_rot sp_fill_a sp_fill_s sp_arc sp_arc_ir sp_side_bg sp_side_a sp_side_p
""".split()
I_FIELDS = """
gid kind path_off seg_lo seg_hi nest_off nest_cnt nest_n miss_lo miss_hi num_g0 num_g1 num_g2 num_n
j_txt fp_on fpo_on hseed head_hit
""".split()
F_FIELDS = "t cur_x cur_y cur_e br_a br_bw br_lx br_rx br_ay br_sc".split()
PI = {k: i for i, k in enumerate(P_FIELDS)}
II = {k: i for i, k in enumerate(I_FIELDS)}
NP, NI, NF = len(P_FIELDS), len(I_FIELDS), len(F_FIELDS)
for _k, _i in PI.items():
    globals()["P_" + _k.upper()] = tl.constexpr(_i)
for _k, _i in II.items():
    globals()["I_" + _k.upper()] = tl.constexpr(_i)
NEST_W = 8                                       # type time x y span_index span_start cos sin
GLY_W = 4
TXT_MAX = 5


# ------------------------------------------------------------------------------------------- triton helpers
@triton.jit
def _cov(s, nx, ny, h):
    """Exact box (half-width h) coverage of a half-plane at signed distance s (inside > 0), normal (nx, ny)."""
    ax = tl.abs(nx) * h
    ay = tl.abs(ny) * h
    a = tl.maximum(tl.maximum(ax, ay), 1e-6)
    b = tl.maximum(tl.minimum(ax, ay), 1e-6)
    sa = tl.abs(s)
    u = tl.maximum(a + b - sa, 0.0)
    tail = u * u / (8.0 * a * b)
    quad = tl.where(s > 0, 1.0 - tail, tail)
    return tl.where(sa <= a - b, 0.5 + s / (2.0 * a), quad)


@triton.jit
def _disk(dx, dy, rad, h):
    d = tl.sqrt(dx * dx + dy * dy)
    inv = 1.0 / tl.maximum(d, 1e-6)
    nx = tl.where(d > 1e-6, dx * inv, 1.0)
    ny = dy * inv
    return _cov(rad - d, nx, ny, h), d, nx, ny


@triton.jit
def _sat_at(sat_ptr, off, w, u, v, m):
    """Bilinear SAT value at continuous (u, v) (already clamped to [0,w]x[0,h]); w = glyph width in texels."""
    iu = tl.minimum(tl.floor(u), w - 1.0)
    iv = tl.floor(v)
    fu = u - iu
    fv = v - iv
    iv = tl.minimum(iv, tl.maximum(iv, 0.0))
    row = (w + 1.0)
    base = off + iv * row + iu
    b0 = base.to(tl.int32)
    s00 = tl.load(sat_ptr + b0, mask=m, other=0.0)
    s01 = tl.load(sat_ptr + b0 + 1, mask=m, other=0.0)
    s10 = tl.load(sat_ptr + b0 + row.to(tl.int32), mask=m & (fv > 0), other=0.0)
    s11 = tl.load(sat_ptr + b0 + row.to(tl.int32) + 1, mask=m & (fv > 0), other=0.0)
    top = s00 + (s01 - s00) * fu
    bot = s10 + (s11 - s10) * fu
    return top + (bot - top) * fv


@triton.jit
def _glyph(sat_ptr, gly_ptr, gidx, lx, ly, hx, hy, ox, oy, sc):
    """Mean glyph alpha over the footprint [lx-hx, lx+hx] x [ly-hy, ly+hy] (local units); glyph top-left at
    (ox, oy), sc local units per texel."""
    off = tl.load(gly_ptr + gidx * 4 + 0)
    w = tl.load(gly_ptr + gidx * 4 + 1)
    hgt = tl.load(gly_ptr + gidx * 4 + 2)
    u0 = (lx - hx - ox) / sc
    u1 = (lx + hx - ox) / sc
    v0 = (ly - hy - oy) / sc
    v1 = (ly + hy - oy) / sc
    m = (u1 > 0) & (u0 < w) & (v1 > 0) & (v0 < hgt)
    cu0 = tl.minimum(tl.maximum(u0, 0.0), w)
    cu1 = tl.minimum(tl.maximum(u1, 0.0), w)
    cv0 = tl.minimum(tl.maximum(v0, 0.0), hgt)
    cv1 = tl.minimum(tl.maximum(v1, 0.0), hgt)
    s11 = _sat_at(sat_ptr, off, w, cu1, cv1, m)
    s01 = _sat_at(sat_ptr, off, w, cu0, cv1, m)
    s10 = _sat_at(sat_ptr, off, w, cu1, cv0, m)
    s00 = _sat_at(sat_ptr, off, w, cu0, cv0, m)
    area = (u1 - u0) * (v1 - v0)
    return tl.where(m, (s11 - s01 - s10 + s00) / area, 0.0)


@triton.jit
def _ease_out_quint(x):
    y = x - 1.0
    return 1.0 + y * y * y * y * y


@triton.jit
def _ease_out_quad(x):
    return x * (2.0 - x)


@triton.jit
def _ease_elastic_half(x):
    # 2^-10x * sin((0.5x - 0.075) * 2pi / 0.3) + 1 - EH * x
    return tl.exp2(-10.0 * x) * tl.sin((0.5 * x - 0.075) * 20.943951023931955) + 1.0 - 0.0009765625 * 0.8660254037844386 * x


@triton.jit
def _over(cr, cg, cb, sr, sg, sb, a):
    """Normal blend of a straight-alpha source colour (sr,sg,sb clamped to [0,1]) with per-pixel alpha a."""
    sr = tl.minimum(tl.maximum(sr, 0.0), 1.0)
    sg = tl.minimum(tl.maximum(sg, 0.0), 1.0)
    sb = tl.minimum(tl.maximum(sb, 0.0), 1.0)
    return cr + (sr - cr) * a, cg + (sg - cg) * a, cb + (sb - cb) * a


@triton.jit
def _add(cr, cg, cb, sr, sg, sb, a):
    sr = tl.minimum(tl.maximum(sr, 0.0), 1.0)
    sg = tl.minimum(tl.maximum(sg, 0.0), 1.0)
    sb = tl.minimum(tl.maximum(sb, 0.0), 1.0)
    return tl.minimum(cr + sr * a, 1.0), tl.minimum(cg + sg * a, 1.0), tl.minimum(cb + sb * a, 1.0)


@triton.jit
def _ring_layer(cr, cg, cb, dx, dy, rad, thick, gy0, gh, ctr, ctg, ctb, cta, cbr, cbg, cbb, cba, h, ADD: tl.constexpr):
    """Masked circle with transparent content and a border band (RingPiece): band [rad-thick, rad] with the
    1-px masking ramps (border weight ramps in over [rad-thick, rad-thick+AA], alpha out over [rad-AA, rad]).
    Border colour = vertical gradient (top -> bottom over the circle's box), alpha premultiplied in c*a."""
    d = tl.sqrt(dx * dx + dy * dy)
    inv = 1.0 / tl.maximum(d, 1e-6)
    nx = tl.where(d > 1e-6, dx * inv, 1.0)
    ny = dy * inv
    co = _cov(rad - 0.5 * AAc - d, nx, ny, h)
    ci = _cov(rad - thick + 0.5 * AAc - d, nx, ny, h)
    band = tl.maximum(co - ci, 0.0)
    f = tl.minimum(tl.maximum((dy - gy0) / gh, 0.0), 1.0)
    br = ctr + (cbr - ctr) * f
    bg = ctg + (cbg - ctg) * f
    bb = ctb + (cbb - ctb) * f
    ba = cta + (cba - cta) * f
    if ADD:
        cr, cg, cb = _add(cr, cg, cb, br, bg, bb, band * ba)
    else:
        cr, cg, cb = _over(cr, cg, cb, br, bg, bb, band * ba)
    return cr, cg, cb


@triton.jit
def _fill_layer(cr, cg, cb, dx, dy, rad, gy0, gh, ctr, ctg, ctb, cbr, cbg, cbb, a, h):
    """Masked filled circle with a vertical gradient (normal blend)."""
    d = tl.sqrt(dx * dx + dy * dy)
    inv = 1.0 / tl.maximum(d, 1e-6)
    nx = tl.where(d > 1e-6, dx * inv, 1.0)
    ny = dy * inv
    c = _cov(rad - 0.5 * AAc - d, nx, ny, h)
    f = tl.minimum(tl.maximum((dy - gy0) / gh, 0.0), 1.0)
    return _over(cr, cg, cb, ctr + (cbr - ctr) * f, ctg + (cbg - ctg) * f, ctb + (cbb - ctb) * f, c * a)


@triton.jit
def _circle_piece(cr, cg, cb, gx, gy, h, P, sat_ptr, gly_ptr, Ib, num_glyphs: tl.constexpr):
    """ArgonMainCirclePiece layers (outer fill, gradients, inner fill, number+shadow, flash glow, border)."""
    cx = tl.load(P + P_CX)
    cy = tl.load(P + P_CY)
    q = tl.load(P + P_Q)
    dx = gx - cx
    dy = gy - cy
    R = 64.0 * q
    # outer fill (127 units)
    a = tl.load(P + P_A_OFILL)
    if a > 0:
        cr, cg, cb = _fill_layer(cr, cg, cb, dx, dy, 63.5 * q, -R, 2 * R, tl.load(P + P_OF_R), tl.load(P + P_OF_G),
                                 tl.load(P + P_OF_B), tl.load(P + P_OF_R), tl.load(P + P_OF_G), tl.load(P + P_OF_B), a, h)
    a = tl.load(P + P_A_OGRAD)
    if a > 0:
        ro = tl.load(P + P_R_OGRAD)
        cr, cg, cb = _fill_layer(cr, cg, cb, dx, dy, ro, -ro, 2 * ro, tl.load(P + P_OG_TR), tl.load(P + P_OG_TG),
                                 tl.load(P + P_OG_TB), tl.load(P + P_OG_BR), tl.load(P + P_OG_BG), tl.load(P + P_OG_BB), a, h)
    a = tl.load(P + P_A_IGRAD)
    if a > 0:
        ri = 44.137931 * q
        cr, cg, cb = _fill_layer(cr, cg, cb, dx, dy, ri, -ri, 2 * ri, tl.load(P + P_IG_TR), tl.load(P + P_IG_TG),
                                 tl.load(P + P_IG_TB), tl.load(P + P_IG_BR), tl.load(P + P_IG_BG), tl.load(P + P_IG_BB), a, h)
    a = tl.load(P + P_A_IFILL)
    if a > 0:
        rf = 33.103448 * q
        cr, cg, cb = _fill_layer(cr, cg, cb, dx, dy, rf, -rf, 2 * rf, tl.load(P + P_OF_R), tl.load(P + P_OF_G),
                                 tl.load(P + P_OF_B), tl.load(P + P_OF_R), tl.load(P + P_OF_G), tl.load(P + P_OF_B), a, h)
    a = tl.load(P + P_A_NUM)
    if a > 0:
        ns = tl.load(P + P_NUM_S)
        ny0 = tl.load(P + P_NUM_Y)
        nc = tl.load(P + P_NUM_C)
        nn = tl.load(Ib + I_NUM_N)
        sha = tl.load(P + P_A_SHADOW)
        sdy = 3.12 * q
        acc = tl.zeros_like(dx)
        accs = tl.zeros_like(dx)
        for k in tl.static_range(3):
            if k < nn:
                gi = tl.load(Ib + I_NUM_G0 + k)
                ox = tl.load(P + P_NUM_X0 + k)
                gw = tl.load(gly_ptr + gi * 4 + 1)
                al = _glyph(sat_ptr, gly_ptr, gi, dx, dy, h, h, ox, ny0, ns)
                acc = acc + al
                als = _glyph(sat_ptr, gly_ptr, gi, dx, dy - sdy, h, h, ox, ny0, ns)
                accs = accs + als
        cr, cg, cb = _over(cr, cg, cb, 0.0, 0.0, 0.0, tl.minimum(accs, 1.0) * sha)
        cr, cg, cb = _over(cr, cg, cb, nc, nc, nc, tl.minimum(acc, 1.0) * a)
    a = tl.load(P + P_A_FLASH)
    if a > 0:
        d = tl.sqrt(dx * dx + dy * dy)
        g = tl.minimum(tl.maximum((108.8 * q - d) / (76.8 * q), 0.0), 1.0)
        cr, cg, cb = _add(cr, cg, cb, tl.load(P + P_FL_R), tl.load(P + P_FL_G), tl.load(P + P_FL_B), g * g * a)
    rb = tl.load(P + P_R_BORDER)
    tb = tl.load(P + P_T_BORDER)
    cta = tl.load(P + P_BD_TA)
    cba = tl.load(P + P_BD_BA)
    if (cta > 0) | (cba > 0):
        cr, cg, cb = _ring_layer(cr, cg, cb, dx, dy, rb, tb, -rb, 2 * rb, tl.load(P + P_BD_TR), tl.load(P + P_BD_TG),
                                 tl.load(P + P_BD_TB), cta, tl.load(P + P_BD_BR), tl.load(P + P_BD_BG),
                                 tl.load(P + P_BD_BB), cba, h, False)
    return cr, cg, cb


@triton.jit
def _tile_hit(P, X0: tl.constexpr, tx0, ty0, tx1, ty1):
    bx0 = tl.load(P + X0)
    by0 = tl.load(P + X0 + 1)
    bx1 = tl.load(P + X0 + 2)
    by1 = tl.load(P + X0 + 3)
    return (bx0 < tx1) & (bx1 > tx0) & (by0 < ty1) & (by1 > ty0)


@triton.jit
def _render_kernel(out_ptr, P_ptr, I_ptr, F_ptr, path_ptr, chunk_ptr, nest_ptr, sat_ptr, gly_ptr, txt_ptr,
                   glow_ptr, spt_ptr, brt_ptr, mask_ptr, NTILES: tl.constexpr,
                   K: tl.constexpr, NPc: tl.constexpr, NIc: tl.constexpr, NFc: tl.constexpr,
                   TH: tl.constexpr, TW: tl.constexpr, SS: tl.constexpr, TILES_X: tl.constexpr,
                   CHUNK: tl.constexpr, G_CHEV_L: tl.constexpr):
    tile = tl.program_id(0)
    m = tl.program_id(1)
    ty = tile // TILES_X
    tx = tile % TILES_X
    BH: tl.constexpr = TH * SS
    BW: tl.constexpr = TW * SS
    iy = tl.arange(0, BH)[:, None]
    ix = tl.arange(0, BW)[None, :]
    ax = tx * TW + (ix.to(tl.float32) + 0.5) / SS          # actor px (continuous)
    ay = ty * TH + (iy.to(tl.float32) + 0.5) / SS
    gx = (ax - 25.6) * 2.5 + tl.zeros([BH, BW], tl.float32)  # gamefield
    gy = (ay - 22.4) * 2.5 + tl.zeros([BH, BW], tl.float32)
    h = 1.25 / SS                                             # footprint half-width (osu px)
    tx0 = (tx * TW).to(tl.float32)
    ty0 = (ty * TH).to(tl.float32)
    tx1 = tx0 + TW
    ty1 = ty0 + TH
    tg_x0 = (tx0 - 25.6) * 2.5
    tg_y0 = (ty0 - 22.4) * 2.5
    tg_x1 = (tx1 - 25.6) * 2.5
    tg_y1 = (ty1 - 22.4) * 2.5
    cr = tl.zeros([BH, BW], tl.float32)
    cg = tl.zeros([BH, BW], tl.float32)
    cb = tl.zeros([BH, BW], tl.float32)
    Fb = F_ptr + m * NFc
    T = tl.load(Fb + 0)

    Mb = mask_ptr + (m * NTILES + tile) * 6

    # ---------------------------------------------------------------- pass A: spinners (bottom)
    mk = tl.load(Mb + 0)
    while mk != 0:
        j = libdevice.ffs(mk) - 1
        mk = mk & (mk - 1)
        P = P_ptr + (m * K + j) * NPc
        Ib = I_ptr + (m * K + j) * NIc
        cr, cg, cb = _spinner(cr, cg, cb, gx, gy, h, P, sat_ptr, gly_ptr, Ib, glow_ptr, spt_ptr, m,
                              tg_x0, tg_y0, tg_x1, tg_y1)

    # ---------------------------------------------------------------- pass B: follow points (additive)
    mk = tl.load(Mb + 1)
    while mk != 0:
        j = libdevice.ffs(mk) - 1
        mk = mk & (mk - 1)
        P = P_ptr + (m * K + j) * NPc
        Ib = I_ptr + (m * K + j) * NIc
        if tl.load(Ib + I_FP_ON) > 0:
            cr, cg, cb = _follow_points(cr, cg, cb, gx, gy, h, T, P, 0, sat_ptr, gly_ptr, tg_x0, tg_y0, tg_x1, tg_y1)
        if tl.load(Ib + I_FPO_ON) > 0:
            cr, cg, cb = _follow_points(cr, cg, cb, gx, gy, h, T, P, 7, sat_ptr, gly_ptr, tg_x0, tg_y0, tg_x1, tg_y1)

    # ---------------------------------------------------------------- pass C: ring explosions (judgement layer)
    mk = tl.load(Mb + 2)
    while mk != 0:
        j = libdevice.ffs(mk) - 1
        mk = mk & (mk - 1)
        P = P_ptr + (m * K + j) * NPc
        Ib = I_ptr + (m * K + j) * NIc
        cr, cg, cb = _explosion(cr, cg, cb, gx, gy, h, P, Ib)

    # ---------------------------------------------------------------- pass D: hit objects, latest first
    mk = tl.load(Mb + 3)
    while mk != 0:
        j = 31 - libdevice.clz(mk)
        mk = mk & ~(1 << j)
        P = P_ptr + (m * K + j) * NPc
        Ib = I_ptr + (m * K + j) * NIc
        kind = tl.load(Ib + I_KIND)
        if kind == 1:
            cr, cg, cb = _slider_body(cr, cg, cb, gx, gy, h, P, Ib, path_ptr, chunk_ptr, tg_x0, tg_y0,
                                      tg_x1, tg_y1, CHUNK)
            cr, cg, cb = _slider_nested(cr, cg, cb, gx, gy, h, T, P, Ib, nest_ptr, sat_ptr, gly_ptr,
                                        tg_x0, tg_y0, tg_x1, tg_y1)
        cr, cg, cb = _circle_piece(cr, cg, cb, gx, gy, h, P, sat_ptr, gly_ptr, Ib, 3)
        if kind == 1:
            cr, cg, cb = _ball(cr, cg, cb, gx, gy, h, P, sat_ptr, gly_ptr)

    # ---------------------------------------------------------------- pass E: judgement text + tick-miss popups
    mk = tl.load(Mb + 4)
    while mk != 0:
        j = libdevice.ffs(mk) - 1
        mk = mk & (mk - 1)
        P = P_ptr + (m * K + j) * NPc
        Ib = I_ptr + (m * K + j) * NIc
        if tl.load(P + P_J_A) > 0:
            cr, cg, cb = _judgement_text(cr, cg, cb, gx, gy, h, P, Ib, sat_ptr, gly_ptr, txt_ptr)
        if tl.load(Ib + I_KIND) == 1:
            cr, cg, cb = _tick_misses(cr, cg, cb, gx, gy, h, T, P, Ib, nest_ptr, tg_x0, tg_y0, tg_x1, tg_y1)

    # ---------------------------------------------------------------- pass F: approach circles
    mk = tl.load(Mb + 5)
    while mk != 0:
        j = libdevice.ffs(mk) - 1
        mk = mk & (mk - 1)
        P = P_ptr + (m * K + j) * NPc
        a = tl.load(P + P_A_APP)
        cx = tl.load(P + P_CX)
        cy = tl.load(P + P_CY)
        ra = tl.load(P + P_R_APP)
        dx = gx - cx
        dy = gy - cy
        d = tl.sqrt(dx * dx + dy * dy)
        inv = 1.0 / tl.maximum(d, 1e-6)
        nx = tl.where(d > 1e-6, dx * inv, 1.0)
        ny = dy * inv
        band = _cov(ra * 1.0670625 - d, nx, ny, h) - _cov(ra * 0.9967031 - d, nx, ny, h)
        cr, cg, cb = _over(cr, cg, cb, tl.load(P + P_APP_R), tl.load(P + P_APP_G), tl.load(P + P_APP_B),
                           tl.maximum(band, 0.0) * a)

    # ---------------------------------------------------------------- break overlay (above playfield, below cursor)
    cr, cg, cb = _break_overlay(cr, cg, cb, gx, gy, h, Fb, brt_ptr, m, sat_ptr, gly_ptr, G_CHEV_L)

    # ---------------------------------------------------------------- pass G: cursor
    ccx = tl.load(Fb + 1)
    ccy = tl.load(Fb + 2)
    if (ccx + 26.0 > tg_x0) & (ccx - 26.0 < tg_x1) & (ccy + 26.0 > tg_y0) & (ccy - 26.0 < tg_y1):
        # 2x2 sub-footprints: the cursor stacks several partial layers inside one actor pixel
        ar_ = tl.zeros_like(cr)
        ag_ = tl.zeros_like(cr)
        ab_ = tl.zeros_like(cr)
        for sub in tl.static_range(4):
            ox_ = ((sub % 2) - 0.5) * h
            oy_ = ((sub // 2) - 0.5) * h
            r_, g_, b_ = _cursor(cr, cg, cb, gx + ox_, gy + oy_, 0.5 * h, Fb)
            ar_ += r_
            ag_ += g_
            ab_ += b_
        cr = ar_ * 0.25
        cg = ag_ * 0.25
        cb = ab_ * 0.25

    # ---------------------------------------------------------------- resolve SS and store
    if SS > 1:
        cr = tl.sum(tl.sum(tl.reshape(cr, (TH, SS, TW, SS)), axis=3), axis=1) / (SS * SS)
        cg = tl.sum(tl.sum(tl.reshape(cg, (TH, SS, TW, SS)), axis=3), axis=1) / (SS * SS)
        cb = tl.sum(tl.sum(tl.reshape(cb, (TH, SS, TW, SS)), axis=3), axis=1) / (SS * SS)
    oy = ty * TH + tl.arange(0, TH)[:, None]
    ox = tx * TW + tl.arange(0, TW)[None, :]
    base = out_ptr + m * (3 * 192 * 256) + oy * 256 + ox
    tl.store(base, (tl.minimum(tl.maximum(cr, 0.0), 1.0) * 255.0 + 0.5).to(tl.uint8))
    tl.store(base + 192 * 256, (tl.minimum(tl.maximum(cg, 0.0), 1.0) * 255.0 + 0.5).to(tl.uint8))
    tl.store(base + 2 * 192 * 256, (tl.minimum(tl.maximum(cb, 0.0), 1.0) * 255.0 + 0.5).to(tl.uint8))


@triton.jit
def _seg_dist2(gx, gy, ax_, ay_, bx_, by_):
    vx = bx_ - ax_
    vy = by_ - ay_
    l2 = tl.maximum(vx * vx + vy * vy, 1e-12)
    t = tl.minimum(tl.maximum(((gx - ax_) * vx + (gy - ay_) * vy) / l2, 0.0), 1.0)
    ex = gx - ax_ - t * vx
    ey = gy - ay_ - t * vy
    return ex * ex + ey * ey, ex, ey


@triton.jit
def _slider_body(cr, cg, cb, gx, gy, h, P, Ib, path_ptr, chunk_ptr, tg_x0, tg_y0, tg_x1, tg_y1, CHUNK: tl.constexpr):
    a = tl.load(P + P_A_BODY)
    if a > 0:
        poff = tl.load(Ib + I_PATH_OFF)
        lo = tl.load(Ib + I_SEG_LO)
        hi = tl.load(Ib + I_SEG_HI)
        d0 = tl.load(P + P_D0)
        d1 = tl.load(P + P_D1)
        rout = tl.load(P + P_R_OUT)
        marg = rout + 3.0
        best = tl.full(gx.shape, 1e12, tl.float32)
        bex = tl.zeros_like(gx) + 1.0
        bey = tl.zeros_like(gx)
        seg = lo
        while seg < hi:
            c0 = poff + ((seg - poff) // CHUNK) * CHUNK
            cend = tl.minimum(c0 + CHUNK, hi)
            bx0 = tl.load(chunk_ptr + c0 * 4 + 0)
            by0 = tl.load(chunk_ptr + c0 * 4 + 1)
            bx1 = tl.load(chunk_ptr + c0 * 4 + 2)
            by1 = tl.load(chunk_ptr + c0 * 4 + 3)
            if (bx0 - marg < tg_x1) & (bx1 + marg > tg_x0) & (by0 - marg < tg_y1) & (by1 + marg > tg_y0):
                for k in range(seg, cend):
                    ax_ = tl.load(path_ptr + k * 3 + 0)
                    ay_ = tl.load(path_ptr + k * 3 + 1)
                    ac = tl.load(path_ptr + k * 3 + 2)
                    bx_ = tl.load(path_ptr + k * 3 + 3)
                    by_ = tl.load(path_ptr + k * 3 + 4)
                    bc = tl.load(path_ptr + k * 3 + 5)
                    span = tl.maximum(bc - ac, 1e-9)
                    if ac < d0:
                        w = (d0 - ac) / span
                        ax_n = ax_ + (bx_ - ax_) * w
                        ay_n = ay_ + (by_ - ay_) * w
                    else:
                        ax_n = ax_
                        ay_n = ay_
                    if bc > d1:
                        w2 = (d1 - ac) / span
                        bx_n = ax_ + (bx_ - ax_) * w2
                        by_n = ay_ + (by_ - ay_) * w2
                    else:
                        bx_n = bx_
                        by_n = by_
                    d2, ex, ey = _seg_dist2(gx, gy, ax_n, ay_n, bx_n, by_n)
                    upd = d2 < best
                    best = tl.where(upd, d2, best)
                    bex = tl.where(upd, ex, bex)
                    bey = tl.where(upd, ey, bey)
            seg = cend
        d = tl.sqrt(best)
        inv = 1.0 / tl.maximum(d, 1e-6)
        nx = tl.where(d > 1e-6, bex * inv, 1.0)
        ny = bey * inv
        co = _cov(rout - d, nx, ny, h)
        ci = _cov(tl.load(P + P_R_BIN) - d, nx, ny, h)
        band = tl.maximum(co - ci, 0.0)
        ia = tl.load(P + P_IN_A)
        pr = tl.minimum(tl.load(P + P_BO_R), 1.0) * band + tl.minimum(tl.load(P + P_IN_R), 1.0) * ia * ci
        pg = tl.minimum(tl.load(P + P_BO_G), 1.0) * band + tl.minimum(tl.load(P + P_IN_G), 1.0) * ia * ci
        pb = tl.minimum(tl.load(P + P_BO_B), 1.0) * band + tl.minimum(tl.load(P + P_IN_B), 1.0) * ia * ci
        al = (band + ia * ci) * a
        cr = cr * (1.0 - al) + pr * a
        cg = cg * (1.0 - al) + pg * a
        cb = cb * (1.0 - al) + pb * a
    return cr, cg, cb


@triton.jit
def _slider_nested(cr, cg, cb, gx, gy, h, T, P, Ib, nest_ptr, sat_ptr, gly_ptr, tg_x0, tg_y0, tg_x1, tg_y1):
    """Ticks (ArgonSliderScorePoint) and reverse arrows (ArgonReverseArrow), in container order."""
    sla = tl.load(P + P_SL_A)
    if sla > 0:
        noff = tl.load(Ib + I_NEST_OFF)
        ncnt = tl.load(Ib + I_NEST_CNT)
        nn = tl.load(Ib + I_NEST_N)
        mlo = tl.load(Ib + I_MISS_LO)
        mhi = tl.load(Ib + I_MISS_HI)
        q = tl.load(P + P_Q)
        s0 = tl.load(P + P_SL_START)
        pre = tl.load(P + P_SL_PRE)
        ar = tl.load(P + P_ACC_R)
        ag = tl.load(P + P_ACC_G)
        ab = tl.load(P + P_ACC_B)
        # ---- ticks (tickContainer is drawn before repeatContainer)
        for i in range(ncnt):
            E = nest_ptr + (noff + i) * 8
            typ = tl.load(E + 0)
            if typ == 0:
                nt = tl.load(E + 1)
                nx_ = tl.load(E + 2)
                ny_ = tl.load(E + 3)
                si = tl.load(E + 4)
                sst = tl.load(E + 5)
                tpre = (nt - sst) * 0.5 + tl.where(si > 0, 200.0, pre * 0.66)
                tstart = nt - tpre
                if (T >= tstart) & (nx_ + 12.0 > tg_x0) & (nx_ - 12.0 < tg_x1) & (ny_ + 12.0 > tg_y0) & (ny_ - 12.0 < tg_y1):
                    pin = tl.minimum((T - tstart) / 150.0, 1.0)
                    al = pin
                    sc = 0.5 + 0.5 * _ease_elastic_half(tl.minimum((T - tstart) / 600.0, 1.0))
                    judged = i < nn
                    if judged:
                        bit = tl.where(i < 32, (mlo >> (i % 32)) & 1, (mhi >> (i % 32)) & 1)
                        pj = tl.minimum(tl.maximum((T - nt) / 150.0, 0.0), 1.0)
                        al = al * (1.0 - _ease_out_quint(pj))
                        if bit == 0:
                            sc_j = 0.5 + 0.5 * _ease_elastic_half(tl.minimum(tl.maximum((nt - tstart) / 600.0, 0.0), 1.0))
                            sc = sc_j + (sc_j * 1.5 - sc_j) * _ease_out_quad(pj)
                    if al > 0:
                        rad = 6.0 * q * sc
                        cr, cg, cb = _ring_layer(cr, cg, cb, gx - nx_, gy - ny_, rad, 3.0 * q * sc, -rad, 2 * rad,
                                                 ar, ag, ab, al * sla, ar, ag, ab, al * sla, h, False)
        # ---- reverse arrows
        for i in range(ncnt):
            E = nest_ptr + (noff + i) * 8
            typ = tl.load(E + 0)
            if typ == 1:
                nt = tl.load(E + 1)
                nx_ = tl.load(E + 2)
                ny_ = tl.load(E + 3)
                si = tl.load(E + 4)
                spd = tl.load(P + P_SPAN_DUR)
                # first repeat: visible with the slider, fade-in delayed by preempt/3 (snaking); later: 2 spans early
                first = si == 0
                life0 = tl.where(first, s0 - pre, nt - 2.0 * spd)
                fin0 = tl.where(first, s0 - pre + pre / 3.0, life0)
                fdur = tl.where(first, 150.0, tl.minimum(spd, 150.0))
                if (T >= life0) & (nx_ + 70.0 * q > tg_x0) & (nx_ - 70.0 * q < tg_x1) & (ny_ + 70.0 * q > tg_y0) & (ny_ - 70.0 * q < tg_y1):
                    al = tl.minimum(tl.maximum((T - fin0) / tl.maximum(fdur, 1e-3), 0.0), 1.0)
                    hs = 1.0
                    anim = tl.minimum(300.0, spd)
                    judged = i < nn
                    hitr = False
                    if judged:
                        bit = tl.where(i < 32, (mlo >> (i % 32)) & 1, (mhi >> (i % 32)) & 1)
                        pj = tl.minimum(tl.maximum((T - nt) / tl.maximum(anim, 1e-3), 0.0), 1.0)
                        if bit == 0:
                            hitr = T >= nt
                            al = al * (1.0 - _ease_out_quad(pj))
                        else:
                            al = al * (1.0 - pj)
                    if hitr:
                        hs = 1.0 + 0.5 * _ease_out_quad(tl.minimum(tl.maximum((T - nt) / tl.maximum(anim, 1e-3), 0.0), 1.0))
                    # pulse (300 ms loop from AnimationStartTime); frozen at the hit time once hit
                    lp = tl.where(hitr, nt, T) - life0
                    lp = lp - tl.floor(lp / 300.0) * 300.0
                    e1 = _ease_out_quad(tl.minimum(lp / 35.0, 1.0))
                    e2 = _ease_out_quad(tl.minimum(tl.maximum((lp - 35.0) / 250.0, 0.0), 1.0))
                    msc = tl.where(lp < 35.0, 1.0 + 0.3 * e1, 1.3 - 0.3 * e2)
                    sidex = tl.where(lp < 35.0, -12.0 * e1, -12.0 + 12.0 * e2)
                    if al > 0:
                        cr, cg, cb = _reverse_arrow(cr, cg, cb, gx, gy, h, P, nx_, ny_, tl.load(E + 6), tl.load(E + 7),
                                                    q * hs, msc, sidex, al * sla, sat_ptr, gly_ptr)
    return cr, cg, cb


@triton.jit
def _reverse_arrow(cr, cg, cb, gx, gy, h, P, px, py, ca, sa, s, msc, sidex, a, sat_ptr, gly_ptr):
    # local frame: rotate the pixel offset into the arrow frame (x along the arrow direction), units = osu/s
    dx = gx - px
    dy = gy - py
    lx = (dx * ca + dy * sa) / s
    ly = (-dx * sa + dy * ca) / s
    hl = h / s
    wht = tl.load(P + P_RP_WHITE)
    # main (scaled by msc): white pill 40x20 (stadium) + AngleDoubleRight icon 16x16, colour accent.Darken(4)
    mx = lx / msc
    my = ly / msc
    hm = hl / msc
    qx = tl.maximum(tl.abs(mx) - 10.0, 0.0)
    dd = tl.sqrt(qx * qx + my * my)
    inv = 1.0 / tl.maximum(dd, 1e-6)
    nx = tl.where(dd > 1e-6, qx * inv, 0.0)
    ny = tl.where(dd > 1e-6, my * inv, 1.0)
    cov = _cov(10.0 - 0.5 * AAc / s / msc - dd, nx, ny, hm)
    cr, cg, cb = _over(cr, cg, cb, wht, wht, wht, cov * a)
    gidx = 22 + 19 + 2                   # AngleDoubleRight (see Assets order)
    gw = tl.load(gly_ptr + gidx * 4 + 1)
    gh = tl.load(gly_ptr + gidx * 4 + 2)
    sc = tl.minimum(16.0 / gw, 16.0 / gh)
    al = _glyph(sat_ptr, gly_ptr, gidx, mx, my, hm, hm, -gw * sc * 0.5, -gh * sc * 0.5, sc)
    cr, cg, cb = _over(cr, cg, cb, tl.load(P + P_RP_IR), tl.load(P + P_RP_IG), tl.load(P + P_RP_IB), al * a)
    # side: repeat-edge-piece texture 110.345 units, offset by sidex along the arrow x axis
    eidx = 22 + 19 + 3
    ew = tl.load(gly_ptr + eidx * 4 + 1)
    esc = 110.34483 / ew
    al2 = _glyph(sat_ptr, gly_ptr, eidx, lx - sidex, ly, hl, hl, -55.172415, -55.172415, esc)
    cr, cg, cb = _over(cr, cg, cb, wht, wht, wht, al2 * a)
    return cr, cg, cb


@triton.jit
def _ball(cr, cg, cb, gx, gy, h, P, sat_ptr, gly_ptr):
    q = tl.load(P + P_Q)
    bx = tl.load(P + P_BX)
    by = tl.load(P + P_BY)
    dx = gx - bx
    dy = gy - by
    # follow circle (additive): border 4 units * scale, fill 0.3 alpha, vertical gradients
    af = tl.load(P + P_A_FOL)
    if af > 0:
        rf = tl.load(P + P_R_FOL)
        d = tl.sqrt(dx * dx + dy * dy)
        inv = 1.0 / tl.maximum(d, 1e-6)
        nx = tl.where(d > 1e-6, dx * inv, 1.0)
        ny = dy * inv
        th = rf / 16.0                      # 4 / 64 of the follow radius
        co = _cov(rf - 0.5 * AAc - d, nx, ny, h)
        ci = _cov(rf - th + 0.5 * AAc - d, nx, ny, h)
        band = tl.maximum(co - ci, 0.0)
        f = tl.minimum(tl.maximum((dy + rf) / (2.0 * rf), 0.0), 1.0)
        fr = tl.load(P + P_FO_TR) + (tl.load(P + P_FO_BR) - tl.load(P + P_FO_TR)) * f
        fg = tl.load(P + P_FO_TG) + (tl.load(P + P_FO_BG) - tl.load(P + P_FO_TG)) * f
        fb = tl.load(P + P_FO_TB) + (tl.load(P + P_FO_BB) - tl.load(P + P_FO_TB)) * f
        w = (band + 0.3 * ci) * af
        cr, cg, cb = _add(cr, cg, cb, fr, fg, fb, w)
    ab = tl.load(P + P_A_BALL)
    if ab > 0:
        rb = 55.172415 * q
        d = tl.sqrt(dx * dx + dy * dy)
        inv = 1.0 / tl.maximum(d, 1e-6)
        nx = tl.where(d > 1e-6, dx * inv, 1.0)
        ny = dy * inv
        co = _cov(rb - 0.5 * AAc - d, nx, ny, h)
        ci = _cov(rb - 11.034483 * q + 0.5 * AAc - d, nx, ny, h)
        band = tl.maximum(co - ci, 0.0)
        f = tl.minimum(tl.maximum((dy + rb) / (2.0 * rb), 0.0), 1.0)
        fr = tl.minimum(tl.load(P + P_BA_TR) + (tl.load(P + P_BA_BR) - tl.load(P + P_BA_TR)) * f, 1.0)
        fg = tl.minimum(tl.load(P + P_BA_TG) + (tl.load(P + P_BA_BG) - tl.load(P + P_BA_TG)) * f, 1.0)
        fb = tl.minimum(tl.load(P + P_BA_TB) + (tl.load(P + P_BA_BB) - tl.load(P + P_BA_TB)) * f, 1.0)
        al = (band + ci) * ab
        cr = cr * (1.0 - al) + (band + fr * ci) * ab
        cg = cg * (1.0 - al) + (band + fg * ci) * ab
        cb = cb * (1.0 - al) + (band + fb * ci) * ab
        # icon AngleRight 48x48 (fit), scale (sx, sy), rotated with the ball
        bc = tl.load(P + P_B_COS)
        bs = tl.load(P + P_B_SIN)
        isx = tl.load(P + P_IC_SX)
        isy = tl.load(P + P_IC_SY)
        if isx > 0.01:
            lx = (dx * bc + dy * bs) / q
            ly = (-dx * bs + dy * bc) / q
            gidx = 22 + 19 + 1
            gw = tl.load(gly_ptr + gidx * 4 + 1)
            gh = tl.load(gly_ptr + gidx * 4 + 2)
            sc = tl.minimum(48.0 / gw, 48.0 / gh)
            al2 = _glyph(sat_ptr, gly_ptr, gidx, lx / isx, ly / isy, h / q / isx, h / q / isy,
                         -gw * sc * 0.5, -gh * sc * 0.5, sc)
            cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, al2 * ab)
    return cr, cg, cb


@triton.jit
def _explosion(cr, cg, cb, gx, gy, h, P, Ib):
    q = tl.load(P + P_Q)
    jx = tl.load(P + P_J_X)
    jy = tl.load(P + P_J_Y)
    exa = tl.load(P + P_EX_A)
    exp_ = tl.load(P + P_EX_P)
    n = tl.load(P + P_EX_N).to(tl.int32)
    trav = tl.load(P + P_EX_TR)
    seed = tl.load(Ib + I_HSEED)
    rr = tl.load(P + P_J_R)
    rg = tl.load(P + P_J_G)
    rb = tl.load(P + P_J_B)
    for i in range(n):
        # deterministic pseudo-random direction / distance (the host uses a seeded RNG stream)
        hsh = (seed * 1103515245 + i * 12345 + 7) & 0x7FFFFFFF
        hsh = (hsh ^ (hsh >> 13)) * 1274126177 & 0x7FFFFFFF
        u1 = (hsh & 0xFFFF).to(tl.float32) / 65536.0
        u2 = ((hsh >> 16) & 0x7FFF).to(tl.float32) / 32768.0
        ang = u1 * 6.283185307
        dist = trav * (0.5 + 0.5 * u2) * exp_
        size = tl.where(i < 4, 9.0, 14.0)
        rx = jx + tl.cos(ang) * dist * q
        ry = jy + tl.sin(ang) * dist * q
        rad = size * 0.5 * q
        cr, cg, cb = _ring_layer(cr, cg, cb, gx - rx, gy - ry, rad, 4.0 * q, -rad, 2 * rad, rr, rg, rb, exa,
                                 rr, rg, rb, exa, h, True)
    return cr, cg, cb


@triton.jit
def _judgement_text(cr, cg, cb, gx, gy, h, P, Ib, sat_ptr, gly_ptr, txt_ptr):
    q = tl.load(P + P_Q)
    jx = tl.load(P + P_J_X)
    jy = tl.load(P + P_J_Y) + tl.load(P + P_J_DY)
    a = tl.load(P + P_J_A)
    s = tl.load(P + P_J_S) * q
    c = tl.load(P + P_J_C)
    sn = tl.load(P + P_J_SN)
    dx = gx - jx
    dy = gy - jy
    lx = (dx * c + dy * sn) / s
    ly = (-dx * sn + dy * c) / s
    hl = h / s
    txt = tl.load(Ib + I_J_TXT)
    acc = tl.zeros_like(gx)
    for k in tl.static_range(5):
        T_ = txt_ptr + (txt * 5 + k) * 4
        gi = tl.load(T_ + 0)
        if gi >= 0:
            acc = acc + _glyph(sat_ptr, gly_ptr, gi.to(tl.int32), lx, ly, hl, hl, tl.load(T_ + 1), tl.load(T_ + 2),
                               0.2)
    cr, cg, cb = _add(cr, cg, cb, tl.load(P + P_J_R), tl.load(P + P_J_G), tl.load(P + P_J_B), tl.minimum(acc, 1.0) * a)
    return cr, cg, cb


@triton.jit
def _tick_misses(cr, cg, cb, gx, gy, h, T, P, Ib, nest_ptr, tg_x0, tg_y0, tg_x1, tg_y1):
    noff = tl.load(Ib + I_NEST_OFF)
    ncnt = tl.load(Ib + I_NEST_CNT)
    nn = tl.load(Ib + I_NEST_N)
    mlo = tl.load(Ib + I_MISS_LO)
    mhi = tl.load(Ib + I_MISS_HI)
    q = tl.load(P + P_Q)
    if (mlo != 0) | (mhi != 0):
        for i in range(tl.minimum(nn, 64)):
            bit = tl.where(i < 32, (mlo >> (i % 32)) & 1, (mhi >> (i % 32)) & 1)
            if bit != 0:
                E = nest_ptr + (noff + i) * 8
                typ = tl.load(E + 0)
                nt = tl.load(E + 1)
                nx_ = tl.load(E + 2)
                ny_ = tl.load(E + 3)
                dt = T - nt
                if (dt >= 0) & (dt < 600.0) & (nx_ + 10.0 > tg_x0) & (nx_ - 10.0 < tg_x1) & (ny_ + 10.0 > tg_y0) & (ny_ - 10.0 < tg_y1):
                    sc = 1.4 - 0.4 * _ease_out_quad(tl.minimum(dt / 150.0, 1.0))
                    al = 1.0 - dt / 600.0
                    rad = 6.0 * q * sc
                    c0 = tl.where(typ == 2, 0.5019608, 0.92941177)
                    c1 = tl.where(typ == 2, 0.5019608, 0.06666667)
                    c2 = tl.where(typ == 2, 0.5019608, 0.12941177)
                    dx = gx - nx_
                    dy = gy - ny_
                    cv, d, nx, ny = _disk(dx, dy, rad - 0.5 * AAc, h)
                    cr, cg, cb = _add(cr, cg, cb, c0, c1, c2, cv * al)
    return cr, cg, cb


@triton.jit
def _fp_chevron(cr, cg, cb, lx, ly, hl, a, sat_ptr, gly_ptr):
    gidx = 22 + 19 + 0                    # ChevronRight
    gw = tl.load(gly_ptr + gidx * 4 + 1)
    gh = tl.load(gly_ptr + gidx * 4 + 2)
    sc = tl.minimum(8.0 / gw, 8.0 / gh)
    w8 = gw * sc
    # container 12x8 centred: icon 1 (Gray 0.2 tint) at x in [-6, 2], icon 2 at [-2, 6]
    al1 = _glyph(sat_ptr, gly_ptr, gidx, lx, ly, hl, hl, -6.0 + (8.0 - w8) * 0.5, -gh * sc * 0.5, sc)
    al2 = _glyph(sat_ptr, gly_ptr, gidx, lx, ly, hl, hl, -2.0 + (8.0 - w8) * 0.5, -gh * sc * 0.5, sc)
    # vertical gradient FC618F -> BB1A41 over the container height (8 units)
    f = tl.minimum(tl.maximum((ly + 4.0) / 8.0, 0.0), 1.0)
    r = 0.98823529 + (0.73333333 - 0.98823529) * f
    g = 0.38039216 + (0.10196078 - 0.38039216) * f
    b = 0.56078431 + (0.25490196 - 0.56078431) * f
    # icon 1: linear-light product with Gray(0.2) -> ~0.0331 * lin(c) (applied approximately in sRGB)
    cr, cg, cb = _add(cr, cg, cb, r * 0.2, g * 0.2, b * 0.2, al1 * a)
    cr, cg, cb = _add(cr, cg, cb, r, g, b, al2 * a)
    return cr, cg, cb


@triton.jit
def _follow_points(cr, cg, cb, gx, gy, h, T, P, OFF: tl.constexpr, sat_ptr, gly_ptr, tg_x0, tg_y0, tg_x1, tg_y1):
    t0 = tl.load(P + P_FP_T0 + OFF)
    t1 = tl.load(P + P_FP_T1 + OFF)
    sx = tl.load(P + P_FP_SX + OFF)
    sy = tl.load(P + P_FP_SY + OFF)
    ex = tl.load(P + P_FP_EX + OFF)
    ey = tl.load(P + P_FP_EY + OFF)
    dist = tl.load(P + P_FP_D + OFF)
    pre = tl.load(P + P_FP_PRE)
    q = tl.load(P + P_Q)
    vx = ex - sx
    vy = ey - sy
    ln = tl.sqrt(vx * vx + vy * vy)
    ca = vx / tl.maximum(ln, 1e-6)
    sa = vy / tl.maximum(ln, 1e-6)
    n_pts = tl.maximum(((dist - 32.0 - 48.0) / 32.0).to(tl.int32) + 2, 0)
    for k in range(n_pts):
        d = 48.0 + 32.0 * k
        if d < dist - 32.0:
            fr = d / dist
            fout = t0 + fr * (t1 - t0)
            fin = fout - pre
            if (T >= fin) & (T < fout + 400.0):
                p = tl.minimum((T - fin) / 400.0, 1.0)
                eo = _ease_out_quad(p)
                al = tl.where(T < fout, p, 1.0 - (T - fout) / 400.0)
                ff = fr - 0.1 + 0.1 * eo
                px = sx + ff * vx
                py = sy + ff * vy
                sc = q * (1.5 - 0.5 * eo)
                if (px + 10.0 > tg_x0) & (px - 10.0 < tg_x1) & (py + 10.0 > tg_y0) & (py - 10.0 < tg_y1):
                    dx = gx - px
                    dy = gy - py
                    lx = (dx * ca + dy * sa) / sc
                    ly = (-dx * sa + dy * ca) / sc
                    cr, cg, cb = _fp_chevron(cr, cg, cb, lx, ly, h / sc, al, sat_ptr, gly_ptr)
    return cr, cg, cb


@triton.jit
def _cursor(cr, cg, cb, gx, gy, h, Fb):
    cx = tl.load(Fb + 1)
    cy = tl.load(Fb + 2)
    e = tl.load(Fb + 3)
    dx = gx - cx
    dy = gy - cy
    R = 14.0 * e
    # ExpandTarget: fill FC618F.Darken(0.6) @0.4, border 6 (gradient FC618F -> BB1A41), inner border 2 white@0.8
    d = tl.sqrt(dx * dx + dy * dy)
    inv = 1.0 / tl.maximum(d, 1e-6)
    nx = tl.where(d > 1e-6, dx * inv, 1.0)
    ny = dy * inv
    co = _cov(R - 0.5 * AAc - d, nx, ny, h)
    ci = _cov(R - 6.0 * e + 0.5 * AAc - d, nx, ny, h)
    band = tl.maximum(co - ci, 0.0)
    f = tl.minimum(tl.maximum((dy + R) / (2.0 * R), 0.0), 1.0)
    br = 0.98823529 + (0.73333333 - 0.98823529) * f
    bg = 0.38039216 + (0.10196078 - 0.38039216) * f
    bb = 0.56078431 + (0.25490196 - 0.56078431) * f
    al = band + 0.4 * ci
    cr = cr * (1.0 - al) + br * band + 0.61764706 * 0.4 * ci
    cg = cg * (1.0 - al) + bg * band + 0.23774510 * 0.4 * ci
    cb = cb * (1.0 - al) + bb * band + 0.35049020 * 0.4 * ci
    ci2 = _cov(R - 2.0 * e + 0.5 * AAc - d, nx, ny, h)
    band2 = tl.maximum(co - ci2, 0.0)
    cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, band2 * 0.8)
    # centre dot: glow (171,255,255,100) radius 20 local (x0.2 scale -> 4 units) around r=2.8, then white circle
    gl = tl.minimum(tl.maximum((6.8 - d) / 4.0, 0.0), 1.0)
    cr, cg, cb = _add(cr, cg, cb, 0.67058824, 1.0, 1.0, gl * gl * 0.39215686)
    cdot = _cov(2.8 - 0.5 * AAc - d, nx, ny, h)
    cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, cdot)
    return cr, cg, cb


@triton.jit
def _arc_dist(dx, dy, d, rc, hth, c, half):
    """Signed distance (units, < 0 inside) to a CircularProgress band: centreline radius rc, half thickness hth,
    centred at angle c (radians, 0 = up, clockwise), half-angle `half`, rounded caps."""
    th = libdevice.atan2(dx, -dy)
    rel = th - c
    rel = rel - 6.283185307 * tl.floor((rel + 3.14159265) / 6.283185307)
    inside = tl.abs(rel) <= half
    band = tl.abs(d - rc) - hth
    c1 = c + half
    c2 = c - half
    e1x = dx - tl.sin(c1) * rc
    e1y = dy + tl.cos(c1) * rc
    e2x = dx - tl.sin(c2) * rc
    e2y = dy + tl.cos(c2) * rc
    cap = tl.sqrt(tl.minimum(e1x * e1x + e1y * e1y, e2x * e2x + e2y * e2y)) - hth
    return tl.where(inside, band, cap)


@triton.jit
def _spinner(cr, cg, cb, gx, gy, h, P, sat_ptr, gly_ptr, Ib, glow_ptr, spt_ptr, m, tg_x0, tg_y0, tg_x1, tg_y1):
    a = tl.load(P + P_SP_A)
    if a > 0:
        dx = gx - 256.0
        dy = gy - 192.0
        d = tl.sqrt(dx * dx + dy * dy)
        inv = 1.0 / tl.maximum(d, 1e-6)
        nx = tl.where(d > 1e-6, dx * inv, 1.0)
        ny = dy * inv
        disc = tl.load(P + P_SP_DISC)
        if disc > 0.001:
            # ---- fill: pink EdgeEffect.Shadow (radius 40 local) of a circle of radius 184*fill_scale
            fs = tl.load(P + P_SP_FILL_S)
            fa = tl.load(P + P_SP_FILL_A)
            rf = 184.0 * fs * disc
            gr = 40.0 * fs * disc
            sh = tl.minimum(tl.maximum((rf + gr - d) / gr, 0.0), 1.0)
            cr, cg, cb = _over(cr, cg, cb, 0.98823529, 0.38039216, 0.56078431, sh * sh * fa * a)
            # ---- ticks: 25 stadiums 30x5 (radius 2.5) at r=138, border 2, white shadow 0.2 radius 30
            rot = tl.load(P + P_SP_TICK_ROT)
            ang = libdevice.atan2(dx, dy) * 57.29577951
            rel = (ang + rot) / 14.4
            k0 = tl.floor(rel + 0.5)
            hl = h / disc
            aal = 0.5 * AAc / disc
            for kk in tl.static_range(5):
                k = k0 + (kk - 2)
                ka = (k * 14.4 - rot) * 0.01745329252
                tcx = tl.sin(ka) * 138.0 * disc
                tcy = tl.cos(ka) * 138.0 * disc
                tr = (-(k * 14.4) - 120.0 + rot) * 0.01745329252
                ctr_ = tl.cos(tr)
                str_ = tl.sin(tr)
                lx0 = (dx - tcx) / disc
                ly0 = (dy - tcy) / disc
                lx = lx0 * ctr_ + ly0 * str_
                ly = -lx0 * str_ + ly0 * ctr_
                qx = tl.maximum(tl.abs(lx) - 12.5, 0.0)
                qd = tl.sqrt(qx * qx + ly * ly)
                qinv = 1.0 / tl.maximum(qd, 1e-6)
                tnx = tl.where(qd > 1e-6, qx * qinv, 0.0)
                tny = tl.where(qd > 1e-6, ly * qinv, 1.0)
                f = tl.minimum(tl.maximum((32.5 - qd) / 30.0, 0.0), 1.0)
                cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, f * f * 0.2 * a)
                tco = _cov(2.5 - aal - qd, tnx, tny, hl)
                tci = _cov(0.5 + aal - qd, tnx, tny, hl)
                cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, tl.maximum(tco - tci, 0.0) * a)
            # ---- ring arcs (top + bottom): outer radius 184, thickness 184*ir, half-angle arc*pi
            arc = tl.load(P + P_SP_ARC)
            hth = 92.0 * tl.load(P + P_SP_ARC_IR) * disc
            rc = 184.0 * disc - hth
            dtop = _arc_dist(dx, dy, d, rc, hth, 0.0, arc * 3.14159265)
            dbot = _arc_dist(dx, dy, d, rc, hth, 3.14159265, arc * 3.14159265)
            cv = _cov(-tl.minimum(dtop, dbot), nx, ny, h)
            cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, cv * a)
            # ---- side progress arcs: radius 192, thickness 23.04; background 0.25 white; fill + additive glow
            hth2 = 11.52 * disc
            rc2 = 192.0 * disc - hth2
            bga = tl.load(P + P_SP_SIDE_BG)
            if bga > 0:
                db = tl.minimum(_arc_dist(dx, dy, d, rc2, hth2, 1.5707963, 0.15 * 3.14159265),
                                _arc_dist(dx, dy, d, rc2, hth2, -1.5707963, 0.15 * 3.14159265))
                cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, _cov(-db, nx, ny, h) * bga * a)
            sfa = tl.load(P + P_SP_SIDE_A)
            if sfa > 0.001:
                half = tl.load(P + P_SP_SIDE_P) * 3.14159265
                df = tl.minimum(_arc_dist(dx, dy, d, rc2, hth2, 1.5707963, half),
                                _arc_dist(dx, dy, d, rc2, hth2, -1.5707963, half))
                cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, _cov(-df, nx, ny, h) * sfa * a)
                Hg = 118.52 * disc
                ty = 1.0 - tl.maximum(df, 0.0) / Hg
                pos = tl.minimum(tl.maximum(ty * 107.0 - 0.5, 0.0), 106.0)
                i0 = tl.minimum(tl.floor(pos), 105.0)
                fr = pos - i0
                g0 = tl.load(glow_ptr + i0.to(tl.int32), mask=ty > 0, other=0.0)
                g1 = tl.load(glow_ptr + i0.to(tl.int32) + 1, mask=ty > 0, other=0.0)
                ga = tl.where((ty > 0) & (df > 0), g0 + (g1 - g0) * fr, 0.0)     # texture lookup beyond the rect -> 0
                # the 1-texel-wide glow texture is sampled at its left edge (x = TexRect.x): linear filtering halves it
                cr, cg, cb = _add(cr, cg, cb, 0.67058824, 1.0, 1.0, ga * 0.5 * 0.70588235 * sfa * a)
        # ---- centre rings: RingPiece(10) at 0.8 size + RingPiece(3) full size, container csize * cscale
        cs = tl.load(P + P_SP_CEN)
        csz = tl.load(P + P_SP_CSIZE)
        if cs > 0.001:
            ro = 0.5 * csz * cs
            cr, cg, cb = _ring_layer(cr, cg, cb, dx, dy, 0.8 * ro, 10.0 * cs, -ro, 2 * ro, 1.0, 1.0, 1.0, a,
                                     1.0, 1.0, 1.0, a, h, False)
            cr, cg, cb = _ring_layer(cr, cg, cb, dx, dy, ro, 3.0 * cs, -ro, 2 * ro, 1.0, 1.0, 1.0, a,
                                     1.0, 1.0, 1.0, a, h, False)
        # ---- texts (SPM counter, "SPINS PER MINUTE", bonus counter): prepared glyph list in gamefield px
        for i in range(48):
            E = spt_ptr + (m * 48 + i) * 8
            gi = tl.load(E + 0)
            if gi >= 0:
                x0 = tl.load(E + 1)
                y0 = tl.load(E + 2)
                sc = tl.load(E + 3)
                gw = tl.load(gly_ptr + gi.to(tl.int32) * 4 + 1)
                gh = tl.load(gly_ptr + gi.to(tl.int32) * 4 + 2)
                if (x0 < tg_x1) & (x0 + gw * sc > tg_x0) & (y0 < tg_y1) & (y0 + gh * sc > tg_y0):
                    al = _glyph(sat_ptr, gly_ptr, gi.to(tl.int32), gx, gy, h, h, x0, y0, sc)
                    cr, cg, cb = _over(cr, cg, cb, tl.load(E + 4), tl.load(E + 5), tl.load(E + 6),
                                       al * tl.load(E + 7) * a)
    return cr, cg, cb


@triton.jit
def _break_overlay(cr, cg, cb, gx, gy, h, Fb, brt_ptr, m, sat_ptr, gly_ptr, G_CHEV_L: tl.constexpr):
    """BreakOverlay (UI space, drawn above the playfield and below the cursor): shadow glow, remaining-time bar,
    glowing arrows and texts (prepared glyph list)."""
    a = tl.load(Fb + 4)
    if a > 0:
        ux = (gx - 256.0) * 1.6                       # UI units from the UI centre (1 UI = 0.625 osu px)
        uy = (gy - 184.0) * 1.6
        hu = h * 1.6
        aau = 1.0666667
        # shadow: grey 0.2 @0.8, rounded rect 80x4 inflated by 260 (corner 274, blend 260, exponent 2)
        qx = tl.maximum(tl.abs(ux) - 26.0, 0.0)
        qy = tl.abs(uy) + 12.0
        f = tl.minimum(tl.maximum((274.0 - tl.sqrt(qx * qx + qy * qy)) / 260.0, 0.0), 1.0)
        cr, cg, cb = _over(cr, cg, cb, 0.2, 0.2, 0.2, f * f * 0.8 * a)
        # remaining-time bar (white pill)
        bw = tl.load(Fb + 5)
        if bw > 0.01:
            bh = tl.minimum(4.0, bw)
            px = tl.maximum(tl.abs(ux) - (bw - bh), 0.0)
            d = tl.sqrt(px * px + uy * uy)
            inv = 1.0 / tl.maximum(d, 1e-6)
            nx = tl.where(d > 1e-6, px * inv, 0.0)
            ny = tl.where(d > 1e-6, uy * inv, 1.0)
            cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, _cov(bh - 0.5 * aau - d, nx, ny, hu) * a)
        # arrows: ChevronRight (left) / ChevronLeft (right) fitted in 60x60, blurred ddffff glow + white glyph
        asc = tl.load(Fb + 9)
        for side in tl.static_range(2):
            gidx = 41 + side * (G_CHEV_L - 41)
            acx = tl.load(Fb + 6 + side)
            acy = tl.load(Fb + 8)
            gw = tl.load(gly_ptr + gidx * 4 + 1)
            gh = tl.load(gly_ptr + gidx * 4 + 2)
            sc = tl.minimum(60.0 / gw, 60.0 / gh) * asc
            lx = ux - acx
            ly = uy - acy
            if tl.min(tl.min(tl.abs(lx), 1), 0) < 120.0:
                blur = _glyph(sat_ptr, gly_ptr, gidx, lx, ly, hu + 17.3 * asc, hu + 17.3 * asc,
                              -gw * sc * 0.5, -gh * sc * 0.5, sc)
                cr, cg, cb = _over(cr, cg, cb, 0.86666667, 1.0, 1.0, blur * a)
                al = _glyph(sat_ptr, gly_ptr, gidx, lx, ly, hu, hu, -gw * sc * 0.5, -gh * sc * 0.5, sc)
                cr, cg, cb = _over(cr, cg, cb, 1.0, 1.0, 1.0, al * a)
        # texts (UI coordinates)
        for i in range(40):
            E = brt_ptr + (m * 40 + i) * 8
            gi = tl.load(E + 0)
            if gi >= 0:
                al = _glyph(sat_ptr, gly_ptr, gi.to(tl.int32), ux, uy, hu, hu, tl.load(E + 1), tl.load(E + 2),
                            tl.load(E + 3))
                cr, cg, cb = _over(cr, cg, cb, tl.load(E + 4), tl.load(E + 5), tl.load(E + 6), al * tl.load(E + 7) * a)
    return cr, cg, cb


@triton.jit
def _box_bits(P, X0: tl.constexpr, tx0, ty0, TH: tl.constexpr, TW: tl.constexpr):
    bx0 = tl.load(P + X0)
    by0 = tl.load(P + X0 + 1)
    bx1 = tl.load(P + X0 + 2)
    by1 = tl.load(P + X0 + 3)
    return (bx0 < tx0 + TW) & (bx1 > tx0) & (by0 < ty0 + TH) & (by1 > ty0)


@triton.jit
def _mask_kernel(mask_ptr, P_ptr, I_ptr, K: tl.constexpr, NPc: tl.constexpr, NIc: tl.constexpr,
                 TH: tl.constexpr, TW: tl.constexpr, TILES_X: tl.constexpr, NT: tl.constexpr, NTP: tl.constexpr):
    """Per (frame, tile) bitmasks of slots whose pass-specific culling box overlaps the tile."""
    m = tl.program_id(0)
    t = tl.arange(0, NTP)
    valid = t < NT
    tx0 = ((t % TILES_X) * TW).to(tl.float32)
    ty0 = ((t // TILES_X) * TH).to(tl.float32)
    m0 = tl.zeros([NTP], tl.int32)
    m1 = tl.zeros([NTP], tl.int32)
    m2 = tl.zeros([NTP], tl.int32)
    m3 = tl.zeros([NTP], tl.int32)
    m4 = tl.zeros([NTP], tl.int32)
    m5 = tl.zeros([NTP], tl.int32)
    for j in range(K):
        P = P_ptr + (m * K + j) * NPc
        Ib = I_ptr + (m * K + j) * NIc
        kind = tl.load(Ib + I_KIND)
        bit = 1 << j
        if kind >= 0:
            ob = _box_bits(P, P_OB_X0, tx0, ty0, TH, TW)
            jb = _box_bits(P, P_JB_X0, tx0, ty0, TH, TW)
            if kind == 2:
                m0 = m0 | tl.where(ob, bit, 0)
            else:
                m3 = m3 | tl.where(ob, bit, 0)
            if (tl.load(Ib + I_FP_ON) > 0) | (tl.load(Ib + I_FPO_ON) > 0):
                m1 = m1 | tl.where(_box_bits(P, P_FB_X0, tx0, ty0, TH, TW), bit, 0)
            if tl.load(P + P_EX_A) > 0:
                m2 = m2 | tl.where(jb, bit, 0)
            miss = (tl.load(Ib + I_MISS_LO) != 0) | (tl.load(Ib + I_MISS_HI) != 0)
            if (tl.load(P + P_J_A) > 0) | ((kind == 1) & miss):
                m4 = m4 | tl.where(jb, bit, 0)
            if tl.load(P + P_A_APP) > 0:
                m5 = m5 | tl.where(_box_bits(P, P_AB_X0, tx0, ty0, TH, TW), bit, 0)
    base = mask_ptr + (m * NT + t) * 6
    tl.store(base + 0, m0, mask=valid)
    tl.store(base + 1, m1, mask=valid)
    tl.store(base + 2, m2, mask=valid)
    tl.store(base + 3, m3, mask=valid)
    tl.store(base + 4, m4, mask=valid)
    tl.store(base + 5, m5, mask=valid)


# ------------------------------------------------------------------------------------------- renderer
class Renderer:
    TH, TW = 16, 16

    def __init__(self, pool_gpu, device="cuda", ss=1, K=None, breaks=None, graphs=True, hit_animations=True):
        self.device = torch.device(device)
        self.ss = int(ss)
        # labelled (session 3): False = lazer OsuRulesetSetting.HitAnimations off (circles and slider heads fade
        # out over 60 ms OutQuad from the hit; DrawableHitCircle.cs:214-223). Fixed at construction: _prepare is
        # captured in CUDA graphs, so changing it later needs self._graphs cleared.
        self.hit_animations = bool(hit_animations)
        dev = self.device
        o = {k: v.to(dev) for k, v in pool_gpu["objs"].items()}
        mp = {k: v.to(dev) for k, v in pool_gpu["maps"].items()}
        ne = {k: v.to(dev) for k, v in pool_gpu["nest"].items()}
        pa = {k: v.to(dev) for k, v in pool_gpu["path"].items()}
        self.o, self.mp = o, mp
        G = o["kind"].shape[0]
        # ---- path table [V+1, 3] (x, y, cum) and chunk boxes per vertex index (valid at chunk starts)
        V = pa["x"].shape[0]
        path = torch.zeros((V + 2, 3), dtype=torch.float32, device=dev)
        path[:V, 0], path[:V, 1], path[:V, 2] = pa["x"], pa["y"], pa["cum"]
        self.path = path.contiguous()
        self.CHUNK = 8
        chunk = torch.zeros((V + 1, 4), dtype=torch.float32, device=dev)
        sl = torch.nonzero(o["kind"] == 1).flatten()
        offs, cnts = o["path_off"][sl].long(), o["path_cnt"][sl].long()
        if V > 0:
            # chunk c of slider covers segments [off + 8c, off + 8c + 8) -> vertices [.., +9)
            vid = torch.arange(V, device=dev)
            owner = torch.full((V,), -1, dtype=torch.long, device=dev)
            seg_sl = torch.repeat_interleave(torch.arange(len(sl), device=dev), cnts)
            local = vid[:len(seg_sl)] if False else None
            starts = torch.repeat_interleave(offs, cnts)
            idx = torch.arange(int(cnts.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(cnts, 0) - cnts, cnts)
            verts = starts + idx
            owner[verts] = seg_sl
            cstart = starts + (idx // self.CHUNK) * self.CHUNK
            x, y = pa["x"][verts], pa["y"][verts]
            inf = torch.full((V,), float("inf"), device=dev)
            bx0, by0 = inf.clone(), inf.clone()
            bx1, by1 = -inf.clone(), -inf.clone()
            bx0.scatter_reduce_(0, cstart, x, "amin")
            by0.scatter_reduce_(0, cstart, y, "amin")
            bx1.scatter_reduce_(0, cstart, x, "amax")
            by1.scatter_reduce_(0, cstart, y, "amax")
            # include the first vertex of the next chunk (segment end)
            nxt = verts + 1
            okn = (idx % self.CHUNK == self.CHUNK - 1) & (idx + 1 < torch.repeat_interleave(cnts, cnts))
            bx0.scatter_reduce_(0, cstart[okn], pa["x"][nxt[okn]], "amin")
            by0.scatter_reduce_(0, cstart[okn], pa["y"][nxt[okn]], "amin")
            bx1.scatter_reduce_(0, cstart[okn], pa["x"][nxt[okn]], "amax")
            by1.scatter_reduce_(0, cstart[okn], pa["y"][nxt[okn]], "amax")
            chunk[:V, 0], chunk[:V, 1], chunk[:V, 2], chunk[:V, 3] = bx0, by0, bx1, by1
            # float64 search keys for snaking (slider rank * 1e6 + cum)
            rank = torch.zeros(G, dtype=torch.float64, device=dev)
            rank[sl] = torch.arange(len(sl), device=dev, dtype=torch.float64)
            self.vkey = torch.repeat_interleave(rank[sl], cnts) * 1e6 + pa["cum"][verts].double()
            self.vkey_pos = verts
        self.chunk = chunk.contiguous()
        self.slider_rank = rank if V > 0 else torch.zeros(G, dtype=torch.float64, device=dev)
        # ---- nest table [E, 8]
        E = ne["type"].shape[0]
        nest = torch.zeros((max(E, 1), NEST_W), dtype=torch.float32, device=dev)
        if E:
            nest[:, 0] = ne["type"].float()
            nest[:, 1] = ne["time"]
            nest[:, 2] = ne["x"]
            nest[:, 3] = ne["y"]
            nest[:, 4] = ne["span_index"].float()
            nest[:, 5] = ne["span_start"]
            # repeat arrow direction: towards the neighbouring path vertex (inwards along the path)
            obj = ne["obj"].long()
            poff, pcnt = o["path_off"][obj].long(), o["path_cnt"][obj].long()
            at_end = ne["progress"] > 0.5
            vi = torch.where(at_end, poff + pcnt - 2, poff + 1).clamp(min=0)
            ax_, ay_ = pa["x"][vi.clamp(max=V - 1)], pa["y"][vi.clamp(max=V - 1)]
            ang = torch.atan2(ay_ - ne["y"], ax_ - ne["x"])
            nest[:, 6] = torch.cos(ang)
            nest[:, 7] = torch.sin(ang)
        self.nest = nest.contiguous()
        # ---- static per object extras
        self.accent = torch.as_tensor(ACCENTS, device=dev)
        # ---- assets
        A = Assets()
        self.assets = A
        self.sat = A.sat.to(dev)
        self.gly = A.glyph_tab.to(dev)
        # judgement texts (GREAT, OK, MEH, MISS): glyph idx, x0, y0 (units, relative to the centred box), font scale
        txt = np.full((5, TXT_MAX, 4), -1.0, np.float32)
        for ti, word in enumerate(["", "GREAT", "OK", "MEH", "MISS"]):
            if not word:
                continue
            lay, (W, Hh) = A.layout(A.TORUS_BOLD, word, 20.0, spacing=5.0)
            for k, (gi, x0, y0, w, hh) in enumerate(lay):
                txt[ti, k] = (gi, x0 - W / 2, y0 - Hh / 2, 0.2)
        self.txt = torch.as_tensor(txt, device=dev)
        # circle numbers 0..999: up to 3 glyphs (idx, x0 relative to centred box), y0 relative
        nums = np.zeros((1000, 3, 2), np.float32)
        numn = np.zeros(1000, np.int32)
        ny0 = 0.0
        for n in range(1000):
            lay, (W, Hh) = A.layout(A.TORUS_BOLD, str(n), 52.0)
            numn[n] = len(lay)
            for k, (gi, x0, y0, w, hh) in enumerate(lay):
                nums[n, k] = (gi, x0 - W / 2)
            ny0 = lay[0][2] - Hh / 2 - 2.0          # Y = -2
        self.num_tab = torch.as_tensor(nums, device=dev)
        self.num_n = torch.as_tensor(numn, device=dev)
        self.num_y0 = ny0                          # all digits share yoffset (27/28 texels; see layout)
        self.num_y0_tab = torch.as_tensor(np.array([A.layout(A.TORUS_BOLD, str(d), 52.0)[0][0][2] - 26.0 - 2.0
                                                    for d in range(10)], np.float32), device=dev)
        # follow-point connections (incoming to g): on, prev end pos, this pos, prev end time, this start, dist
        g = torch.arange(G, device=dev)
        prev = (g - 1).clamp(min=0)
        on = (g > 0) & (o["map"][prev] == o["map"]) & (o["new_combo"] == 0) & (o["kind"] != 2) & (o["kind"][prev] != 2)
        sx, sy = o["ex"][prev], o["ey"][prev]
        ex, ey = o["x"], o["y"]
        dist = torch.floor(torch.sqrt((ex - sx) ** 2 + (ey - sy) ** 2))
        on = on & (dist - 32 > 48)
        self.fp = {"on": on, "sx": sx, "sy": sy, "ex": ex, "ey": ey, "t0": o["end"][prev], "t1": o["start"],
                   "d": dist}
        self.G = G
        self._digit_cache = {}
        self.graphs = bool(graphs)
        self._graphs = {}
        rc = torch.zeros((5, 3), device=dev)
        for k, c in RESULT_COL.items():
            rc[k] = torch.tensor(c, device=dev, dtype=torch.float32) / 255.0
        self.res_col = rc
        self.has_spinners = bool((o["kind"] == 2).any())
        A_ = self.assets
        lay, (W_spm, _) = A_.layout(A_.TORUS_SEMI, "SPINS PER MINUTE", 16.0)
        self._spm_lay = (torch.tensor([g[0] for g in lay], device=dev),
                         torch.tensor([g[1] for g in lay], device=dev, dtype=torch.float32),
                         torch.tensor([g[2] for g in lay], device=dev, dtype=torch.float32), W_spm)
        lay, (W_max, _) = A_.layout(A_.TORUS_BOLD, "MAX", 28.0)
        self._max_lay = (torch.tensor([g[0] for g in lay] + [0, 0], device=dev),
                         torch.tensor([g[1] for g in lay] + [0, 0], device=dev, dtype=torch.float32),
                         torch.tensor([g[2] for g in lay] + [0, 0], device=dev, dtype=torch.float32),
                         torch.tensor([True] * 3 + [False] * 2, device=dev), W_max)
        self._pink_l = torch.tensor(_to_lin(np.array(PINK) / 255.0), device=dev, dtype=torch.float32)
        if breaks is None and isinstance(pool_gpu, dict):
            breaks = pool_gpu.get("breaks")
        self._init_breaks(breaks, int(mp["scale"].shape[0]))
        for font, size in ((self.assets.TORUS_SEMI, 28.0), (self.assets.TORUS_BOLD, 28.0), (self.assets.VENERA_BOLD, 33.0)):
            self._digit_layout(font, size, torch.zeros(1, device=dev), 1)      # populate constant caches
        import cv2
        gim = cv2.imread(str(ASSET_DIR / "osu.Game.Resources.Textures.Gameplay.osu.spinner-glow.png"), cv2.IMREAD_UNCHANGED)
        self.glow = torch.as_tensor(gim[:, 0, 3].astype(np.float32) / 255.0, device=dev)

    # --------------------------------------------------------------------------------------------- path eval
    def _path_pos(self, gidx, dist_along):
        """SliderPath.PositionAt by distance for slider objects gidx (tensor), stacked absolute."""
        o = self.o
        poff = o["path_off"][gidx].long()
        pcnt = o["path_cnt"][gidx].long()
        key = self.slider_rank[gidx] * 1e6 + dist_along.double()
        j = torch.searchsorted(self.vkey, key, right=False)            # global position in vkey
        # vkey is laid out in pool path order for sliders -> map back to vertex index
        j = j.clamp(max=self.vkey.shape[0] - 1)
        vi = self.vkey_pos[j]
        vi = torch.minimum(torch.maximum(vi, poff), poff + pcnt - 1)
        i0 = torch.maximum(vi - 1, poff)
        x0, y0, c0 = self.path[i0, 0], self.path[i0, 1], self.path[i0, 2]
        x1, y1, c1 = self.path[vi, 0], self.path[vi, 1], self.path[vi, 2]
        w = ((dist_along - c0) / (c1 - c0).clamp(min=1e-7)).clamp(0, 1)
        w = torch.where((c1 - c0).abs() <= 1e-7, torch.zeros_like(w), w)
        return x0 + (x1 - x0) * w, y0 + (y1 - y0) * w

    def _curve_pos(self, gidx, prog):
        o = self.o
        spans = o["span_count"][gidx].float()
        p = prog * spans
        span = torch.floor(p)
        frac = p - span
        last = span >= spans
        span = torch.where(last, spans - 1, span)
        frac = torch.where(last, torch.ones_like(frac), frac)
        pp = torch.where(torch.remainder(span, 2) == 1, 1 - frac, frac)
        return self._path_pos(gidx, pp * o["distance"][gidx])

    def _seg_index(self, gidx, d):
        """First vertex index whose cum >= d (searchsorted left) within each slider."""
        o = self.o
        poff = o["path_off"][gidx].long()
        pcnt = o["path_cnt"][gidx].long()
        key = self.slider_rank[gidx] * 1e6 + d.double()
        j = torch.searchsorted(self.vkey, key, right=False).clamp(max=self.vkey.shape[0] - 1)
        vi = self.vkey_pos[j]
        return torch.minimum(torch.maximum(vi, poff), poff + pcnt - 1)

    # --------------------------------------------------------------------------------------------- prepare
    @torch.no_grad()
    def _prepare(self, rs):
        dev = self.device
        o, mp = self.o, self.mp
        t = rs["t"].to(dev).float()
        M = t.shape[0]
        gid = rs["gid"].to(dev).long()
        K = gid.shape[1]
        valid = gid >= 0
        g = gid.clamp(min=0)
        mi = rs["map"].to(dev).long()
        T = t[:, None].expand(M, K)
        q = mp["scale"][mi][:, None].expand(M, K)
        pre = mp["preempt"][mi][:, None].expand(M, K)
        fin = mp["fade_in"][mi][:, None].expand(M, K)
        kind = torch.where(valid, o["kind"][g].long(), torch.full_like(g, -1))
        s = o["start"][g]
        e = o["end"][g]
        x, y = o["x"][g], o["y"][g]
        head_r = rs["head_r"].to(dev).long()
        ht = rs["head_t"].to(dev).float()
        end_t = rs["end_t"].to(dev).float()
        trk = rs["trk"].to(dev).long()
        trk_t = rs["trk_t"].to(dev).float()
        nest_n = rs["nest_n"].to(dev).long()
        nmiss = rs["nest_miss"].to(dev).long()
        nest_t = rs["nest_t"].to(dev).float()
        tail_r = rs["tail_r"].to(dev).long()
        hit = (head_r >= 1) & (head_r <= 3)
        miss = head_r == 4
        judged = head_r > 0
        end_j = end_t > -1e8
        is_sl = kind == 1
        is_ci = kind == 0
        is_sp = kind == 2
        acc = self.accent[(o["combo_index"][g] % 6).long()]                    # [M,K,3] sRGB unclamped
        accl = lin(acc)
        P = torch.zeros((M, K, NP), dtype=torch.float32, device=dev)
        I = torch.zeros((M, K, NI), dtype=torch.int32, device=dev)
        Pv = {}

        def put(name, val):
            P[..., PI[name]] = val if torch.is_tensor(val) else torch.full((M, K), float(val), device=dev)

        def puti(name, val):
            I[..., II[name]] = val.int() if torch.is_tensor(val) else int(val)

        c01 = _clamp01
        # ------------------------------------------------ common
        put("cx", x); put("cy", y); put("q", q); put("R", 64 * q)
        puti("gid", gid); puti("kind", kind)
        life = T >= s - pre
        sl_a = torch.where(end_j & ~is_ci, 1 - c01((T - end_t) / 240.0), torch.ones_like(T))
        sl_a = torch.where(is_sl | is_sp, sl_a, torch.ones_like(T)) * life
        # dim tint (linear light): 195 -> white over 100 ms from start-400
        dim_l = float(_to_lin(195 / 255.0))
        tint_l = dim_l + (1 - dim_l) * c01((T - (s - 400.0)) / 100.0)                  # [M,K] linear factor
        tint_s = srgb(tint_l)

        def tinted(c):        # c [M,K,3] sRGB -> sRGB after linear tint
            return srgb(lin(c) * tint_l[..., None])

        # ------------------------------------------------ circle piece (circles and slider heads)
        circ = (is_ci | is_sl) & life
        dho = torch.ones_like(T)
        if self.hit_animations:
            dho = torch.where(hit, (T < ht + 800).float(), dho)
        else:
            # labelled (session 3): HitAnimations off: FadeOut(60, Easing.Out=OutQuad) at the hit replaces the
            # Delay(800).FadeOut() (later same-property transform removed); alpha (1 - dt/60)^2, 0 from ht+60
            dho = torch.where(hit, 1 - e_out_quad(c01((T - ht) / 60.0)), dho)
        dho = torch.where(miss, 1 - c01((T - ht) / 100.0), dho)
        cp_a = c01((T - (s - pre)) / fin)
        ph = c01((T - ht) / 800.0)
        piece = torch.where(hit, 1 - e_out_quad(ph), torch.ones_like(T))
        base_a = dho * cp_a * piece * sl_a * circ
        f150 = torch.where(hit, 1 - e_out_quint(c01((T - ht) / 150.0)), torch.ones_like(T))
        ofill = tinted((acc / 5).clamp(max=1))
        put("a_ofill", base_a * f150 * is_ci)
        put("of_r", ofill[..., 0]); put("of_g", ofill[..., 1]); put("of_b", ofill[..., 2])
        # outer gradient: colour -> white over 80 ms from ht+12.5, then fade 150; resize elastic from ht+12.5
        og_t, og_b = tinted(acc), tinted((acc / 1.1).clamp(max=1))
        pw = torch.where(hit, c01((T - ht - 12.5) / 80.0), torch.zeros_like(T))[..., None]
        og_t = srgb(lin(og_t) + (1 - lin(og_t)) * pw)
        og_b = srgb(lin(og_b) + (1 - lin(og_b)) * pw)
        put("a_ograd", base_a * torch.where(hit, 1 - c01((T - ht - 92.5) / 150.0), torch.ones_like(T)))
        pz = torch.where(hit, c01((T - ht - 12.5) / 400.0), torch.zeros_like(T))
        osize = 110.34483 + (88.275864 - 110.34483) * e_out_elastic_half(pz)
        put("r_ograd", osize * 0.5 * q)
        for nm, c in (("og_t", og_t), ("og_b", og_b)):
            put(nm + "r", c[..., 0]); put(nm + "g", c[..., 1]); put(nm + "b", c[..., 2])
        ig_t, ig_b = tinted((acc / 1.5).clamp(max=1)), tinted((acc / 1.6).clamp(max=1))
        put("a_igrad", base_a * f150)
        for nm, c in (("ig_t", ig_t), ("ig_b", ig_b)):
            put(nm + "r", c[..., 0]); put(nm + "g", c[..., 1]); put(nm + "b", c[..., 2])
        put("a_ifill", base_a * f150)
        # number
        num = o["number"][g].long().clamp(0, 999)
        a_num = base_a * torch.where(hit, 1 - c01((T - ht) / 75.0), torch.ones_like(T))
        put("a_num", a_num)
        put("num_c", tint_s)
        put("num_s", 0.52 * q)
        nt = self.num_tab[num]                                              # [M,K,3,2]
        put("num_x0", nt[..., 0, 1] * q); put("num_x1", nt[..., 1, 1] * q); put("num_x2", nt[..., 2, 1] * q)
        put("num_y", self.num_y0_tab[0] * q)
        put("a_shadow", a_num * 0.2 * tint_s ** 2)
        puti("num_g0", nt[..., 0, 0]); puti("num_g1", nt[..., 1, 0]); puti("num_g2", nt[..., 2, 0])
        puti("num_n", self.num_n[num])
        # flash glow
        put("a_flash", base_a * torch.where(hit, e_out_quint(c01((T - ht) / 150.0)), torch.zeros_like(T)))
        put("fl_r", acc[..., 0]); put("fl_g", acc[..., 1]); put("fl_b", acc[..., 2])
        # border ring: white -> gradient(accent@0.5, accent@0) over 800 (linear light incl. alpha), elastic resize
        pb = torch.where(hit, c01((T - ht) / 800.0), torch.zeros_like(T))[..., None]
        top = srgb(1 + (accl - 1) * pb)
        topa = 1 + (0.5 - 1) * pb[..., 0]
        bota = 1 - pb[..., 0]
        top = srgb(lin(top) * tint_l[..., None])
        put("bd_tr", top[..., 0]); put("bd_tg", top[..., 1]); put("bd_tb", top[..., 2])
        put("bd_br", top[..., 0]); put("bd_bg", top[..., 1]); put("bd_bb", top[..., 2])
        put("bd_ta", topa * base_a); put("bd_ba", bota * base_a)
        pr_ = torch.where(hit, c01((T - ht) / 400.0), torch.zeros_like(T))
        bsize = 128 + (106.81379 - 128) * e_out_elastic_half(pr_)
        put("r_border", bsize * 0.5 * q); put("t_border", 4.4137931 * q)
        # approach circle
        app = (is_ci | is_sl) & life
        a_in = 0.9 * c01((T - (s - pre)) / torch.minimum(fin * 2, pre))
        a_at_s = 0.9 * c01(pre / torch.minimum(fin * 2, pre))
        a_ap = torch.where(T < s, a_in, a_at_s * (1 - c01((T - s) / 50.0)))
        a_ap = torch.where(hit & (T >= ht), torch.zeros_like(T), a_ap)
        a_ap = a_ap * dho * sl_a * app
        put("a_app", a_ap)
        sc_ap = 4 - 3 * c01((T - (s - pre)) / pre)
        put("r_app", 64.0 * sc_ap * q)
        put("app_r", acc[..., 0]); put("app_g", acc[..., 1]); put("app_b", acc[..., 2])
        # ------------------------------------------------ slider
        dist = o["distance"][g]
        spans = o["span_count"][g].float()
        dur = (e - s).clamp(min=1e-6)
        cpr = c01((T - s) / dur)
        head_hit = hit
        span_i = torch.minimum(torch.floor(cpr * spans), spans - 1)
        sp_prog = cpr * spans - span_i
        sp_prog = torch.where(torch.remainder(span_i, 2) == 1, 1 - sp_prog, sp_prog)
        cp_used = torch.where(head_hit, cpr, torch.zeros_like(cpr))
        span_u = torch.minimum(torch.floor(cp_used * spans), spans - 1)
        sp_u = cp_used * spans - span_u
        sp_u = torch.where(torch.remainder(span_u, 2) == 1, 1 - sp_u, sp_u)
        sp_u = torch.where(cp_used >= 1, torch.where(torch.remainder(spans - 1, 2) == 1, torch.zeros_like(sp_u),
                                                     torch.ones_like(sp_u)), sp_u)
        end_p = c01((T - (s - pre)) / (pre / 3))
        st_p = torch.zeros_like(T)
        lastspan = span_u >= spans - 1
        odd_last = torch.remainder(spans - 1, 2) == 1
        end_p = torch.where(lastspan & odd_last, sp_u, end_p)
        st_p = torch.where(lastspan & ~odd_last, sp_u, st_p)
        d0, d1 = st_p * dist, end_p * dist
        put("d0", d0); put("d1", d1)
        # nested hits for the slider state
        lo32 = nmiss[..., 0] & 0xFFFFFFFF
        hi32 = nmiss[..., 1] & 0xFFFFFFFF
        idx64 = torch.arange(64, device=dev)
        bits = torch.where(idx64 < 32, (lo32[..., None] >> idx64.clamp(max=31)) & 1,
                           (hi32[..., None] >> (idx64 - 32).clamp(min=0)) & 1)
        judged_mask = idx64 < nest_n[..., None]
        n_nest_miss = (bits * judged_mask).sum(-1)
        nested_hit = (nest_n - n_nest_miss) > 0
        sl_hit = head_hit | nested_hit
        body_out = torch.where(end_j & sl_hit & head_hit, 1 - c01((T - end_t) / 40.0), torch.ones_like(T))
        a_body = cp_a * body_out * sl_a * is_sl
        put("a_body", a_body)
        body_g = tint_s                                                     # sRGB-space multiply (path shader)
        bo = acc * body_g[..., None]
        inn = (acc / 5).clamp(max=1) * body_g[..., None]
        put("bo_r", bo[..., 0]); put("bo_g", bo[..., 1]); put("bo_b", bo[..., 2])
        put("in_r", inn[..., 0]); put("in_g", inn[..., 1]); put("in_b", inn[..., 2]); put("in_a", 0.98)
        prad = (55.172413 * q)
        wtex = 2 * torch.floor(prad.clamp(min=1))
        # texel alpha ramp deficit (texture AA) and border/interior transition
        i_t = torch.arange(6, device=dev, dtype=torch.float32)
        alph = torch.minimum(i_t / (wtex[..., None] - 1) / 0.02, torch.ones_like(i_t))
        alph[..., 0] = 0
        deficit = 0.5 * (1 - alph[..., 0]) + (1 - 0.5 * (alph[..., :-1] + alph[..., 1:])).sum(-1)
        put("r_out", prad - deficit * prad / wtex)
        ib = torch.floor(0.2 * (wtex - 1) + 1e-6)
        put("r_bin", prad * (1 - (ib + 1) / wtex))
        if self.vkey.numel() > 0:
            gs = torch.where(is_sl, g, torch.zeros_like(g))
            lo = self._seg_index(gs, d0)
            hi = self._seg_index(gs, d1)
            poff = o["path_off"][gs].long()
            pcnt = o["path_cnt"][gs].long()
            lo = torch.maximum(lo - 1, poff)
            hi = torch.minimum(hi, poff + pcnt - 1)
            puti("seg_lo", torch.where(is_sl, lo, torch.zeros_like(lo)))
            puti("seg_hi", torch.where(is_sl, hi, torch.zeros_like(hi)))
            puti("path_off", poff)
            bxy = self._curve_pos(gs, cpr)
            dlt = 0.1 / dist.clamp(min=1e-3)
            pa_ = self._curve_pos(gs, torch.minimum(1 - dlt, cpr))
            pb_ = self._curve_pos(gs, torch.minimum(torch.ones_like(cpr), cpr + dlt))
        else:
            bxy = (x, y)
            pa_, pb_ = (x, y), (x, y)
        puti("nest_off", o["nest_off"][g]); puti("nest_cnt", torch.where(is_sl, o["nest_cnt"][g], 0 * g))
        puti("nest_n", nest_n); puti("miss_lo", nmiss[..., 0]); puti("miss_hi", nmiss[..., 1])
        put("sl_a", sl_a * is_sl); put("sl_start", s); put("sl_pre", pre)
        put("span_dur", o["span_dur"][g])
        put("acc_r", acc[..., 0]); put("acc_g", acc[..., 1]); put("acc_b", acc[..., 2])
        put("rp_white", tint_s)
        rpi = tinted((acc / 5).clamp(max=1))
        put("rp_ir", rpi[..., 0]); put("rp_ig", rpi[..., 1]); put("rp_ib", rpi[..., 2])
        # ball
        bvis = (T >= s) & is_sl
        a_bc = e_out_quint(c01((T - s) / 200.0))
        a_end = e_out_quint(c01((end_t - s) / 200.0)) * (1 - e_out_quint(c01((T - end_t) / 50.0)))
        a_bc = torch.where(end_j & (T >= end_t), a_end, a_bc)
        put("a_ball", a_bc * sl_a * bvis)
        put("bx", bxy[0]); put("by", bxy[1])
        ddx, ddy = pa_[0] - pb_[0], pa_[1] - pb_[1]
        rot = (-90.0 - torch.atan2(ddx, ddy) * 180 / math.pi) * math.pi / 180
        put("b_cos", torch.cos(rot)); put("b_sin", torch.sin(rot))
        ie = e_out_elastic_half(c01((T - s) / 200.0))
        isx, isy = 0.6 * ie, 0.8 * ie
        ie_end = e_out_elastic_half(c01((end_t - s) / 200.0))
        pe = e_out_quint(c01((T - end_t) / 200.0))
        isx = torch.where(end_j & (T >= end_t), 0.6 * ie_end + (0.54 - 0.6 * ie_end) * pe, isx)
        isy = torch.where(end_j & (T >= end_t), 0.8 * ie_end + (0.72 - 0.8 * ie_end) * pe, isy)
        put("ic_sx", isx); put("ic_sy", isy)
        bt, bb_ = acc, (acc / 1.5).clamp(max=1)
        put("ba_tr", bt[..., 0]); put("ba_tg", bt[..., 1]); put("ba_tb", bt[..., 2])
        put("ba_br", bb_[..., 0]); put("ba_bg", bb_[..., 1]); put("ba_bb", bb_[..., 2])
        # follow circle state machine from the last tracking change (t0) and, if provided, the one before (tp)
        t0 = torch.maximum(trk_t, s)
        ever = trk_t > -1e8
        # (changes before StartTime all collapse onto StartTime: FollowCircle uses max(now, StartTime))
        if "trk_tp" in rs:
            tp = rs["trk_tp"].to(dev).float()
            has_tp = tp > -1e8
        else:
            tp = torch.full_like(T, -1e9)
            has_tp = (t0 > s + 1.0) & (trk > 0)       # unknown history: assume a short (100 ms) release
            tp = torch.where(has_tp, t0 - 100.0, tp)
        tpc = torch.maximum(tp, s)
        dprev = (t0 - tpc).clamp(min=0)
        # press after a release that began at tp (from a completed press; a release collapsed onto StartTime
        # starts from rest): OnSliderPress resets the scale to 1 when the alpha is ~0
        rel_s = 2.4 + 0.48 * e_out_quint(c01(dprev / 150.0))
        rel_a = 1 - e_out_quint(c01(dprev / 150.0))
        real_rel = has_tp & (tp > s + 1e-3) & (rel_a >= 1e-3)
        s0_on = torch.where(real_rel, rel_s, torch.ones_like(T))
        a0_on = torch.where(real_rel, rel_a, torch.zeros_like(T))
        # release after a press that began at tp (from rest)
        s0_off = torch.where(has_tp, 1 + 1.4 * e_out_quint(c01(dprev / 300.0)), torch.full_like(T, 2.4))
        a0_off = torch.where(has_tp, e_out_quint(c01(dprev / 300.0)), torch.ones_like(T))

        def fol_state(tau):
            p3 = e_out_quint(c01((tau - t0) / 300.0))
            p1 = e_out_quint(c01((tau - t0) / 150.0))
            sc_on = s0_on + (2.4 - s0_on) * p3
            al_on = a0_on + (1 - a0_on) * p3
            sc_off = s0_off + (2.88 - s0_off) * p1
            al_off = a0_off * (1 - p1)
            scv = torch.where(trk > 0, sc_on, torch.where(ever, sc_off, torch.ones_like(tau)))
            alv = torch.where(trk > 0, al_on, torch.where(ever, al_off, torch.zeros_like(tau)))
            return scv, alv
        fsc, fal = fol_state(T)
        # tick pulse at the latest nested hit while tracking
        last_i = (nest_n - 1).clamp(min=0)
        last_bit = torch.gather(bits, -1, last_i[..., None].clamp(max=63))[..., 0]
        ncnt = o["nest_cnt"][g].long()
        last_is_tail = last_i >= ncnt - 1
        pulse = (trk > 0) & (nest_n > 0) & (last_bit == 0) & ~last_is_tail & (T >= nest_t)
        sc_at, _ = fol_state(nest_t)
        pulse = pulse & (sc_at >= 2.352)
        dtp = T - nest_t
        psc = torch.where(dtp < 40, sc_at + (2.592 - sc_at) * e_out_quint(c01(dtp / 40.0)),
                          2.592 + (2.4 - 2.592) * e_out_quint(c01((dtp - 40) / 200.0)))
        fsc = torch.where(pulse, psc, fsc)
        # slider end with tail hit: scale -> 1 (300 OutQuint), alpha -> 0 (150 OutQuint) from state at end_t
        sce, ale = fol_state(end_t)
        tail_hit = (tail_r == 1) & end_j & (T >= end_t)
        # OnSliderEnd is scheduled at end_t when the tail is judged; a tracking transform added afterwards (visual
        # tracking change at/after the tail judgement) starts earlier than end_t and cancels it (osu-framework
        # removes same-member transforms that start later than a newly added one)
        all_nested = nest_n >= ncnt
        # strict: a tracking change in the SAME update as the tail judgement does not cancel OnSliderEnd
        # (DrawableSlider.Update copies Tracking before the tail's CheckForResult; verifier issue 1)
        tail_hit = tail_hit & ~(all_nested & (trk_t > nest_t + 1e-3) & (trk_t <= end_t))
        pe3 = e_out_quint(c01((T - end_t) / 300.0))
        pe1 = e_out_quint(c01((T - end_t) / 150.0))
        fsc = torch.where(tail_hit, sce + (1 - sce) * pe3, fsc)
        fal = torch.where(tail_hit, ale * (1 - pe1), fal)
        # after the slider is judged further tracking changes are ignored: freeze at end_t state
        frozen = end_j & (T >= end_t) & ~tail_hit
        fsc = torch.where(frozen & (trk_t > end_t), sce, fsc)
        put("a_fol", fal * sl_a * bvis)
        put("r_fol", 64.0 * q * fsc)
        fo_b = (acc / 1.5).clamp(max=1)
        put("fo_tr", acc[..., 0]); put("fo_tg", acc[..., 1]); put("fo_tb", acc[..., 2])
        put("fo_br", fo_b[..., 0]); put("fo_bg", fo_b[..., 1]); put("fo_bb", fo_b[..., 2])
        puti("head_hit", head_hit)
        # ------------------------------------------------ judgement popups (circle / head / spinner result)
        jx = torch.where(is_sp, torch.full_like(x, 256.0), x)
        jy = torch.where(is_sp, torch.full_like(y, 192.0), y)
        dtj = T - ht
        jact = judged & (dtj >= 0) & (dtj < 800)
        pa_j = 1 - c01(dtj / 800.0)
        ta = e_out_quint(c01(dtj / 300.0))
        tsc = 1 + 0.2 * e_out_quint(c01(dtj / 1800.0))
        msc = 1.6 - 0.6 * e_in_quad(c01(dtj / 100.0))
        mdy = 100 * e_in_quint(c01(dtj / 800.0)) * q
        mrot = 40 * e_in_quint(c01(dtj / 800.0)) * math.pi / 180
        put("j_x", jx); put("j_y", jy)
        put("j_a", torch.where(jact, torch.where(miss, pa_j, pa_j * ta), torch.zeros_like(T)))
        put("j_s", torch.where(miss, msc, tsc))
        put("j_c", torch.where(miss, torch.cos(mrot), torch.ones_like(T)))
        put("j_sn", torch.where(miss, torch.sin(mrot), torch.zeros_like(T)))
        put("j_dy", torch.where(miss, mdy, torch.zeros_like(T)))
        jc = self.res_col[head_r.clamp(0, 4)]
        put("j_r", jc[..., 0]); put("j_g", jc[..., 1]); put("j_b", jc[..., 2])
        puti("j_txt", head_r.clamp(0, 4))
        exn = torch.where(head_r == 1, 8, torch.where(head_r == 2, 4, torch.where(head_r == 3, 3, 0)))
        trav = torch.where(head_r == 1, 52.0, torch.where(head_r == 2, 31.2, 15.6))
        exa = torch.where(hit & (dtj >= 0) & (dtj < 1000), pa_j * (1 - e_out_quint(c01(dtj / 1000.0))),
                          torch.zeros_like(T))
        put("ex_a", exa); put("ex_p", 0.3 + 0.7 * e_out_quint(c01(dtj / 600.0))); put("ex_n", exn.float())
        put("ex_tr", trav)
        puti("hseed", g)
        # ------------------------------------------------ spinner
        self._spt = torch.full((M, 48, 8), -1.0, device=dev)
        if self.has_spinners:
            self._prepare_spinner(rs, put, T, s, e, pre, is_sp, end_j, end_t, head_r, dev)
        # ------------------------------------------------ follow points (incoming connection of g / outgoing of g+1)
        fp = self.fp
        put("fp_pre", 800.0 * torch.clamp(pre / 450.0, max=1.0))
        on_in = fp["on"][g] & valid
        nxt = (g + 1).clamp(max=self.G - 1)
        has_next_slot = torch.zeros_like(valid)
        has_next_slot[:, :-1] = gid[:, 1:] == (gid[:, :-1] + 1)
        on_out = fp["on"][nxt] & valid & (g + 1 < self.G) & ~has_next_slot
        for pref, gg, on in (("fp_", g, on_in), ("fpo_", nxt, on_out)):
            put(pref + "t0", fp["t0"][gg]); put(pref + "t1", fp["t1"][gg])
            put(pref + "sx", fp["sx"][gg]); put(pref + "sy", fp["sy"][gg])
            put(pref + "ex", fp["ex"][gg]); put(pref + "ey", fp["ey"][gg]); put(pref + "d", fp["d"][gg])
        puti("fp_on", on_in); puti("fpo_on", on_out)
        # ------------------------------------------------ culling boxes (actor px)
        R = 64 * q

        def box(name, x0, y0, x1, y1):
            put(name + "_x0", AX0 + APX * x0 - 1); put(name + "_y0", AY0 + APX * y0 - 1)
            put(name + "_x1", AX0 + APX * x1 + 1); put(name + "_y1", AY0 + APX * y1 + 1)
        big = 1e4
        if self.vkey.numel() > 0:
            pbx0, pby0, pbx1, pby1 = self._slider_box(g)
        else:
            pbx0 = pby0 = pbx1 = pby1 = torch.zeros_like(x)
        fr = 64 * q * torch.maximum(fsc, torch.ones_like(fsc)) + 2
        ox0 = torch.where(is_sl, torch.minimum(pbx0 - R, bxy[0] - fr), x - R * 1.75)
        oy0 = torch.where(is_sl, torch.minimum(pby0 - R, bxy[1] - fr), y - R * 1.75)
        ox1 = torch.where(is_sl, torch.maximum(pbx1 + R, bxy[0] + fr), x + R * 1.75)
        oy1 = torch.where(is_sl, torch.maximum(pby1 + R, bxy[1] + fr), y + R * 1.75)
        ox0 = torch.where(is_sp, torch.full_like(x, 256 - 260.0), ox0)
        oy0 = torch.where(is_sp, torch.full_like(x, 192 - 260.0), oy0)
        ox1 = torch.where(is_sp, torch.full_like(x, 256 + 260.0), ox1)
        oy1 = torch.where(is_sp, torch.full_like(x, 192 + 260.0), oy1)
        vis_obj = valid & (life | is_sp)
        ox0 = torch.where(vis_obj, ox0, torch.full_like(x, big))
        box("ob", ox0, oy0, ox1, oy1)
        ra = 64.0 * sc_ap * q * 1.07 + 2
        box("ab", x - ra, y - ra, x + ra, y + ra)
        jr = torch.maximum(R * 1.6 + 60 * q * 1.0, 60.0 * q + 10) + torch.where(miss, mdy, torch.zeros_like(T))
        jx0, jy0, jx1, jy1 = jx - jr, jy - jr, jx + jr, jy + jr
        # slider tick-miss popups lie on the path
        jx0 = torch.where(is_sl, torch.minimum(jx0, pbx0 - 12), jx0)
        jy0 = torch.where(is_sl, torch.minimum(jy0, pby0 - 12), jy0)
        jx1 = torch.where(is_sl, torch.maximum(jx1, pbx1 + 12), jx1)
        jy1 = torch.where(is_sl, torch.maximum(jy1, pby1 + 12), jy1)
        jx0 = torch.where(valid, jx0, torch.full_like(x, big))
        box("jb", jx0, jy0, jx1, jy1)
        inf_ = torch.full_like(x, 1e9)
        fx0 = torch.minimum(torch.where(on_in, torch.minimum(fp["sx"][g], fp["ex"][g]), inf_),
                            torch.where(on_out, torch.minimum(fp["sx"][nxt], fp["ex"][nxt]), inf_)) - 12
        fy0 = torch.minimum(torch.where(on_in, torch.minimum(fp["sy"][g], fp["ey"][g]), inf_),
                            torch.where(on_out, torch.minimum(fp["sy"][nxt], fp["ey"][nxt]), inf_)) - 12
        fx1 = torch.maximum(torch.where(on_in, torch.maximum(fp["sx"][g], fp["ex"][g]), -inf_),
                            torch.where(on_out, torch.maximum(fp["sx"][nxt], fp["ex"][nxt]), -inf_)) + 12
        fy1 = torch.maximum(torch.where(on_in, torch.maximum(fp["sy"][g], fp["ey"][g]), -inf_),
                            torch.where(on_out, torch.maximum(fp["sy"][nxt], fp["ey"][nxt]), -inf_)) + 12
        box("fb", fx0, fy0, fx1, fy1)
        # ------------------------------------------------ frame params
        F = torch.zeros((M, NF), dtype=torch.float32, device=dev)
        F[:, 0] = t
        cur = rs["cursor"].to(dev).float()
        F[:, 1], F[:, 2] = cur[:, 0], cur[:, 1]
        keys = rs["keys"].to(dev).long()
        pt = rs["press_t"].to(dev).float()
        rt = rs["release_t"].to(dev).float()
        ex_ = 1 + 0.2 * e_out_elastic_half(c01((t - pt) / 400.0))
        ex_ = torch.where(pt > -1e8, ex_, torch.ones_like(t))
        e0 = 1 + 0.2 * e_out_elastic_half(c01((rt - pt) / 400.0))
        ctr = e0 + (1 - e0) * e_out_quad(c01((t - rt) / 400.0))
        F[:, 3] = torch.where(keys > 0, ex_, torch.where((rt > -1e8) & (pt > -1e8), ctr, torch.ones_like(t)))
        self._prepare_breaks(t, rs["map"].to(dev).long(), cur, F)
        return P.contiguous(), I.contiguous(), F.contiguous()

    def _init_breaks(self, breaks, n_maps):
        """Per-map break periods with effect (duration >= 650 ms) as GPU tables."""
        dev = self.device
        rows, off, cnt = [], [], []
        for mi in range(n_maps):
            b = [] if breaks is None or mi >= len(breaks) else [tuple(x) for x in np.asarray(breaks[mi]).reshape(-1, 2)]
            b = sorted((float(s0), float(s1)) for s0, s1 in b if s1 - s0 >= 650.0)
            off.append(len(rows))
            cnt.append(len(b))
            rows.extend(b)
        self.brk = torch.tensor(rows if rows else [(0.0, 0.0)], dtype=torch.float32, device=dev)
        self.brk_off = torch.tensor(off, dtype=torch.long, device=dev)
        self.brk_cnt = torch.tensor(cnt, dtype=torch.long, device=dev)
        self.brk_max = max(cnt) if cnt else 0
        A = self.assets
        # static break texts: (font, text, size, x-anchor mode, y top, colour)
        W = 1365.3334

        def lay(font, text, size):
            g, (w, hh) = A.layout(font, text, size)
            return g, w
        self._brk_static = []
        gt, wt = lay(A.TORUS_BOLD, "CURRENT PROGRESS", 15.0)
        for gi, x0, y0, w, hh in gt:
            self._brk_static.append((gi, x0 - wt / 2, 15.0 + y0, 0.15, (1.0, 1.0, 1.0)))
        yel = tuple(np.array((0xff, 0xcc, 0x22)) / 255.0)
        for k, label in enumerate(("Accuracy", "Grade")):
            g, w = lay(A.TORUS_REG, label, 17.0)
            ytop = 35.0 + 17.0 * k
            for gi, x0, y0, ww, hh in g:
                self._brk_static.append((gi, -2.0 - w + x0, ytop + y0, 0.17, yel))

    def _prepare_breaks(self, t, mi, cur, F):
        dev = self.device
        M = t.shape[0]
        brt = torch.full((M, 40, 8), -1.0, device=dev)
        self._brt = brt
        if self.brk_max == 0:
            return
        b0 = torch.full_like(t, -1e9)
        b1 = torch.full_like(t, -1e9)
        for k in range(self.brk_max):
            idx = (self.brk_off[mi] + k).clamp(max=self.brk.shape[0] - 1)
            ok = k < self.brk_cnt[mi]
            s0, s1 = self.brk[idx, 0], self.brk[idx, 1]
            act = ok & (t >= s0) & (t < s1)
            b0 = torch.where(act, s0, b0)
            b1 = torch.where(act, s1, b1)
        on = b0 > -1e8
        c01 = _clamp01
        FADE = 325.0
        pe = b1 - FADE                               # period end (BreakTracker: EndTime - BREAK_FADE_DURATION)
        pdur = (pe - b0).clamp(min=1.0)
        fin = c01((t - b0) / FADE)
        alpha = torch.where(t < pe, fin, 1 - c01((t - pe) / FADE)) * on
        F[:, 4] = alpha
        W = 1365.3334
        # remaining time bar: adjustment box 0 -> 0.3 W (OutQuint, 325) then 0 at the period end; inner damped fraction
        adj = torch.where(t < pe, 0.3 * W * e_out_quint(fin), torch.zeros_like(t))
        frac = ((pe - (t - 57.7) - FADE) / pdur).clamp(min=0, max=1) * (1 - torch.pow(0.5, (t - b0).clamp(min=0) / 40.0))
        F[:, 5] = 0.5 * adj * frac * on
        # arrows (glow icons, parallax -0.01: scale 1.01 and a small mouse offset)
        offr = torch.where(t < pe, 0.6 + (0.22 - 0.6) * e_out_quint(fin),
                           (0.22 + (0.6 - 0.22) * e_out_quint(c01((t - pe) / FADE))))
        mux = (256.0 + 1.5 * cur[:, 0] - 640.0) / 0.9375
        muy = (84.0 + 1.5 * cur[:, 1] - 360.0) / 0.9375
        rx = torch.sign(mux) * (1 - torch.pow(0.999, mux.abs()))
        ry = torch.sign(muy) * (1 - torch.pow(0.999, muy.abs()))
        px, py = rx * (W / 2) * -0.01, ry * 384.0 * -0.01
        F[:, 6] = (-offr * W - 30.0) * 1.01 + px
        F[:, 7] = (offr * W + 30.0) * 1.01 + px
        F[:, 8] = py
        F[:, 9] = 1.01
        # texts
        A = self.assets
        n = 0
        cx_cnt = -50.0 + 50.0 * e_out_quint(fin)
        cx_info = 50.0 - 50.0 * e_out_quint(fin)
        secs = torch.ceil((b1 - t) / 1000.0).clamp(min=0)
        gi, x0, y0, val, w = self._digit_layout(A.VENERA_BOLD, 33.0, secs, 3)
        for k in range(3):
            brt[:, n, 0] = torch.where(val[:, k] & on, gi[:, k].float(), torch.full_like(t, -1.0))
            brt[:, n, 1] = cx_cnt - w / 2 + x0[:, k]
            brt[:, n, 2] = -15.0 - 33.0 + y0[:, k]
            brt[:, n, 3] = 0.33
            brt[:, n, 4:7] = 1.0
            brt[:, n, 7] = 1.0
            n += 1
        for gi_, x0_, y0_, sc_, col in self._brk_static:
            brt[:, n, 0] = torch.where(on, torch.full_like(t, float(gi_)), torch.full_like(t, -1.0))
            brt[:, n, 1] = cx_info + x0_
            brt[:, n, 2] = y0_
            brt[:, n, 3] = sc_
            brt[:, n, 4], brt[:, n, 5], brt[:, n, 6] = col
            brt[:, n, 7] = 1.0
            n += 1
        # values (Torus Bold 17, YellowLight): accuracy "100.00%" and grade "SS" unless rs provides accuracy
        yl = (1.0, 0xdd / 255.0, 0x55 / 255.0)
        for k, text in enumerate(("100.00%", "SS")):
            g, (wv, _) = A.layout(A.TORUS_BOLD, text, 17.0)
            ytop = 35.0 + 17.0 * k
            for gi_, x0_, y0_, ww, hh in g:
                if n >= 40:
                    break
                brt[:, n, 0] = torch.where(on, torch.full_like(t, float(gi_)), torch.full_like(t, -1.0))
                brt[:, n, 1] = cx_info + 2.0 + x0_
                brt[:, n, 2] = ytop + y0_
                brt[:, n, 3] = 0.17
                brt[:, n, 4], brt[:, n, 5], brt[:, n, 6] = yl
                brt[:, n, 7] = 1.0
                n += 1

    def _slider_box(self, g):
        if not hasattr(self, "_sbox"):
            o = self.o
            G = self.G
            bx0 = torch.zeros(G, device=self.device); by0 = torch.zeros(G, device=self.device)
            bx1 = torch.zeros(G, device=self.device); by1 = torch.zeros(G, device=self.device)
            sl = torch.nonzero(o["kind"] == 1).flatten()
            if len(sl):
                offs, cnts = o["path_off"][sl].long(), o["path_cnt"][sl].long()
                owner = torch.repeat_interleave(sl, cnts)
                idx = torch.arange(int(cnts.sum()), device=self.device) - torch.repeat_interleave(torch.cumsum(cnts, 0) - cnts, cnts)
                verts = torch.repeat_interleave(offs, cnts) + idx
                xx, yy = self.path[verts, 0], self.path[verts, 1]
                inf = torch.full((G,), float("inf"), device=self.device)
                bx0, by0 = inf.clone(), inf.clone()
                bx1, by1 = -inf.clone(), -inf.clone()
                bx0.scatter_reduce_(0, owner, xx, "amin"); by0.scatter_reduce_(0, owner, yy, "amin")
                bx1.scatter_reduce_(0, owner, xx, "amax"); by1.scatter_reduce_(0, owner, yy, "amax")
            self._sbox = (bx0, by0, bx1, by1)
        b = self._sbox
        return b[0][g], b[1][g], b[2][g], b[3][g]

    def _digit_layout(self, font, size, n, max_digits=6):
        """Per-frame layout of integer n (tensor [M]) as text: glyph idx [M,D], x0 [M,D] (units, text left at 0),
        y0 (units), valid [M,D], total width [M]."""
        A = self.assets
        key = (font, size)
        if key not in self._digit_cache:
            s = size / 100.0
            f = A.fonts[font]
            idx = torch.tensor([A.index[(font, ord(str(d)))] for d in range(10)], device=self.device)
            xo = torch.tensor([f["chars"][ord(str(d))][4] * s for d in range(10)], device=self.device)
            yo = torch.tensor([f["chars"][ord(str(d))][5] * s for d in range(10)], device=self.device)
            xa = torch.tensor([f["chars"][ord(str(d))][6] * s for d in range(10)], device=self.device)
            kern = torch.tensor([[f["kern"].get((ord(str(a)), ord(str(b))), 0) * s for b in range(10)]
                                 for a in range(10)], device=self.device)
            self._digit_cache[key] = (idx, xo, yo, xa, kern)
        idx, xo, yo, xa, kern = self._digit_cache[key]
        n = n.long().clamp(min=0, max=10 ** max_digits - 1)
        nd = torch.ones_like(n)
        for k in range(1, max_digits):
            nd = nd + (n >= 10 ** k).long()
        pos = torch.arange(max_digits, device=self.device)
        valid = pos[None, :] < nd[:, None]
        power = (nd[:, None] - 1 - pos[None, :]).clamp(min=0)
        dig = (n[:, None] // (10 ** power)) % 10
        dig = torch.where(valid, dig, torch.zeros_like(dig))
        adv = torch.where(valid, xa[dig], torch.zeros_like(xa[dig]))
        kp = torch.zeros_like(adv)
        kp[:, 1:] = torch.where(valid[:, 1:], kern[dig[:, :-1], dig[:, 1:]], torch.zeros_like(kp[:, 1:]))
        start = torch.cumsum(adv + kp, 1) - adv
        x0 = start + xo[dig]
        width = (adv + kp).sum(1)
        return idx[dig], x0, yo[dig], valid, width

    def _prepare_spinner(self, rs, put, T, s, e, pre, is_sp, end_j, end_t, head_r, dev):
        c01 = _clamp01
        P_ = put
        M, K = T.shape
        # DHO alpha: from start-preempt; FadeOut(240) at the result time
        a = torch.where(end_j, 1 - c01((T - end_t) / 240.0), torch.ones_like(T)) * (T >= s - pre) * is_sp
        P_("sp_a", a)
        t0 = s - pre
        # disc / centre scale: 0 -> 0.2|0.3 (P/4 OutQuint from t0+P/2) -> 1|0.8 (P/2 OutQuint from t0+P)
        p1 = c01((T - (t0 + pre / 2)) / (pre / 4))
        p2 = c01((T - (t0 + pre)) / (pre / 2))
        disc = 0.2 * e_out_quint(p1)
        disc = torch.where(T >= t0 + pre, 0.2 + 0.8 * e_out_quint(p2), disc)
        cen = 0.3 * e_out_quint(p1)
        cen = torch.where(T >= t0 + pre, 0.3 + 0.5 * e_out_quint(p2), cen)
        hit = (head_r >= 1) & (head_r <= 3)
        pr_ = c01((T - end_t) / 320.0)
        disc_end = torch.where(end_t >= t0 + pre, 0.2 + 0.8 * e_out_quint(c01((end_t - (t0 + pre)) / (pre / 2))),
                               0.2 * e_out_quint(c01((end_t - (t0 + pre / 2)) / (pre / 4))))
        disc = torch.where(end_j & hit & (T >= end_t), disc_end + (1.2 - disc_end) * e_out_quad(pr_), disc)
        disc = torch.where(end_j & (head_r == 4) & (T >= end_t), disc_end + (0.8 - disc_end) * e_in_quad(pr_), disc)
        P_("sp_disc", disc)
        gidx = rs["gid"].to(dev).long().clamp(min=0)
        trk = rs["spin_trk"].to(dev).float()
        total = rs["spin_total"].to(dev).float()
        rot = rs["spin_rot"].to(dev).float()
        req = self.o["spins_req"][gidx].float()
        maxb = self.o["max_bonus"][gidx].float()
        prog = torch.where(req > 0, c01(total / 360.0 / req.clamp(min=1)), torch.ones_like(total))
        done = prog >= 1
        # rotation rate estimate from the damping lag (visual rotation = credited rotation smoothed with tau 99.5 ms)
        tau = 99.5
        t_ref = torch.minimum(T, torch.where(end_j, end_t, e))
        lag = (total - rot.abs()).clamp(min=0) * torch.exp(((T - t_ref).clamp(min=0)) / tau)
        rate = lag / tau                                                    # deg / ms
        inv_rate = 1.0 / rate.clamp(min=1e-3)
        moving = rate > 0.05
        spun_for = torch.where(moving, total * inv_rate, torch.full_like(T, 1e6))      # ms since spinning began
        # tracking interpolation (Damp 0.985/ms): fill alpha 0.2..0.4, centre 80..40 (frozen once complete)
        trk_i = torch.where(trk > 0, 1 - torch.pow(0.985, (T - s).clamp(min=0)), torch.zeros_like(T))
        req1 = req.clamp(min=1)
        since_done = torch.where(moving, (total - 360.0 * req1) * inv_rate, torch.full_like(T, 1e6))
        since_done = since_done + (T - t_ref).clamp(min=0) + STEP_MS
        if "spin_tc" in rs:
            tcf = rs["spin_tc"].to(dev).float()
            since_done = torch.where(tcf > -1e8, T - tcf, since_done)
        # spin-start / completion times from a constant-rate model (progress rate rp per ms)
        rp = rate / (360.0 * req1)
        tc_abs = T - since_done
        t_a = torch.where(done, tc_abs - 1.0 / rp.clamp(min=1e-6), T - prog / rp.clamp(min=1e-6))
        t_a = torch.maximum(t_a, s)
        done_i = torch.where(done, 1 - torch.pow(0.985, (tc_abs - t_a).clamp(min=0)), trk_i)
        if "spin_ti" in rs:
            trk_i = rs["spin_ti"].to(dev).float()
            done_i = trk_i
        P_("sp_cen", cen)
        P_("sp_csize", 80.0 - 40.0 * done_i)
        # tick rotation: player rotation (damped visual) + ambient 25*dur/2000 deg over P+dur from t0+P/2; +180 on hit
        amb = 25 * (e - s) / 2000.0 * c01((T - (t0 + pre / 2)) / (pre + (e - s)))
        amb = amb + torch.where(end_j & hit, 180.0 * c01((T - end_t) / 320.0), torch.zeros_like(T))
        P_("sp_tick_rot", rot + amb)
        age = torch.where(moving, torch.remainder(total, 360.0) * inv_rate, torch.full_like(T, 1e6))
        age = age + (T - t_ref).clamp(min=0) + STEP_MS
        age = torch.where(trk > 0, age, torch.full_like(age, 1e6))        # not tracking: no new whole spins
        if "spin_last" in rs:
            sl_ = rs["spin_last"].to(dev).float()
            age = torch.where(sl_ > -1e8, T - sl_, torch.full_like(age, 1e6))
        # each new whole spin: FinishTransforms (-> 0.4), FadeTo(0.6, 60, OutExpo), then FadeTo(0.4, 250, OutQuint)
        flash = torch.where(age < 60, 0.4 + 0.2 * (1 - torch.pow(2.0, -10 * c01(age / 60.0))),
                            0.6 - 0.2 * e_out_quint(c01((age - 60) / 250.0)))
        # the hit-state change rewinds+clears the disc transforms: a completed fill drops to 0.4; an incomplete one
        # keeps damping its tracking interpolation towards 0 (tracking stops with the result)
        after = end_j & (T >= end_t)
        fill_a = torch.where(done, flash, 0.2 + 0.2 * trk_i)
        fill_end = torch.where(done, torch.full_like(T, 0.4),
                               0.2 + 0.2 * (total > 0).float() * torch.pow(0.985, (T - end_t).clamp(min=0)))
        P_("sp_fill_a", torch.where(after, fill_end, fill_a))
        # fill scale lerps towards its target with factor dt/100 (time constant ~100 ms): use the lagged progress
        # (exponential smoothing, time constant 100 ms, of the progress ramp; decays after completion)
        tf = 100.0
        lag_run = rp * tf * (1 - torch.exp(-(T - t_a).clamp(min=0) / tf))
        lag_c = rp * tf * (1 - torch.exp(-(tc_abs - t_a).clamp(min=0) / tf))
        prog_l = torch.where(done, 1 - lag_c * torch.exp(-since_done.clamp(min=0) / tf), prog - lag_run)
        prog_l = torch.where(req > 0, c01(prog_l), torch.ones_like(total))
        P_("sp_fill_s", 0.1 + 0.88 * prog_l)
        # ring arcs: DampContinuously(half-life 40 ms) from (0.31, 0.02) to (0.5, 0.044) once complete
        kd = torch.where(done, 1 - torch.pow(0.5, since_done.clamp(min=0) / 40.0), torch.zeros_like(T))
        P_("sp_arc", 0.31 + 0.19 * kd)
        P_("sp_arc_ir", 0.02 + 0.024 * kd)
        P_("sp_side_bg", torch.where(done, torch.zeros_like(prog), torch.full_like(prog, 0.25)))
        # side fill: alpha damps to 1 while 0 < progress < 1 (half-life 40 ms), then back to 0; progress lags ~58 ms
        up = 1 - torch.pow(0.5, spun_for.clamp(min=0) / 40.0)
        side_a = torch.where(done, up * torch.pow(0.5, since_done.clamp(min=0) / 40.0), up * (prog > 0))
        P_("sp_side_a", side_a)
        prog_l2 = torch.where(req > 0, c01((total - rate * 57.7).clamp(min=0) / 360.0 / req1), torch.ones_like(total))
        side_p = torch.where(done, 0.15 * torch.pow(0.5, since_done.clamp(min=0) / 40.0), 0.15 * prog_l2)
        P_("sp_side_p", side_p)
        # ---- texts: first spinner slot per frame
        spt = self._spt
        spt[:M].fill_(-1.0)
        has = is_sp.any(1)
        kk = torch.arange(K, device=dev, dtype=torch.float32)
        j = torch.argmax(torch.where(is_sp, a * 2.0 + kk / 1000.0, torch.full_like(a, -1.0)), 1)
        ar = torch.arange(M, device=dev)
        A = self.assets
        started = (total > 0) | (trk > 0)
        fin = self.mp["fade_in"][rs["map"].to(dev).long()]
        spm_a = (c01((T - s) / fin[:, None]) * started)[ar, j] * has
        if "spin_ts" in rs:
            ts_ = rs["spin_ts"].to(dev).float()
            spm_a = torch.where(ts_ > -1e8, c01((T - ts_) / fin[:, None]), torch.zeros_like(T))[ar, j] * has
        # SPM over a 595 ms window that also covers the time before spinning started
        spm = torch.floor(rate * torch.clamp(spun_for / 595.0, max=1.0) * 1000 * 60 / 360)[ar, j]
        if "spin_spm" in rs:
            spm = torch.trunc(rs["spin_spm"].to(dev).float())[ar, j]
        n = 0

        def emit(gi, x0, y0, sc, col, al, valid):
            nonlocal n
            D = gi.shape[1]
            for k in range(D):
                ok = valid[:, k] & (al > 0.002)
                spt[:M, n, 0] = torch.where(ok, gi[:, k].float(), torch.full_like(al, -1.0))
                spt[:M, n, 1] = x0[:, k]
                spt[:M, n, 2] = y0[:, k]
                spt[:M, n, 3] = sc
                spt[:M, n, 4] = col[0]
                spt[:M, n, 5] = col[1]
                spt[:M, n, 6] = col[2]
                spt[:M, n, 7] = al
                n += 1

        semi = A.TORUS_SEMI
        gi, x0, y0, val, w = self._digit_layout(semi, 28.0, spm, 4)
        for shadow in (True, False):
            dy_ = 0.06 * 28.0 if shadow else 0.0
            emit(gi, 256.0 - w[:, None] / 2 + x0, 192.0 + 60.0 + y0 + dy_, torch.full_like(spm_a, 0.28),
                 (0.0, 0.0, 0.0) if shadow else (1.0, 1.0, 1.0), spm_a * (0.2 if shadow else 1.0), val)
        gi2, xs, ys, W = self._spm_lay
        gi2, xs, ys = gi2[None].expand(M, -1), xs[None].expand(M, -1), ys[None].expand(M, -1)
        v2 = torch.ones_like(gi2, dtype=torch.bool)
        for shadow in (True, False):
            dy_ = 0.06 * 16.0 if shadow else 0.0
            emit(gi2, 256.0 - W / 2 + xs, 192.0 + 90.0 + ys + dy_, torch.full_like(spm_a, 0.16),
                 (0.0, 0.0, 0.0) if shadow else (1.0, 1.0, 1.0), spm_a * (0.2 if shadow else 1.0), v2)
        # bonus counter (Bold 28 at Y=-100): 50 * bonus spins, or "MAX"
        comp = torch.floor(total / 360.0)
        bspins = (comp - (req + 2)).clamp(min=0)
        bspins = torch.minimum(bspins, maxb)
        bact = ((bspins > 0) & ~(end_j & (T >= end_t)))[ar, j] & has      # counter transforms cleared at the result
        is_max = ((bspins >= maxb) & (maxb > 0))[ar, j]
        bage = age[ar, j]
        b_sc = torch.where(is_max, 1.5 + 1.3 * e_out_quint(c01(bage / 1000.0)), 1.5 - 0.5 * e_out_quint(c01(bage / 1000.0)))
        b_al = torch.where(is_max, 1 - c01(bage / 500.0), 1 - c01(bage / 1500.0)) * bact
        pk = c01(bage / 400.0)
        pink_l = self._pink_l
        mcol = srgb(pink_l[None] + (1 - pink_l[None]) * pk[:, None])
        bcol = torch.where(is_max[:, None], mcol, torch.ones_like(mcol))
        bold = A.TORUS_BOLD
        gi3, x3, y3, v3, w3 = self._digit_layout(bold, 28.0, (50 * bspins)[ar, j], 5)
        gim, xm, ym, vm, Wm = self._max_lay
        gim, xm, ym, vm = (v[None].expand(M, -1) for v in (gim, xm, ym, vm))
        gi3 = torch.where(is_max[:, None], gim, gi3)
        x3 = torch.where(is_max[:, None], xm, x3)
        y3 = torch.where(is_max[:, None], ym, y3)
        v3 = torch.where(is_max[:, None], vm, v3)
        w3 = torch.where(is_max, torch.full_like(w3, Wm), w3)
        # scale about the text centre (box 28 tall centred at y = -100)
        bx = 256.0 + (x3 - w3[:, None] / 2) * b_sc[:, None]
        by = 192.0 - 100.0 + (y3 - 14.0) * b_sc[:, None]
        emit(gi3, bx, by, 0.28 * b_sc, (bcol[:, 0], bcol[:, 1], bcol[:, 2]), b_al, v3)

    # --------------------------------------------------------------------------------------------- render
    @torch.no_grad()
    def render(self, rs, aug=None):
        """rs: dict of tensors (RENDER_INTERFACE.md) -> uint8 [M, 3, 192, 256] on the renderer device.

        The whole pipeline (parameter preparation, tile masks, raster kernel) is captured once per (M, rs layout)
        in a CUDA graph and replayed (the preparation is ~1k small torch ops; eager launches would dominate)."""
        rs = {k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))) for k, v in rs.items()}
        if self.graphs and self.device.type == "cuda" and getattr(self, "debug_hook", None) is None:
            out = self._render_graph(rs)
        else:
            out = self._render_eager(rs)
        if aug is not None:
            out = _augment(out, aug)
        return out

    def _render_eager(self, rs):
        P, I, F = self._prepare(rs)
        if getattr(self, "debug_hook", None) is not None:      # calibration experiments only
            self.debug_hook(P, I, F, rs)
        return self._launch(P, I, F)

    def _render_graph(self, rs):
        M = int(rs["t"].shape[0])
        key = (M,) + tuple(sorted((k, str(v.dtype), tuple(v.shape[1:])) for k, v in rs.items()))
        entry = self._graphs.get(key)
        if entry is None:
            static = {k: torch.zeros(v.shape, dtype=v.dtype, device=self.device) for k, v in rs.items()}
            for k, v in rs.items():
                static[k].copy_(v)
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):                                # warm-up: caches, autotune, allocator
                    self._render_eager(static)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._render_eager(static)
            entry = (graph, static, out)
            if len(self._graphs) > 8:
                self._graphs.clear()
            self._graphs[key] = entry
        graph, static, out = entry
        for k, v in rs.items():
            static[k].copy_(v, non_blocking=True)
        graph.replay()
        return out.clone()

    def _masks(self, P, I):
        """Per (frame, tile, pass) bitmasks of the slots whose culling box overlaps the tile:
        0 spinner, 1 follow points, 2 ring explosions, 3 hit objects, 4 judgements/tick misses, 5 approach."""
        M, K = P.shape[0], P.shape[1]
        nx, ny = W_ACT // self.TW, H_ACT // self.TH
        out = torch.empty((M, nx * ny, 6), dtype=torch.int32, device=P.device)
        _mask_kernel[(M,)](out, P, I, K=K, NPc=NP, NIc=NI, TH=self.TH, TW=self.TW, TILES_X=nx, NT=nx * ny,
                           NTP=triton.next_power_of_2(nx * ny), num_warps=4)
        return out

    def _launch(self, P, I, F):
        M, K = P.shape[0], P.shape[1]
        out = torch.empty((M, 3, H_ACT, W_ACT), dtype=torch.uint8, device=self.device)
        TH, TW = self.TH, self.TW
        tiles_x, tiles_y = W_ACT // TW, H_ACT // TH
        grid = (tiles_x * tiles_y, M)
        _render_kernel[grid](out, P, I, F, self.path, self.chunk, self.nest, self.sat, self.gly, self.txt,
                             self.glow, self._spt, self._brt, self._masks(P, I), tiles_x * tiles_y, K=K, NPc=NP, NIc=NI, NFc=NF, TH=TH, TW=TW, SS=self.ss,
                             TILES_X=tiles_x, CHUNK=self.CHUNK, G_CHEV_L=self.assets.index[("fa", 0xF053)],
                             num_warps=4)
        return out


def _augment(img, aug):
    """Optional domain randomisation: brightness/contrast multipliers and gaussian noise (per frame)."""
    x = img.float()
    if "brightness" in aug:
        x = x * aug["brightness"].view(-1, 1, 1, 1)
    if "contrast" in aug:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * aug["contrast"].view(-1, 1, 1, 1) + mean
    if "noise" in aug:
        g = torch.Generator(device=x.device).manual_seed(int(aug.get("seed", 0)))
        x = x + torch.randn(x.shape, generator=g, device=x.device) * aug["noise"].view(-1, 1, 1, 1)
    return x.clamp(0, 255).round().to(torch.uint8)
