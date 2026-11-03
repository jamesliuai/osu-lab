"""Fast recurrent visual policy for osu-sim.

Cursor heads (both issue the same absolute cursor + 4-way held-key command as the spec's B0):
  * "b0": spec Section 7.2 global tanh-Normal from actor memory;
  * "s1": spec Section 21.2-21.4 spatial cell/residual head: per 12x16 CNN cell a shared MLP on the local
          64-channel feature + projected LSTM context + fixed cell centre -> (cell logit, residual mean/log-std),
          calibrated cell prior (canonical B0 prior, sigma0 = exp(-0.7)), exact density and all-cell entropy.
Key head (all arms): Linear(hidden+2 -> 128) ReLU Linear(128 -> 4), conditioned on the chosen cursor.
  labelled (session 3) architecture deviation, opt-in key_local=True (s1 only): the spec 21 key MLP reads memory +
  chosen cursor only; key_local adds a bias-free Linear(trunk -> 128) of the chosen cell's S1 local feature to the key
  MLP's first-layer pre-activation. Zero-initialised, so an agent grown into it is function-identical.

Labelled executive throughput choices: optional shared actor/critic trunk, optional patch4 stem, bf16
autocast for the convolutional encoder only (LSTM, heads, log-probs, losses stay float32), cuDNN over
packed reset-free segments for multi-step passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

H, W = 192, 256
AUX = 8
KEY_PRIOR = (0.96, 0.018, 0.018, 0.004)
GX, GY = 16, 12
NCELL = GX * GY
LOG_STD_BOUNDS = (-7.0, 0.5)


def layer_init(module, gain=math.sqrt(2)):
    nn.init.orthogonal_(module.weight, gain)
    nn.init.zeros_(module.bias)
    return module


def log_tanh_jacobian(latent: Tensor) -> Tensor:
    return 2 * (math.log(2) - latent - F.softplus(-2 * latent))


def normal_logp(x, mean, log_std):
    return -0.5 * ((x - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)


class ResBlock(nn.Module):
    """x + conv(relu(conv(x))) on the 12x16 map; the second conv starts at zero so a block added to a trained
    encoder leaves its function unchanged (labelled capacity growth: +32-64 px receptive field per block)."""

    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.c1 = layer_init(nn.Conv2d(ch, ch, 3, 1, dilation, dilation=dilation))
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x)))


class Encoder(nn.Module):
    def __init__(self, kind: str = "b0", width: float = 1, res_blocks: int = 0, proj_dim: int = 256,
                 fp32_projection: bool = False, trunk: int = 64):
        super().__init__()
        self.trunk = trunk      # channels at the 24x32 and 12x16 levels (labelled growth path: 64 -> wider)
        # Optional float32 projection controls rounding at the encoder projection.
        self.fp32_projection = fp32_projection
        c = lambda n: int(n * width)
        if kind == "b0":
            convs = [nn.Conv2d(3, c(16), 5, 2, 2), nn.Conv2d(c(16), c(32), 3, 2, 1),
                     nn.Conv2d(c(32), trunk, 3, 2, 1), nn.Conv2d(trunk, trunk, 3, 2, 1)]
        elif kind == "patch4":
            convs = [nn.Conv2d(3, c(32), 4, 4), nn.Conv2d(c(32), trunk, 3, 2, 1), nn.Conv2d(trunk, trunk, 3, 2, 1)]
        else:
            raise ValueError(f"unknown encoder {kind}")
        layers = []
        for conv in convs:
            layers += [layer_init(conv), nn.ReLU()]
        self.kind = kind
        self.convs = nn.Sequential(*layers)
        self.res = nn.Sequential(*[ResBlock(trunk, 1 + i % 2) for i in range(res_blocks)])
        self.projection = nn.Sequential(nn.Flatten(), layer_init(nn.Linear(trunk * 12 * 16, proj_dim)), nn.ReLU())

    def forward(self, rgb: Tensor):
        """rgb uint8 [M,3,192,256] -> (features float32 [M,proj_dim], spatial bf16/f32 [M,64,12,16])."""
        dt = torch.bfloat16 if rgb.is_cuda else torch.float32
        with torch.autocast("cuda", torch.bfloat16, enabled=rgb.is_cuda):
            if self.kind == "patch4":
                # the 4x4/stride-4 stem is a per-patch linear map; computing it as one GEMM on [M,48,64,(c,kh,kw)]
                # patches is the same function with the same parameters.
                M = rgb.shape[0]
                conv = self.convs[0]
                p = rgb.reshape(M, 3, 48, 4, 64, 4).permute(0, 2, 4, 1, 3, 5).reshape(M, 48, 64, 48)
                y = F.linear(p.to(dt) * (1.0 / 255.0), conv.weight.reshape(conv.out_channels, 48), conv.bias)
                x = F.relu(y).permute(0, 3, 1, 2)              # NCHW view with channels_last strides
                spatial = self.convs[2:](x)
            else:
                x = rgb.contiguous(memory_format=torch.channels_last)
                x = x.to(dt) * (1.0 / 255.0)
                spatial = self.convs(x)
            spatial = self.res(spatial)
            if not self.fp32_projection:
                y = self.projection(spatial)
        if self.fp32_projection:
            y = self.projection(spatial.float())
        return y.float(), spatial


class RecurrentCore(nn.Module):
    def __init__(self, encoder: str = "b0", hidden: int = 256, width: float = 1, res_blocks: int = 0,
                 proj_dim: int = 256, fp32_projection: bool = False, trunk: int = 64):
        super().__init__()
        self.encoder = Encoder(encoder, width, res_blocks, proj_dim, fp32_projection, trunk)
        self.hidden = hidden
        self.proj_dim = proj_dim
        self.memory = nn.LSTM(proj_dim + AUX, hidden)
        for name, p in self.memory.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            else:
                nn.init.orthogonal_(p)

    def step(self, rgb, aux, state, reset):
        """One decision. rgb [B,3,H,W] u8, aux [B,8], state [B,2,hid], reset [B]. -> hidden, spatial, state."""
        f, spatial = self.encoder(rgb)
        feat = torch.cat((f, aux), -1)
        keep = (~reset).float().view(-1, 1, 1)
        state = state * keep
        h0 = state[:, 0].unsqueeze(0).contiguous()
        c0 = state[:, 1].unsqueeze(0).contiguous()
        out, (h1, c1) = self.memory(feat.unsqueeze(0), (h0, c0))
        return out[0], spatial, torch.stack((h1[0], c1[0]), 1)

    def sequence(self, rgb, aux, state0, reset):
        """rgb [T,B,3,H,W], aux [T,B,8], state0 [B,2,hid], reset [T,B] -> hidden [T,B,hid], spatial [T*B,64,12,16].

        Splits every column at its resets and runs all segments in one packed cuDNN call; segments that start
        at t=0 without a reset use state0.
        """
        T, B = reset.shape
        f, spatial = self.encoder(rgb.reshape(T * B, 3, H, W))
        feat = torch.cat((f.reshape(T, B, self.proj_dim), aux), -1)
        dev = reset.device
        starts = reset.clone()
        starts[0] = True
        seg_local = starts.long().cumsum(0) - 1
        nseg = seg_local[-1] + 1
        offset = nseg.cumsum(0) - nseg
        seg = seg_local + offset[None]
        total = int(nseg.sum())
        tidx = torch.arange(T, device=dev).view(T, 1).expand(T, B)
        seg_start = torch.zeros(total, dtype=torch.long, device=dev)
        seg_start[seg[starts]] = tidx[starts]
        pos = tidx - seg_start[seg]
        lengths = torch.bincount(seg.flatten(), minlength=total)
        lmax = int(lengths.max())
        padded = feat.new_zeros(lmax, total, feat.shape[-1])
        padded[pos, seg] = feat
        h0 = feat.new_zeros(1, total, self.hidden)
        c0 = feat.new_zeros(1, total, self.hidden)
        first = seg[0]
        keep = (~reset[0]).float().view(B, 1)
        h0[0, first] = state0[:, 0] * keep
        c0[0, first] = state0[:, 1] * keep
        packed = pack_padded_sequence(padded, lengths.cpu(), enforce_sorted=False)
        out, _ = self.memory(packed, (h0, c0))
        out, _ = pad_packed_sequence(out, total_length=lmax)
        return out[pos, seg], spatial


@dataclass
class Action:
    latent: Tensor    # [...,2] pre-tanh (global for b0, residual for s1)
    cursor: Tensor    # [...,2] in [-1,1]
    key: Tensor       # [...] int64 0..3
    log_prob: Tensor  # [...]
    cell: Tensor      # [...] int64 (s1 cell index; 0 for b0)


def _cell_centres(device):
    idx = torch.arange(NCELL, device=device)
    return torch.stack((-1 + (2 * (idx % GX) + 1).float() / GX, -1 + (2 * (idx // GX) + 1).float() / GY), -1)


def cell_of(pos):
    """labelled (session 5): S1 cell index of action-space positions [..., 2] (column-major x, row y, as _cell_centres)."""
    col = ((pos[..., 0] + 1) * (GX / 2)).floor().long().clamp(0, GX - 1)
    row = ((pos[..., 1] + 1) * (GY / 2)).floor().long().clamp(0, GY - 1)
    return row * GX + col


def key_input(cursor, cell, prev, vmax):
    """labelled (session 5) CURSOR PHYSICS: the position and S1 cell the key head reads. Without physics (vmax None) the
    chosen cursor and cell (old behaviour); with it the EXECUTED position (osu_sim.motion.physics_step_torch from the
    previous executed position prev = aux[..., 0:2]) and its cell: a click lands where the cursor actually is. Where the
    cap does not bind this is the chosen cursor and cell exactly. cursor/prev [..., 2], cell/vmax [...]."""
    if vmax is None:
        return cursor, cell
    from .motion import physics_step_torch
    ex, bind = physics_step_torch(prev, cursor, vmax)
    return ex, (None if cell is None else torch.where(bind, cell_of(ex), cell))


class Agent(nn.Module):
    def __init__(self, encoder: str = "b0", shared: bool = True, hidden: int = 256,
                 cursor_log_std_init: float = -1.6, width: float = 1, cursor_head: str = "b0",
                 residual_log_std_min: float = LOG_STD_BOUNDS[0], res_blocks: int = 0, proj_dim: int = 256,
                 fp32_projection: bool = False, trunk: int = 64, key_local: bool = False):
        super().__init__()
        key_local = bool(key_local)
        if key_local and cursor_head != "s1":
            raise ValueError("key_local needs the s1 cursor head (the b0 head has no per-cell local feature)")
        self.config = dict(encoder=encoder, shared=shared, hidden=hidden, cursor_log_std_init=cursor_log_std_init,
                           width=width, cursor_head=cursor_head, residual_log_std_min=residual_log_std_min,
                           res_blocks=res_blocks, proj_dim=proj_dim, fp32_projection=fp32_projection, trunk=trunk,
                           key_local=key_local)
        # labelled: cursor commands are rounded to integer client pixels by the host, so sub-pixel residual noise
        # carries no meaning but makes PPO ratios hypersensitive; -3.0 ~ 1.5 client px at S1 cell scale
        self.log_std_lo = float(residual_log_std_min)
        self.head = cursor_head
        self.actor_core = RecurrentCore(encoder, hidden, width, res_blocks, proj_dim, fp32_projection, trunk)
        self.critic_core = None if shared else RecurrentCore(encoder, hidden, width, res_blocks, proj_dim,
                                                             fp32_projection, trunk)
        if cursor_head == "b0":
            self.cursor_mean = layer_init(nn.Linear(hidden, 2), 0.01)
            self.cursor_log_std = layer_init(nn.Linear(hidden, 2), 0.01)
            nn.init.constant_(self.cursor_log_std.bias, cursor_log_std_init)
        elif cursor_head == "s1":
            from osu_rl.policy.priors import calibrated_prior
            prior = calibrated_prior()
            self.register_buffer("prior_log_masses", torch.tensor(prior.log_masses, dtype=torch.float32))
            self.register_buffer("prior_means", torch.tensor(prior.means, dtype=torch.float32))
            self.register_buffer("prior_log_stds", torch.tensor(prior.log_stds, dtype=torch.float32))
            self.register_buffer("cell_centres", _cell_centres("cpu"))
            self.register_buffer("half_sizes", torch.tensor((1 / GX, 1 / GY)))
            self.context = layer_init(nn.Linear(hidden, trunk))
            self.readout = nn.Sequential(layer_init(nn.Linear(trunk + 2, 64)), nn.ReLU(),
                                         layer_init(nn.Linear(64, 5), 0.01))
        else:
            raise ValueError(cursor_head)
        self.key_head = nn.Sequential(layer_init(nn.Linear(hidden + 2, 128)), nn.ReLU(),
                                      layer_init(nn.Linear(128, 4), 0.01))
        with torch.no_grad():
            self.key_head[-1].bias.copy_(torch.tensor(KEY_PRIOR).log())
        self.value_head = nn.Sequential(layer_init(nn.Linear(hidden, 128)), nn.ReLU(),
                                        layer_init(nn.Linear(128, 1), 1.0))
        # labelled (session 3) deviation, see module docstring. Registered last so every pre-existing parameter keeps
        # its index; None (no parameter, no state_dict entry) by default so old checkpoints load strictly.
        self.key_local = None
        if key_local:
            self.key_local = nn.Linear(trunk, 128, bias=False)
            nn.init.zeros_(self.key_local.weight)

    # ---------------------------------------------------------------- state
    @property
    def state_slots(self) -> int:
        return 2 if self.critic_core is None else 4

    def initial_state(self, batch: int, device) -> Tensor:
        return torch.zeros(batch, self.state_slots, self.actor_core.hidden, device=device)

    # ---------------------------------------------------------------- heads
    def _key_logits(self, hidden, cursor, local=None):
        """Exact factored W_h h + W_c c + b; cursor may carry an extra cell dimension.

        key_local agents also add key_local(local), local = the S1 local feature of the cursor's cell ([...,trunk],
        with the same optional cell dimension as cursor); it is required then, so a caller cannot silently drop it."""
        l0 = self.key_head[0]
        hd = hidden.shape[-1]
        a = F.linear(hidden, l0.weight[:, :hd], l0.bias)
        if cursor.dim() == hidden.dim() + 1:
            a = a.unsqueeze(-2)
        if self.key_local is None:
            return self.key_head[2](F.relu(a + F.linear(cursor, l0.weight[:, hd:])))
        if local is None:
            raise ValueError("key_local agent: _key_logits needs the chosen cell's local feature")
        # accumulate in place into one [...,128] buffer: the all-cell entropy pass ([TB,192,128]) then holds no more
        # than the old W_h h + W_c c path (no autograd hazard: neither linear saves its output for backward)
        pre = self.key_local(local)
        pre.add_(F.linear(cursor, l0.weight[:, hd:])).add_(a)
        return self.key_head[2](F.relu(pre))

    def _s1_params(self, hidden, spatial, return_local: bool = False):
        """hidden [B,hid], spatial [B,64,12,16] -> logits [B,192], mean [B,192,2], log_std [B,192,2]
        (+ local [B,192,trunk], the per-cell feature the readout sees, with return_local)."""
        local = spatial.float().flatten(-2).transpose(-1, -2) + self.context(hidden).unsqueeze(-2)
        coords = self.cell_centres.expand(hidden.shape[0], NCELL, 2)
        out = self.readout(torch.cat((local, coords), -1))
        logits = self.prior_log_masses + out[..., 0]
        mean = self.prior_means + out[..., 1:3]
        log_std = (self.prior_log_stds + out[..., 3:5]).clamp(self.log_std_lo, LOG_STD_BOUNDS[1])
        if return_local:
            return logits, mean, log_std, local
        return logits, mean, log_std

    def _cursor_dist(self, hidden, spatial):
        if self.head == "b0":
            return self.cursor_mean(hidden), self.cursor_log_std(hidden).clamp(self.log_std_lo, LOG_STD_BOUNDS[1])
        return self._s1_params(hidden, spatial)

    def _sample(self, hidden, spatial, mode, temperature=1.0, prev=None, vmax=None):
        """prev/vmax (labelled (session 5) cursor physics): previous executed position [B,2] and cap [B] for the key
        head's input (key_input); None = old behaviour."""
        B = hidden.shape[0]
        klocal = None
        if self.head == "b0":
            mean, log_std = self._cursor_dist(hidden, spatial)
            latent = mean + log_std.exp() * temperature * torch.randn_like(mean) if mode == "sample" else mean
            cursor = latent.tanh()
            cell = torch.zeros(B, dtype=torch.long, device=hidden.device)
            kcur = key_input(cursor, cell, prev, vmax)[0]
            cursor_lp = (normal_logp(latent, mean, log_std) - log_tanh_jacobian(latent)).sum(-1)
        else:
            logits, mean, log_std, local = self._s1_params(hidden, spatial, return_local=True)
            # labelled (session 3) evaluation decoders: "csample" samples the cursor (cell + residual) and decodes keys
            # deterministically; "gated:<tau>" takes the arg-max cell when its probability is >= tau and samples the
            # cursor otherwise (the policy's spread-out spinner distribution completes spinners only when sampled)
            samp_cursor = None
            if mode == "sample" or mode == "csample":
                samp_cursor = torch.ones(B, dtype=torch.bool, device=hidden.device)
            elif mode.startswith("gated"):
                tau = float(mode.split(":")[1]) if ":" in mode else 0.5
                samp_cursor = logits.softmax(-1).max(-1).values < tau
            if samp_cursor is not None:
                drawn = torch.distributions.Categorical(logits=logits / temperature).sample()
                cell = torch.where(samp_cursor, drawn, logits.argmax(-1))
            else:
                cell = logits.argmax(-1)
            m = mean.gather(1, cell.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
            ls = log_std.gather(1, cell.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
            if samp_cursor is not None:
                latent = torch.where(samp_cursor.view(B, 1), m + ls.exp() * temperature * torch.randn_like(m), m)
            else:
                latent = m
            cursor = self.cell_centres[cell] + latent.tanh() * self.half_sizes
            cursor_lp = (logits.log_softmax(-1).gather(1, cell.view(B, 1)).squeeze(1)
                         + (normal_logp(latent, m, ls) - log_tanh_jacobian(latent)).sum(-1) + math.log(NCELL))
            kcur, kcell = key_input(cursor, cell, prev, vmax)
            if self.key_local is not None:
                klocal = local[torch.arange(B, device=cell.device), kcell]         # [B,trunk] of the key head's cell
        klog = self._key_logits(hidden, kcur, klocal)
        if mode == "sample":
            key = torch.distributions.Categorical(logits=klog / temperature).sample()
        elif mode == "deterministic" or mode == "csample" or mode.startswith("gated"):
            key = klog.argmax(-1)
        elif mode == "any_key":
            p = klog.softmax(-1)
            press = p[:, 1:].sum(-1) > p[:, 0]
            key = torch.where(press, p[:, 1:].argmax(-1) + 1, torch.zeros_like(press, dtype=torch.long))
        else:
            raise ValueError(mode)
        lp = cursor_lp + klog.log_softmax(-1).gather(-1, key.view(-1, 1)).squeeze(-1)
        return Action(latent, cursor, key, lp, cell)

    @torch.no_grad()
    def act(self, rgb, aux, state, reset, mode: str = "sample", temperature: float = 1.0, vmax=None):
        """vmax [B] (labelled (session 5) cursor physics, osu px; None = off): the key head reads the executed position."""
        ha, spatial, sa = self.actor_core.step(rgb, aux, state[:, :2], reset)
        if self.critic_core is None:
            hc, new_state = ha, sa
        else:
            hc, _, sc = self.critic_core.step(rgb, aux, state[:, 2:], reset)
            new_state = torch.cat((sa, sc), 1)
        value = self.value_head(hc).squeeze(-1)
        return self._sample(ha, spatial, mode, temperature, prev=aux[:, :2], vmax=vmax), value, new_state

    @torch.no_grad()
    def anchor_step(self, rgb, aux, state, reset, cursor, cell, vmax=None):
        """labelled (session 5) KL-anchor reference pass (frozen pre-fine-tune actor, stepped alongside the policy on the
        same frame and aux): log-softmax of the key head given the TAKEN cursor (the same key input as the policy:
        key_input) and log-softmax of the S1 cell distribution. -> (key_lp [B,4], cell_lp [B,192], new actor state)."""
        if self.head != "s1":
            raise ValueError("anchor_step needs the s1 cursor head")
        ha, spatial, sa = self.actor_core.step(rgb, aux, state[:, :2], reset)
        logits, _, _, local = self._s1_params(ha, spatial, return_local=True)
        kcur, kcell = key_input(cursor, cell, aux[:, :2], vmax)
        klocal = None
        if self.key_local is not None:
            klocal = local[torch.arange(ha.shape[0], device=ha.device), kcell]
        klog = self._key_logits(ha, kcur, klocal)
        return klog.log_softmax(-1), logits.log_softmax(-1), sa

    def evaluate(self, rgb, aux, state0, reset, latent, cursor, key, cell, return_any: bool = False, vmax=None,
                 return_dists: bool = False):
        """PPO re-evaluation over [T,B]. Returns log_prob, cursor_entropy, key_entropy, value (all [T,B]);
        with return_any also log P(any key down | taken cursor) [T,B] (for the training-only hold floor).
        labelled (session 5): vmax [T,B] (cursor physics; None = off) places the key head's input at the executed
        position (key_input); return_dists appends a dict with the key log-softmax [T,B,4] and (s1) the cell
        log-softmax [T,B,192] of the taken steps (KL anchor)."""
        T, B = reset.shape
        ha, spatial = self.actor_core.sequence(rgb, aux, state0[:, :2], reset)
        hc = ha if self.critic_core is None else self.critic_core.sequence(rgb, aux, state0[:, 2:], reset)[0]
        value = self.value_head(hc).squeeze(-1)
        h = ha.reshape(T * B, -1)
        lat = latent.reshape(T * B, 2)
        cur = cursor.reshape(T * B, 2)
        k = key.reshape(T * B)
        prev = aux[..., :2].reshape(T * B, 2)
        vm = None if vmax is None else vmax.reshape(T * B)
        klocal = None
        logp_cell = None
        if self.head == "b0":
            mean, log_std = self._cursor_dist(h, None)
            cursor_lp = (normal_logp(lat, mean, log_std) - log_tanh_jacobian(lat)).sum(-1)
            fresh = mean + log_std.exp() * torch.randn_like(mean)
            cursor_ent = (0.5 + 0.5 * math.log(2 * math.pi) + log_std + log_tanh_jacobian(fresh)).sum(-1)
            key_ent = torch.distributions.Categorical(
                logits=self._key_logits(h, key_input(fresh.tanh(), None, prev, vm)[0])).entropy()
            kcur = key_input(cur, None, prev, vm)[0]
        else:
            c = cell.reshape(T * B)
            logits, mean, log_std, local = self._s1_params(h, spatial, return_local=True)
            logp_cell = logits.log_softmax(-1)
            m = mean.gather(1, c.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            ls = log_std.gather(1, c.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            cursor_lp = (logp_cell.gather(1, c.view(-1, 1)).squeeze(1)
                         + (normal_logp(lat, m, ls) - log_tanh_jacobian(lat)).sum(-1) + math.log(NCELL))
            # spec 21.4 all-cell entropy with one fresh residual per cell (cell-logit gradients retained)
            p_cell = logp_cell.exp()
            fresh = mean + log_std.exp() * torch.randn_like(mean)
            fresh_cursor = self.cell_centres + fresh.tanh() * self.half_sizes
            resid_ent = (0.5 + 0.5 * math.log(2 * math.pi) + log_std + log_tanh_jacobian(fresh)).sum(-1) \
                - math.log(NCELL)
            cursor_ent = -(p_cell * logp_cell).sum(-1) + (p_cell * resid_ent).sum(-1)
            kcur, kcell = key_input(cur, c, prev, vm)
            if self.key_local is not None:
                klocal = local[torch.arange(T * B, device=c.device), kcell]         # [TB,trunk] of the key head's cell
            if vm is None:
                kl_all = self._key_logits(h, fresh_cursor, None if klocal is None else local)   # [TB,192,4]
            else:
                # labelled (session 5): each candidate cell's key distribution at ITS executed position and cell
                allc = torch.arange(NCELL, device=c.device).expand(T * B, NCELL)
                fc_ex, fc_cell = key_input(fresh_cursor, allc, prev.unsqueeze(1), vm.unsqueeze(1))
                loc_all = None
                if klocal is not None:
                    loc_all = local.gather(1, fc_cell.unsqueeze(-1).expand(-1, -1, local.shape[-1]))
                kl_all = self._key_logits(h, fc_ex, loc_all)
            key_ent = (p_cell * torch.distributions.Categorical(logits=kl_all).entropy()).sum(-1)
        klog = self._key_logits(h, kcur, klocal)
        klsm = klog.log_softmax(-1)
        log_prob = cursor_lp + klsm.gather(-1, k.view(-1, 1)).squeeze(-1)
        rs = lambda x: x.reshape(T, B)
        out = [rs(log_prob), rs(cursor_ent), rs(key_ent), value]
        if return_any:
            out.append(rs(klsm[:, 1:].logsumexp(-1)))
        if return_dists:
            out.append(dict(key_lp=klsm.reshape(T, B, -1),
                            cell_lp=None if logp_cell is None else logp_cell.reshape(T, B, -1)))
        return tuple(out)


def _embed(k: str, v: Tensor, t: Tensor, H1: int, H2: int, P1: int, P2: int, T1: int = 64, T2: int = 64,
           zero_cross: bool = True) -> Tensor:
    """Place old tensor v (checkpoint shape) into t (grown shape); entries t keeps are new units / fresh init.

    H = LSTM hidden, P = projection width, T = encoder trunk channels. Old units never read new units (zero cross
    blocks), so the grown function equals the old one exactly in fp32; new units read everything (fresh init) where
    that keeps the function unchanged because nobody reads them yet."""
    t = t.clone()
    if "encoder.convs." in k or "encoder.res." in k:
        # conv weight [O, I, kh, kw] / bias [O]: old block top-left; old outputs ignore new input channels
        if v.dim() == 4:
            if zero_cross:
                t[:v.shape[0], v.shape[1]:] = 0.0
            t[:v.shape[0], :v.shape[1]] = v
        else:
            t[:v.shape[0]] = v
    elif k.endswith("projection.1.weight"):
        # [P, T*12*16], flatten is channel-major: old channels are the first T1*192 columns
        if zero_cross:
            t[:P1, T1 * 192:] = 0.0                               # old features ignore new channels
        t[:P1, :T1 * 192] = v                                     # new feature rows keep their fresh init
    elif k.endswith("projection.1.bias"):
        t[:P1] = v
    elif "memory.weight_ih" in k or "memory.weight_hh" in k or "memory.bias" in k:
        for g in range(4):                                        # cuDNN gate blocks i, f, g, o
            rows_new, rows_old = slice(g * H2, g * H2 + H1), slice(g * H1, (g + 1) * H1)
            if "weight_ih" in k:
                if zero_cross:
                    t[rows_new] = 0.0                             # old units ignore new features
                t[rows_new, :P1] = v[rows_old, :P1]
                t[rows_new, P2:] = v[rows_old, P1:]               # aux columns
            elif "weight_hh" in k:
                if zero_cross:
                    t[rows_new] = 0.0                             # old units ignore new units
                t[rows_new, :H1] = v[rows_old]
            else:
                t[rows_new] = v[rows_old]
    elif k == "context.weight":                                   # [T, H]: LSTM context added to each cell feature
        if zero_cross:
            t.zero_()
        t[:T1, :H1] = v
    elif k == "context.bias":
        if zero_cross:
            t.zero_()
        t[:T1] = v
    elif k == "readout.0.weight":                                 # [64, T + 2]: (cell feature, cell centre)
        if zero_cross:
            t[:, T1:T2] = 0.0                                     # nobody reads the new channels yet
        t[:, :T1] = v[:, :T1]
        t[:, T2:] = v[:, T1:]
    elif k == "key_local.weight":                                 # [128, T]: chosen cell's local feature -> key MLP
        if zero_cross:
            t[:, T1:] = 0.0                                       # nobody reads the new channels yet
        t[:, :T1] = v
    elif k in ("value_head.0.weight", "key_head.0.weight"):
        if zero_cross:
            t.zero_()                                             # nobody reads the new units yet
        t[:, :H1] = v[:, :H1]
        if k == "key_head.0.weight":
            t[:, H2:] = v[:, H1:]                                 # chosen-cursor columns
    else:
        raise ValueError(f"cannot grow {k}: {tuple(v.shape)} -> {tuple(t.shape)}")
    return t


def _dims(old_cfg: dict, new_cfg: dict):
    return (old_cfg.get("hidden", 256), new_cfg["hidden"], old_cfg.get("proj_dim", 256), new_cfg["proj_dim"],
            old_cfg.get("trunk", 64), new_cfg.get("trunk", 64))


def grow_state_dict(old_sd: dict, old_cfg: dict, new: "Agent") -> tuple[dict, set]:
    """Embed a checkpoint into a larger agent so the function is unchanged (labelled capacity growth).

    Supports more encoder residual blocks (zero-init, from the fresh agent), wider encoder trunk channels (new
    channels computed from everything, read by nobody), a wider projection (new features feed zero LSTM input
    columns) and a wider LSTM (new units read anything but are read by nobody: zero recurrent columns into old units
    and zero columns in every hidden-state consumer). Exact in fp32. Returns (state_dict, resized names).
    Tensors the checkpoint lacks (extra residual blocks, key_local.weight) keep the new agent's fresh init, which is
    function-preserving because each starts at zero.
    """
    sd = {k: v.clone() for k, v in new.state_dict().items()}
    dims = _dims(old_cfg, new.config)
    resized = set()
    for k, v in old_sd.items():
        if k not in sd:
            raise ValueError(f"cannot grow: the new agent has no {k} (shrinking or dropping key_local is unsupported)")
        if sd[k].shape == v.shape:
            sd[k] = v.clone()
        else:
            resized.add(k)
            sd[k] = _embed(k, v, sd[k], *dims)
    return sd, resized


def grow_moment(name: str, v: Tensor, new_shape, old_cfg: dict, new_cfg: dict, fill: float = 0.0) -> Tensor:
    """Adam moment of a resized parameter: old entries keep their moments, new entries start at `fill`.

    labelled (session 3): fill lets the caller warm the second moment of new entries (default 0.0 = old behaviour);
    with zero_cross=False _embed never overwrites new entries, so every new entry equals fill."""
    return _embed(name, v, torch.full(new_shape, fill, dtype=v.dtype, device=v.device), *_dims(old_cfg, new_cfg),
                  zero_cross=False)


def apply_ema_payload(agent, payload, decay):
    """labelled (session 4): load EMA `decay` parameters saved by PPOTrainer.save (payload["ema"][str(decay)], keyed by
    parameter name) into `agent` (which must already hold payload["agent"] for its buffers). Raises if absent."""
    import torch
    ema = payload.get("ema", {}).get(str(float(decay)))
    if ema is None:
        raise KeyError(f"checkpoint has no EMA {decay}; available: {sorted(payload.get('ema', {}))}")
    with torch.no_grad():
        for name, p in agent.named_parameters():
            p.copy_(ema[name].to(p.device, p.dtype))
    return agent
