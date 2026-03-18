# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy.io
import torch.nn.functional as fun
import torch.nn.functional as F
from .evaluation import MetricsCal
import math


# =========================================================================
# 1. Coordinates
# =========================================================================
def make_coord_2d(h, w, device=None):
    if isinstance(h, torch.Tensor):
        h = int(h.item())
    if isinstance(w, torch.Tensor):
        w = int(w.item())
    h, w = int(h), int(w)

    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    coord = torch.stack([xx, yy], dim=-1).view(-1, 2)
    return coord


def make_coord_1d(l, device=None):
    if isinstance(l, torch.Tensor):
        l = int(l.item())
    l = int(l)
    return torch.linspace(-1, 1, steps=l, device=device).view(-1, 1)


# =========================================================================
# 1.1 MLP helpers
# =========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, use_layernorm=False):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.activation(x + self.ln(self.linear(x)))


def build_mlp(in_dim, out_dim, hidden_dim, depth, use_layernorm=False):
    layers = []
    if depth == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU(0.2, True))
        for _ in range(depth - 2):
            layers.append(ResidualBlock(hidden_dim, use_layernorm))
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# =========================================================================
# 1.2 Forward models for degradations (unchanged)
# =========================================================================
class PSF_down():
    def __call__(self, input_tensor, psf, ratio):
        _, C, _, _ = input_tensor.shape
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1)
        return fun.conv2d(input_tensor, psf, None, (ratio, ratio), groups=C)


class SRF_down():
    def __call__(self, input_tensor, srf):
        return fun.conv2d(input_tensor, srf, None)


# =========================================================================
# Relative L1 Loss
# =========================================================================
class RelativeL1Loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(RelativeL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        band_means = torch.mean(target, dim=(2, 3), keepdim=True).detach()
        diff = torch.abs(pred - target)
        rel_diff = diff / (band_means + self.eps)
        return torch.mean(rel_diff)


# =========================================================================
# 2. Spatial Feature Extractor
# =========================================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)


class SimpleUNet_Light_Attn(nn.Module):
    def __init__(self, in_ch, out_ch, base_dim=16, depth=3):
        super().__init__()
        self.depth = depth
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 1, 1, 0),
            nn.BatchNorm2d(base_dim), nn.LeakyReLU(0.2, True)
        )

        self.encs = nn.ModuleList()
        self.encs.append(nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, 1, 1),
            nn.BatchNorm2d(base_dim), nn.LeakyReLU(0.2, True)
        ))
        for i in range(1, depth + 1):
            in_c = base_dim * (2 ** (i - 1))
            out_c = base_dim * (2 ** i)
            self.encs.append(nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, True)
            ))

        self.attn = SpatialAttention()

        self.decs = nn.ModuleList()
        for i in range(depth, 0, -1):
            in_c = base_dim * (2 ** i) + base_dim * (2 ** (i - 1))
            out_c = base_dim * (2 ** (i - 1)) if i > 1 else out_ch
            self.decs.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, True)
            ))

    def forward(self, x):
        e_list = []
        x = self.bottleneck(x)
        for i in range(self.depth + 1):
            x = self.encs[i](x)
            e_list.append(x)

        x = self.attn(e_list[-1])
        for i in range(self.depth):
            skip = e_list[self.depth - 1 - i]
            up_x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = self.decs[i](torch.cat([skip, up_x], dim=1))
        return x


# =========================================================================
# 2.1 Deformable Dynamic Gather (2D)
# =========================================================================
class DeformableDynamicGather2D(nn.Module):
    def __init__(self, feat_dim, K=9, hidden_dim=64):
        super().__init__()
        assert K == 9
        self.K = K

        base = torch.tensor([
            [-1, -1], [0, -1], [1, -1],
            [-1,  0], [0,  0], [1,  0],
            [-1,  1], [0,  1], [1,  1],
        ], dtype=torch.float32)
        self.register_buffer('base_offsets', base, persistent=False)

        out_dim = 1 + 1 + 2 * K + K
        self.router = build_mlp(in_dim=feat_dim + 4, out_dim=out_dim, hidden_dim=hidden_dim, depth=3)
        if isinstance(self.router[-1], nn.Linear):
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

        self.eps = 1e-8
        self.tau2d = 2.0

    def forward(self, feat_map, coords_2d, cell_2d):
        B, C, H, W = feat_map.shape
        N = coords_2d.shape[1]
        if cell_2d.dim() == 3 and cell_2d.shape[1] == 1:
            cell_2d = cell_2d.expand(B, N, 2)

        grid_anchor = coords_2d.view(B, 1, N, 2)
        f_anchor = F.grid_sample(feat_map, grid_anchor, mode='bilinear', padding_mode='border', align_corners=True)
        f_anchor = f_anchor.squeeze(2).permute(0, 2, 1)

        router_in = torch.cat([f_anchor, coords_2d, cell_2d], dim=-1)
        out = self.router(router_in)

        idx = 0
        r_raw = out[..., idx:idx + 1]; idx += 1
        sigma_raw = out[..., idx:idx + 1]; idx += 1
        res_raw = out[..., idx:idx + 2 * self.K]; idx += 2 * self.K
        gate_raw = out[..., idx:idx + self.K]; idx += self.K

        r = torch.clamp(F.softplus(r_raw) + 0.1, 0.1, 4.0)
        sigma = torch.clamp(F.softplus(sigma_raw) + 0.5, 0.5, 6.0)

        res = torch.tanh(res_raw).view(B, N, self.K, 2) * 0.5
        base = self.base_offsets.view(1, 1, self.K, 2)
        offsets_pix = r.view(B, N, 1, 1) * base + res

        scale_x = 2.0 / max(W - 1, 1)
        scale_y = 2.0 / max(H - 1, 1)
        offsets_norm = torch.empty_like(offsets_pix)
        offsets_norm[..., 0] = offsets_pix[..., 0] * scale_x
        offsets_norm[..., 1] = offsets_pix[..., 1] * scale_y

        coords_exp = coords_2d.unsqueeze(2).expand(B, N, self.K, 2)
        deform_coords = coords_exp + offsets_norm
        grid = deform_coords.view(B, 1, N * self.K, 2)

        f = F.grid_sample(feat_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
        f = f.view(B, C, N, self.K).permute(0, 2, 3, 1)

        d2 = offsets_pix[..., 0] ** 2 + offsets_pix[..., 1] ** 2
        sig_eff = sigma.squeeze(-1).unsqueeze(-1) * self.tau2d
        w_geo = torch.exp(-0.5 * d2 / (sig_eff ** 2 + self.eps))

        gate = torch.sigmoid(gate_raw)
        w = w_geo * gate
        w = w / (w.sum(dim=-1, keepdim=True) + self.eps)

        out_feat = torch.sum(f * w.unsqueeze(-1), dim=2)
        debug = {
            'r_mean': r.mean().detach(),
            'sigma_mean': sigma.mean().detach(),
            'w_entropy': (-(w * torch.log(torch.clamp(w, min=1e-8))).sum(dim=-1).mean()).detach()
        }
        return out_feat, debug


# =========================================================================
# 2.2 Spatial INR (Abundance): Dirichlet parameterization (strict sum=1)
# =========================================================================
class Spatial_Abundance_Net(nn.Module):
    def __init__(self, args, in_channels, feat_dim=32):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)
        msi_base_dim = getattr(args, 'msi_base_dim', 64)
        msi_depth = getattr(args, 'msi_depth', 3)
        inr_spat_depth = getattr(args, 'inr_spat_depth', 4)
        inr_spat_dim = getattr(args, 'inr_spat_dim', 128)

        self.unet = SimpleUNet_Light_Attn(in_channels, feat_dim, base_dim=msi_base_dim, depth=msi_depth)
        self.gather = DeformableDynamicGather2D(feat_dim=feat_dim, K=9, hidden_dim=64)
        self.mlp = build_mlp(in_dim=feat_dim + 4, out_dim=self.R, hidden_dim=inr_spat_dim, depth=inr_spat_depth)

        self.alpha_eps = 1e-6
        self.alpha_cap = getattr(args, 'alpha_cap', 0.0)  # 0 => disabled

    def forward(self, img_input, coords_2d, target_H, target_W):
        B = img_input.shape[0]
        N = coords_2d.shape[1]
        device = img_input.device

        feat_map = self.unet(img_input)
        cell_2d = torch.tensor([2.0 / float(target_W), 2.0 / float(target_H)],
                               device=device).view(1, 1, 2).expand(B, 1, 2)

        q_feat, dbg = self.gather(feat_map, coords_2d, cell_2d)
        cell_expand = cell_2d.expand(B, N, 2)

        mlp_in = torch.cat([q_feat, coords_2d, cell_expand], dim=-1)
        U_logits = self.mlp(mlp_in)

        alpha = F.softplus(U_logits) + self.alpha_eps
        if isinstance(self.alpha_cap, (int, float)) and self.alpha_cap > 0:
            alpha = torch.clamp(alpha, max=float(self.alpha_cap))

        U = alpha / (alpha.sum(dim=-1, keepdim=True) + self.alpha_eps)
        return U, dbg


# =========================================================================
# 3. Spectral Prototype Block
# =========================================================================
class SpectralPrototypeBlock(nn.Module):
    def __init__(self, channels, num_prototypes=8, attn_dim=64, dropout=0.0):
        super().__init__()
        self.attn_dim = attn_dim

        self.local_dw = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.local_pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.local_bn = nn.BatchNorm1d(channels)
        self.local_act = nn.LeakyReLU(0.2, True)

        self.prototype_tokens = nn.Parameter(torch.randn(num_prototypes, channels) * 0.02)
        self.q_proj = nn.Conv1d(channels, attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Linear(channels, attn_dim, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.fuse = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, True)
        )

        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels)
        )

        self.gate_local = nn.Parameter(torch.tensor(1.0))
        self.gate_proto = nn.Parameter(torch.tensor(1.0))
        self.gate_ffn = nn.Parameter(torch.tensor(1.0))
        self.out_act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        identity = x
        local = self.local_act(self.local_bn(self.local_pw(self.local_dw(x))))

        q = self.q_proj(x).transpose(1, 2)
        proto = self.prototype_tokens
        k = self.k_proj(proto).unsqueeze(0).expand(x.shape[0], -1, -1)
        v = self.v_proj(proto).unsqueeze(0).expand(x.shape[0], -1, -1)

        attn = torch.matmul(q, k.transpose(1, 2)) / (self.attn_dim ** 0.5)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        proto_feat = torch.matmul(attn, v).transpose(1, 2)

        fused = self.fuse(torch.cat([local, proto_feat], dim=1))
        out = identity + self.gate_local * local + self.gate_proto * fused
        out = out + self.gate_ffn * self.ffn(out)
        return self.out_act(out)


# =========================================================================
# 3.1 Deformable Dynamic Gather (1D)
# =========================================================================
class DeformableDynamicGather1D(nn.Module):
    def __init__(self, feat_dim, K=5, hidden_dim=64):
        super().__init__()
        assert K == 5
        self.K = K
        base = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=torch.float32)
        self.register_buffer('base_offsets', base, persistent=False)

        out_dim = 1 + 1 + K + K
        self.router = build_mlp(in_dim=feat_dim + 2, out_dim=out_dim, hidden_dim=hidden_dim, depth=3)
        if isinstance(self.router[-1], nn.Linear):
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        self.eps = 1e-8

    def forward(self, feat_1d, coords_1d, cell_1d):
        B, C, L_in = feat_1d.shape
        N = coords_1d.shape[1]
        if cell_1d.dim() == 3 and cell_1d.shape[1] == 1:
            cell_1d = cell_1d.expand(B, N, 1)

        feat_2d = feat_1d.unsqueeze(2)
        grid_x = coords_1d.view(B, 1, N, 1)
        grid_y = torch.zeros_like(grid_x)
        grid_anchor = torch.cat([grid_x, grid_y], dim=-1)

        f_anchor = F.grid_sample(feat_2d, grid_anchor, mode='bilinear', padding_mode='border', align_corners=True)
        f_anchor = f_anchor.squeeze(2).permute(0, 2, 1)

        router_in = torch.cat([f_anchor, coords_1d, cell_1d], dim=-1)
        out = self.router(router_in)

        idx = 0
        r_raw = out[..., idx:idx + 1]; idx += 1
        sigma_raw = out[..., idx:idx + 1]; idx += 1
        res_raw = out[..., idx:idx + self.K]; idx += self.K
        gate_raw = out[..., idx:idx + self.K]; idx += self.K

        r = torch.clamp(F.softplus(r_raw) + 0.3, 0.3, 2.0)
        sigma = torch.clamp(F.softplus(sigma_raw) + 0.5, 0.5, 3.0)

        res = torch.tanh(res_raw).view(B, N, self.K, 1) * 0.5
        base = self.base_offsets.view(1, 1, self.K, 1)
        offsets_band = r.view(B, N, 1, 1) * base + res

        scale_x = 2.0 / max(L_in - 1, 1)
        offsets_norm_x = offsets_band[..., 0] * scale_x

        coords_exp_x = coords_1d.unsqueeze(2).expand(B, N, self.K, 1)[..., 0]
        deform_x = coords_exp_x + offsets_norm_x
        deform_y = torch.zeros_like(deform_x)
        deform_grid = torch.stack([deform_x, deform_y], dim=-1)

        grid = deform_grid.view(B, 1, N * self.K, 2)
        f = F.grid_sample(feat_2d, grid, mode='bilinear', padding_mode='border', align_corners=True)
        f = f.view(B, C, N, self.K).permute(0, 2, 3, 1)

        d2 = offsets_band[..., 0] ** 2
        tau = 2.0
        w_geo = torch.exp(-0.5 * d2 / ((sigma.squeeze(-1).unsqueeze(-1) * tau) ** 2 + self.eps))
        gate = torch.sigmoid(gate_raw)

        w = w_geo * gate
        w = w / (w.sum(dim=-1, keepdim=True) + self.eps)

        out_feat = torch.sum(f * w.unsqueeze(-1), dim=2)
        debug = {
            'r1d_mean': r.mean().detach(),
            'sigma1d_mean': sigma.mean().detach(),
            'w1d_entropy': (-(w * torch.log(torch.clamp(w, min=1e-8))).sum(dim=-1).mean()).detach()
        }
        return out_feat, debug


# =========================================================================
# 3.2 Spectral INR (Endmember)
# =========================================================================
class Spectral_Endmember_Net(nn.Module):
    def __init__(self, args, in_channels, feat_dim=32):
        super().__init__()
        hsi_base_dim = getattr(args, 'hsi_base_dim', 32)
        hsi_depth = getattr(args, 'hsi_depth', 2)
        self.R = getattr(args, 'num_endmembers', 32)

        self.head = nn.Sequential(
            nn.Conv1d(16, hsi_base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hsi_base_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.blocks = nn.ModuleList([
            SpectralPrototypeBlock(
                channels=hsi_base_dim,
                num_prototypes=getattr(args, 'proto_num', 8),
                attn_dim=getattr(args, 'proto_attn_dim', 64),
                dropout=getattr(args, 'proto_dropout', 0.0)
            )
            for _ in range(hsi_depth)
        ])
        self.tail = nn.Conv1d(hsi_base_dim, feat_dim, kernel_size=1)

        self.gather_1d = DeformableDynamicGather1D(feat_dim=feat_dim, K=5, hidden_dim=64)

        inr_spec_depth = getattr(args, 'inr_spec_depth', 4)
        inr_spec_dim = getattr(args, 'inr_spec_dim', 128)
        self.mlp = build_mlp(in_dim=feat_dim + 2, out_dim=self.R, hidden_dim=inr_spec_dim, depth=inr_spec_depth)

        self.v_temp = nn.Parameter(torch.tensor(2.0))

    def forward(self, coords_1d, img_input):
        B, L_in, H, W = img_input.shape
        target_L = coords_1d.shape[0]
        device = img_input.device

        regional_spectra = F.adaptive_avg_pool2d(img_input, (4, 4))
        spectral_signal = regional_spectra.view(B, L_in, 16).permute(0, 2, 1)

        feat_1d = self.head(spectral_signal)
        for block in self.blocks:
            feat_1d = block(feat_1d)
        feat_1d = self.tail(feat_1d)

        coords_expand = coords_1d.unsqueeze(0).expand(B, target_L, 1)
        cell_1d = torch.full((B, 1, 1), 2.0 / float(target_L), device=device)

        q_feat, dbg = self.gather_1d(feat_1d, coords_expand, cell_1d)
        cell_expand = cell_1d.expand(B, target_L, 1)
        mlp_in = torch.cat([q_feat, coords_expand, cell_expand], dim=-1)

        V_logits = self.mlp(mlp_in)
        t = torch.clamp(self.v_temp, 0.5, 5.0)
        V = 0.05 + 0.95 * torch.sigmoid(V_logits / t)
        return V, dbg


# =========================================================================
# 4. End-to-End Projection
# =========================================================================
class LinearUnmixingProjection(nn.Module):
    def __init__(self, args, in_channels_spatial):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)
        self.spatial_net = Spatial_Abundance_Net(args, in_channels_spatial, feat_dim=32)
        self.spectral_net = Spectral_Endmember_Net(args, in_channels=0, feat_dim=32)

    def forward(self, spatial_input, spectral_input, coords_2d, coords_1d, target_shape):
        B = target_shape[0].item() if torch.is_tensor(target_shape[0]) else int(target_shape[0])
        target_L = target_shape[1].item() if torch.is_tensor(target_shape[1]) else int(target_shape[1])
        target_H = target_shape[2].item() if torch.is_tensor(target_shape[2]) else int(target_shape[2])
        target_W = target_shape[3].item() if torch.is_tensor(target_shape[3]) else int(target_shape[3])

        U, dbg_u = self.spatial_net(spatial_input, coords_2d, target_H, target_W)
        V, dbg_v = self.spectral_net(coords_1d, spectral_input)

        Z_flat = torch.matmul(V, U.transpose(1, 2)) / math.sqrt(self.R)
        Z_img = Z_flat.view(B, target_L, target_H, target_W)

        E_out = V[0]
        A_out = U.transpose(1, 2).view(B, self.R, target_H, target_W)[0]

        dbg = {}
        dbg.update(dbg_u)
        dbg.update(dbg_v)
        return Z_img, E_out, A_out, dbg


# =========================================================================
# 5. Fusion (NO residual learning)
# =========================================================================
class End2EndUnmixingFusion(nn.Module):
    def __init__(self, args, L_bands, C_msi):
        super().__init__()
        in_channels_fused = L_bands + C_msi
        self.projector = LinearUnmixingProjection(args, in_channels_spatial=in_channels_fused)

    def forward(self, Y_hsi, Y_msi, target_H, target_W, target_L):
        B, _, _, _ = Y_hsi.shape
        device = Y_hsi.device

        coords_2d = make_coord_2d(target_H, target_W, device).unsqueeze(0).expand(B, -1, -1)
        coords_1d = make_coord_1d(target_L, device)

        _, _, H_hr, W_hr = Y_msi.shape
        up_hsi_for_feat = F.interpolate(Y_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        fused_spatial_input = torch.cat([up_hsi_for_feat, Y_msi], dim=1)

        Z_img, E_out, A_out, dbg = self.projector(
            spatial_input=fused_spatial_input,
            spectral_input=up_hsi_for_feat,
            coords_2d=coords_2d,
            coords_1d=coords_1d,
            target_shape=(B, target_L, target_H, target_W)
        )

        X_out = torch.clamp(Z_img, 1e-6, 1.0)
        out_list = [X_out]
        residual_tensor = torch.zeros_like(X_out)
        return out_list, E_out, A_out, residual_tensor, dbg


# =========================================================================
# 6. DIP Training Pipeline with probes (adds observability probes)
# =========================================================================
class dip():
    def __init__(self, args, psf, srf, blind):
        self.args = args
        self.hr_msi = blind.tensor_hr_msi
        self.lr_hsi = blind.tensor_lr_hsi
        self.gt = blind.gt

        psf_est = np.reshape(psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor))
        self.psf_est = torch.tensor(psf_est).to(self.args.device).float()
        srf_est = np.reshape(srf.T, newshape=(srf.shape[1], srf.shape[0], 1, 1))
        self.srf_est = torch.tensor(srf_est).to(self.args.device).float()

        self.psf_down = PSF_down()
        self.srf_down = SRF_down()

        L_bands = self.lr_hsi.shape[1]
        C_msi = self.hr_msi.shape[1]

        self.target_scale = getattr(args, 'target_scale', 1.0)
        _, _, H_hr, W_hr = self.hr_msi.shape
        self.target_H = int(H_hr * self.target_scale)
        self.target_W = int(W_hr * self.target_scale)
        self.target_L = getattr(args, 'target_L', L_bands)

        self.net = End2EndUnmixingFusion(args=self.args, L_bands=L_bands, C_msi=C_msi).to(self.args.device)

        def lambda_rule(epoch):
            raw = 1.0 - max(0, epoch + 1 - self.args.niter3_dip) / float(self.args.niter_decay3_dip + 1)
            min_lr_factor = getattr(self.args, 'min_lr_factor', 0.1)
            return max(raw, min_lr_factor)

        base_lr = max(getattr(self.args, 'lr_stage3_dip', 1e-3), 3e-4)

        spec_params = list(self.net.projector.spectral_net.parameters())
        spec_ids = set(map(id, spec_params))
        other_params = [p for p in self.net.parameters() if id(p) not in spec_ids]

        spec_lr_mult = getattr(self.args, 'spec_lr_mult', 2.0)
        self.optimizer = optim.Adam([
            {'params': other_params, 'lr': base_lr},
            {'params': spec_params, 'lr': base_lr * spec_lr_mult},
        ])
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

    def angle_similarity_loss(self, Sh, Sm):
        B, C, H, W = Sh.shape
        Sh_vec = Sh.view(B, C, -1).permute(0, 2, 1)
        Sm_vec = Sm.view(B, C, -1).permute(0, 2, 1)
        dot = (Sh_vec * Sm_vec).sum(dim=-1)
        norm_Sh = torch.sqrt(torch.sum(Sh_vec ** 2, dim=-1) + 1e-12)
        norm_Sm = torch.sqrt(torch.sum(Sm_vec ** 2, dim=-1) + 1e-12)
        cos = dot / (norm_Sh * norm_Sm)
        cos = torch.clamp(cos, -0.9999, 0.9999)
        return (torch.acos(cos).mean() / math.pi)

    @staticmethod
    def _grad_norm(model):
        total_norm = 0.0
        nonzero = 0
        total = 0
        max_grad = 0.0
        for _, p in model.named_parameters():
            if p.grad is not None:
                total += 1
                g = p.grad.detach()
                if torch.isfinite(g).all() and g.abs().sum().item() > 0:
                    nonzero += 1
                if torch.isfinite(g).all():
                    max_grad = max(max_grad, g.abs().max().item())
                total_norm += g.norm(2).item() ** 2
        return total_norm ** 0.5, nonzero, total, max_grad

    @staticmethod
    def _tv2d(x):
        return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()

    def train(self):
        flag_best_fhsi = [10.0, 0.0, None, 0]

        L1Loss = nn.L1Loss(reduction='mean')
        RelLoss = RelativeL1Loss(eps=1e-3).to(self.args.device)
        sam_weight = getattr(self.args, 'sam_weight', 0.1)

        lambda_a_tv = getattr(self.args, 'lambda_a_tv', 2e-4)
        lambda_dir = getattr(self.args, 'lambda_dir', 1e-4)
        alpha0 = getattr(self.args, 'dirichlet_alpha0', 2.0)

        total_epochs = self.args.niter3_dip + self.args.niter_decay3_dip

        for epoch in range(1, total_epochs + 1):
            self.optimizer.zero_grad()

            out_list, self.endmember, self.abundance, residual_tensor, dbg = self.net(
                self.lr_hsi, self.hr_msi,
                target_H=self.target_H, target_W=self.target_W, target_L=self.target_L
            )
            self.hr_hsi_rec = out_list[-1]

            X_obs = self.hr_hsi_rec
            if self.target_H != self.hr_msi.shape[2] or self.target_W != self.hr_msi.shape[3]:
                X_obs = F.interpolate(X_obs, size=(self.hr_msi.shape[2], self.hr_msi.shape[3]),
                                      mode='bilinear', align_corners=False)
            if self.target_L != self.lr_hsi.shape[1]:
                X_obs = F.interpolate(
                    X_obs.unsqueeze(1),
                    size=(self.lr_hsi.shape[1], X_obs.shape[2], X_obs.shape[3]),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(1)

            hr_msi_pred = self.srf_down(X_obs, self.srf_est)
            lr_hsi_pred = self.psf_down(X_obs, self.psf_est, self.args.scale_factor)

            loss_hsi_l1 = L1Loss(self.lr_hsi, lr_hsi_pred)
            loss_hsi_rel = RelLoss(lr_hsi_pred, self.lr_hsi)
            loss_hsi = loss_hsi_l1 + 1.5 * loss_hsi_rel

            loss_msi_l1 = L1Loss(self.hr_msi, hr_msi_pred)
            loss_msi_rel = RelLoss(hr_msi_pred, self.hr_msi)
            loss_msi = loss_msi_l1 + 1.5 * loss_msi_rel

            loss_sam = self.angle_similarity_loss(lr_hsi_pred, self.lr_hsi)

            A = self.abundance.unsqueeze(0) if self.abundance.dim() == 3 else self.abundance
            loss_a_tv = self._tv2d(A)

            U = torch.clamp(A, min=1e-8)
            loss_dir = -((alpha0 - 1.0) * torch.log(U)).mean()

            loss_total = loss_hsi + loss_msi + sam_weight * loss_sam + lambda_a_tv * loss_a_tv + lambda_dir * loss_dir

            if not torch.isfinite(loss_total):
                print(f"[FATAL] loss_total is not finite at epoch {epoch}")
                break

            loss_total.backward()
            grad_norm_all, nz_grad, tot_grad, max_grad = self._grad_norm(self.net)
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 == 0:
                # NOTE: probes are in no_grad for printing; but we also compute a tiny autograd probe below.
                with torch.no_grad():
                    print('\n==================================================')
                    print(f'epoch:{epoch}/{total_epochs} lr:{self.optimizer.param_groups[0]["lr"]:.6f}')
                    print(f'loss_total:{loss_total.item():.6f} | loss_hsi:{loss_hsi.item():.6f} | loss_msi:{loss_msi.item():.6f} | loss_sam:{loss_sam.item():.6f} * {sam_weight}')
                    print(f'loss_a_tv:{loss_a_tv.item():.6f} * {lambda_a_tv} | loss_dir:{loss_dir.item():.6f} * {lambda_dir} (alpha0={alpha0})')
                    print(f'grad_norm_all:{grad_norm_all:.6f} | nonzero_grad:{nz_grad}/{tot_grad} | max_grad:{max_grad:.3e}')

                    if len(dbg) > 0:
                        dbg_str = ' | '.join([f'{k}:{float(v):.4f}' for k, v in dbg.items()])
                        print(f'[Probe-Gather] {dbg_str}')

                    gt_tensor = torch.tensor(self.gt.transpose(2, 0, 1)).unsqueeze(0).to(self.args.device).float()

                    X_eval = self.hr_hsi_rec
                    if X_eval.shape[2:] != gt_tensor.shape[2:]:
                        X_eval = F.interpolate(X_eval, size=gt_tensor.shape[2:], mode='bilinear', align_corners=False)
                    if X_eval.shape[1] != gt_tensor.shape[1]:
                        X_eval = F.interpolate(X_eval.unsqueeze(1),
                                               size=(gt_tensor.shape[1], gt_tensor.shape[2], gt_tensor.shape[3]),
                                               mode='trilinear', align_corners=False).squeeze(1)

                    spatial_rmse = torch.sqrt(torch.mean((X_eval - gt_tensor) ** 2, dim=1))
                    rmse_map = spatial_rmse[0]
                    max_rmse = rmse_map.max()
                    mean_rmse = rmse_map.mean()
                    idx = torch.argmax(rmse_map).item()
                    Hh, Ww = rmse_map.shape
                    worst_h = int(idx // Ww)
                    worst_w = int(idx % Ww)

                    print(f'[Probe-Spatial] Max RMSE: {max_rmse.item():.5f} | Mean RMSE: {mean_rmse.item():.5f} | Ratio: {max_rmse.item() / (mean_rmse.item() + 1e-8):.1f}x')
                    print(f'[Probe-WorstPixel] max_rmse={max_rmse.item():.5f} at (h={worst_h}, w={worst_w}) of (H={Hh}, W={Ww})')

                    # distance to boundary
                    dist_top = worst_h
                    dist_left = worst_w
                    dist_bottom = (Hh - 1) - worst_h
                    dist_right = (Ww - 1) - worst_w
                    print(f'[Probe-WorstPixelBoundaryDist] top:{dist_top} left:{dist_left} bottom:{dist_bottom} right:{dist_right}')

                    # values at worst pixel
                    gt_pix = gt_tensor[0, :, worst_h, worst_w]
                    pred_pix = X_eval[0, :, worst_h, worst_w]
                    print(f'[Probe-WorstPixelValues] gt(mean={gt_pix.mean().item():.5f}, min={gt_pix.min().item():.5f}, max={gt_pix.max().item():.5f}) | '
                          f'pred(mean={pred_pix.mean().item():.5f}, min={pred_pix.min().item():.5f}, max={pred_pix.max().item():.5f})')

                    err_vec = (pred_pix - gt_pix).abs()
                    print(f'[Probe-WorstPixelBands] mean_abs={err_vec.mean().item():.5f} max_abs={err_vec.max().item():.5f}')

                    # abundance at worst pixel (top-k)
                    A_pix = A[0, :, worst_h, worst_w]  # [R]
                    topk = torch.topk(A_pix, k=min(5, A_pix.numel()))
                    top_vals = topk.values.detach().cpu().numpy().tolist()
                    top_idx = topk.indices.detach().cpu().numpy().tolist()
                    print(f'[Probe-WorstPixelAbundanceTop5] idx={top_idx} val={[round(v, 6) for v in top_vals]}')

                    # MSI error at worst pixel (if same spatial size after X_obs alignment)
                    # Use X_obs/hr_msi_pred already aligned to hr_msi size; map worst pixel into hr_msi grid if sizes equal.
                    if X_obs.shape[2:] == self.hr_msi.shape[2:]:
                        # worst_h/w are on gt/X_eval grid; if that matches hr_msi size, ok, else skip
                        if (worst_h < X_obs.shape[2]) and (worst_w < X_obs.shape[3]) and (X_obs.shape[2:] == (Hh, Ww)):
                            msi_err = (hr_msi_pred[0, :, worst_h, worst_w] - self.hr_msi[0, :, worst_h, worst_w]).abs()
                            print(f'[Probe-WorstPixelMSI] mean_abs={msi_err.mean().item():.5f} max_abs={msi_err.max().item():.5f}')
                        else:
                            print('[Probe-WorstPixelMSI] skipped (grid mismatch)')
                    else:
                        print('[Probe-WorstPixelMSI] skipped (X_obs not on hr_msi grid)')

                    finite_ok = torch.isfinite(self.hr_hsi_rec).all().item()
                    print(f'[Probe-Numerics] hr_hsi_rec finite: {bool(finite_ok)}')

                    X_stage_np = X_eval.data.cpu().numpy()[0].transpose(1, 2, 0)
                    _, stage_psnr, _, _, _, _, _ = MetricsCal(self.gt, X_stage_np, self.args.scale_factor)
                    print(f'>> End-to-End PSNR: {stage_psnr:.2f}')

                # ----------------- Observability probe (autograd) -----------------
                # Compute gradient of loss wrt X_eval at worst pixel to see if it is constrained.
                # Do NOT change training; only prints. This adds overhead once per 50 epochs.
                try:
                    # rebuild a scalar loss on X_eval that matches your metric location
                    # (use simple MSE on X_eval vs gt_tensor, just for sensitivity)
                    X_eval_probe = self.hr_hsi_rec
                    gt_tensor_probe = torch.tensor(self.gt.transpose(2, 0, 1)).unsqueeze(0).to(self.args.device).float()
                    if X_eval_probe.shape[2:] != gt_tensor_probe.shape[2:]:
                        X_eval_probe = F.interpolate(X_eval_probe, size=gt_tensor_probe.shape[2:], mode='bilinear', align_corners=False)
                    if X_eval_probe.shape[1] != gt_tensor_probe.shape[1]:
                        X_eval_probe = F.interpolate(X_eval_probe.unsqueeze(1),
                                                     size=(gt_tensor_probe.shape[1], gt_tensor_probe.shape[2], gt_tensor_probe.shape[3]),
                                                     mode='trilinear', align_corners=False).squeeze(1)

                    # IMPORTANT: X_eval_probe depends on net params. We measure grad wrt X_eval_probe tensor.
                    probe_loss = (X_eval_probe - gt_tensor_probe).pow(2).mean()
                    g = torch.autograd.grad(probe_loss, X_eval_probe, retain_graph=False, create_graph=False, allow_unused=True)[0]
                    if g is None:
                        print('[Probe-Observability] grad wrt X_eval is None')
                    else:
                        g_pix = g[0, :, worst_h, worst_w].abs()
                        print(f'[Probe-Observability] grad@worst_pixel mean_abs={g_pix.mean().item():.6e} max_abs={g_pix.max().item():.6e}')
                except Exception as e:
                    print(f'[Probe-Observability] failed: {e}')

        best_output = flag_best_fhsi[2] if isinstance(flag_best_fhsi[2], torch.Tensor) else self.hr_hsi_rec

        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'),
                         {'Out': best_output.data.cpu().numpy()[0].transpose(1, 2, 0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Endmember.mat'),
                         {'end': self.endmember.data.cpu().numpy()})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Abundance.mat'),
                         {'abun': self.abundance.data.cpu().numpy()})

        return best_output