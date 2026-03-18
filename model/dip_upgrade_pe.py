# -*- coding: utf-8 -*-
"""
dip_upgrade.py �� Unrolled Tucker Fusion with LIIF-style INR for arbitrary-scale
                  HSI-MSI fusion.

Key changes vs original:
  [Spatial]  Replace ODConv + bilinear-upsample + bare MLP
             with INR2D (grid_sample + positional encoding + local ensemble)
  [Spectral] Add sinusoidal positional encoding to spectral INR
  [Scale]    All coordinate / cell generation is resolution-agnostic
"""

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
from .INR2D import INR2D, make_coord
import math


# =========================================================================
# 1. Utility: coordinate generation & MLP builder
# =========================================================================
def make_coord_1d(length, device=None):
    """Pixel-centre coordinates in [-1, 1] for a 1-D signal of given length."""
    r = 1.0 / length
    return (-1 + r + 2 * r * torch.arange(length, device=device).float()).view(-1, 1)


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
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# =========================================================================
# Degradation operators
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
# Loss
# =========================================================================
class RelativeL1Loss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        band_means = torch.mean(target, dim=(2, 3), keepdim=True).detach()
        return torch.mean(torch.abs(pred - target) / (band_means + self.eps))


# =========================================================================
# 2. Spatial Encoder �� Multi-scale Feature Pyramid + INR2D querying
# =========================================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class FeaturePyramidEncoder(nn.Module):
    """
    Multi-scale encoder producing features at 1��, 1/2��, 1/4�� ... resolutions.
    """

    def __init__(self, in_ch, base_dim=16, depth=2):
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
                nn.MaxPool2d(2),
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, True)
            ))
        self.attn = SpatialAttention()

    def forward(self, x):
        x = self.bottleneck(x)
        features = []
        for i in range(self.depth + 1):
            x = self.encs[i](x)
            if i == self.depth:
                x = self.attn(x)
            features.append(x)
        return features


class Spatial_INR2D_Tucker(nn.Module):
    """
    Spatial factor (U) generator using INR2D for each pyramid scale.

    Instead of bilinear-upsampling features to HR and feeding a bare MLP,
    each scale keeps its native LR resolution and uses INR2D.query_2D to
    produce HR-resolution outputs via grid_sample + PE + local ensemble.

    Supports **arbitrary target resolution** �� coords and cells are generated
    on-the-fly from (target_h, target_w).
    """

    def __init__(self, args, in_channels):
        super().__init__()
        msi_base_dim = getattr(args, 'msi_base_dim', 64)
        msi_depth = getattr(args, 'msi_depth', 2)
        self.Rs = getattr(args, 'Rs', 64)
        self.msi_depth = msi_depth

        inr_spat_dim = getattr(args, 'inr_spat_dim', 128)
        inr_spat_depth = getattr(args, 'inr_spat_depth', 3)
        pe_levels_spatial = getattr(args, 'pe_levels_spatial', 4)
        inr_feat_unfold = getattr(args, 'inr_feat_unfold', False)
        inr_local_ensemble = getattr(args, 'inr_local_ensemble', True)
        inr_cell_decode = getattr(args, 'inr_cell_decode', True)

        # 1. Multi-scale encoder
        self.encoder = FeaturePyramidEncoder(in_channels, base_dim=msi_base_dim, depth=msi_depth)

        # 2. Per-scale INR2D decoder (replaces ODConv + bare MLP)
        self.inr_decoders = nn.ModuleList()
        for i in range(msi_depth + 1):
            cur_dim = msi_base_dim * (2 ** i)
            self.inr_decoders.append(INR2D(
                dim=cur_dim,
                out_dim=self.Rs,
                hidden_dim=inr_spat_dim,
                hidden_layers=inr_spat_depth,
                L=pe_levels_spatial,
                local_ensemble=inr_local_ensemble,
                feat_unfold=inr_feat_unfold,
                cell_decode=inr_cell_decode,
            ))

        # 3. Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.Rs * (msi_depth + 1), self.Rs * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.Rs * 2, self.Rs),
        )

        # 4. Global feature for core tensor
        deepest_dim = msi_base_dim * (2 ** msi_depth)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(deepest_dim, 64), nn.LeakyReLU(0.2, True)
        )

    def forward(self, img_input, target_h, target_w):
        """
        Args:
            img_input: (B, C_msi, H_lr, W_lr)  �� input to encoder (could be S_k)
            target_h:  int �� target spatial height
            target_w:  int �� target spatial width

        Returns:
            U:                (B, target_h*target_w, Rs)
            spatial_global:   (B, 64)
        """
        B = img_input.shape[0]
        device = img_input.device

        # 1. Encode multi-scale features
        features = self.encoder(img_input)   # list of (B, C_i, h_i, w_i)

        # Global feature
        spatial_global = self.global_proj(
            self.global_pool(features[-1]).flatten(1)
        )

        # 2. Generate HR query coords + cell (once, shared across scales)
        coord = make_coord((target_h, target_w), device=device)          # (N, 2)
        coord = coord.unsqueeze(0).expand(B, -1, -1)                      # (B, N, 2)

        cell = torch.zeros(B, coord.shape[1], 2, device=device)
        cell[:, :, 0] = 2.0 / target_h
        cell[:, :, 1] = 2.0 / target_w

        # 3. Query each scale's INR2D at the same HR coordinates
        U_scales = []
        for i, feat_i in enumerate(features):
            # feat_i is at its native LR resolution; INR2D handles the
            # continuous querying internally via grid_sample
            U_i = self.inr_decoders[i].query_2D(
                feat_i, coord, cell,
                target_h=target_h, target_w=target_w
            )   # (B, Rs, target_h, target_w)
            U_i_flat = U_i.flatten(2).permute(0, 2, 1)                   # (B, N, Rs)
            U_scales.append(U_i_flat)

        # 4. Fuse scales (2-layer MLP instead of single linear)
        U_concat = torch.cat(U_scales, dim=-1)                            # (B, N, Rs*(D+1))
        U = torch.tanh(self.fusion(U_concat))                              # (B, N, Rs)

        return U, spatial_global


# =========================================================================
# 3. Spectral Encoder �� ResDCT + PE-enhanced INR
# =========================================================================
class Spectral_ResDCT_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

        self.freq_attn = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.LeakyReLU(0.2, True),
            nn.Conv1d(channels, channels, 1, bias=False), nn.Sigmoid()
        )
        self.freq_gate = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, L = x.shape
        device = x.device

        identity = x
        out_conv = self.relu(self.bn1(self.conv1(x)))
        out_conv = self.bn2(self.conv2(out_conv))

        n = torch.arange(L, dtype=torch.float32, device=device)
        k = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)
        dct_m = torch.cos(math.pi / L * (n + 0.5) * k)
        dct_m[0] *= 1 / math.sqrt(2)
        dct_m *= math.sqrt(2 / L)

        x_freq = torch.matmul(x, dct_m.t())
        cutoff = L // 2
        mask_H = torch.ones(L, device=device)
        mask_H[:cutoff] = 0.0

        freq_H = x_freq * mask_H.view(1, 1, L)
        freq_H_enhanced = freq_H * self.freq_attn(freq_H) * self.freq_gate
        out_dct_detail = torch.matmul(freq_H_enhanced, dct_m)

        return self.relu(identity + out_conv + out_dct_detail)


class Spectral_ResDCT_Extractor(nn.Module):
    def __init__(self, in_ch=16, out_ch=32, base_dim=32, depth=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, base_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(base_dim), nn.LeakyReLU(0.2, True)
        )
        self.blocks = nn.ModuleList([Spectral_ResDCT_Block(base_dim) for _ in range(depth)])
        self.tail = nn.Conv1d(base_dim, out_ch, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Sequential(nn.Linear(out_ch, 64), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        out_seq = self.tail(x)
        global_feat = self.global_proj(self.global_pool(out_seq).flatten(1))
        return out_seq, global_feat


class Spectral_INR_Tucker_Adaptive(nn.Module):
    """
    Spectral factor (V) generator with positional encoding on spectral coords.

    Change vs original: adds sinusoidal PE (2*pe_levels dims) to the 1D
    spectral coordinate, resolving the spectral-bias problem of bare MLP.
    """

    def __init__(self, args, in_channels, feat_dim=32):
        super().__init__()
        hsi_base_dim = getattr(args, 'hsi_base_dim', 32)
        hsi_depth = getattr(args, 'hsi_depth', 2)
        Rl = getattr(args, 'Rl', 20)
        inr_spec_depth = getattr(args, 'inr_spec_depth', 4)
        inr_spec_dim = getattr(args, 'inr_spec_dim', 128)
        self.pe_levels = getattr(args, 'pe_levels_spectral', 4)

        self.spectral_extractor = Spectral_ResDCT_Extractor(
            in_ch=16, out_ch=feat_dim, base_dim=hsi_base_dim, depth=hsi_depth
        )

        # MLP input: feat_dim + 1 (raw coord) + 2*pe_levels (PE)
        mlp_in_dim = feat_dim + 1 + 2 * self.pe_levels
        self.mlp = build_mlp(mlp_in_dim, Rl, inr_spec_dim, inr_spec_depth)

    def positional_encoding_1d(self, x):
        """x: (..., 1) �� (..., 2*pe_levels)"""
        freq = (2 ** torch.arange(self.pe_levels, dtype=torch.float32,
                                  device=x.device)) * np.pi           # (L,)
        spectrum = x * freq                                            # (..., L)
        return torch.cat([spectrum.sin(), spectrum.cos()], dim=-1)      # (..., 2L)

    def forward(self, coords_1d, img_input):
        B, L_in, H, W = img_input.shape
        L_coords = coords_1d.shape[0]

        regional_spectra = F.adaptive_avg_pool2d(img_input, (4, 4))
        spectral_signal = regional_spectra.view(B, L_in, 16).permute(0, 2, 1)

        feat_map_1d, spectral_global_feat = self.spectral_extractor(spectral_signal)
        local_feat = feat_map_1d.permute(0, 2, 1)                     # (B, L, feat_dim)

        coords_expand = coords_1d.unsqueeze(0).expand(B, L_coords, -1)  # (B, L, 1)
        pe = self.positional_encoding_1d(coords_expand)                   # (B, L, 2*pe_levels)

        mlp_in = torch.cat([local_feat, coords_expand, pe], dim=-1)
        V = torch.tanh(self.mlp(mlp_in))
        return V, spectral_global_feat


# =========================================================================
# 4. Core Tensor & Projection
# =========================================================================
class CoreMatrixGenerator(nn.Module):
    def __init__(self, args, feat_dim=64):
        super().__init__()
        self.Rl = getattr(args, 'Rl', 20)
        self.Rs = getattr(args, 'Rs', 64)
        core_depth = getattr(args, 'core_depth', 3)
        core_dim = getattr(args, 'core_dim', 128)

        self.spatial_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Sigmoid()
        )
        self.spectral_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Sigmoid()
        )
        self.fusion_mlp = build_mlp(
            feat_dim * 2, self.Rl * self.Rs, core_dim, core_depth, use_layernorm=True
        )

    def forward(self, spatial_feat, spectral_feat):
        B = spatial_feat.shape[0]
        w_s = self.spatial_gate(spatial_feat)
        w_l = self.spectral_gate(spectral_feat)
        joint = torch.cat([spatial_feat * w_l, spectral_feat * w_s], dim=-1)
        return self.fusion_mlp(joint).view(B, self.Rl, self.Rs)


class ContinuousTuckerProjection(nn.Module):
    """
    Generates the Tucker residual Z = V @ G @ U^T at arbitrary resolution.

    Changes:
      - spatial_net is now Spatial_INR2D_Tucker (INR2D-based)
      - spectral_net has PE
      - forward receives (target_h, target_w) instead of pre-computed coords_2d
    """

    def __init__(self, args, in_channels_unet):
        super().__init__()
        self.Rl = getattr(args, 'Rl', 20)
        self.Rs = getattr(args, 'Rs', 64)

        self.spatial_net = Spatial_INR2D_Tucker(args, in_channels_unet)
        self.spectral_net = Spectral_INR_Tucker_Adaptive(args, in_channels_unet, feat_dim=32)
        self.core_generator = CoreMatrixGenerator(args, feat_dim=64)

    def forward(self, S_k, coords_1d, target_shape):
        """
        Args:
            S_k:          (B, L, H_hr, W_hr) �� current iterate (at HR resolution)
            coords_1d:    (L, 1) �� spectral coordinates
            target_shape: (B, L, H_hr, W_hr)
        """
        B, L, H, W = target_shape

        # Spatial factor U: (B, H*W, Rs)
        U, spatial_global = self.spatial_net(S_k, target_h=H, target_w=W)

        # Spectral factor V: (B, L, Rl)
        V, spectral_global = self.spectral_net(coords_1d, S_k)

        # Core tensor G: (B, Rl, Rs)
        core_G = self.core_generator(spatial_global, spectral_global)
        scale = math.sqrt(self.Rl * self.Rs)

        # Reconstruct: Z = V @ G @ U^T / sqrt(Rl*Rs)
        VG = torch.matmul(V, core_G)                         # (B, L, Rs)
        Z_flat = torch.matmul(VG, U.transpose(1, 2)) / scale # (B, L, H*W)
        Z_img = Z_flat.view(B, L, H, W)

        E_out = V[0]                                          # (L, Rl) endmember
        A_out = torch.matmul(core_G, U.transpose(1, 2)).view(B, self.Rl, H, W)[0]
        return Z_img, E_out, A_out


# =========================================================================
# 5. Unrolled Optimization Network
# =========================================================================
class UnrolledTuckerFusion(nn.Module):
    def __init__(self, args, L_bands):
        super().__init__()
        self.K = getattr(args, 'K_iters', 5)
        self.eta = nn.Parameter(torch.tensor([0.1] * self.K))
        self.res_alpha = nn.Parameter(torch.tensor([0.1] * self.K))

        self.projectors = nn.ModuleList([
            ContinuousTuckerProjection(args, L_bands) for _ in range(self.K)
        ])

    def forward(self, Y_hsi, Y_msi, psf_est, srf_est, scale_factor):
        B, L, H_lr, W_lr = Y_hsi.shape
        B, C_m, H_hr, W_hr = Y_msi.shape
        device = Y_hsi.device

        coords_1d = make_coord_1d(L, device)

        X_k = F.interpolate(Y_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        psf_grouped = psf_est.repeat(L, 1, 1, 1) if psf_est.shape[0] == 1 else psf_est

        out_list = []
        soft_eta = torch.sigmoid(self.eta) + 1e-4
        soft_alpha = torch.sigmoid(self.res_alpha) + 1e-4

        last_grad_norm = 0.
        last_res_norm = 0.

        for k in range(self.K):
            # Data-fidelity gradient
            diff_hsi = F.conv2d(X_k, psf_grouped, stride=scale_factor, groups=L) - Y_hsi
            grad_hsi = F.conv_transpose2d(diff_hsi, psf_grouped, stride=scale_factor, groups=L)
            if grad_hsi.shape != X_k.shape:
                grad_hsi = F.interpolate(grad_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)

            diff_msi = F.conv2d(X_k, srf_est) - Y_msi
            grad_msi = F.conv_transpose2d(diff_msi, srf_est)

            S_k = X_k - soft_eta[k] * (grad_hsi + grad_msi)
            S_k = torch.clamp(S_k, 0.0, 1.0)

            # Tucker projection (INR2D-based)
            residual, E_k, A_k = self.projectors[k](
                S_k, coords_1d, target_shape=(B, L, H_hr, W_hr)
            )

            X_k = S_k + soft_alpha[k] * residual
            X_k = torch.clamp(X_k, 1e-4, 1.0)
            out_list.append(X_k)

            if k == self.K - 1:
                last_grad_norm = torch.norm(grad_hsi + grad_msi).item()
                last_res_norm = torch.norm(residual).item()

        return out_list, E_k, A_k, last_grad_norm, last_res_norm


# =========================================================================
# 6. DIP Training Pipeline
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
        self.net = UnrolledTuckerFusion(args=self.args, L_bands=L_bands).to(args.device)

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + 1 - self.args.niter3_dip) / float(self.args.niter_decay3_dip + 1)

        self.optimizer = optim.Adam(self.net.parameters(), lr=max(self.args.lr_stage3_dip, 3e-4))
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

    def angle_similarity_loss(self, Sh, Sm):
        B, C, H, W = Sh.shape
        Sh_vec = Sh.view(B, C, -1).permute(0, 2, 1)
        Sm_vec = Sm.view(B, C, -1).permute(0, 2, 1)
        dot = (Sh_vec * Sm_vec).sum(dim=-1)
        norm_Sh = torch.sqrt(torch.sum(Sh_vec ** 2, dim=-1) + 1e-12)
        norm_Sm = torch.sqrt(torch.sum(Sm_vec ** 2, dim=-1) + 1e-12)
        cos = torch.clamp(dot / (norm_Sh * norm_Sm), -0.9999, 0.9999)
        return torch.acos(cos).mean() / math.pi

    def train(self):
        flag_best_fhsi = [10, 0, 'data', 0]

        L1Loss = nn.L1Loss(reduction='mean')
        RelLoss = RelativeL1Loss(eps=1e-3).to(self.args.device)

        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
            self.optimizer.zero_grad()

            out_list, self.endmember, self.abundance, last_grad_norm, last_res_norm = self.net(
                self.lr_hsi, self.hr_msi, self.psf_est, self.srf_est, self.args.scale_factor
            )

            self.hr_hsi_rec = out_list[-1]
            loss_total = 0
            gamma = getattr(self.args, 'gamma', 0.8)

            for i, X_pred in enumerate(out_list):
                weight = gamma ** (self.net.K - 1 - i)
                hr_msi_pred = self.srf_down(X_pred, self.srf_est)
                lr_hsi_pred = self.psf_down(X_pred, self.psf_est, self.args.scale_factor)

                loss_hsi = L1Loss(self.lr_hsi, lr_hsi_pred) + 0.1 * RelLoss(lr_hsi_pred, self.lr_hsi)
                loss_msi = L1Loss(self.hr_msi, hr_msi_pred) + 0.1 * RelLoss(hr_msi_pred, self.hr_msi)

                loss_step = loss_hsi + loss_msi
                loss_step += getattr(self.args, 'sam_weight', 0.01) * self.angle_similarity_loss(
                    lr_hsi_pred, self.lr_hsi
                )
                loss_total += weight * loss_step

            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 == 0:
                with torch.no_grad():
                    print('\n==================================================')
                    print(f'epoch:{epoch} lr:{self.optimizer.param_groups[0]["lr"]:.6f}')

                    gt_tensor = torch.tensor(self.gt.transpose(2, 0, 1)).unsqueeze(0).to(self.args.device).float()
                    spatial_rmse = torch.sqrt(torch.mean((self.hr_hsi_rec - gt_tensor) ** 2, dim=1))
                    max_spatial_err = spatial_rmse.max().item()
                    mean_spatial_err = spatial_rmse.mean().item()
                    print(
                        f'[Probe-Spatial] Max RMSE: {max_spatial_err:.5f} | Mean RMSE: {mean_spatial_err:.5f} '
                        f'| Ratio: {max_spatial_err / (mean_spatial_err + 1e-8):.1f}x'
                    )

                    print(f'[Probe-Unfolding] PSNR progress across {self.net.K} stages:')
                    stage_psnrs = []
                    for k_idx, X_stage in enumerate(out_list):
                        X_stage_np = X_stage.data.cpu().numpy()[0].transpose(1, 2, 0)
                        _, stage_psnr, _, _, _, _, _ = MetricsCal(self.gt, X_stage_np, self.args.scale_factor)
                        stage_psnrs.append(f"S{k_idx + 1}:{stage_psnr:.2f}")
                    print(" -> ".join(stage_psnrs))

                    abun_min = self.abundance.min().item()
                    abun_max = self.abundance.max().item()
                    abun_sum_mean = torch.sum(self.abundance, dim=1).mean().item()
                    print(
                        f'[Probe-Latent] Abundance Range: [{abun_min:.2f}, {abun_max:.2f}] '
                        f'| Mean Channel Sum: {abun_sum_mean:.2f}'
                    )
                    print(
                        f'[Probe-Gradient] Data Grad Norm: {last_grad_norm:.4f} '
                        f'| Prior Residual Norm: {last_res_norm:.4f}'
                    )

                    rec_np = self.hr_hsi_rec.data.cpu().numpy()[0]
                    gt_np = self.gt.transpose(2, 0, 1)
                    L_bands = rec_np.shape[0]
                    band_ergas_list = []
                    for b in range(L_bands):
                        rmse_b = np.sqrt(np.mean((rec_np[b] - gt_np[b]) ** 2))
                        mu_b = np.mean(gt_np[b])
                        band_ergas_list.append(rmse_b / mu_b if mu_b > 1e-4 else 0.0)

                    worst_band_idx = np.argmax(band_ergas_list)
                    worst_band_val = band_ergas_list[worst_band_idx]
                    gt_worst_mean = np.mean(gt_np[worst_band_idx])

                    alpha_vals = torch.sigmoid(self.net.res_alpha).data.cpu().numpy()
                    eta_vals = torch.sigmoid(self.net.eta).data.cpu().numpy()

                    print(
                        f'[Diagnosis] Worst ERGAS Band: {worst_band_idx} '
                        f'| Relative Err: {worst_band_val:.4f} '
                        f'| GT Mean of this band: {gt_worst_mean:.5f}'
                    )
                    print(f'[Network] Unfolding Alphas: {np.round(alpha_vals, 3)}')
                    print(f'[Network] Unfolding Etas:   {np.round(eta_vals, 3)}')
                    print('--------------------------------------------------')

                    hr_msi_frec = self.srf_down(self.hr_hsi_rec, self.srf_est)
                    lr_hsi_frec = self.psf_down(self.hr_hsi_rec, self.psf_est, self.args.scale_factor)

                    hr_msi_numpy = self.hr_msi.data.cpu().numpy()[0].transpose(1, 2, 0)
                    hr_msi_frec_numpy = hr_msi_frec.data.cpu().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_numpy = self.lr_hsi.data.cpu().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_frec_numpy = lr_hsi_frec.data.cpu().numpy()[0].transpose(1, 2, 0)
                    hr_hsi_rec_numpy = self.hr_hsi_rec.data.cpu().numpy()[0].transpose(1, 2, 0)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(
                        lr_hsi_numpy, lr_hsi_frec_numpy, self.args.scale_factor
                    )
                    L1 = np.mean(np.abs(lr_hsi_numpy - lr_hsi_frec_numpy))
                    info1 = (f"lr_hsi vs pred\n L1 {L1:.4f} sam {sam:.4f},psnr {psnr:.4f},"
                             f"ergas {ergas:.4f},cc {cc:.4f},rmse {rmse:.4f},Ssim {Ssim:.4f},Uqi {Uqi:.4f}")
                    print(info1)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(
                        hr_msi_numpy, hr_msi_frec_numpy, self.args.scale_factor
                    )
                    L1 = np.mean(np.abs(hr_msi_numpy - hr_msi_frec_numpy))
                    info2 = (f"hr_msi vs pred\n L1 {L1:.4f} sam {sam:.4f},psnr {psnr:.4f},"
                             f"ergas {ergas:.4f},cc {cc:.4f},rmse {rmse:.4f},Ssim {Ssim:.4f},Uqi {Uqi:.4f}")
                    print(info2)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(
                        self.gt, hr_hsi_rec_numpy, self.args.scale_factor
                    )
                    L1 = np.mean(np.abs(self.gt - hr_hsi_rec_numpy))
                    info3 = (f"hr_hsi vs gt\n L1 {L1:.4f} sam {sam:.4f},psnr {psnr:.4f},"
                             f"ergas {ergas:.4f},cc {cc:.4f},rmse {rmse:.4f},Ssim {Ssim:.4f},Uqi {Uqi:.4f}")
                    print(info3)

                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
                    with open(file_name, 'a') as f:
                        f.write(f'--- epoch:{epoch} ---\n')
                        f.write(info1 + '\n')
                        f.write(info2 + '\n')
                        f.write(info3 + '\n')

                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:
                        flag_best_fhsi[0] = sam
                        flag_best_fhsi[1] = psnr
                        flag_best_fhsi[2] = self.hr_hsi_rec
                        flag_best_fhsi[3] = epoch
                        info_a, info_b, info_c = info1, info2, info3

        scipy.io.savemat(
            os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'),
            {'Out': flag_best_fhsi[2].data.cpu().numpy()[0].transpose(1, 2, 0)}
        )
        scipy.io.savemat(
            os.path.join(self.args.expr_dir, 'Endmember.mat'),
            {'end': self.endmember.data.cpu().numpy()}
        )
        scipy.io.savemat(
            os.path.join(self.args.expr_dir, 'Abundance.mat'),
            {'abun': self.abundance.data.cpu().numpy()}
        )

        file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
        with open(file_name, 'a') as f:
            f.write('========================== BEST ==========================\n')
            f.write(f'epoch_fhsi_best:{flag_best_fhsi[3]}\n')
            f.write(info_a + '\n')
            f.write(info_b + '\n')
            f.write(info_c + '\n')

        return flag_best_fhsi[2]