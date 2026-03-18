# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy.io
import torch.nn.functional as F
from .evaluation import MetricsCal
import math


# =========================================================================
# 1. ������������������
# =========================================================================
def make_coord_2d(h, w, device=None):
    if isinstance(h, torch.Tensor) or isinstance(w, torch.Tensor):
        h, w = 128, 128
    else:
        h, w = int(h), int(w)
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    coord = torch.stack([xx, yy], dim=-1).view(-1, 2)
    return coord


def make_coord_1d(l, device=None):
    if isinstance(l, torch.Tensor):
        l = 31
    else:
        l = int(l)
    return torch.linspace(-1, 1, steps=l, device=device).view(-1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, use_layernorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim) if not use_layernorm else nn.GroupNorm(1, dim)
        self.act = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim) if not use_layernorm else nn.GroupNorm(1, dim)

    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class ResidualBlock_1D(nn.Module):
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
            layers.append(ResidualBlock_1D(hidden_dim, use_layernorm))

        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# =========================================================================
# 2. ������������������
# =========================================================================
class PSF_down():
    def __call__(self, input_tensor, psf, ratio):
        _, C, _, _ = input_tensor.shape
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1)
        output_tensor = F.conv2d(input_tensor, psf, None, (ratio, ratio), groups=C)
        return output_tensor


class SRF_down():
    def __call__(self, input_tensor, srf):
        output_tensor = F.conv2d(input_tensor, srf, None)
        return output_tensor


def angle_similarity_loss(Sh, Sm):
    B, C, H, W = Sh.shape
    Sh_vec = Sh.view(B, C, -1).permute(0, 2, 1)
    Sm_vec = Sm.view(B, C, -1).permute(0, 2, 1)
    dot = (Sh_vec * Sm_vec).sum(dim=-1)
    norm_Sh = torch.sqrt(torch.sum(Sh_vec ** 2, dim=-1) + 1e-12)
    norm_Sm = torch.sqrt(torch.sum(Sm_vec ** 2, dim=-1) + 1e-12)
    cos = dot / (norm_Sh * norm_Sm)
    return (1.0 - cos).mean()


def tv_loss(x):
    dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return dh + dw


def abundance_entropy(A, eps=1e-12):
    A_clamped = torch.clamp(A, min=eps)
    ent = -(A_clamped * torch.log(A_clamped)).sum(dim=1)
    return ent.mean()


def pairwise_offdiag_mean_cosine(E):
    E_n = F.normalize(E, dim=-1)
    sim = torch.matmul(E_n, E_n.transpose(1, 2))
    R = sim.shape[-1]
    eye = torch.eye(R, device=sim.device).unsqueeze(0)
    off_diag = sim * (1.0 - eye)
    denom = max(R * (R - 1), 1)
    return off_diag.sum() / (sim.shape[0] * denom)


# =========================================================================
# 3. ����������������
# =========================================================================
class StickBreaking(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.temp = nn.Parameter(torch.tensor(5.0))

    def forward(self, logits):
        t = torch.clamp(self.temp, min=0.5, max=10.0)
        v = torch.sigmoid(logits / t)
        remains = 1.0 - v + self.eps
        cum_remains = torch.cumprod(remains, dim=-1)
        ones = torch.ones_like(cum_remains[..., :1])
        cum_remains_shifted = torch.cat([ones, cum_remains[..., :-1]], dim=-1)
        abundances_R_minus_1 = v * cum_remains_shifted
        last_abundance = cum_remains[..., -1:]
        abundances = torch.cat([abundances_R_minus_1, last_abundance], dim=-1)
        return abundances


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(scale))


class ResUNet_Light(nn.Module):
    def __init__(self, in_ch, out_ch, base_dim=32, depth=3):
        super().__init__()
        self.depth = depth
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, 1, 1),
            nn.BatchNorm2d(base_dim), nn.LeakyReLU(0.2, True)
        )
        self.encs = nn.ModuleList()
        self.encs.append(ResidualBlock(base_dim))

        for i in range(1, depth + 1):
            in_c = base_dim * (2 ** (i - 1))
            out_c = base_dim * (2 ** i)
            self.encs.append(nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_c, out_c, 1, 1, 0),  # �������� padding
                nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, True),
                ResidualBlock(out_c)
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


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = 2 ** torch.arange(num_freqs).float() * math.pi

    def forward(self, x):
        freqs = self.freqs.to(x.device)
        x_proj = x.unsqueeze(-1) * freqs
        x_proj = x_proj.view(*x.shape[:-1], -1)
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# =========================================================================
# 4. Deformable Dynamic Kernels
# =========================================================================
class DeformableDynamicKernel2D(nn.Module):
    def __init__(self, feat_dim, K=9, hidden_dim=64):
        super().__init__()
        self.K = K
        self.router = build_mlp(in_dim=feat_dim + 2, out_dim=3 * K, hidden_dim=hidden_dim, depth=3)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, feat_map, coords_2d):
        B, C, H, W = feat_map.shape
        N = coords_2d.shape[1]

        grid_anchor = coords_2d.view(B, 1, N, 2)
        f_anchor = F.grid_sample(feat_map, grid_anchor, mode='bilinear', padding_mode='border', align_corners=True)
        f_anchor = f_anchor.squeeze(2).permute(0, 2, 1)

        router_in = torch.cat([f_anchor, coords_2d], dim=-1)
        routing_out = self.router(router_in)

        max_offset_x = 16.0 / W
        max_offset_y = 16.0 / H

        offsets = torch.tanh(routing_out[..., :2 * self.K])
        offsets_x = offsets[..., 0::2] * max_offset_x
        offsets_y = offsets[..., 1::2] * max_offset_y
        offsets = torch.stack([offsets_x, offsets_y], dim=-1)

        raw_weights = routing_out[..., 2 * self.K:]
        dynamic_weights = F.softmax(raw_weights, dim=-1).unsqueeze(1)

        coords_expanded = coords_2d.unsqueeze(2).expand(B, N, self.K, 2)
        deform_coords = coords_expanded + offsets
        grid_deform = deform_coords.view(B, N * self.K, 1, 2)

        f_sampled = F.grid_sample(feat_map, grid_deform, mode='bilinear', padding_mode='border', align_corners=True)
        f_sampled = f_sampled.squeeze(-1).view(B, C, N, self.K)

        f_agg = torch.sum(f_sampled * dynamic_weights, dim=-1)
        return f_agg.permute(0, 2, 1)


class DeformableDynamicKernel1D(nn.Module):
    def __init__(self, feat_dim, K=5, hidden_dim=64):
        super().__init__()
        self.K = K
        self.router = build_mlp(in_dim=feat_dim + 1, out_dim=2 * K, hidden_dim=hidden_dim, depth=3)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, feat_1d, coords_1d):
        B, C, L_in = feat_1d.shape
        N = coords_1d.shape[1]

        feat_2d = feat_1d.unsqueeze(2)
        grid_x = coords_1d.view(B, 1, N, 1)
        grid_y = torch.zeros_like(grid_x)
        grid_anchor = torch.cat([grid_x, grid_y], dim=-1)

        f_anchor = F.grid_sample(feat_2d, grid_anchor, mode='bilinear', padding_mode='border', align_corners=True)
        f_anchor = f_anchor.squeeze(2).permute(0, 2, 1)

        router_in = torch.cat([f_anchor, coords_1d], dim=-1)
        routing_out = self.router(router_in)

        max_offset_x = 6.0 / L_in
        offsets_x = torch.tanh(routing_out[..., :self.K]) * max_offset_x
        raw_weights = routing_out[..., self.K:]
        dynamic_weights = F.softmax(raw_weights, dim=-1).unsqueeze(1)

        coords_expanded_x = coords_1d.expand(B, N, self.K)
        deform_coords_x = coords_expanded_x + offsets_x
        deform_coords_y = torch.zeros_like(deform_coords_x)
        deform_coords = torch.stack([deform_coords_x, deform_coords_y], dim=-1)

        grid_deform = deform_coords.view(B, N * self.K, 1, 2)
        f_sampled = F.grid_sample(feat_2d, grid_deform, mode='bilinear', padding_mode='border', align_corners=True)
        f_sampled = f_sampled.squeeze(-1).view(B, C, N, self.K)

        f_agg = torch.sum(f_sampled * dynamic_weights, dim=-1)
        return f_agg.permute(0, 2, 1)

# =========================================================================
# 5. INR �������� (������ INR-1D ������������)
# =========================================================================
# (Spatial_Abundance_Net �������������� DifferentiableEndmemberExtractor)

class Spatial_Abundance_Net(nn.Module):
    def __init__(self, args, in_channels, out_dim, feat_dim=32):
        super().__init__()
        msi_base_dim = getattr(args, 'msi_base_dim', 32)
        msi_depth = getattr(args, 'msi_depth', 3)
        inr_spat_depth = getattr(args, 'inr_spat_depth', 3)
        inr_spat_dim = getattr(args, 'inr_spat_dim', 128)

        self.unet = ResUNet_Light(in_channels, feat_dim, base_dim=msi_base_dim, depth=msi_depth)
        self.deformable_kernel = DeformableDynamicKernel2D(feat_dim=feat_dim, K=9)

        self.pos_encoder_2d = PositionalEncoding(num_freqs=4)
        coord_dim = 2 + 2 * 2 * 4

        self.mlp = build_mlp(
            in_dim=feat_dim + coord_dim,
            out_dim=out_dim,
            hidden_dim=inr_spat_dim,
            depth=inr_spat_depth
        )

    def forward(self, img_input, coords_2d):
        feat_map = self.unet(img_input)
        f_agg = self.deformable_kernel(feat_map, coords_2d)
        coords_encoded = self.pos_encoder_2d(coords_2d)
        mlp_in = torch.cat([f_agg, coords_encoded], dim=-1)
        logits = self.mlp(mlp_in)
        return logits


# �������������������� DifferentiableEndmemberExtractor��������������
class DifferentiableEndmemberExtractor(nn.Module):
    def __init__(self, in_channels, R):
        super().__init__()
        self.R = R
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, R, kernel_size=1)
        )

    def forward(self, Y_hsi):
        B, L_in, H, W = Y_hsi.shape
        attn_logits = self.attention(Y_hsi)
        attn_flat = attn_logits.view(B, self.R, -1)
        attn_weights = F.softmax(attn_flat, dim=-1)
        Y_flat = Y_hsi.view(B, L_in, -1).transpose(1, 2)
        E_discrete = torch.matmul(attn_weights, Y_flat)
        return E_discrete


# =========================================================================
# ������������������ Shared_Endmember_INR
# =========================================================================
class Shared_Endmember_INR(nn.Module):
    def __init__(self, args, in_channels, feat_dim=64):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)
        inr_spec_depth = getattr(args, 'inr_spec_depth', 4)
        inr_spec_dim = getattr(args, 'inr_spec_dim', 128)

        self.pos_encoder = PositionalEncoding(num_freqs=6)
        coord_dim = 1 + 2 * 6

        # ���������������������������� HSI ��������������
        self.extractor = DifferentiableEndmemberExtractor(in_channels=in_channels, R=self.R)

        self.feat_extractor = nn.Sequential(
            nn.Conv1d(self.R, feat_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1)
        )
        self.deformable_kernel = DeformableDynamicKernel1D(feat_dim=feat_dim, K=5)
        self.inr_mlp = build_mlp(
            in_dim=feat_dim + coord_dim,
            out_dim=self.R,
            hidden_dim=inr_spec_dim,
            depth=inr_spec_depth
        )

        self.temp = nn.Parameter(torch.tensor(5.0))

    def forward(self, coords_1d, Y_hsi):
        B, L_in, _, _ = Y_hsi.shape
        target_L = coords_1d.shape[0]

        # 1. ������ HSI ��������������
        E_discrete = self.extractor(Y_hsi)

        # 2. ��������
        feat_1d = self.feat_extractor(E_discrete)

        # 3. 1D ������������
        coords_1d_expand = coords_1d.unsqueeze(0).expand(B, target_L, -1)
        f_agg = self.deformable_kernel(feat_1d, coords_1d_expand)

        # 4. ���������� MLP ������������
        coords_encoded = self.pos_encoder(coords_1d)
        coords_encoded_expand = coords_encoded.unsqueeze(0).expand(B, target_L, -1)

        mlp_in = torch.cat([f_agg, coords_encoded_expand], dim=-1)

        t = torch.clamp(self.temp, min=0.5, max=10.0)
        V = torch.sigmoid(self.inr_mlp(mlp_in) / t)

        return V


# =========================================================================
# 6. ������������������ (��������������)
# =========================================================================
class uMD_INR_Fusion(nn.Module):
    def __init__(self, args, L_bands, C_msi):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)

        # ��������MSI -> INR-2D -> ������������
        self.spatial_net = Spatial_Abundance_Net(args, in_channels=C_msi, out_dim=self.R - 1)
        self.stick_breaking = StickBreaking()

        # ��������HSI -> INR-1D -> ������������
        self.spectral_net = Shared_Endmember_INR(args, in_channels=L_bands)

    def forward(self, Y_hsi, Y_msi, target_H, target_W, target_L):
        B, _, H_lr, W_lr = Y_hsi.shape
        device = Y_hsi.device

        # ��������
        coords_2d_HR = make_coord_2d(target_H, target_W, device).unsqueeze(0).expand(B, -1, -1)
        coords_1d_target = make_coord_1d(target_L, device)

        # 1. ����������
        logits_msi = self.spatial_net(Y_msi, coords_2d_HR)
        A_HR = self.stick_breaking(logits_msi)  # [B, H*W, R]

        # 2. ����������
        E_target = self.spectral_net(coords_1d_target, Y_hsi)  # [B, target_L, R]

        # 3. ������������������X = E * A^T
        Z_flat_HR = torch.matmul(E_target, A_HR.transpose(1, 2))
        X_obs = Z_flat_HR.view(B, target_L, target_H, target_W)

        # ��������
        A_HR_img = A_HR.transpose(1, 2).view(B, self.R, target_H, target_W)
        A_LR_img = F.interpolate(A_HR_img, size=(H_lr, W_lr), mode='bilinear', align_corners=False)

        return X_obs, A_HR_img, A_LR_img, E_target


# =========================================================================
# 7. DIP Training Pipeline (��������������������������������)
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

        self.target_scale = getattr(args, 'target_scale', 4.0)
        _, _, H_hr, W_hr = self.hr_msi.shape
        self.target_H = int(H_hr * self.target_scale)
        self.target_W = int(W_hr * self.target_scale)
        self.target_L = getattr(args, 'target_L', L_bands)

        self.net = uMD_INR_Fusion(args=self.args, L_bands=L_bands, C_msi=C_msi).to(args.device)

        # =======================================================
        # ������������������
        # =======================================================
        self.lambda_tv = getattr(self.args, 'tv_weight', 5e-5)
        self.lambda_endmember_sep = getattr(self.args, 'endmember_sep_weight', 0.05)  # ��������������������
        self.lambda_endmember_smooth = getattr(self.args, 'endmember_smooth_weight', 0.01)  # ������������
        self.lambda_sparsity = getattr(self.args, 'sparse_weight', 0.005)  # �������� (��������)

        # ����������������������������0
        self.lambda_align = getattr(self.args, 'align_weight', 0.0)

        self.min_lr_factor = getattr(self.args, 'min_lr_factor', 0.1)

        def lambda_rule(epoch):
            raw = 1.0 - max(0, epoch + 1 - self.args.niter3_dip) / float(self.args.niter_decay3_dip + 1)
            return max(raw, self.min_lr_factor)

        router_params = []
        base_params = []
        for name, param in self.net.named_parameters():
            if 'router' in name:
                router_params.append(param)
            else:
                base_params.append(param)

        base_lr = getattr(self.args, 'lr_stage3_dip', 5e-4)

        self.optimizer = optim.Adam([
            {'params': base_params, 'lr': base_lr},
            {'params': router_params, 'lr': base_lr * 10.0}
        ])

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

    def train(self):
        flag_best_fhsi = [10, 0, 'data', 0]
        L1Loss = nn.L1Loss(reduction='mean')
        info_a, info_b, info_c = '', '', ''

        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
            self.optimizer.zero_grad()

            # ������������������
            X_obs, A_HR_img, _, E_target = self.net(
                self.lr_hsi, self.hr_msi,
                target_H=self.target_H, target_W=self.target_W, target_L=self.target_L
            )

            self.hr_hsi_rec = X_obs
            self.abundance = A_HR_img
            self.endmember = E_target[0]

            # ��������
            hr_msi_pred = self.srf_down(X_obs, self.srf_est)
            lr_hsi_pred = self.psf_down(X_obs, self.psf_est, self.args.scale_factor)

            # 1. ��������������
            loss_hsi = L1Loss(self.lr_hsi, lr_hsi_pred) + 0.1 * angle_similarity_loss(lr_hsi_pred, self.lr_hsi)
            loss_msi = L1Loss(self.hr_msi, hr_msi_pred)

            # 2. �������� (����)
            loss_align = torch.tensor(0.0, device=X_obs.device)  # ����������������
            loss_tv_abundance = tv_loss(A_HR_img)
            loss_sparsity = abundance_entropy(A_HR_img)

            # 3. �������� (����)
            loss_endmember_sep = pairwise_offdiag_mean_cosine(E_target)
            if E_target.shape[-1] > 1:
                loss_endmember_smooth = torch.mean(torch.abs(E_target[:, :, 1:] - E_target[:, :, :-1]))
            else:
                loss_endmember_smooth = torch.tensor(0.0, device=X_obs.device)

            # ������
            loss_total = (
                    loss_hsi
                    + loss_msi
                    + self.lambda_align * loss_align
                    + self.lambda_tv * loss_tv_abundance
                    + self.lambda_sparsity * loss_sparsity
                    + self.lambda_endmember_sep * loss_endmember_sep
                    + self.lambda_endmember_smooth * loss_endmember_smooth
            )

            loss_total.backward()

            # --- �������������� ---
            if epoch % 50 == 0:
                print('\n--- Diagnostic Probe ---')

                def get_grad_norm(model):
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    return total_norm ** 0.5

                def get_module_grad_norm(module):
                    total_norm = 0.0
                    count = 0
                    for p in module.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                            count += 1
                    return total_norm ** 0.5, count

                grad_norm_spat = get_grad_norm(self.net.spatial_net)
                grad_norm_spec = get_grad_norm(self.net.spectral_net)
                print(f"  [Gradients] Spatial-Path: {grad_norm_spat:.6f} | Spectral-Path: {grad_norm_spec:.6f}")

                A_HR_mean = A_HR_img.mean().item()
                A_HR_max = A_HR_img.max().item()
                A_HR_min = A_HR_img.min().item()
                A_entropy = abundance_entropy(A_HR_img).item()

                A_avg_activation = A_HR_img.mean(dim=(0, 2, 3)).detach().cpu().numpy()
                topk_idx = np.argsort(-A_avg_activation)[:min(5, len(A_avg_activation))]
                topk_str = ', '.join([f'{idx}:{A_avg_activation[idx]:.4f}' for idx in topk_idx])

                active_001 = int((A_avg_activation > 0.01).sum())
                active_0005 = int((A_avg_activation > 0.005).sum())

                print(
                    f"  [Abundance HR] Mean: {A_HR_mean:.6f} | Min: {A_HR_min:.6f} | Max: {A_HR_max:.6f} | Entropy: {A_entropy:.6f}")
                print(f"  [Abundance Avg Activation TopK] {topk_str}")
                print(f"  [Active Endmembers] >0.01: {active_001} | >0.005: {active_0005}")

                E_mean = E_target.mean().item()
                E_var = E_target.var().item()
                E_min = E_target.min().item()
                E_max = E_target.max().item()
                E_sep_monitor = pairwise_offdiag_mean_cosine(E_target).item()

                d1 = torch.mean(torch.abs(E_target[:, :, 1:] - E_target[:, :, :-1])).item() if E_target.shape[
                                                                                                   -1] > 1 else 0.0
                if E_target.shape[-1] > 2:
                    d2 = torch.mean(torch.abs((E_target[:, :, 2:] - E_target[:, :, 1:-1]) - (
                                E_target[:, :, 1:-1] - E_target[:, :, :-2]))).item()
                else:
                    d2 = 0.0

                print(
                    f"  [Endmember] Mean: {E_mean:.6f} | Variance: {E_var:.6f} | Range: [{E_min:.6f}, {E_max:.6f}] | OffDiagCos: {E_sep_monitor:.6f}")
                print(f"  [Endmember Smoothness] FirstDiff: {d1:.6f} | SecondDiff: {d2:.6f}")

                sb_temp = torch.clamp(self.net.stick_breaking.temp, min=0.5, max=10.0).item()
                spec_temp = torch.clamp(self.net.spectral_net.temp, min=0.5, max=10.0).item()
                print(f"  [Temperature] StickBreaking: {sb_temp:.6f} | SpectralDecoder: {spec_temp:.6f}")

                router_2d_grad_all, router_2d_count = get_module_grad_norm(
                    self.net.spatial_net.deformable_kernel.router)
                router_1d_grad_all, router_1d_count = get_module_grad_norm(
                    self.net.spectral_net.deformable_kernel.router)
                print(
                    f"  [Deform-2D Router-All] Grad Norm: {router_2d_grad_all:.6f} | ParamsWithGrad: {router_2d_count}")
                print(
                    f"  [Deform-1D Router-All] Grad Norm: {router_1d_grad_all:.6f} | ParamsWithGrad: {router_1d_count}")

                try:
                    router_layer_2d = self.net.spatial_net.deformable_kernel.router[-1]
                    router_2d_last = router_layer_2d.weight.grad.norm(
                        2).item() if router_layer_2d.weight.grad is not None else 0.0
                    print(f"  [Deform-2D Router-Last] Grad Norm: {router_2d_last:.6f}")
                except Exception:
                    pass

                try:
                    router_layer_1d = self.net.spectral_net.deformable_kernel.router[-1]
                    router_1d_last = router_layer_1d.weight.grad.norm(
                        2).item() if router_layer_1d.weight.grad is not None else 0.0
                    print(f"  [Deform-1D Router-Last] Grad Norm: {router_1d_last:.6f}")
                except Exception:
                    pass

                print(
                    f"  [Loss Breakdown] "
                    f"HSI: {loss_hsi.item():.6f} | "
                    f"MSI: {loss_msi.item():.6f} | "
                    f"TV: {loss_tv_abundance.item():.6f} | "
                    f"Sep: {loss_endmember_sep.item():.6f} | "
                    f"Total: {loss_total.item():.6f}"
                )
                print('------------------------------------------')

            self.optimizer.step()
            self.scheduler.step()

            # --- �������������� ---
            if epoch % 50 == 0:
                with torch.no_grad():
                    print('\n==================================================')
                    print(
                        f'epoch:{epoch} '
                        f'lr_base:{self.optimizer.param_groups[0]["lr"]:.6f} '
                        f'lr_router:{self.optimizer.param_groups[1]["lr"]:.6f}'
                    )

                    gt_tensor = torch.tensor(self.gt.transpose(2, 0, 1)).unsqueeze(0).to(self.args.device).float()
                    X_eval = self.hr_hsi_rec

                    if X_eval.shape[2:] != gt_tensor.shape[2:]:
                        X_eval = F.interpolate(X_eval, size=gt_tensor.shape[2:], mode='bilinear', align_corners=False)

                    if X_eval.shape[1] != gt_tensor.shape[1]:
                        X_eval = F.interpolate(
                            X_eval.unsqueeze(1),
                            size=(gt_tensor.shape[1], gt_tensor.shape[2], gt_tensor.shape[3]),
                            mode='trilinear',
                            align_corners=False
                        ).squeeze(1)

                    X_stage_np = X_eval.data.cpu().numpy()[0].transpose(1, 2, 0)
                    _, stage_psnr, _, _, _, _, _ = MetricsCal(self.gt, X_stage_np, self.args.scale_factor)
                    print(f'>> [Probe] End-to-End PSNR: {stage_psnr:.2f}')

                    hr_msi_frec = self.srf_down(X_obs, self.srf_est)
                    lr_hsi_frec = self.psf_down(X_obs, self.psf_est, self.args.scale_factor)

                    hr_msi_numpy = self.hr_msi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_msi_frec_numpy = hr_msi_frec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_numpy = self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_frec_numpy = lr_hsi_frec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_hsi_rec_numpy = X_eval.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(
                        lr_hsi_numpy, lr_hsi_frec_numpy, self.args.scale_factor
                    )
                    L1 = np.mean(np.abs(lr_hsi_numpy - lr_hsi_frec_numpy))
                    info1 = "lr_hsi vs pred\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi
                    )
                    print(info1)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(
                        hr_msi_numpy, hr_msi_frec_numpy, self.args.scale_factor
                    )
                    L1 = np.mean(np.abs(hr_msi_numpy - hr_msi_frec_numpy))
                    info2 = "hr_msi vs pred\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi
                    )
                    print(info2)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(
                        self.gt, hr_hsi_rec_numpy, self.args.scale_factor
                    )
                    L1 = np.mean(np.abs(self.gt - hr_hsi_rec_numpy))
                    info3 = "hr_hsi vs gt\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi
                    )
                    print(info3)

                    # ������������������������
                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
                    with open(file_name, 'a') as opt_file:
                        opt_file.write(f'--- epoch:{epoch} ---\n')
                        opt_file.write(
                            f'lr_base:{self.optimizer.param_groups[0]["lr"]:.8f} '
                            f'lr_router:{self.optimizer.param_groups[1]["lr"]:.8f}\n'
                        )
                        opt_file.write(
                            f'loss_total:{loss_total.item():.8f} '
                            f'loss_hsi:{loss_hsi.item():.8f} '
                            f'loss_msi:{loss_msi.item():.8f} '
                            f'loss_align:{loss_align.item():.8f} '
                            f'loss_tv:{loss_tv_abundance.item():.8f}\n'
                        )
                        opt_file.write(
                            f'A_entropy:{A_entropy:.8f} '
                            f'active_gt_0.01:{active_001} '
                            f'active_gt_0.005:{active_0005} '
                            f'stick_temp:{sb_temp:.8f} '
                            f'spec_temp:{spec_temp:.8f} '
                            f'E_offdiag_cos:{E_sep_monitor:.8f} '
                            f'E_min:{E_min:.8f} '
                            f'E_max:{E_max:.8f} '
                            f'E_d1:{d1:.8f} '
                            f'E_d2:{d2:.8f} '
                            f'router2d_all_grad:{router_2d_grad_all:.8f} '
                            f'router1d_all_grad:{router_1d_grad_all:.8f}\n'
                        )
                        opt_file.write(info1 + '\n')
                        opt_file.write(info2 + '\n')
                        opt_file.write(info3 + '\n')

                    # �������� PSNR/SAM
                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:
                        flag_best_fhsi[0] = sam
                        flag_best_fhsi[1] = psnr
                        flag_best_fhsi[2] = self.hr_hsi_rec
                        flag_best_fhsi[3] = epoch
                        info_a, info_b, info_c = info1, info2, info3

        # ��������
        best_output = flag_best_fhsi[2] if isinstance(flag_best_fhsi[2], torch.Tensor) else self.hr_hsi_rec

        scipy.io.savemat(
            os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'),
            {'Out': best_output.data.cpu().numpy()[0].transpose(1, 2, 0)}
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
        with open(file_name, 'a') as opt_file:
            opt_file.write('========================== BEST ==========================\n')
            opt_file.write(f'epoch_fhsi_best:{flag_best_fhsi[3]}\n')
            opt_file.write(info_a + '\n')
            opt_file.write(info_b + '\n')
            opt_file.write(info_c + '\n')

        return flag_best_fhsi[2]