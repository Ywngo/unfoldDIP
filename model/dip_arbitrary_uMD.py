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
    """ ���� MLP �� 1D ������ """

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
        if use_layernorm: layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU(0.2, True))

        for _ in range(depth - 2):
            layers.append(ResidualBlock_1D(hidden_dim, use_layernorm))

        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# =========================================================================
# ��������������
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


# =========================================================================
# 2. ���������������� (������������������)
# =========================================================================
def angle_similarity_loss(Sh, Sm):
    B, C, H, W = Sh.shape
    Sh_vec = Sh.view(B, C, -1).permute(0, 2, 1)
    Sm_vec = Sm.view(B, C, -1).permute(0, 2, 1)
    dot = (Sh_vec * Sm_vec).sum(dim=-1)
    norm_Sh = torch.sqrt(torch.sum(Sh_vec ** 2, dim=-1) + 1e-12)
    norm_Sm = torch.sqrt(torch.sum(Sm_vec ** 2, dim=-1) + 1e-12)
    cos = dot / (norm_Sh * norm_Sm)
    cos = torch.clamp(cos, -0.9999, 0.9999)
    angles = torch.acos(cos)
    return (angles.mean() / math.pi)


# =========================================================================
# 3. ���������������� (������ Res-UNet)
# =========================================================================
class StickBreaking(nn.Module):
    """ Dirichlet-Net ��������������������������1 """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits):
        v = torch.sigmoid(logits)
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
    """ �������������������������������������������� """

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
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, True),
                ResidualBlock(out_c)  # ������������
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


class Spatial_Abundance_Net(nn.Module):
    def __init__(self, args, in_channels, out_dim, feat_dim=32):
        super().__init__()
        msi_base_dim = getattr(args, 'msi_base_dim', 32)
        msi_depth = getattr(args, 'msi_depth', 3)
        inr_spat_depth = getattr(args, 'inr_spat_depth', 3)
        inr_spat_dim = getattr(args, 'inr_spat_dim', 128)

        self.unet = ResUNet_Light(in_channels, feat_dim, base_dim=msi_base_dim, depth=msi_depth)
        self.mlp = build_mlp(in_dim=feat_dim + 2, out_dim=out_dim, hidden_dim=inr_spat_dim, depth=inr_spat_depth)

    def forward(self, img_input, coords_2d):
        B, _, H_in, W_in = img_input.shape
        target_N = coords_2d.shape[1]

        feat_map = self.unet(img_input)
        grid = coords_2d.view(B, 1, target_N, 2)
        q_feat = F.grid_sample(feat_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
        q_feat = q_feat.squeeze(2).permute(0, 2, 1)

        mlp_in = torch.cat([q_feat, coords_2d], dim=-1)
        logits = self.mlp(mlp_in)
        return logits


# =========================================================================
# 4. �������������� (��������������������������������)
# =========================================================================
class PositionalEncoding(nn.Module):
    """ ���������������������� INR �������������������������� """

    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = 2 ** torch.arange(num_freqs).float() * math.pi

    def forward(self, x):
        freqs = self.freqs.to(x.device)
        x_proj = x * freqs
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Shared_Endmember_INR(nn.Module):
    def __init__(self, args, in_channels, feat_dim=64):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)
        inr_spec_depth = getattr(args, 'inr_spec_depth', 4)
        inr_spec_dim = getattr(args, 'inr_spec_dim', 128)

        self.pos_encoder = PositionalEncoding(num_freqs=6)
        coord_dim = 1 + 2 * 6  # �������� + sin/cos ��������

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(in_channels, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, feat_dim)
        )
        # ��������: �������� + ����������������
        self.inr_mlp = build_mlp(in_dim=feat_dim + coord_dim, out_dim=self.R, hidden_dim=inr_spec_dim,
                                 depth=inr_spec_depth)

    def forward(self, coords_1d, Y_hsi):
        B = Y_hsi.shape[0]
        target_L = coords_1d.shape[0]

        global_stat = self.global_pool(Y_hsi).view(B, -1)
        global_feat = self.global_mlp(global_stat)
        global_feat_expand = global_feat.unsqueeze(1).expand(-1, target_L, -1)

        # ������������
        coords_encoded = self.pos_encoder(coords_1d)  # (target_L, coord_dim)
        coords_expand = coords_encoded.unsqueeze(0).expand(B, target_L, -1)

        mlp_in = torch.cat([global_feat_expand, coords_expand], dim=-1)
        V = torch.sigmoid(self.inr_mlp(mlp_in))
        return V


# =========================================================================
# 5. uMD-INR ����������
# =========================================================================
class uMD_INR_Fusion(nn.Module):
    def __init__(self, args, L_bands, C_msi):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)
        self.enc_msi = Spatial_Abundance_Net(args, in_channels=C_msi, out_dim=self.R - 1)
        self.enc_hsi = Spatial_Abundance_Net(args, in_channels=L_bands, out_dim=self.R - 1)
        self.stick_breaking = StickBreaking()
        self.shared_decoder = Shared_Endmember_INR(args, in_channels=L_bands)

    def forward(self, Y_hsi, Y_msi, target_H, target_W, target_L):
        B, _, H_lr, W_lr = Y_hsi.shape
        device = Y_hsi.device

        coords_2d_HR = make_coord_2d(target_H, target_W, device).unsqueeze(0).expand(B, -1, -1)
        coords_2d_LR = make_coord_2d(H_lr, W_lr, device).unsqueeze(0).expand(B, -1, -1)
        coords_1d_target = make_coord_1d(target_L, device)

        # ��������������
        logits_msi = self.enc_msi(Y_msi, coords_2d_HR)
        A_HR = self.stick_breaking(logits_msi)

        logits_hsi = self.enc_hsi(Y_hsi, coords_2d_LR)
        A_LR = self.stick_breaking(logits_hsi)

        # ������������ (����������)
        E_target = self.shared_decoder(coords_1d_target, Y_hsi)

        # ����������������
        Z_flat_HR = torch.matmul(E_target, A_HR.transpose(1, 2))
        X_obs = Z_flat_HR.view(B, target_L, target_H, target_W)

        A_HR_img = A_HR.transpose(1, 2).view(B, self.R, target_H, target_W)
        A_LR_img = A_LR.transpose(1, 2).view(B, self.R, H_lr, W_lr)

        return X_obs, A_HR_img, A_LR_img, E_target


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
        C_msi = self.hr_msi.shape[1]

        self.target_scale = getattr(args, 'target_scale', 4.0)
        _, _, H_hr, W_hr = self.hr_msi.shape
        self.target_H = int(H_hr * self.target_scale)
        self.target_W = int(W_hr * self.target_scale)
        self.target_L = getattr(args, 'target_L', L_bands)

        self.net = uMD_INR_Fusion(args=self.args, L_bands=L_bands, C_msi=C_msi).to(args.device)

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + 1 - self.args.niter3_dip) / float(self.args.niter_decay3_dip + 1)

        # ������������������������������������
        self.optimizer = optim.Adam(self.net.parameters(), lr=max(self.args.lr_stage3_dip, 5e-4))
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

    def train(self):
        flag_best_fhsi = [10, 0, 'data', 0]
        L1Loss = nn.L1Loss(reduction='mean')

        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
            self.optimizer.zero_grad()

            X_obs, A_HR_img, A_LR_img, E_target = self.net(
                self.lr_hsi, self.hr_msi,
                target_H=self.target_H, target_W=self.target_W, target_L=self.target_L
            )

            self.hr_hsi_rec = X_obs
            self.abundance = A_HR_img
            self.endmember = E_target[0]

            hr_msi_pred = self.srf_down(X_obs, self.srf_est)
            lr_hsi_pred = self.psf_down(X_obs, self.psf_est, self.args.scale_factor)

            # ������L1 �������� + ���������������� (SAM)
            loss_hsi = L1Loss(self.lr_hsi, lr_hsi_pred) + 0.1 * angle_similarity_loss(lr_hsi_pred, self.lr_hsi)
            loss_msi = L1Loss(self.hr_msi, hr_msi_pred)

            # uSDN ��������������������������������������������
            A_HR_down = self.psf_down(A_HR_img, self.psf_est, self.args.scale_factor)
            loss_align = angle_similarity_loss(A_HR_down, A_LR_img)

            # ������ entropy ����������StickBreaking ������������������������
            lambda_align = getattr(self.args, 'align_weight', 0.1)
            loss_total = loss_hsi + loss_msi + lambda_align * loss_align

            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            # --- ��������������������������---
            if epoch % 50 == 0:
                with torch.no_grad():
                    print('\n==================================================')
                    print(f'epoch:{epoch} lr:{self.optimizer.param_groups[0]["lr"]:.6f}')
                    print(
                        f'Losses - HSI: {loss_hsi.item():.4f} | MSI: {loss_msi.item():.4f} | Align: {loss_align.item():.4f}')

                    gt_tensor = torch.tensor(self.gt.transpose(2, 0, 1)).unsqueeze(0).to(self.args.device).float()
                    X_eval = self.hr_hsi_rec

                    if X_eval.shape[2:] != gt_tensor.shape[2:]:
                        X_eval = F.interpolate(X_eval, size=gt_tensor.shape[2:], mode='bilinear')
                    if X_eval.shape[1] != gt_tensor.shape[1]:
                        X_eval = F.interpolate(X_eval.unsqueeze(1),
                                               size=(gt_tensor.shape[1], gt_tensor.shape[2], gt_tensor.shape[3]),
                                               mode='trilinear').squeeze(1)

                    X_stage_np = X_eval.data.cpu().numpy()[0].transpose(1, 2, 0)
                    _, stage_psnr, _, _, _, _, _ = MetricsCal(self.gt, X_stage_np, self.args.scale_factor)
                    print(f'>> [Probe-Ablation] End-to-End PSNR: {stage_psnr:.2f}')

                    hr_msi_frec = self.srf_down(X_obs, self.srf_est)
                    lr_hsi_frec = self.psf_down(X_obs, self.psf_est, self.args.scale_factor)

                    hr_msi_numpy = self.hr_msi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_msi_frec_numpy = hr_msi_frec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_numpy = self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_frec_numpy = lr_hsi_frec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_hsi_rec_numpy = X_eval.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(lr_hsi_numpy, lr_hsi_frec_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(lr_hsi_numpy - lr_hsi_frec_numpy))
                    info1 = "lr_hsi vs pred\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(info1)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(hr_msi_numpy, hr_msi_frec_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(hr_msi_numpy - hr_msi_frec_numpy))
                    info2 = "hr_msi vs pred\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(info2)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.gt, hr_hsi_rec_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(self.gt - hr_hsi_rec_numpy))
                    info3 = "hr_hsi vs gt\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(info3)

                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
                    with open(file_name, 'a') as opt_file:
                        opt_file.write(f'--- epoch:{epoch} ---\n')
                        opt_file.write(info1 + '\n')
                        opt_file.write(info2 + '\n')
                        opt_file.write(info3 + '\n')

                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:
                        flag_best_fhsi[0] = sam
                        flag_best_fhsi[1] = psnr
                        flag_best_fhsi[2] = self.hr_hsi_rec
                        flag_best_fhsi[3] = epoch
                        info_a, info_b, info_c = info1, info2, info3

        best_output = flag_best_fhsi[2] if isinstance(flag_best_fhsi[2], torch.Tensor) else self.hr_hsi_rec
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'),
                         {'Out': best_output.data.cpu().numpy()[0].transpose(1, 2, 0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Endmember.mat'), {'end': self.endmember.data.cpu().numpy()})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Abundance.mat'), {'abun': self.abundance.data.cpu().numpy()})

        file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('========================== BEST ==========================\n')
            opt_file.write(f'epoch_fhsi_best:{flag_best_fhsi[3]}\n')
            opt_file.write(info_a + '\n')
            opt_file.write(info_b + '\n')
            opt_file.write(info_c + '\n')

        return flag_best_fhsi[2]