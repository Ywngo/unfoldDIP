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
# 1. �������� (���� INR ������ coords �� mlp)
# =========================================================================
class PSF_down():
    def __call__(self, input_tensor, psf, ratio):
        _, C, _, _ = input_tensor.shape
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1)
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio), groups=C)
        return output_tensor


class SRF_down():
    def __call__(self, input_tensor, srf):
        output_tensor = fun.conv2d(input_tensor, srf, None)
        return output_tensor


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
# 2. ������������ (�� CNN)
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
    """ U-Net ������������������������������ out_ch """

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

        deepest_feat = self.attn(e_list[-1])
        x = deepest_feat

        for i in range(self.depth):
            skip = e_list[self.depth - 1 - i]
            up_x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = self.decs[i](torch.cat([skip, up_x], dim=1))

        return x


class Spatial_Abundance_Net(nn.Module):
    """ ������������������������������ MLP """

    def __init__(self, args, in_channels):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)
        msi_base_dim = getattr(args, 'msi_base_dim', 64)
        msi_depth = getattr(args, 'msi_depth', 3)

        # U-Net �������� R ��������������
        self.unet = SimpleUNet_Light_Attn(in_channels, self.R, base_dim=msi_base_dim, depth=msi_depth)

        # �������������������� MLP ������
        self.refine = nn.Sequential(
            nn.Conv2d(self.R, self.R, 3, 1, 1),
            nn.BatchNorm2d(self.R),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.R, self.R, 3, 1, 1)
        )

    def forward(self, img_input):
        U = self.unet(img_input)
        U = self.refine(U)
        U = torch.tanh(U)  # (B, R, H, W)
        return U


# =========================================================================
# 3. ������������ (�� CNN)
# =========================================================================
class Spectral_ResDCT_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

        self.freq_attn = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.freq_gate = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, L = x.shape
        device = x.device

        identity = x
        out_conv = self.conv1(x)
        out_conv = self.bn1(out_conv)
        out_conv = self.relu(out_conv)
        out_conv = self.conv2(out_conv)
        out_conv = self.bn2(out_conv)

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


class Spectral_Endmember_Net(nn.Module):
    """ �������������������������������������� """

    def __init__(self, args):
        super().__init__()
        hsi_base_dim = getattr(args, 'hsi_base_dim', 32)
        hsi_depth = getattr(args, 'hsi_depth', 2)
        self.R = getattr(args, 'num_endmembers', 32)

        self.head = nn.Sequential(
            nn.Conv1d(16, hsi_base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hsi_base_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.blocks = nn.ModuleList([Spectral_ResDCT_Block(hsi_base_dim) for _ in range(hsi_depth)])
        self.tail = nn.Conv1d(hsi_base_dim, self.R, kernel_size=1)

    def forward(self, img_input):
        B, L_in, H, W = img_input.shape

        regional_spectra = F.adaptive_avg_pool2d(img_input, (4, 4))
        x = regional_spectra.view(B, L_in, 16).permute(0, 2, 1)  # (B, 16, L_in)

        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        V = self.tail(x)  # (B, R, L_in)
        V = torch.tanh(V)

        return V.transpose(1, 2)  # ������ (B, L_in, R)


# =========================================================================
# 4. ����������
# =========================================================================
class LinearUnmixingProjection(nn.Module):
    """ Z = V @ U^T """

    def __init__(self, args, in_channels_spatial):
        super().__init__()
        self.R = getattr(args, 'num_endmembers', 32)
        self.spatial_net = Spatial_Abundance_Net(args, in_channels_spatial)
        self.spectral_net = Spectral_Endmember_Net(args)

    def forward(self, spatial_input, spectral_input):
        B, _, H, W = spatial_input.shape

        # U ��������: (B, R, H, W) -> ������ (B, R, H*W)
        U = self.spatial_net(spatial_input)
        U_flat = U.view(B, self.R, -1)

        # V ��������: (B, L_in, R)
        V = self.spectral_net(spectral_input)

        # ��������: (B, L_in, R) @ (B, R, H*W) -> (B, L_in, H*W)
        Z_flat = torch.matmul(V, U_flat) / math.sqrt(self.R)

        # �������������� (B, L_in, H, W)
        Z_img = Z_flat.view(B, V.shape[1], H, W)

        E_out = V[0]
        A_out = U[0]

        return Z_img, E_out, A_out


# =========================================================================
# 5. �������������� (����������������)
# =========================================================================
class End2EndUnmixingFusion(nn.Module):
    def __init__(self, args, L_bands, C_msi):
        super().__init__()
        in_channels_fused = L_bands + C_msi
        self.projector = LinearUnmixingProjection(args, in_channels_spatial=in_channels_fused)

    def forward(self, Y_hsi, Y_msi):
        B, L_in, H_lr, W_lr = Y_hsi.shape
        B, C_m, H_hr, W_hr = Y_msi.shape

        # 1. ��������������
        up_hsi_for_feat = F.interpolate(Y_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)

        # 2. ��������
        fused_spatial_input = torch.cat([up_hsi_for_feat, Y_msi], dim=1)  # (B, L_in + C_m, H_hr, W_hr)

        # 3. �������� (������)
        residual, E_out, A_out = self.projector(fused_spatial_input, up_hsi_for_feat)

        # 4. ��������������������
        X_out = up_hsi_for_feat + residual
        X_out = torch.clamp(X_out, 1e-4, 1.0)

        out_list = [X_out]
        fake_grad_norm = 0.0
        last_res_norm = torch.norm(residual).item()

        return out_list, E_out, A_out, fake_grad_norm, last_res_norm


# =========================================================================
# 6. DIP Training Pipeline (��������������������)
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

        # ������ CNN ��������
        self.net = End2EndUnmixingFusion(args=self.args, L_bands=L_bands, C_msi=C_msi).to(args.device)

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
        cos = dot / (norm_Sh * norm_Sm)
        cos = torch.clamp(cos, -0.9999, 0.9999)
        angles = torch.acos(cos)
        return (angles.mean() / math.pi)

    def train(self):
        flag_best_fhsi = [10, 0, 'data', 0]

        L1Loss = nn.L1Loss(reduction='mean')
        RelLoss = RelativeL1Loss(eps=1e-3).to(self.args.device)

        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
            self.optimizer.zero_grad()

            # �������� (�������� target_H, target_W ������)
            out_list, self.endmember, self.abundance, last_grad_norm, last_res_norm = self.net(
                self.lr_hsi, self.hr_msi
            )

            self.hr_hsi_rec = out_list[-1]

            # =========================================================================
            # Loss ���� (����������������������������������������������������������)
            # =========================================================================
            X_obs = self.hr_hsi_rec

            hr_msi_pred = self.srf_down(X_obs, self.srf_est)
            lr_hsi_pred = self.psf_down(X_obs, self.psf_est, self.args.scale_factor)

            loss_hsi = L1Loss(self.lr_hsi, lr_hsi_pred) + 1.5 * RelLoss(lr_hsi_pred, self.lr_hsi)
            loss_msi = L1Loss(self.hr_msi, hr_msi_pred) + 1.5 * RelLoss(hr_msi_pred, self.hr_msi)

            loss_total = loss_hsi + loss_msi
            loss_total += getattr(self.args, 'sam_weight', 0.01) * self.angle_similarity_loss(lr_hsi_pred, self.lr_hsi)

            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            # =========================================================================
            # �������������� (����������������)
            # =========================================================================
            if epoch % 50 == 0:
                with torch.no_grad():
                    print('\n==================================================')
                    print(f'epoch:{epoch} lr:{self.optimizer.param_groups[0]["lr"]:.6f}')

                    gt_tensor = torch.tensor(self.gt.transpose(2, 0, 1)).unsqueeze(0).to(self.args.device).float()

                    X_eval = self.hr_hsi_rec
                    if X_eval.shape[2:] != gt_tensor.shape[2:]:
                        X_eval = F.interpolate(X_eval, size=gt_tensor.shape[2:], mode='bilinear')
                    if X_eval.shape[1] != gt_tensor.shape[1]:
                        X_eval = F.interpolate(X_eval.unsqueeze(1),
                                               size=(gt_tensor.shape[1], gt_tensor.shape[2], gt_tensor.shape[3]),
                                               mode='trilinear').squeeze(1)

                    spatial_rmse = torch.sqrt(torch.mean((X_eval - gt_tensor) ** 2, dim=1))
                    max_spatial_err = spatial_rmse.max().item()
                    mean_spatial_err = spatial_rmse.mean().item()
                    print(
                        f'?? [Probe-Spatial] Max RMSE: {max_spatial_err:.5f} | Mean RMSE: {mean_spatial_err:.5f} | Ratio: {max_spatial_err / (mean_spatial_err + 1e-8):.1f}x')

                    X_stage_np = X_eval.data.cpu().numpy()[0].transpose(1, 2, 0)
                    _, stage_psnr, _, _, _, _, _ = MetricsCal(self.gt, X_stage_np, self.args.scale_factor)
                    print(f'?? [Probe-Ablation] End-to-End PSNR: {stage_psnr:.2f} (Pure CNN)')

                    abun_min = self.abundance.min().item()
                    abun_max = self.abundance.max().item()
                    abun_sum_mean = torch.sum(self.abundance, dim=1).mean().item()
                    print(
                        f'?? [Probe-Latent] Abundance Range: [{abun_min:.2f}, {abun_max:.2f}] | Mean Channel Sum: {abun_sum_mean:.2f}')

                    rec_np = X_eval.data.cpu().numpy()[0]
                    gt_np = self.gt.transpose(2, 0, 1)
                    L_bands = rec_np.shape[0]
                    band_ergas_list = []
                    for b in range(L_bands):
                        mse_b = np.mean((rec_np[b] - gt_np[b]) ** 2)
                        rmse_b = np.sqrt(mse_b)
                        mu_b = np.mean(gt_np[b])
                        if mu_b > 1e-4:
                            band_ergas_list.append(rmse_b / mu_b)
                        else:
                            band_ergas_list.append(0.0)

                    worst_band_idx = np.argmax(band_ergas_list)
                    worst_band_val = band_ergas_list[worst_band_idx]
                    gt_worst_mean = np.mean(gt_np[worst_band_idx])

                    print(
                        f'?? [Diagnosis] Worst ERGAS Band: {worst_band_idx} | Relative Err: {worst_band_val:.4f} | GT Mean of this band: {gt_worst_mean:.5f}')
                    print('--------------------------------------------------')

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
                        info_a = info1
                        info_b = info2
                        info_c = info3

        # ��������
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