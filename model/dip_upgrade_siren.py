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
# 1. ��������������������
# =========================================================================
def make_coord_2d(h, w, device=None):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    coord = torch.stack([xx, yy], dim=-1).view(-1, 2)
    return coord


def make_coord_1d(l, device=None):
    return torch.linspace(-1, 1, steps=l, device=device).view(-1, 1)


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


# =========================================================================
# [����] Edge Loss (��������) & SIREN ������ (������������)
# =========================================================================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        k_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('weight_x', k_x)
        self.register_buffer('weight_y', k_y)

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        weight_x = self.weight_x.repeat(C, 1, 1, 1)
        weight_y = self.weight_y.repeat(C, 1, 1, 1)

        pred_grad_x = F.conv2d(pred, weight_x, padding=1, groups=C)
        pred_grad_y = F.conv2d(pred, weight_y, padding=1, groups=C)
        target_grad_x = F.conv2d(target, weight_x, padding=1, groups=C)
        target_grad_y = F.conv2d(target, weight_y, padding=1, groups=C)

        loss_grad = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        return loss_grad


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLinear(nn.Module):
    """ ���������������������������� """

    def __init__(self, in_features, out_features, is_first=False, w0=30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.w0 = w0
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.w0,
                                            np.sqrt(6 / self.linear.in_features) / self.w0)

    def forward(self, x):
        return self.activation(self.linear(x))


# =========================================================================
# 2. ������������ (�� SIREN)
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
    def __init__(self, in_ch, out_ch, base_dim=16):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(in_ch, base_dim, 1, 1, 0), nn.BatchNorm2d(base_dim),
                                        nn.LeakyReLU(0.2, True))
        self.enc1 = nn.Sequential(nn.Conv2d(base_dim, base_dim, 3, 1, 1), nn.BatchNorm2d(base_dim),
                                  nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base_dim, base_dim * 2, 3, 1, 1),
                                  nn.BatchNorm2d(base_dim * 2), nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base_dim * 2, base_dim * 4, 3, 1, 1),
                                  nn.BatchNorm2d(base_dim * 4), nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base_dim * 4, base_dim * 8, 3, 1, 1),
                                  nn.BatchNorm2d(base_dim * 8), nn.LeakyReLU(0.2, True))

        self.attn = SpatialAttention()

        self.dec3 = nn.Sequential(nn.Conv2d(base_dim * 12, base_dim * 4, 3, 1, 1), nn.BatchNorm2d(base_dim * 4),
                                  nn.LeakyReLU(0.2, True))
        self.dec2 = nn.Sequential(nn.Conv2d(base_dim * 6, base_dim * 2, 3, 1, 1), nn.BatchNorm2d(base_dim * 2),
                                  nn.LeakyReLU(0.2, True))
        self.dec1 = nn.Sequential(nn.Conv2d(base_dim * 3, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch),
                                  nn.LeakyReLU(0.2, True))

    def forward(self, x):
        x_base = self.bottleneck(x)
        e1 = self.enc1(x_base)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.attn(self.enc4(e3))

        up_e4 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([e3, up_e4], dim=1))
        up_d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([e2, up_d3], dim=1))
        up_d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([e1, up_d2], dim=1))
        return d1


class Spatial_LIIF_Tucker_Light(nn.Module):
    def __init__(self, in_channels, Rs, feat_dim=32):
        super().__init__()
        self.unet = SimpleUNet_Light_Attn(in_channels, feat_dim, base_dim=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(nn.Linear(16 * 8, 64), nn.LeakyReLU(0.2, True))

        # [����] ������ SIREN ��������
        self.mlp = nn.Sequential(
            SirenLinear(feat_dim + 2, 64, is_first=True, w0=30.0),
            SirenLinear(64, 64, is_first=False, w0=30.0),
            nn.Linear(64, Rs)  # ������������������������
        )

    def forward(self, img_input, coords_2d):
        feat_map = self.unet(img_input)
        x_enc4 = self.unet.enc4(self.unet.enc3(self.unet.enc2(self.unet.enc1(self.unet.bottleneck(img_input)))))
        spatial_global_feat = self.global_pool(x_enc4).flatten(1)
        spatial_global_feat = self.global_proj(spatial_global_feat)

        local_feat = feat_map.flatten(2).permute(0, 2, 1)
        mlp_in = torch.cat([local_feat, coords_2d], dim=-1)
        U = self.mlp(mlp_in)
        return U, spatial_global_feat


# =========================================================================
# 3. ������������ (�� SIREN)
# =========================================================================
class DCI_1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.freq_attn = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels),
                                       nn.LeakyReLU(0.2, True), nn.Conv1d(channels, channels, 1, bias=False),
                                       nn.Sigmoid())
        self.freq_gate = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, L = x.shape
        device = x.device
        n = torch.arange(L, dtype=torch.float32, device=device)
        k = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)
        dct_m = torch.cos(math.pi / L * (n + 0.5) * k)
        dct_m[0] *= 1 / math.sqrt(2)
        dct_m *= math.sqrt(2 / L)

        x_freq = torch.matmul(x, dct_m.t())

        cutoff = L // 2
        mask_L = torch.zeros(L, device=device)
        mask_L[:cutoff] = 1.0
        mask_H = 1.0 - mask_L

        freq_L = x_freq * mask_L.view(1, 1, L)
        freq_H = x_freq * mask_H.view(1, 1, L)

        freq_H_enhanced = freq_H * self.freq_attn(freq_H) * self.freq_gate
        freq_combined = freq_L + freq_H_enhanced
        out = torch.matmul(freq_combined, dct_m)
        return out + x


class Spectral_DCI_Extractor(nn.Module):
    def __init__(self, in_ch=16, out_ch=32, base_dim=32):
        super().__init__()
        self.head = nn.Sequential(nn.Conv1d(in_ch, base_dim, 3, padding=1, bias=False), nn.BatchNorm1d(base_dim),
                                  nn.LeakyReLU(0.2, True))
        self.dci_1 = DCI_1D(base_dim)
        self.conv_1 = nn.Sequential(nn.Conv1d(base_dim, base_dim, 3, padding=1, bias=False), nn.BatchNorm1d(base_dim),
                                    nn.LeakyReLU(0.2, True))
        self.dci_2 = DCI_1D(base_dim)
        self.conv_2 = nn.Sequential(nn.Conv1d(base_dim, base_dim, 3, padding=1, bias=False), nn.BatchNorm1d(base_dim),
                                    nn.LeakyReLU(0.2, True))
        self.tail = nn.Conv1d(base_dim, out_ch, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Sequential(nn.Linear(out_ch, 64), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        f_base = self.head(x)
        f_dci = self.dci_1(f_base)
        f_conv = self.conv_1(f_dci)
        f_dci = self.dci_2(f_conv)
        f_conv = self.conv_2(f_dci)
        out_seq = self.tail(f_conv)
        global_feat = self.global_pool(out_seq).flatten(1)
        spectral_global_feat = self.global_proj(global_feat)
        return out_seq, spectral_global_feat


class Spectral_INR_Tucker_Adaptive(nn.Module):
    def __init__(self, in_channels, Rl, feat_dim=32):
        super().__init__()
        self.spectral_extractor = Spectral_DCI_Extractor(in_ch=16, out_ch=feat_dim, base_dim=32)

        # [����] ������ SIREN ��������
        self.mlp = nn.Sequential(
            SirenLinear(feat_dim + 1, 64, is_first=True, w0=30.0),
            SirenLinear(64, 64, is_first=False, w0=30.0),
            nn.Linear(64, Rl)
        )

    def forward(self, coords_1d, img_input):
        B, L_in, H, W = img_input.shape
        L_coords = coords_1d.shape[0]

        regional_spectra = F.adaptive_avg_pool2d(img_input, (4, 4))
        spectral_signal = regional_spectra.view(B, L_in, 16).permute(0, 2, 1)

        feat_map_1d, spectral_global_feat = self.spectral_extractor(spectral_signal)
        local_feat = feat_map_1d.permute(0, 2, 1)

        coords_expand = coords_1d.unsqueeze(0).expand(B, L_coords, -1)
        mlp_in = torch.cat([local_feat, coords_expand], dim=-1)
        V = self.mlp(mlp_in)
        return V, spectral_global_feat


# =========================================================================
# 4. Core Tensor & 5. �������� & 6. DIP ��������
# (������������������)
# =========================================================================
class CoreMatrixGenerator(nn.Module):
    def __init__(self, Rl, Rs, feat_dim=64):
        super().__init__()
        self.Rl = Rl
        self.Rs = Rs
        self.spatial_gate = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Sigmoid())
        self.spectral_gate = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2, True),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2, True),
            nn.Linear(128, Rl * Rs)
        )

    def forward(self, spatial_feat, spectral_feat):
        B = spatial_feat.shape[0]
        w_spatial = self.spatial_gate(spatial_feat)
        w_spectral = self.spectral_gate(spectral_feat)
        inter_spatial = spatial_feat * w_spectral
        inter_spectral = spectral_feat * w_spatial
        joint_feat = torch.cat([inter_spatial, inter_spectral], dim=-1)
        core_flat = self.fusion_mlp(joint_feat)
        return core_flat.view(B, self.Rl, self.Rs)


class ContinuousTuckerProjection(nn.Module):
    def __init__(self, in_channels_unet, Rl=15, Rs=64):
        super().__init__()
        self.Rl = Rl
        self.Rs = Rs
        self.spatial_net = Spatial_LIIF_Tucker_Light(in_channels_unet, Rs=Rs, feat_dim=32)
        self.spectral_net = Spectral_INR_Tucker_Adaptive(in_channels=in_channels_unet, Rl=Rl)
        self.core_generator = CoreMatrixGenerator(Rl, Rs, feat_dim=64)

    def forward(self, S_k, coords_2d, coords_1d, target_shape):
        B, L, H, W = target_shape
        U, spatial_global_feat = self.spatial_net(S_k, coords_2d)
        V, spectral_global_feat = self.spectral_net(coords_1d, S_k)
        core_G = self.core_generator(spatial_global_feat, spectral_global_feat)
        scale_factor = math.sqrt(self.Rl * self.Rs)
        VG = torch.matmul(V, core_G)
        Z_flat = torch.matmul(VG, U.transpose(1, 2)) / scale_factor
        Z_img = Z_flat.view(B, L, H, W)
        E_out = V[0]
        A_out = torch.matmul(core_G, U.transpose(1, 2)).view(B, self.Rl, H, W)[0]
        return Z_img, E_out, A_out


class UnrolledTuckerFusion(nn.Module):
    def __init__(self, L_bands, K_iters=5, Rl=15, Rs=64):
        super().__init__()
        self.K = K_iters
        self.eta = nn.Parameter(torch.tensor([0.1] * K_iters))
        self.res_alpha = nn.Parameter(torch.tensor([0.1] * K_iters))
        self.projectors = nn.ModuleList([ContinuousTuckerProjection(L_bands, Rl, Rs) for _ in range(self.K)])

    def forward(self, Y_hsi, Y_msi, psf_est, srf_est, scale_factor):
        B, L, H_lr, W_lr = Y_hsi.shape
        B, C_m, H_hr, W_hr = Y_msi.shape
        device = Y_hsi.device
        coords_2d = make_coord_2d(H_hr, W_hr, device).unsqueeze(0).expand(B, -1, -1)
        coords_1d = make_coord_1d(L, device)
        X_k = F.interpolate(Y_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        psf_grouped = psf_est.repeat(L, 1, 1, 1) if psf_est.shape[0] == 1 else psf_est
        out_list = []
        soft_eta = torch.sigmoid(self.eta) + 1e-4
        soft_alpha = torch.sigmoid(self.res_alpha) + 1e-4

        for k in range(self.K):
            diff_hsi = F.conv2d(X_k, psf_grouped, stride=scale_factor, groups=L) - Y_hsi
            grad_hsi = F.conv_transpose2d(diff_hsi, psf_grouped, stride=scale_factor, groups=L)
            if grad_hsi.shape != X_k.shape:
                grad_hsi = F.interpolate(grad_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
            diff_msi = F.conv2d(X_k, srf_est) - Y_msi
            grad_msi = F.conv_transpose2d(diff_msi, srf_est)
            S_k = X_k - soft_eta[k] * (grad_hsi + grad_msi)
            S_k = torch.clamp(S_k, -0.05, 1.05)
            residual, E_k, A_k = self.projectors[k](S_k, coords_2d, coords_1d, target_shape=(B, L, H_hr, W_hr))
            X_k = S_k + soft_alpha[k] * residual
            X_k = torch.clamp(X_k, -0.05, 1.05)
            out_list.append(X_k)
        return out_list, E_k, A_k


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
        self.net = UnrolledTuckerFusion(L_bands=L_bands, K_iters=self.args.K_iters, Rl=self.args.Rl,
                                        Rs=self.args.Rs).to(args.device)

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
        edge_criterion = EdgeLoss().to(self.args.device)

        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
            self.optimizer.zero_grad()
            out_list, self.endmember, self.abundance = self.net(self.lr_hsi, self.hr_msi, self.psf_est, self.srf_est,
                                                                self.args.scale_factor)
            self.hr_hsi_rec = out_list[-1]
            loss_total = 0
            gamma = self.args.gamma

            for i, X_pred in enumerate(out_list):
                weight = gamma ** (self.net.K - 1 - i)
                hr_msi_pred = self.srf_down(X_pred, self.srf_est)
                lr_hsi_pred = self.psf_down(X_pred, self.psf_est, self.args.scale_factor)
                loss_step = L1Loss(self.lr_hsi, lr_hsi_pred) + L1Loss(self.hr_msi, hr_msi_pred)
                loss_step += self.args.sam_weight * self.angle_similarity_loss(lr_hsi_pred, self.lr_hsi)
                loss_step += 0.05 * edge_criterion(hr_msi_pred, self.hr_msi)
                loss_total += weight * loss_step

            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 == 0:
                with torch.no_grad():
                    print('epoch:{} lr:{}'.format(epoch, self.optimizer.param_groups[0]['lr']))
                    print('************')
                    hr_msi_frec = self.srf_down(self.hr_hsi_rec, self.srf_est)
                    lr_hsi_frec = self.psf_down(self.hr_hsi_rec, self.psf_est, self.args.scale_factor)
                    hr_msi_numpy = self.hr_msi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_msi_frec_numpy = hr_msi_frec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_numpy = self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_hsi_frec_numpy = lr_hsi_frec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    hr_hsi_rec_numpy = self.hr_hsi_rec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(lr_hsi_numpy, lr_hsi_frec_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(lr_hsi_numpy - lr_hsi_frec_numpy))
                    information1 = "lr_hsi vs pred\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information1)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(hr_msi_numpy, hr_msi_frec_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(hr_msi_numpy - hr_msi_frec_numpy))
                    information2 = "hr_msi vs pred\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information2)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.gt, hr_hsi_rec_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(self.gt - hr_hsi_rec_numpy))
                    information3 = "hr_hsi vs gt\n L1 {:.4f} sam {:.4f},psnr {:.4f},ergas {:.4f},cc {:.4f},rmse {:.4f},Ssim {:.4f},Uqi {:.4f}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information3)
                    print('--------------------------------')

                    file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
                    with open(file_name, 'a') as opt_file:
                        opt_file.write(f'--- epoch:{epoch} ---\n')
                        opt_file.write(information1 + '\n')
                        opt_file.write(information2 + '\n')
                        opt_file.write(information3 + '\n')

                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:
                        flag_best_fhsi[0] = sam
                        flag_best_fhsi[1] = psnr
                        flag_best_fhsi[2] = self.hr_hsi_rec
                        flag_best_fhsi[3] = epoch
                        information_a = information1
                        information_b = information2
                        information_c = information3

        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'),
                         {'Out': flag_best_fhsi[2].data.cpu().numpy()[0].transpose(1, 2, 0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Endmember.mat'), {'end': self.endmember.data.cpu().numpy()})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Abundance.mat'), {'abun': self.abundance.data.cpu().numpy()})

        file_name = os.path.join(self.args.expr_dir, 'Stage3.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('========================== BEST ==========================\n')
            opt_file.write(f'epoch_fhsi_best:{flag_best_fhsi[3]}\n')
            opt_file.write(information_a + '\n')
            opt_file.write(information_b + '\n')
            opt_file.write(information_c + '\n')

        return flag_best_fhsi[2]