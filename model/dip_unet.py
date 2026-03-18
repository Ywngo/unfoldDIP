# -*- coding: utf-8 -*-
"""
dip_unet.py - U-Net + Coord-Aware INR (DIP-Compatible)
              V3: ���������������� + SAM��������
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy.io
import torch.nn.functional as F
from .evaluation import MetricsCal


# =========================================================================
# 1. ��������
# =========================================================================
class PSF_down:
    def __call__(self, input_tensor, psf, ratio):
        _, C, _, _ = input_tensor.shape
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1)
        return F.conv2d(input_tensor, psf, None, (ratio, ratio), groups=C)


class SRF_down:
    def __call__(self, input_tensor, srf):
        return F.conv2d(input_tensor, srf, None)


def spectral_interpolate(x, target_L):
    B, L_in, H, W = x.shape
    if L_in == target_L:
        return x
    x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, L_in)
    x_interp = F.interpolate(x_flat, size=target_L, mode='linear', align_corners=False)
    return x_interp.reshape(B, H, W, target_L).permute(0, 3, 1, 2)


# =========================================================================
# 2. ��������
# =========================================================================
class SAMLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_norm = torch.norm(pred, dim=1)
        if pred_norm.max() < self.eps:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        dot_product = torch.sum(pred * target, dim=1)
        norm_target = torch.norm(target, dim=1)
        cos_theta = dot_product / (pred_norm * norm_target + self.eps)
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        return torch.mean(torch.acos(cos_theta))


# =========================================================================
# 3. ��������
# =========================================================================
def make_coord_2d(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)


def make_coord_1d(L, device):
    coord = torch.linspace(-1, 1, L, device=device)
    return coord.reshape(1, 1, -1)


# =========================================================================
# 4. 2D ��������
# =========================================================================
class ResConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class AttentionGate2D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        return x * self.psi(F.gelu(g1 + x1))


# =========================================================================
# 5. 1D ��������
# =========================================================================
class ResConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class AttentionGate1D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv1d(F_g, F_int, 1, bias=False), nn.BatchNorm1d(F_int))
        self.W_x = nn.Sequential(nn.Conv1d(F_l, F_int, 1, bias=False), nn.BatchNorm1d(F_int))
        self.psi = nn.Sequential(nn.Conv1d(F_int, 1, 1, bias=False), nn.BatchNorm1d(1), nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='linear', align_corners=False)
        return x * self.psi(F.gelu(g1 + x1))


# =========================================================================
# 6. UNet2D ��������
# =========================================================================
class UNet2D_Abundance(nn.Module):
    def __init__(self, in_ch, K=30, base_ch=64, dropout=0.1):
        super().__init__()
        self.K = K
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.inc = ResConv2D(in_ch + 2, c1, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResConv2D(c1, c2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResConv2D(c2, c3, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResConv2D(c3, c4, dropout))

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.ag3 = AttentionGate2D(F_g=c3, F_l=c3, F_int=c3 // 2)
        self.conv_up3 = ResConv2D(c3 * 2, c3, dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.ag2 = AttentionGate2D(F_g=c2, F_l=c2, F_int=c2 // 2)
        self.conv_up2 = ResConv2D(c2 * 2, c2, dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.ag1 = AttentionGate2D(F_g=c1, F_l=c1, F_int=c1 // 2)
        self.conv_up1 = ResConv2D(c1 * 2, c1, dropout)

        self.inr_head = nn.Sequential(
            nn.Conv2d(c1 + 2, c1, 1), nn.GELU(),
            nn.Conv2d(c1, K, 1),
        )

    def forward(self, x, target_h, target_w):
        B, device = x.shape[0], x.device
        coord = make_coord_2d(x.shape[2], x.shape[3], device).expand(B, -1, -1, -1)
        x_in = torch.cat([x, coord], dim=1)

        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u3 = self.up3(x4)
        if u3.shape[2:] != x3.shape[2:]:
            u3 = F.interpolate(u3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.conv_up3(torch.cat([self.ag3(u3, x3), u3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[2:] != x2.shape[2:]:
            u2 = F.interpolate(u2, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.conv_up2(torch.cat([self.ag2(u2, x2), u2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[2:] != x1.shape[2:]:
            u1 = F.interpolate(u1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.conv_up1(torch.cat([self.ag1(u1, x1), u1], dim=1))

        if d1.shape[2:] != (target_h, target_w):
            d1 = F.interpolate(d1, size=(target_h, target_w), mode='bilinear', align_corners=False)

        coord_out = make_coord_2d(target_h, target_w, device).expand(B, -1, -1, -1)
        logits = self.inr_head(torch.cat([d1, coord_out], dim=1))
        self._logits = logits.detach()
        A = F.softmax(logits, dim=1)
        return A


# =========================================================================
# 7. UNet1D �������� �� �������� + ��������
# =========================================================================
class UNet1D_Endmember(nn.Module):
    """
    v3 ����:
      1. ��������������������������������������
         E = anchor + network_output �� ����������������������
      2. ���� softplus + max����������������������
         ���������� clamp(0, 1)
      3. ����dropout �� ������������������������������������
    """
    def __init__(self, in_ch, K=30, base_ch=64, dropout=0.02):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.inc = ResConv1D(in_ch + 1, c1, dropout)
        self.down1_pool = nn.Conv1d(c1, c1, 2, stride=2, bias=False)
        self.down1_conv = ResConv1D(c1, c2, dropout)
        self.down2_pool = nn.Conv1d(c2, c2, 2, stride=2, bias=False)
        self.down2_conv = ResConv1D(c2, c3, dropout)

        self.up2 = nn.ConvTranspose1d(c3, c2, 2, stride=2)
        self.ag2 = AttentionGate1D(F_g=c2, F_l=c2, F_int=c2 // 2)
        self.conv_up2 = ResConv1D(c2 * 2, c2, dropout)

        self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)
        self.ag1 = AttentionGate1D(F_g=c1, F_l=c1, F_int=c1 // 2)
        self.conv_up1 = ResConv1D(c1 * 2, c1, dropout)

        # ����������������
        self.inr_head = nn.Sequential(
            nn.Conv1d(c1 + 1, c1, 1), nn.GELU(),
            nn.Conv1d(c1, K, 1),
        )
        self._init_head()

        # �������������� �� �� init_from_data() ������
        self.E_anchor = None  # ���� init_from_data ������

    def _init_head(self):
        """�������� �� �������������� �� ���� �� anchor"""
        for m in self.inr_head:
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_from_data(self, lr_hsi, K):
        """
        �� lr_hsi ������������������
        lr_hsi: (1, L, H, W)
        """
        with torch.no_grad():
            B, L, H, W = lr_hsi.shape
            pixels = lr_hsi.reshape(B, L, -1)  # (1, L, N)

            # �������������� + �������������� K ����������
            N = pixels.shape[2]
            indices = torch.linspace(0, N - 1, K).long()
            sampled = pixels[0, :, indices]  # (L, K)

            # ����������������������������������������
            global_mean = pixels[0].mean(dim=1, keepdim=True).expand_as(sampled)  # (L, K)
            anchor = 0.7 * sampled + 0.3 * global_mean  # (L, K)

            # ����������������
            self.E_anchor = nn.Parameter(anchor.unsqueeze(0).clone())  # (1, L, K)
            print(f"[EndmemberInit] anchor range: {anchor.min():.4f} ~ {anchor.max():.4f}, "
                  f"mean: {anchor.mean():.4f}")

    def forward(self, x, target_L):
        B, device = x.shape[0], x.device

        coord_in = make_coord_1d(x.shape[2], device).expand(B, -1, -1)
        x_in = torch.cat([x, coord_in], dim=1)

        x1 = self.inc(x_in)
        x2 = self.down1_conv(self.down1_pool(x1))
        x3 = self.down2_conv(self.down2_pool(x2))

        u2 = self.up2(x3)
        if u2.shape[2] != x2.shape[2]:
            u2 = F.interpolate(u2, size=x2.shape[2:], mode='linear', align_corners=False)
        d2 = self.conv_up2(torch.cat([self.ag2(u2, x2), u2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[2] != x1.shape[2]:
            u1 = F.interpolate(u1, size=x1.shape[2:], mode='linear', align_corners=False)
        d1 = self.conv_up1(torch.cat([self.ag1(u1, x1), u1], dim=1))

        if d1.shape[2] != target_L:
            d1 = F.interpolate(d1, size=target_L, mode='linear', align_corners=False)

        coord_out = make_coord_1d(target_L, device).expand(B, -1, -1)
        residual = self.inr_head(torch.cat([d1, coord_out], dim=1))  # (B, K, L)
        self._pre_act = residual.detach()

        # ���� + ����
        anchor = self.E_anchor  # (1, L, K)
        if anchor.shape[1] != target_L:
            anchor = F.interpolate(
                anchor.permute(0, 2, 1), size=target_L, mode='linear', align_corners=False
            ).permute(0, 2, 1)

        # residual: (B, K, L) �� (B, L, K)
        E = anchor + residual.permute(0, 2, 1) * 0.1  # ��������������������������
        E = torch.clamp(E, 0.0, 1.0)

        return E  # (B, L, K)


# =========================================================================
# 8. ������������
# =========================================================================
class UNetINRFusion(nn.Module):
    def __init__(self, args, L_in, C_m):
        super().__init__()
        self.K = getattr(args, 'K_endmembers', 30)
        self.grid_size = getattr(args, 'endmember_grid_size', 8)

        self.spatial_net = UNet2D_Abundance(
            in_ch=L_in + C_m, K=self.K,
            base_ch=getattr(args, 'unet2d_base_ch', 64),
            dropout=getattr(args, 'unet_dropout', 0.1),
        )
        self.spectral_net = UNet1D_Endmember(
            in_ch=self.grid_size ** 2, K=self.K,
            base_ch=getattr(args, 'unet1d_base_ch', 64),
            dropout=0.02,
        )

    def init_endmember_from_data(self, lr_hsi):
        """��������������������������������"""
        self.spectral_net.init_from_data(lr_hsi, self.K)
        # ������init_from_data ���������� nn.Parameter��
        # �������������������� device
        device = next(self.parameters()).device
        self.spectral_net.E_anchor = nn.Parameter(
            self.spectral_net.E_anchor.data.to(device)
        )

    def forward(self, lr_hsi, hr_msi, target_h=None, target_w=None, target_L=None):
        B, L_in, H_lr, W_lr = lr_hsi.shape
        _, C_m, H_hr, W_hr = hr_msi.shape
        target_h = target_h or H_hr
        target_w = target_w or W_hr
        target_L = target_L or L_in

        lr_up = F.interpolate(lr_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        A = self.spatial_net(torch.cat([lr_up, hr_msi], dim=1), target_h, target_w)

        gs = self.grid_size
        reg_spectra = F.adaptive_avg_pool2d(lr_hsi, (gs, gs))
        reg_spectra = reg_spectra.view(B, L_in, gs * gs).permute(0, 2, 1)
        E = self.spectral_net(reg_spectra, target_L)  # (B, L, K)

        A_flat = A.reshape(B, self.K, -1)
        Z_flat = torch.matmul(E, A_flat)
        Z_img = Z_flat.reshape(B, target_L, target_h, target_w)

        return Z_img, E[0], A[0]


# =========================================================================
# 9. ����������
# =========================================================================
class TrainingDiagnostics:
    def __init__(self, K=30):
        self.K = K
        self.history = {
            'loss': [], 'loss_hsi': [], 'loss_msi': [], 'loss_sam': [],
            'grad_norm_spatial': [], 'grad_norm_spectral': [],
            'grad_norm_spatial_head': [], 'grad_norm_spectral_head': [],
            'grad_norm_anchor': [],
            'E_mean': [], 'E_std': [], 'E_zeros_ratio': [], 'E_ones_ratio': [],
            'A_entropy': [], 'A_max': [], 'A_effective_K': [],
            'E_pre_act_mean': [], 'E_pre_act_std': [],
            'E_anchor_mean': [], 'E_anchor_std': [],
            'A_logits_mean': [], 'A_logits_std': [],
            'Z_mean': [], 'Z_std': [], 'Z_vs_input_ratio': [],
        }

    @staticmethod
    def grad_norm(module):
        total = 0.0
        count = 0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
                count += 1
        return total ** 0.5 if count > 0 else 0.0

    @staticmethod
    def param_grad_norm(param):
        if param.grad is not None:
            return param.grad.data.norm(2).item()
        return 0.0

    @staticmethod
    def abundance_entropy(A):
        A_flat = A.reshape(A.shape[0], -1).permute(1, 0)
        log_A = torch.log(A_flat + 1e-10)
        entropy = -torch.sum(A_flat * log_A, dim=1)
        return entropy.mean().item()

    @staticmethod
    def effective_endmembers(A, threshold=0.05):
        A_flat = A.reshape(A.shape[0], -1).permute(1, 0)
        active = (A_flat > threshold).float().sum(dim=1)
        return active.mean().item()

    def collect(self, net, E, A, Z, loss_dict, lr_hsi):
        with torch.no_grad():
            self.history['loss'].append(loss_dict['total'])
            self.history['loss_hsi'].append(loss_dict['hsi'])
            self.history['loss_msi'].append(loss_dict['msi'])
            self.history['loss_sam'].append(loss_dict['sam'])

            self.history['grad_norm_spatial'].append(self.grad_norm(net.spatial_net))
            self.history['grad_norm_spectral'].append(self.grad_norm(net.spectral_net))
            self.history['grad_norm_spatial_head'].append(self.grad_norm(net.spatial_net.inr_head))
            self.history['grad_norm_spectral_head'].append(self.grad_norm(net.spectral_net.inr_head))

            # anchor ����
            if net.spectral_net.E_anchor is not None:
                self.history['grad_norm_anchor'].append(
                    self.param_grad_norm(net.spectral_net.E_anchor))
                self.history['E_anchor_mean'].append(
                    net.spectral_net.E_anchor.data.mean().item())
                self.history['E_anchor_std'].append(
                    net.spectral_net.E_anchor.data.std().item())
            else:
                self.history['grad_norm_anchor'].append(0.0)
                self.history['E_anchor_mean'].append(0.0)
                self.history['E_anchor_std'].append(0.0)

            self.history['E_mean'].append(E.mean().item())
            self.history['E_std'].append(E.std().item())
            self.history['E_zeros_ratio'].append((E < 0.01).float().mean().item())
            self.history['E_ones_ratio'].append((E > 0.99).float().mean().item())

            pre_act = net.spectral_net._pre_act
            self.history['E_pre_act_mean'].append(pre_act.mean().item())
            self.history['E_pre_act_std'].append(pre_act.std().item())

            self.history['A_entropy'].append(self.abundance_entropy(A))
            self.history['A_max'].append(A.max().item())
            self.history['A_effective_K'].append(self.effective_endmembers(A))

            logits = net.spatial_net._logits
            self.history['A_logits_mean'].append(logits.mean().item())
            self.history['A_logits_std'].append(logits.std().item())

            self.history['Z_mean'].append(Z.mean().item())
            self.history['Z_std'].append(Z.std().item())

            # Z ������ lr_hsi ��������
            input_mean = lr_hsi.mean().item()
            z_mean = Z.mean().item()
            self.history['Z_vs_input_ratio'].append(
                z_mean / (input_mean + 1e-8))

    def print_report(self, epoch):
        idx = -1
        K = self.K
        max_entropy = np.log(K)
        print(f"\n{'='*70}")
        print(f"  DIAGNOSTICS @ EPOCH {epoch}")
        print(f"{'='*70}")

        print(f"  [Loss] total: {self.history['loss'][idx]:.6f} | "
              f"hsi: {self.history['loss_hsi'][idx]:.6f} | "
              f"msi: {self.history['loss_msi'][idx]:.6f} | "
              f"sam: {self.history['loss_sam'][idx]:.6f}")

        gs = self.history['grad_norm_spatial'][idx]
        ge = self.history['grad_norm_spectral'][idx]
        gsh = self.history['grad_norm_spatial_head'][idx]
        geh = self.history['grad_norm_spectral_head'][idx]
        ga = self.history['grad_norm_anchor'][idx]
        print(f"  [Grad] spatial_unet: {gs:.6f} | spectral_unet: {ge:.6f} | anchor: {ga:.6f}")
        print(f"         spatial_head: {gsh:.6f} | spectral_head: {geh:.6f}")
        if ge < 1e-6:
            print(f"  ??  ��������������������")

        em = self.history['E_mean'][idx]
        es = self.history['E_std'][idx]
        ez = self.history['E_zeros_ratio'][idx]
        eo = self.history['E_ones_ratio'][idx]
        pam = self.history['E_pre_act_mean'][idx]
        pas = self.history['E_pre_act_std'][idx]
        eam = self.history['E_anchor_mean'][idx]
        eas = self.history['E_anchor_std'][idx]
        print(f"  [Endmember] mean: {em:.4f} | std: {es:.4f} | "
              f"zeros(<0.01): {ez*100:.1f}% | ones(>0.99): {eo*100:.1f}%")
        print(f"              pre_act(residual) mean: {pam:.4f} | std: {pas:.4f}")
        print(f"              anchor mean: {eam:.4f} | std: {eas:.4f}")
        if em < 0.2:
            print(f"  ??  ������������({em:.4f})����������������������")

        ae = self.history['A_entropy'][idx]
        am = self.history['A_max'][idx]
        ak = self.history['A_effective_K'][idx]
        alm = self.history['A_logits_mean'][idx]
        als = self.history['A_logits_std'][idx]
        print(f"  [Abundance] entropy: {ae:.4f}/{max_entropy:.4f} | "
              f"max: {am:.4f} | effective_K: {ak:.1f}/{K}")
        print(f"              logits mean: {alm:.4f} | std: {als:.4f}")

        zm = self.history['Z_mean'][idx]
        zs = self.history['Z_std'][idx]
        zr = self.history['Z_vs_input_ratio'][idx]
        print(f"  [Recon Z] mean: {zm:.4f} | std: {zs:.4f} | Z/input_ratio: {zr:.4f}")
        if zr < 0.5:
            print(f"  ??  ������������������{zr*100:.0f}%��������������")
        if zr > 2.0:
            print(f"  ??  ����������������{zr*100:.0f}%������������")

        if len(self.history['loss']) >= 3:
            losses = self.history['loss'][-3:]
            sams = self.history['loss_sam'][-3:]
            print(f"  [Trend] loss: {[f'{l:.6f}' for l in losses]}")
            print(f"          sam:  {[f'{s:.6f}' for s in sams]}")

        print(f"{'='*70}\n")


# =========================================================================
# 10. DIP ����������
# =========================================================================
class dip:
    def __init__(self, args, psf, srf, blind):
        self.args = args
        self.hr_msi = blind.tensor_hr_msi
        self.lr_hsi = blind.tensor_lr_hsi
        self.gt = blind.gt

        self.psf_est = torch.tensor(
            np.reshape(psf, (1, 1, args.scale_factor, args.scale_factor))
        ).to(args.device).float()
        self.srf_est = torch.tensor(
            np.reshape(srf.T, (srf.shape[1], srf.shape[0], 1, 1))
        ).to(args.device).float()
        self.psf_down = PSF_down()
        self.srf_down = SRF_down()

        self.L_in = self.lr_hsi.shape[1]
        self.net = UNetINRFusion(args, self.L_in, self.hr_msi.shape[1]).to(args.device)

        # �� ������������������
        self.net.init_endmember_from_data(self.lr_hsi)

        self.target_h = int(round(self.hr_msi.shape[2] * getattr(args, 'target_scale', 1.0)))
        self.target_w = int(round(self.hr_msi.shape[3] * getattr(args, 'target_scale', 1.0)))
        self.target_L = getattr(args, 'target_L', None)

        self.L1Loss = nn.L1Loss()
        self.SAMLoss = SAMLoss().to(args.device)

        # ��������������������������
        base_lr = max(args.lr_stage3_dip, 3e-4)
        param_groups = [
            {'params': self.net.spatial_net.parameters(), 'lr': base_lr},
            # spectral UNet ����������
            {'params': [p for n, p in self.net.spectral_net.named_parameters()
                        if 'E_anchor' not in n], 'lr': base_lr * 3},
            # anchor ��������������������������������
            {'params': [self.net.spectral_net.E_anchor], 'lr': base_lr * 0.5},
        ]
        self.optimizer = optim.Adam(param_groups)
        self.scheduler = lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda e: 1.0 - max(0, e + 1 - args.niter3_dip) / float(args.niter_decay3_dip + 1)
        )

        K = getattr(args, 'K_endmembers', 30)
        self.diag = TrainingDiagnostics(K=K)

    def _to_observation_space(self, X):
        if X.shape[2:] != self.hr_msi.shape[2:]:
            X_obs = F.interpolate(X, size=self.hr_msi.shape[2:], mode='bilinear', align_corners=False)
        else:
            X_obs = X
        if X_obs.shape[1] != self.L_in:
            X_obs = spectral_interpolate(X_obs, self.L_in)
        return X_obs

    def _compute_loss_decomposed(self, X):
        X_obs = self._to_observation_space(X)
        hsi_pred = self.psf_down(X_obs, self.psf_est, self.args.scale_factor)
        msi_pred = self.srf_down(X_obs, self.srf_est)

        loss_hsi = self.L1Loss(hsi_pred, self.lr_hsi)
        loss_msi = self.L1Loss(msi_pred, self.hr_msi)
        loss_sam = self.SAMLoss(hsi_pred, self.lr_hsi)

        # ���� SAM ��������������������������
        total = loss_hsi + loss_msi + 0.15 * loss_sam

        return total, {
            'total': total.item(),
            'hsi': loss_hsi.item(),
            'msi': loss_msi.item(),
            'sam': loss_sam.item(),
        }

    def print_and_format_metrics(self, name, pred_np, target_np):
        sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(target_np, pred_np, self.args.scale_factor)
        l1_err = np.mean(np.abs(target_np - pred_np))
        info = (
            f"[{name}] L1: {l1_err:.4f} | SAM: {sam:.4f} | PSNR: {psnr:.4f} | ERGAS: {ergas:.4f} | "
            f"CC: {cc:.4f} | RMSE: {rmse:.4f} | SSIM: {Ssim:.4f} | UQI: {Uqi:.4f}"
        )
        print(info)
        return info, {'sam': sam, 'psnr': psnr, 'ergas': ergas}

    def train(self):
        best = [100, 0, None]
        total_epochs = self.args.niter3_dip + self.args.niter_decay3_dip
        for epoch in range(1, total_epochs + 1):
            self.optimizer.zero_grad()
            self.hr_hsi_rec, self.endmember, self.abundance = self.net(
                self.lr_hsi, self.hr_msi, self.target_h, self.target_w, self.target_L
            )
            loss, loss_dict = self._compute_loss_decomposed(self.hr_hsi_rec)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.spectral_net.parameters(), max_norm=1.0)

            if epoch % 50 == 0:
                self.diag.collect(
                    self.net, self.endmember, self.abundance,
                    self.hr_hsi_rec.detach(), loss_dict, self.lr_hsi
                )

            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 == 0:
                with torch.no_grad():
                    rec_for_metric = F.interpolate(
                        self.hr_hsi_rec, size=self.gt.shape[:2], mode='bilinear'
                    ).cpu().numpy()[0].transpose(1, 2, 0)
                    print(f'\nEPOCH {epoch:04d} | Loss: {loss.item():.4f}')
                    _, m = self.print_and_format_metrics("hr_hsi vs GT  ", rec_for_metric, self.gt)

                    e_min, e_max = self.endmember.min().item(), self.endmember.max().item()
                    a_min, a_max = self.abundance.min().item(), self.abundance.max().item()
                    print(f"[Monitor] E: {e_min:.4f}~{e_max:.4f} | A: {a_min:.4f}~{a_max:.4f}")

                    self.diag.print_report(epoch)

                    if m['sam'] < best[0]:
                        best = [m['sam'], m['psnr'], self.hr_hsi_rec]

        best_np = best[2].cpu().numpy()[0].transpose(1, 2, 0)
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'), {'Out': best_np})
        return best[2]