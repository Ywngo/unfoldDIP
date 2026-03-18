import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from .INR1D_factorized import FactorizedINR1D as EndmemberINR1D
from .INR2D import INR2D as INR2D


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, device):
    net.to(device)
    init_weights(net, init_type='kaiming', gain=0.02)
    return net


class DoubleUNetMultiScaleINR(nn.Module):
    def __init__(self, Out_fhsi, Out_fmsi, args):
        super().__init__()
        self.band_feat = args.band
        self.K = args.band_k
        self.L_real = Out_fhsi.shape[1]
        self.H_hr, self.W_hr = Out_fmsi.shape[2], Out_fmsi.shape[3]
        self.H_lr, self.W_lr = Out_fhsi.shape[2], Out_fhsi.shape[3]

        # --- HSI ??? ---
        C_hsi_in = Out_fhsi.shape[1]
        self.hsi_enc1 = nn.Sequential(nn.Conv2d(C_hsi_in, self.band_feat, 3, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))
        self.hsi_enc2 = nn.Sequential(nn.Conv2d(self.band_feat, self.band_feat, 3, stride=2, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))
        self.hsi_enc3 = nn.Sequential(nn.Conv2d(self.band_feat, self.band_feat, 3, stride=2, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))

        self.hsi_pool = nn.AdaptiveAvgPool2d(1)
        self.L0 = min(self.L_real // 2, 64) if self.L_real > 4 else self.L_real
        self.hsi_scale_mlp = nn.Sequential(nn.Linear(self.band_feat, self.band_feat), nn.ReLU(inplace=True),
                                           nn.Linear(self.band_feat, self.K * self.L0))
        self.inr1d = EndmemberINR1D(self.L0, self.L_real, K=self.K, hidden_dim=256)

        # --- MSI ??? ---
        C_msi_in = Out_fmsi.shape[1]
        self.msi_enc1 = nn.Sequential(nn.Conv2d(C_msi_in, self.band_feat, 3, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))
        self.msi_enc2 = nn.Sequential(nn.Conv2d(self.band_feat, self.band_feat, 3, stride=2, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))
        self.msi_enc3 = nn.Sequential(nn.Conv2d(self.band_feat, self.band_feat, 3, stride=2, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))

        # --- ??? ---
        self.msi_head = nn.Conv2d(self.band_feat, self.K, 1)
        self.hsi_head = nn.Conv2d(self.band_feat, self.K, 1)
        self.inr2d_msi = INR2D(dim=self.K, out_dim=self.K, hidden_dim=256)
        self.inr2d_hsi = INR2D(dim=self.K, out_dim=self.K, hidden_dim=256)

        # ??????????????
        self.modulate_conv = nn.Sequential(
            nn.Conv2d(self.K * 2, self.K, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, Out_fhsi, Out_fmsi):
        # 1. ?????? (???????????????)
        Fh1 = self.hsi_enc1(Out_fhsi)
        z1 = self.hsi_pool(Fh1).view(Fh1.shape[0], -1)
        E_base = self.hsi_scale_mlp(z1).view(Fh1.shape[0], self.K, self.L0)
        E_full = self.inr1d(E_base).mean(dim=0).permute(1, 0)
        # ?????? Softmax ??????????????
        E_full = 2.0 * torch.tanh(E_full)

        # 2. HSI ??? (?????????)
        Ah_coarse = self.hsi_head(Fh1)
        Ah_lr = self.inr2d_hsi(Ah_coarse)

        # 3. MSI ??? (?????????)
        Fm1 = self.msi_enc1(Out_fmsi)
        Am_coarse = self.msi_head(Fm1)
        Am_hr = self.inr2d_msi(Am_coarse)

        # ================= ???????????? =================
        # ???????????????????????????????????
        Ah_up = F.interpolate(Ah_lr, size=(self.H_hr, self.W_hr), mode='bilinear', align_corners=False)

        # ????-??????? (Confidence Gate)
        gate = self.modulate_conv(torch.cat([Am_hr, Ah_up], dim=1))

        # ???????? MSI??????? HSI????????????
        A_fused_hr = gate * Am_hr + (1 - gate) * Ah_up
        # ============================================================

        return E_full, A_fused_hr, Ah_lr


class ADMM_Unfolding_Network(nn.Module):
    def __init__(self, Out_fhsi, Out_fmsi, args, psf, srf, num_stages=3):
        super().__init__()
        self.num_stages = num_stages
        self.args = args
        self.ratio = args.scale_factor

        self.register_buffer('psf', psf)
        self.register_buffer('srf', srf)
        self.ae_net = DoubleUNetMultiScaleINR(Out_fhsi, Out_fmsi, args)

        # ADMM ????
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(num_stages)])
        self.rho = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(num_stages)])

        # ?????ADMM ? Nesterov ??????
        self.momentum = nn.ParameterList([nn.Parameter(torch.tensor(0.2)) for _ in range(num_stages)])

    def H_down(self, x):
        C = x.shape[1]
        psf = self.psf if self.psf.shape[0] == C else self.psf.repeat(C, 1, 1, 1)
        # ?? ReplicationPad ????????????? scipy ????
        pad = self.psf.shape[-1] // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        return F.conv2d(x_pad, psf, None, stride=self.ratio, groups=C)

    def H_up(self, y, target_size):
        C = y.shape[1]
        psf = self.psf if self.psf.shape[0] == C else self.psf.repeat(C, 1, 1, 1)
        # ???????? (Adjoint Operator)????? output_size??????
        pad = self.psf.shape[-1] // 2
        out = F.conv_transpose2d(y, psf, None, stride=self.ratio, groups=C,
                                 output_size=(target_size[0] + 2 * pad, target_size[1] + 2 * pad))
        # ???? padding ???
        return out[:, :, pad:-pad, pad:-pad]

    def M_down(self, x):
        return F.conv2d(x, self.srf, None)

    def M_up(self, y):
        return F.conv_transpose2d(y, self.srf, None)

    def forward(self, Y_h, Y_m):
        B, C, H_h, W_h = Y_h.shape
        _, _, H_m, W_m = Y_m.shape
        target_size = (H_m, W_m)

        Z_k = F.interpolate(Y_h, size=target_size, mode='bilinear', align_corners=False)
        U_k = torch.zeros_like(Z_k)
        Z_prev = Z_k.clone()  # ??????

        for k in range(self.num_stages):
            # 1. ????????
            inp_hsi = F.interpolate(Z_k + U_k, size=(H_h, W_h), mode='bilinear', align_corners=False)
            E_full, A_fused_hr, _ = self.ae_net(inp_hsi, Y_m)

            K_hr = A_fused_hr.shape[1]
            A_hr_flat = A_fused_hr.squeeze(0).permute(1, 2, 0).reshape(-1, K_hr)
            X_prior_flat = torch.matmul(A_hr_flat, E_full.t())
            X_prior = X_prior_flat.reshape(H_m, W_m, -1).permute(2, 0, 1).unsqueeze(0)

            # 2. ADMM ????? (????????)
            res_h = self.H_down(Z_k) - Y_h
            res_m = self.M_down(Z_k) - Y_m

            grad_fidelity = 2 * self.H_up(res_h, target_size) + 2 * self.M_up(res_m)

            # ================= ?????Nesterov ???? ADMM =================
            # ?????
            Z_new = Z_k - self.eta[k] * grad_fidelity - self.rho[k] * (Z_k - X_prior + U_k)
            # ?????????????
            Z_k = Z_new + self.momentum[k] * (Z_new - Z_prev)
            Z_prev = Z_new.clone()
            # ====================================================================

            # 3. ????
            U_k = U_k + Z_k - X_prior

        return Z_k, X_prior, E_full, A_fused_hr


def build_admm_unfolding_net(Out_fhsi, Out_fmsi, args, psf, srf, num_stages=3):
    net = ADMM_Unfolding_Network(Out_fhsi, Out_fmsi, args, psf, srf, num_stages)
    return init_net(net, args.device)