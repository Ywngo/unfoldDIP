# -*- coding: utf-8 -*-
"""
���������� + ������ INR ������ Stage-3 ����

����:
    E_full, A_msi_hr, A_hsi_lr = double_U_net_multiscale(Out_fhsi, Out_fmsi, args)

    E_full   : (L_real, K)          # ��������
    A_msi_hr : (B, K, H_hr, W_hr)   # HR abundance
    A_hsi_lr : (B, K, H_lr, W_lr)   # LR abundance
"""

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


from .INR1D import EndmemberINR1D as EndmemberINR1D


from .INR2D import INR2D as INR2D

def init_weights(net, init_type='kaiming', gain=0.02):
    print('in init_weights')

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type
                )
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, device, init_type='kaiming', init_gain=0.02, initializer=True):
    print('in init_net')
    net.to(device)
    if initializer:
        init_weights(net, init_type, init_gain)
    else:
        print('Stage3 with default initialize')
    return net


def double_U_net_multiscale(
    Out_fhsi, Out_fmsi, args, init_type='kaiming', init_gain=0.02, initializer=True
):
    """
    Out_fhsi: (B, L_real, H_lr, W_lr)
    Out_fmsi: (B, C_msi, H_hr, W_hr)
    """
    net = DoubleUNetMultiScaleINR(Out_fhsi, Out_fmsi, args)
    return init_net(net, args.device, init_type, init_gain, initializer)


class DoubleUNetMultiScaleINR(nn.Module):
    """
    ���������� + ������ INR ����������

    - ���� E:
        HSI ���������� (3 ����) -> ���������� pooling ���� Zs
        -> ������ MLP ���� E_base_s (B,K,L0)
        -> ������ EndmemberINR1D_s ������ E_s (B,K,L_real)
        -> ���������������� E_full (L_real,K)

    - MSI abundance:
        MSI ���������� (3 ����) -> ������ head ���� coarse abundance_s
        -> INR2D_s refine �� HR ����
        -> ���������������� A_msi_hr

    - HSI abundance:
        HSI ���������� (2 ����: Fh1,Fh2) -> ������ head ���� coarse abundance_s
        -> INR2D_s refine �� LR ����
        -> ������������ A_hsi_lr
    """

    def __init__(self, Out_fhsi, Out_fmsi, args):
        super().__init__()

        self.band_feat = args.band       # ���������� (hidden dim)
        self.K = args.band_k             # ��������

        self.Out_fhsi = Out_fhsi         # (B, L_real, H_lr, W_lr)
        self.Out_fmsi = Out_fmsi         # (B, C_msi, H_hr, W_hr)

        self.L_real = Out_fhsi.shape[1]
        self.H_hr, self.W_hr = Out_fmsi.shape[2], Out_fmsi.shape[3]
        self.H_lr, self.W_lr = Out_fhsi.shape[2], Out_fhsi.shape[3]

        print("MSI HR size:", (self.H_hr, self.W_hr))
        print("HSI LR size:", (self.H_lr, self.W_lr))
        print("L_real (bands):", self.L_real, "K (endmembers):", self.K)

        # ---------------- HSI ���������� (��������) ----------------
        C_hsi_in = self.Out_fhsi.shape[1]

        self.hsi_enc1 = nn.Sequential(
            nn.Conv2d(C_hsi_in, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hsi_enc2 = nn.Sequential(
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hsi_enc3 = nn.Sequential(
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )

        latent_dim = self.band_feat
        self.hsi_pool = nn.AdaptiveAvgPool2d(1)   # (H,W)->(1,1)

        # L0: ��������������
        self.L0 = min(self.L_real // 2, 64) if self.L_real > 4 else self.L_real

        # ������: latent -> E_base_s (B,K,L0)
        self.hsi_scale1_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, self.K * self.L0),
        )
        self.hsi_scale2_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, self.K * self.L0),
        )
        self.hsi_scale3_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, self.K * self.L0),
        )


        self.inr1d_s1 = EndmemberINR1D(self.L0, self.L_real,hidden_dim=256)
        self.inr1d_s2 = EndmemberINR1D(self.L0, self.L_real,hidden_dim=256)
        self.inr1d_s3 = EndmemberINR1D(self.L0, self.L_real,hidden_dim=256)


        self.scale_weights_E = nn.Parameter(torch.ones(3))

        # ---------------- MSI ���������� + INR2D abundance ----------------
        C_msi_in = self.Out_fmsi.shape[1]

        self.msi_enc1 = nn.Sequential(
            nn.Conv2d(C_msi_in, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.msi_enc2 = nn.Sequential(
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.msi_enc3 = nn.Sequential(
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.msi_head1 = nn.Sequential(
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.K, kernel_size=1),
        )
        self.msi_head2 = nn.Sequential(
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.K, kernel_size=1),
        )
        self.msi_head3 = nn.Sequential(
            nn.Conv2d(self.band_feat, self.band_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.band_feat),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band_feat, self.K, kernel_size=1),
        )


        self.inr2d_msi_1 = INR2D(dim=self.K, out_dim=self.K, hidden_dim=256)
        self.inr2d_msi_2 = INR2D(dim=self.K, out_dim=self.K, hidden_dim=256)
        self.inr2d_msi_3 = INR2D(dim=self.K, out_dim=self.K, hidden_dim=256)

        self.scale_weights_A_msi = nn.Parameter(torch.ones(3))



    def _build_E_from_hsi(self, Fh1, Fh2, Fh3):
        """
        �� HSI ������������������ E_full (L_real, K)��
        ������ latent -> E_base_s (B,K,L0) -> EndmemberINR1D_s -> E_s (B,K,L_real) -> ������
        """
        B = Fh1.shape[0]

        z1 = self.hsi_pool(Fh1).view(B, -1)   # (B, C)
        z2 = self.hsi_pool(Fh2).view(B, -1)
        z3 = self.hsi_pool(Fh3).view(B, -1)

        E_base1 = self.hsi_scale1_mlp(z1).view(B, self.K, self.L0)  # (B, K, L0)
        E_base2 = self.hsi_scale2_mlp(z2).view(B, self.K, self.L0)
        E_base3 = self.hsi_scale3_mlp(z3).view(B, self.K, self.L0)

        E1 = self.inr1d_s1(E_base1)   # (B, K, L_real)
        E2 = self.inr1d_s2(E_base2)
        E3 = self.inr1d_s3(E_base3)

        w = torch.softmax(self.scale_weights_E, dim=0)  # (3,)
        E_all = w[0] * E1 + w[1] * E2 + w[2] * E3       # (B, K, L_real)

        E_full = E_all.mean(dim=0).permute(1, 0)        # (L_real, K)

        import torch.nn.functional as F
        E_full = F.softplus(E_full)  # ��������
        # ������������������������ (���������������� L_real, K������������ L_real ������ dim=0 �� max)
        E_full = E_full / (E_full.max(dim=0, keepdim=True)[0] + 1e-6)
        return E_full

    def forward(self, Out_fhsi, Out_fmsi):
        B = Out_fhsi.shape[0]

        # HSI ����������
        Fh1 = self.hsi_enc1(Out_fhsi)    # (B, C, H_lr,   W_lr)
        Fh2 = self.hsi_enc2(Fh1)         # (B, C, H_lr/2, W_lr/2)
        Fh3 = self.hsi_enc3(Fh2)         # (B, C, H_lr/4, W_lr/4)

        E_full = self._build_E_from_hsi(Fh1, Fh2, Fh3)  # (L_real, K)

        # MSI ���������� + INR2D abundance
        Fm1 = self.msi_enc1(Out_fmsi)                   # (B, C, H_hr,   W_hr)
        Fm2 = self.msi_enc2(Fm1)                        # (B, C, H_hr/2, W_hr/2)
        Fm3 = self.msi_enc3(Fm2)                        # (B, C, H_hr/4, W_hr/4)

        A1_coarse = self.msi_head1(Fm1)                 # (B, K, H_hr,   W_hr)
        A1_hr     = self.inr2d_msi_1(A1_coarse)         # (B, K, H_hr,   W_hr)

        A2_coarse = self.msi_head2(Fm2)                 # (B, K, H_hr/2, W_hr/2)
        A2_hr_in  = F.interpolate(A2_coarse, size=(self.H_hr, self.W_hr),
                                  mode='bilinear', align_corners=False)
        A2_hr     = self.inr2d_msi_2(A2_hr_in)          # (B, K, H_hr,   W_hr)

        A3_coarse = self.msi_head3(Fm3)                 # (B, K, H_hr/4, W_hr/4)
        A3_hr_in  = F.interpolate(A3_coarse, size=(self.H_hr, self.W_hr),
                                  mode='bilinear', align_corners=False)
        A3_hr     = self.inr2d_msi_3(A3_hr_in)          # (B, K, H_hr,   W_hr)

        w_A_msi = torch.softmax(self.scale_weights_A_msi, dim=0)
        A_msi_hr = w_A_msi[0] * A1_hr + w_A_msi[1] * A2_hr + w_A_msi[2] * A3_hr  # (B,K,H_hr,W_hr)
        A_msi_hr = F.softmax(A_msi_hr, dim=1)


        return E_full, A_msi_hr