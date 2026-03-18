# -*- coding: utf-8 -*-
"""
������������������������������������ + ������������
- ������������ E (L_real, K)�������������� Out_fhsi ����
- MSI ������coarse abundance + INR2D_abun -> HR abundance
- HSI ������coarse abundance + INR2D_3   -> LR abundance
"""

import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .INR1D import EndmemberINR1D as INR1D
from .INR2D import INR2D

def init_weights(net, init_type, gain):
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
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / channel)
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


def init_net(net, device, init_type, init_gain, initializer):
    print('in init_net')
    net.to(device)
    if initializer:
        init_weights(net, init_type, init_gain)
    else:
        print('Stage3 with default initialize')
    return net


########################## double_U_net_skip ############################


def double_U_net_skip(
    Out_fhsi, Out_fmsi, args, init_type='kaiming', init_gain=0.02, initializer=True
):
    net = double_u_net_skip(Out_fhsi, Out_fmsi, args)
    return init_net(net, args.device, init_type, init_gain, initializer)


class double_u_net_skip(nn.Module):
    """
    ������ double U-Net��
    - ������������ E (L_real, K)�������������� Out_fhsi.shape[1] ����
    - MSI ������coarse abundance logits -> INR2D_abun -> HR abundance
    - HSI ������coarse abundance logits -> INR2D_3   -> LR abundance
    forward ����: E_full, A_msi_hr, A_hsi_lr
    """

    def __init__(self, Out_fhsi, Out_fmsi, args):
        super().__init__()

        # 1) ��������
        self.band = args.band         # ����������������U-Net hidden dim��
        self.band_k = args.band_k     # �������� K

        # ���������������������������� / ��������
        self.Out_fhsi = Out_fhsi      # (B, L_real, H_lr, W_lr)
        self.Out_fmsi = Out_fmsi      # (B, C_msi, H_hr, W_hr)

        # �������������� HSI ����
        self.L_real = self.Out_fhsi.shape[1]   # ���������������� L
        self.K = self.band_k                   # �������� K

        # HR-MSI ����
        self.scale = [
            (self.Out_fmsi.shape[2], self.Out_fmsi.shape[3]),
            (int(self.Out_fmsi.shape[2] / 2), int(self.Out_fmsi.shape[3] / 2)),
            (int(self.Out_fmsi.shape[2] / 4), int(self.Out_fmsi.shape[3] / 4)),
        ]
        # LR-HSI ����
        self.scale2 = [
            (self.Out_fhsi.shape[2], self.Out_fhsi.shape[3]),
            (int(self.Out_fhsi.shape[2] / 2), int(self.Out_fhsi.shape[3] / 2)),
        ]

        print("MSI scales:", self.scale)
        print("HSI scales:", self.scale2)

        # ==================================================
        # 1. ������������ E (L_real, K)
        #    ���������� INR1D��������������������
        # ==================================================
        self.E = nn.Parameter(torch.randn(self.L_real, self.K))

        # ==================================================
        # 2. MSI ����������coarse abundance + INR2D_abun
        # ==================================================
        C_msi = self.Out_fmsi.shape[1]

        self.ex6 = nn.Sequential(
            nn.Conv2d(C_msi, self.band, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.ex7 = nn.Sequential(
            nn.Conv2d(self.band, self.band, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.ex8 = nn.Sequential(
            nn.Conv2d(self.band, self.band, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # skip ����
        self.skip3 = nn.Sequential(
            nn.Conv2d(self.band, 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.skip4 = nn.Sequential(
            nn.Conv2d(self.band, 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ����/��������
        self.HLF1 = nn.Sequential(
            nn.Conv2d(self.band, self.band // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.band // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band // 2, self.band, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.HLF2 = nn.Sequential(
            nn.Conv2d(self.band, self.band // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.band // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band // 2, self.band, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # MSI coarse abundance head: ���� coarse abundance logits
        self.msi_abun_head = nn.Sequential(
            nn.Conv2d(self.band + 2, self.band, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band, self.band_k, kernel_size=1, stride=1, padding=0),
        )

        # INR2D_abun: �� MSI abundance ������/��������������������
        self.INR2D_abun = INR2D(
            dim=self.band_k,
            out_dim=self.band_k,
            hidden_dim=256,
            hidden_layers=3,
            L=4,
            weight_mode='gaussian',
        )

        # ==================================================
        # 3. HSI ����������coarse abundance + INR2D_3
        # ==================================================
        C_hsi = self.Out_fhsi.shape[1]

        self.ex11 = nn.Sequential(
            nn.Conv2d(C_hsi, self.band, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.ex12 = nn.Sequential(
            nn.Conv2d(self.band, self.band, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.skip5 = nn.Sequential(
            nn.Conv2d(self.band, 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.HLF3 = nn.Sequential(
            nn.Conv2d(self.band, self.band // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.band // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band // 2, self.band, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # HSI coarse abundance head: ���� coarse abundance logits (LR ����)
        self.hsi_abun_head = nn.Sequential(
            nn.Conv2d(self.band + 2, self.band, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band, self.band_k, kernel_size=1, stride=1, padding=0),
        )

        # INR2D_3: �� HSI abundance �� refine����������������
        self.INR2D_3 = INR2D(
            dim=self.band_k,
            out_dim=self.band_k,
            hidden_dim=256,
            hidden_layers=3,
            L=4,
            weight_mode='gaussian',
        )

    def forward(self, Out_fhsi, Out_fmsi):
        """
        ������
          E_full      : (L_real, K)
          abun_msi_hr : (B, K, H_hr, W_hr)
          abun_hsi_lr : (B, K, H_lr, W_lr)
        """
        B, _, H_hr, W_hr = Out_fmsi.shape
        _, _, H_lr, W_lr = Out_fhsi.shape

        # ================== 1. �������� E_full ==================
        # ���������������� E������ (L_real, K)
        E_full = self.E

        # ================== 2. MSI ������HR abundance ==================
        x9 = self.ex6(Out_fmsi)                                   # (B, band, H, W)
        x10 = F.adaptive_avg_pool2d(x9, self.scale[1])            # (B, band, H/2, W/2)
        x10_high = F.adaptive_max_pool2d(x9, self.scale[1])       # (B, band, H/2, W/2)

        x11 = self.ex7(x10)                                       # (B, band, H/2, W/2)
        x12 = F.adaptive_avg_pool2d(x11, self.scale[2])           # (B, band, H/4, W/4)
        x12_high = F.adaptive_max_pool2d(x11, self.scale[2])      # (B, band, H/4, W/4)
        x13 = self.ex8(x12)                                       # (B, band, H/4, W/4)

        up_1 = nn.Upsample(self.scale[1], mode='bilinear', align_corners=False)
        s3 = self.skip3(x11)                                      # (B, 2, H/2, W/2)
        x14 = up_1(x13)                                           # (B, band, H/2, W/2)
        x12_high_up = up_1(x12_high)                              # (B, band, H/2, W/2)
        x14 = x14 + x12_high_up                                   # ��������
        x14 = self.HLF1(x14)                                      # (B, band, H/2, W/2)

        x15 = torch.cat([s3, x14], dim=1)                         # (B, band+2, H/2, W/2)

        up_2 = nn.Upsample(self.scale[0], mode='bilinear', align_corners=False)
        s4 = self.skip4(x9)                                       # (B, 2, H, W)
        x16 = up_2(x15)                                           # (B, band+2, H, W)

        # ���������� band ������������ HLF2
        x16 = x16[:, : self.band, :, :]                           # (B, band, H, W)
        x16 = self.HLF2(x16)                                      # (B, band, H, W)

        msi_abun_in = torch.cat([s4, x16], dim=1)                 # (B, band+2, H, W)
        abun_coarse_logits = self.msi_abun_head(msi_abun_in)      # (B, K, H, W)

        abun_hr_logits = self.INR2D_abun(abun_coarse_logits)      # (B, K, H, W)
        abun_msi_hr = abun_hr_logits                              # (B, K, H, W)



        return E_full, abun_msi_hr


########################## double_U_net_skip ############################


if __name__ == "__main__":
    pass