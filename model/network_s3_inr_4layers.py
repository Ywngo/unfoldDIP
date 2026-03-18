# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug
空天信息创新研究院20-25直博生，导师高连如

"""
from model.restormer import Restormer, RestormerBlock

"""
❗❗❗❗❗❗#此py作用：第三阶段所需要的网络模块
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
import os
import scipy
import torch.nn.functional as fun
from .network_s3_siren1d import INR1D
from .nerword_s3_siren import INR2D


def init_weights(net, init_type, gain):
    print('in init_weights')

    def init_func(m):
        classname = m.__class__.__name__
        # print(classname,m,'_______')
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
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
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, device, init_type, init_gain, initializer):
    print('in init_net')
    net.to(device)  # gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer:
        # print(2,initializer)
        init_weights(net, init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net


class SharedLatentFusion(nn.Module):
    """
    Build an explicit shared latent z from spat and spec features and provide
    injectors to feed z into spatial & spectral decoders.
    - produce z_shared (B, z_dim)
    - injectors return tensors broadcastable to feature shapes
    """

    def __init__(self, in_spat_ch, in_spec_ch, z_dim=128):
        super().__init__()
        self.spat_pool = nn.AdaptiveAvgPool2d(1)
        self.spec_pool = nn.AdaptiveAvgPool1d(1)
        self.proj_spat = nn.Linear(in_spat_ch, z_dim)
        self.proj_spec = nn.Linear(in_spec_ch, z_dim)
        self.fuse = nn.Sequential(nn.Linear(z_dim * 2, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim))
        # injectors
        self.inject_spat = nn.Linear(z_dim, in_spat_ch)
        self.inject_spec = nn.Linear(z_dim, in_spec_ch)

    def forward(self, F_spat, F_spec):
        """
        F_spat: (B, C_spat, H, W)
        F_spec: (B, C_spec, L)
        returns:
           z_shared (B, z_dim),
           spat_inject (B, C_spat, 1, 1) broadcastable,
           spec_inject (B, C_spec, 1) broadcastable
        """
        B, C_spat, H, W = F_spat.shape
        B2, C_spec, L = F_spec.shape
        assert B == B2

        z_s = self.spat_pool(F_spat).view(B, C_spat)  # (B, C_spat)
        z_p = self.spec_pool(F_spec).view(B, C_spec)  # (B, C_spec)

        z_s_proj = self.proj_spat(z_s)  # (B, z)
        z_p_proj = self.proj_spec(z_p)  # (B, z)
        z_cat = torch.cat([z_s_proj, z_p_proj], dim=1)  # (B, 2z)
        z_shared = self.fuse(z_cat)  # (B, z)

        # injectors
        spat_inject = self.inject_spat(z_shared).view(B, C_spat, 1, 1)
        spec_inject = self.inject_spec(z_shared).view(B, C_spec, 1)

        return z_shared, spat_inject, spec_inject, z_s_proj, z_p_proj


########################## double_U_net_skip ############################

def double_U_net_skip(Out_fhsi, Out_fmsi, args, init_type='kaiming', init_gain=0.02, initializer=True):
    net = double_u_net_skip_4layer(Out_fhsi, Out_fmsi, args)

    return init_net(net, args.device, init_type, init_gain, initializer)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_s3_siren1d import INR1D
from .nerword_s3_siren import INR2D

class double_u_net_skip_4layer(nn.Module):
    def __init__(self, Out_fhsi, Out_fmsi, args):
        super().__init__()
        self.band = args.band
        self.band_k = args.band_k

        # ================= HSI ���� =================
        self.scale1 = [
            Out_fhsi.shape[1],
            Out_fhsi.shape[1]//2,
            Out_fhsi.shape[1]//4,
            Out_fhsi.shape[1]//8
        ]
        # INR ����
        self.INR1D_1 = INR1D(self.band, self.scale1[0], 256, 4, 5)
        self.INR1D_2 = INR1D(self.band, self.scale1[1], 256, 4, 5)
        self.INR1D_3 = INR1D(self.band, self.scale1[2], 256, 4, 5)
        self.INR1D_4 = INR1D(self.band, self.scale1[3], 256, 4, 5)

        # ����������
        self.ex1 = nn.Sequential(nn.Conv1d(Out_fhsi.shape[2]*Out_fhsi.shape[3], self.band, 5, 1, 2),
                                 nn.BatchNorm1d(self.band), nn.LeakyReLU(0.2, inplace=True))
        self.ex2 = nn.Sequential(nn.Conv1d(self.band, self.band, 5, 1, 2),
                                 nn.BatchNorm1d(self.band), nn.LeakyReLU(0.2, inplace=True))
        self.ex3 = nn.Sequential(nn.Conv1d(self.band, self.band, 5, 1, 2),
                                 nn.BatchNorm1d(self.band), nn.LeakyReLU(0.2, inplace=True))
        self.ex4 = nn.Sequential(nn.Conv1d(self.band, self.band, 5, 1, 2),
                                 nn.BatchNorm1d(self.band), nn.LeakyReLU(0.2, inplace=True))

        # ����������
        self.ex5 = nn.Sequential(nn.Conv1d(self.band + self.band//2, self.band, 5, 1, 2),
                                 nn.BatchNorm1d(self.band), nn.LeakyReLU(0.2, inplace=True))
        self.ex6 = nn.Sequential(nn.Conv1d(self.band + self.band//2, self.band, 5, 1, 2),
                                 nn.BatchNorm1d(self.band), nn.LeakyReLU(0.2, inplace=True))

        self.ex7 = nn.Sequential(nn.Conv1d(self.band + self.band//2, self.band, 1, 1, 0),
                                 nn.BatchNorm1d(self.band),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv1d(self.band, self.band_k, kernel_size=1, stride=1, padding=0),
                                 nn.Sigmoid())

        # skip connections
        self.skip1 = nn.Sequential(nn.Conv1d(self.band, self.band//2, 3, 1, 1),
                                   nn.BatchNorm1d(self.band//2), nn.LeakyReLU(0.2, inplace=True))
        self.skip2 = nn.Sequential(nn.Conv1d(self.band, self.band//2, 3, 1, 1),
                                   nn.BatchNorm1d(self.band//2), nn.LeakyReLU(0.2, inplace=True))
        self.skip3 = nn.Sequential(nn.Conv1d(self.band, self.band//2, 3, 1, 1),
                                   nn.BatchNorm1d(self.band//2), nn.LeakyReLU(0.2, inplace=True))


        # ================= MSI ���� =================
        self.scale = [
            (Out_fmsi.shape[2], Out_fmsi.shape[3]),
            (Out_fmsi.shape[2]//2, Out_fmsi.shape[3]//2),
            (Out_fmsi.shape[2]//4, Out_fmsi.shape[3]//4),
            (Out_fmsi.shape[2]//8, Out_fmsi.shape[3]//8)
        ]

        # INR ����
        self.INR2D_1 = INR2D(240, 240, 256, 3, 4)
        self.INR2D_2 = INR2D(240, 240, 256, 3, 4)
        self.INR2D_3 = INR2D(240, 240, 256, 3, 4)
        self.INR2D_4 = INR2D(240, 240, 256, 3, 4)

        # ����������
        self.ex9 = nn.Sequential(nn.Conv2d(Out_fmsi.shape[1], self.band, 5, 1, 2),
                                 nn.BatchNorm2d(self.band), nn.LeakyReLU(0.2, inplace=True))
        self.ex10 = nn.Sequential(nn.Conv2d(self.band, self.band, 5, 1, 2),
                                 nn.BatchNorm2d(self.band), nn.LeakyReLU(0.2, inplace=True))
        self.ex11 = nn.Sequential(nn.Conv2d(self.band, self.band, 5, 1, 2),
                                 nn.BatchNorm2d(self.band), nn.LeakyReLU(0.2, inplace=True))
        self.ex12 = nn.Sequential(nn.Conv2d(self.band, self.band, 5, 1, 2),
                                 nn.BatchNorm2d(self.band), nn.LeakyReLU(0.2, inplace=True))

        # ����������
        self.HLF1 = nn.Sequential(nn.Conv2d(self.band, self.band//2, 3, 1, 1),
                                  nn.BatchNorm2d(self.band//2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(self.band//2, self.band, 1, 1, 0),
                                  nn.BatchNorm2d(self.band),
                                  nn.LeakyReLU(0.2, inplace=True))
        self.HLF2 = nn.Sequential(nn.Conv2d(self.band, self.band//2, 3, 1, 1),
                                  nn.BatchNorm2d(self.band//2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(self.band//2, self.band, 1, 1, 0),
                                  nn.BatchNorm2d(self.band),
                                  nn.LeakyReLU(0.2, inplace=True))
        self.HLF3 = nn.Sequential(nn.Conv2d(self.band, self.band//2, 3, 1, 1),
                                  nn.BatchNorm2d(self.band//2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(self.band//2, self.band, 1, 1, 0),
                                  nn.BatchNorm2d(self.band),
                                  nn.LeakyReLU(0.2, inplace=True))

        self.ex13 = nn.Sequential(nn.Conv2d(self.band + 2, self.band, 1, 1, 0),nn.BatchNorm2d(self.band), nn.Sigmoid())
        self.ex14 = nn.Sequential(nn.Conv2d(self.band + 2, self.band, 1, 1, 0),nn.BatchNorm2d(self.band), nn.Sigmoid())
        self.ex15 = nn.Sequential(
            nn.Conv2d(self.band + 2, self.band, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            # nn.Sigmoid()  #nn.LeakyReLU(0.2, inplace=True)
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band, self.band_k, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.Sigmoid()
        )


        # skip connections
        self.skip1_msi = nn.Sequential(nn.Conv2d(self.band, 2, 1, 1, 0), nn.BatchNorm2d(2), nn.LeakyReLU(0.2, inplace=True))
        self.skip2_msi = nn.Sequential(nn.Conv2d(self.band, 2, 1, 1, 0), nn.BatchNorm2d(2), nn.LeakyReLU(0.2, inplace=True))
        self.skip3_msi = nn.Sequential(nn.Conv2d(self.band, 2, 1, 1, 0), nn.BatchNorm2d(2), nn.LeakyReLU(0.2, inplace=True))


    def forward(self, Out_fhsi, Out_fmsi):
        # ---------------- HSI ���� ----------------
        b, c, h, w = Out_fhsi.shape
        x = Out_fhsi.permute(0,2,3,1).reshape(b, h*w, c).cuda()
        x1 = self.ex1(x)                          # ������1
        x2 = nn.AdaptiveAvgPool1d(self.scale1[1])(x1)
        hsi_inr1 = self.INR1D_1(x2.permute(0,2,1)).permute(0,2,1)

        x3 = self.ex2(x2)                         # ������2
        x4 = nn.AdaptiveAvgPool1d(self.scale1[2])(x3)
        hsi_inr2 = self.INR1D_2(x4.permute(0,2,1)).permute(0,2,1)

        x5 = self.ex3(x4)                         # ������3
        x6 = nn.AdaptiveAvgPool1d(self.scale1[3])(x5)
        hsi_inr3 = self.INR1D_3(x6.permute(0,2,1)).permute(0,2,1)

        x7 = self.ex4(x6)                         # ������4


        # Decoder HSI
        up1 = nn.Upsample(self.scale1[2], mode='linear')
        x8 = up1(x7) + hsi_inr3
        x9 = self.ex5(torch.cat([self.skip3(x5), x8], dim=1))

        up2 = nn.Upsample(self.scale1[1], mode='linear')
        x10 = up2(x9) + hsi_inr2
        x11 = self.ex6(torch.cat([self.skip2(x3), x10], dim=1))

        up3 = nn.Upsample(self.scale1[0], mode='linear')
        x12 = up3(x11) + hsi_inr1
        out_fhsi = self.ex7(torch.cat([self.skip1(x1), x12], dim=1))


        # ---------------- MSI ���� ----------------
        x6_msi = self.ex9(Out_fmsi)
        x7_msi = nn.AdaptiveAvgPool2d(self.scale[1])(x6_msi)
        msi_inr1 = self.INR2D_1(nn.AdaptiveMaxPool2d(self.scale[1])(x6_msi))

        x8_msi = self.ex10(x7_msi)
        x9_msi = nn.AdaptiveAvgPool2d(self.scale[2])(x8_msi)
        msi_inr2 = self.INR2D_2(nn.AdaptiveMaxPool2d(self.scale[2])(x8_msi))

        x10_msi = self.ex11(x9_msi)
        x11_msi = nn.AdaptiveAvgPool2d(self.scale[3])(x10_msi)
        msi_inr3 = self.INR2D_3(nn.AdaptiveMaxPool2d(self.scale[3])(x10_msi))

        x12_msi = self.ex12(x11_msi)


        # Decoder MSI
        up1_msi = nn.Upsample(self.scale[2], mode='bilinear')
        x13_msi = up1_msi(x12_msi) + msi_inr3
        x13_msi = self.HLF1(x13_msi)
        x14_msi = self.ex13(torch.cat([self.skip3_msi(x10_msi), x13_msi], dim=1))

        up2_msi = nn.Upsample(self.scale[1], mode='bilinear')
        x15_msi = up2_msi(x14_msi) + msi_inr2
        x15_msi = self.HLF2(x15_msi)
        x16_msi = self.ex14(torch.cat([self.skip2_msi(x8_msi), x15_msi], dim=1))

        up3_msi = nn.Upsample(self.scale[0], mode='bilinear')
        x17_msi = up3_msi(x16_msi) + msi_inr1
        x17_msi = self.HLF2(x17_msi)

        out_fmsi = self.ex15(torch.cat([self.skip1_msi(x6_msi), x17_msi], dim=1))

        return out_fhsi, out_fmsi
