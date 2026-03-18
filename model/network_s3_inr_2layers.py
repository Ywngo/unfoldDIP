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
    net = double_u_net_skip_2layer(Out_fhsi, Out_fmsi, args)

    return init_net(net, args.device, init_type, init_gain, initializer)


class double_u_net_skip_2layer(nn.Module):
    def __init__(self, Out_fhsi, Out_fmsi, args):
        super().__init__()
        self.band = args.band
        self.band_k = args.band_k
        self.Out_fhsi = Out_fhsi
        self.Out_fmsi = Out_fmsi

        # HSI ������/����
        self.scale1 = [Out_fhsi.shape[1], int(Out_fhsi.shape[1]/2)]
        self.INR1D_1 = INR1D(self.band, self.scale1[0], 256, 4, 5)

        self.ex1 = nn.Sequential(
            nn.Conv1d(Out_fhsi.shape[2]*Out_fhsi.shape[3], self.band, 5, 1, 2),
            nn.BatchNorm1d(self.band),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ex2 = nn.Sequential(
            nn.Conv1d(self.band, self.band, 5, 1, 2),
            nn.BatchNorm1d(self.band),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # HSI ������/����
        self.ex3 = nn.Sequential(
            nn.Conv1d(self.band + self.band//2, self.band, 5, 1, 2),
            nn.BatchNorm1d(self.band),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ex4 = nn.Sequential(
            nn.Conv1d(self.band + self.band//2, self.band_k, 1, 1, 0),
            nn.Sigmoid()
        )
        self.skip1 = nn.Sequential(
            nn.Conv1d(self.band, self.band//2, 3, 1, 1),
            nn.BatchNorm1d(self.band//2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # MSI ������/����
        self.scale = [
            (Out_fmsi.shape[2], Out_fmsi.shape[3]),
            (Out_fmsi.shape[2]//2, Out_fmsi.shape[3]//2)
        ]
        self.INR2D_1 = INR2D(dim=240, out_dim=240, hidden_dim=256, hidden_layers=3, L=4)
        self.ex6 = nn.Sequential(
            nn.Conv2d(Out_fmsi.shape[1], self.band, 5, 1, 2),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ex7 = nn.Sequential(
            nn.Conv2d(self.band, self.band, 5, 1, 2),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # MSI ������/����
        self.HLF1 = nn.Sequential(
            nn.Conv2d(self.band, self.band//2, 3, 1, 1),
            nn.BatchNorm2d(self.band//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.band//2, self.band, 1, 1, 0),
            nn.BatchNorm2d(self.band),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ex8 = nn.Sequential(
            nn.Conv2d(self.band + 2, self.band_k, 1, 1, 0),
            nn.Sigmoid()
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(self.band, 2, 1, 1, 0),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, Out_fhsi, Out_fmsi):
        # HSI
        b, c, h, w = Out_fhsi.shape
        x = Out_fhsi.permute(0,2,3,1).reshape(b, h*w, c).cuda()
        x1 = self.ex1(x)
        x2 = nn.AdaptiveAvgPool1d(self.scale1[1])(x1)
        hsi_inr = self.INR1D_1(x2.permute(0,2,1)).permute(0,2,1)
        x3 = self.ex2(x2)
        up = nn.Upsample(self.scale1[0], mode='linear')
        s1 = self.skip1(x1)
        x4 = up(x3) + hsi_inr
        out_fhsi = self.ex4(torch.cat([s1, x4], dim=1))

        # MSI
        x6 = self.ex6(Out_fmsi)
        x7 = nn.AdaptiveAvgPool2d(self.scale[1])(x6)
        msi_inr = self.INR2D_1(nn.AdaptiveMaxPool2d(self.scale[1])(x6))
        x8 = self.ex7(x7)
        up2 = nn.Upsample(self.scale[0], mode='bilinear')
        s3 = self.skip3(x6)
        x9 = up2(x8) + msi_inr
        out_fmsi = self.ex8(torch.cat([s3, x9], dim=1))
        return out_fhsi, out_fmsi



########################## double_U_net_skip############################


if __name__ == "__main__":
    pass

