# -*- coding: utf-8 -*-
"""
Batch Augmentation & Blind Estimation Layer (Simplest Version)
Restored to basic initialization and L1 Loss with full metrics monitoring.
"""

import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from .read_data import readdata
from .evaluation import MetricsCal


class BlurDown(object):
    """
    Performs spatial blurring (convolution) and downsampling.
    """

    def __init__(self):
        pass

    def __call__(self, input_tensor: torch.Tensor, psf, groups, ratio):
        if psf.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1)
        output_tensor = F.conv2d(input_tensor, psf, None, (ratio, ratio), groups=groups)
        return output_tensor


# ==========================================
# 1. SRF-Net (Sine Activation)
# ==========================================
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SrfNet(nn.Module):
    def __init__(self, hs_bands, ms_bands):
        super(SrfNet, self).__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands

        self.fc = nn.Sequential(
            nn.Linear(hs_bands, 64),
            Sine(w0=5.0),
            nn.Linear(64, 32),
            Sine(w0=1.0),
            nn.Linear(32, ms_bands)
        )

    def forward(self):
        device = next(self.parameters()).device
        inp = torch.eye(self.hs_bands, device=device)
        out = self.fc(inp)
        out = out.t()
        return out.view(self.ms_bands, self.hs_bands, 1, 1)


# ==========================================
# 2. BlindNet (Simplest PSF Est)
# ==========================================
class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio = ratio

        # [Simplest Initialization]
        # Initialize with zeros. After Softmax, this becomes a Uniform Distribution.
        # This is the standard "blind" starting point.
        self.psf_logits = nn.Parameter(torch.zeros([1, 1, self.ker_size, self.ker_size]))

        self.srf_net = SrfNet(hs_bands, ms_bands)
        self.blur_down = BlurDown()

    @property
    def psf(self):
        b, c, h, w = self.psf_logits.shape
        return F.softmax(self.psf_logits.view(b, c, -1), dim=-1).view(b, c, h, w)

    @property
    def srf(self):
        logits = self.srf_net()
        return F.softmax(logits, dim=1)

    def forward(self, lr_hsi, hr_msi):
        srf = self.srf
        psf = self.psf

        # Branch 1: Spectral
        lr_msi_fhsi = F.conv2d(lr_hsi, srf, None)
        lr_msi_fhsi = torch.clamp(lr_msi_fhsi, 0.0, 1.0)

        # Branch 2: Spatial
        lr_msi_fmsi = self.blur_down(hr_msi, psf, self.ms_bands, self.ratio)
        lr_msi_fmsi = torch.clamp(lr_msi_fmsi, 0.0, 1.0)

        return lr_msi_fhsi, lr_msi_fmsi


# ==========================================
# 3. Training Wrapper (With Full Monitoring)
# ==========================================
class Blind(readdata):
    def __init__(self, args):
        super().__init__(args)

        self.S1_lr = 1e-3 if self.args.lr_stage1 < 1e-3 else self.args.lr_stage1
        self.ker_size = self.args.scale_factor
        self.ratio = self.args.scale_factor
        self.hs_bands = self.srf_gt.shape[0]
        self.ms_bands = self.srf_gt.shape[1]

        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ker_size, self.ratio).to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.S1_lr)

        # [Loss] Standard L1 Loss
        self.criterion = nn.L1Loss()

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter1) / float(args.niter_decay1 + 1)
            return lr_l

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

    def train(self):
        lr_hsi, hr_msi = self.tensor_lr_hsi.to(self.args.device), self.tensor_hr_msi.to(self.args.device)

        print(f"Start Training (Simplest PSF + L1 Loss)... Total Epochs: {self.args.niter1 + self.args.niter_decay1}")

        for epoch in range(1, self.args.niter1 + self.args.niter_decay1 + 1):

            self.model.train()
            self.optimizer.zero_grad()

            lr_msi_fhsi_est, lr_msi_fmsi_est = self.model(lr_hsi, hr_msi)
            loss = self.criterion(lr_msi_fhsi_est, lr_msi_fmsi_est)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # --- Full Monitoring (Restored) ---
            if (epoch) % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    print("____________________________________________")
                    print('epoch:{} lr:{}'.format(epoch, self.optimizer.param_groups[0]['lr']))
                    print('Loss: {:.9f}'.format(loss.item()))
                    print('************')

                    # Prepare for evaluation
                    lr_msi_fhsi_est = torch.clamp(lr_msi_fhsi_est, 0.0, 1.0)
                    lr_msi_fmsi_est = torch.clamp(lr_msi_fmsi_est, 0.0, 1.0)

                    lr_msi_fhsi_est_numpy = lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
                    lr_msi_fmsi_est_numpy = lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    # 1. Consistency (Generated vs Generated)
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(lr_msi_fhsi_est_numpy - lr_msi_fmsi_est_numpy))
                    information1 = "Consistency (Gen vs Gen)\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information1)
                    print('************')

                    # 2. SRF Branch vs Real LR-MSI
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(self.lr_msi_fhsi - lr_msi_fhsi_est_numpy))
                    information2 = "SRF Branch vs Real LR-MSI \n  L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information2)
                    print('************')

                    # 3. PSF Branch vs Real LR-MSI
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(self.lr_msi_fmsi - lr_msi_fmsi_est_numpy))
                    information3 = "PSF Branch vs Real LR-MSI\n L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information3)

                    # Print Values
                    psf_val = self.model.psf.data.cpu().detach().numpy()
                    srf_val = self.model.srf.data.cpu().detach().numpy()

                    print('************')

                    # 4. PSF Validation (Apply Est. PSF to GT HSI -> Compare with Real LR-HSI)
                    curr_psf = self.model.psf.repeat(self.hs_bands, 1, 1, 1)
                    lr_hsi_est = F.conv2d(self.tensor_gt.to(self.args.device), curr_psf, None,
                                          (self.ker_size, self.ker_size), groups=self.hs_bands)
                    lr_hsi_est = torch.clamp(lr_hsi_est, 0.0, 1.0)
                    lr_hsi_est_numpy = lr_hsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.lr_hsi, lr_hsi_est_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(self.lr_hsi - lr_hsi_est_numpy))
                    information4 = "PSF Validation (GT->LRHSI)\n L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information4)
                    print('************')

                    # 5. SRF Validation (Apply Est. SRF to GT HSI -> Compare with Real HR-MSI)
                    if srf_val.shape[0] != 1:
                        srf_est = np.squeeze(srf_val).T
                    else:
                        srf_est_tmp = np.squeeze(srf_val).T
                        srf_est = srf_est_tmp[:, np.newaxis]

                    w, h, c = self.gt.shape
                    # Ensure dimensions match
                    if srf_est.shape[0] == c:
                        hr_msi_est_numpy = np.dot(self.gt.reshape(w * h, c), srf_est).reshape(w, h, srf_est.shape[1])
                        hr_msi_est_numpy = np.clip(hr_msi_est_numpy, 0.0, 1.0)
                    else:
                        hr_msi_est_numpy = np.zeros_like(self.hr_msi)

                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(self.hr_msi, hr_msi_est_numpy,
                                                                       self.args.scale_factor)
                    L1 = np.mean(np.abs(self.hr_msi - hr_msi_est_numpy))
                    information5 = "SRF Validation (GT->HRMSI)\n L1 {} sam {},psnr {} ,ergas {},cc {},rmse {},Ssim {},Uqi {}".format(
                        L1, sam, psnr, ergas, cc, rmse, Ssim, Uqi)
                    print(information5)

                    # Save Logs
                    file_name = os.path.join(self.args.expr_dir, 'Stage1.txt')
                    with open(file_name, 'a') as opt_file:
                        opt_file.write(f'Epoch: {epoch}\n')
                        opt_file.write(information1 + '\n')
                        opt_file.write(information2 + '\n')
                        opt_file.write(information3 + '\n')
                        opt_file.write(information4 + '\n')
                        opt_file.write(information5 + '\n')
                        opt_file.write('-' * 20 + '\n')

        # Save Model
        PATH = os.path.join(self.args.expr_dir, self.model.__class__.__name__ + '.pth')
        torch.save(self.model.state_dict(), PATH)

        # Save Result Mats
        lr_msi_fhsi_est_numpy = lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
        lr_msi_fmsi_est_numpy = lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1, 2, 0)
        sio.savemat(os.path.join(self.args.expr_dir, 'estimated_lr_msi.mat'),
                    {'lr_msi_fhsi': lr_msi_fhsi_est_numpy, 'lr_msi_fmsi': lr_msi_fmsi_est_numpy})

        return lr_msi_fhsi_est, lr_msi_fmsi_est

    def get_est_values(self):
        psf = self.model.psf.data.cpu().detach().numpy()
        srf = self.model.srf.data.cpu().detach().numpy()
        psf = np.squeeze(psf)
        srf = np.squeeze(srf).T
        return psf, srf

    def get_save_result(self):
        psf, srf = self.get_est_values()

        # Save to experiment dir
        sio.savemat(os.path.join(self.args.expr_dir, 'estimated_psf_srf.mat'),
                    {'psf_est': psf, 'srf_est': srf})

        # Save to checkpoints
        pth_name = f"{self.args.data_name}_scale{self.args.scale_factor}_estimated_psf_srf.mat"
        stage1_save_path = os.path.join('./checkpoints', pth_name)
        os.makedirs(os.path.dirname(stage1_save_path), exist_ok=True)
        sio.savemat(stage1_save_path, {'psf_est': psf, 'srf_est': srf})