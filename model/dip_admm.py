import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy
import torch.nn.functional as F
from .evaluation import MetricsCal
from .admm import build_admm_unfolding_net


# ????????????????
class PSF_down():
    def __call__(self, input_tensor, psf, ratio):
        _, C, _, _ = input_tensor.shape
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1)
        pad = psf.shape[-1] // 2
        x_pad = F.pad(input_tensor, (pad, pad, pad, pad), mode='replicate')
        return F.conv2d(x_pad, psf, None, stride=(ratio, ratio), groups=C)


class SRF_down():
    def __call__(self, input_tensor, srf):
        return F.conv2d(input_tensor, srf, None)


class dip():
    def __init__(self, args, psf, srf, blind):
        self.args = args
        self.lr_hsi = blind.tensor_lr_hsi.to(args.device)
        self.hr_msi = blind.tensor_hr_msi.to(args.device)
        self.gt = blind.gt

        psf_est = np.reshape(psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor))
        self.psf_est = torch.tensor(psf_est).to(self.args.device).float()
        srf_est = np.reshape(srf.T, newshape=(srf.shape[1], srf.shape[0], 1, 1))
        self.srf_est = torch.tensor(srf_est).to(self.args.device).float()

        self.psf_down = PSF_down()
        self.srf_down = SRF_down()

        self.net = build_admm_unfolding_net(
            self.lr_hsi, self.hr_msi, self.args, self.psf_est, self.srf_est, num_stages=3
        )

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + 1 - self.args.niter3_dip) / float(self.args.niter_decay3_dip + 1)

        # ?????????????????
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr_stage3_dip * 0.5)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

    def train(self):
        flag_best_fhsi = [10, 0, 'data', 0]
        L1Loss = nn.L1Loss(reduction='mean')

        for epoch in range(1, self.args.niter3_dip + self.args.niter_decay3_dip + 1):
            self.optimizer.zero_grad()

            Z_final, X_prior, E_full, A_fused_hr = self.net(self.lr_hsi, self.hr_msi)

            self.endmember = E_full
            self.abundance = A_fused_hr.squeeze(0)
            self.hr_hsi_rec = Z_final

            # ================= ???????? Loss =================
            # ??????????(??TV, ?????)??????????
            lr_hsi_frec = self.psf_down(Z_final, self.psf_est, self.args.scale_factor)
            hr_msi_frec = self.srf_down(Z_final, self.srf_est)

            # ???? (???????????????)
            loss_fhsi = L1Loss(self.lr_hsi, lr_hsi_frec) * 5.0
            loss_fmsi = L1Loss(self.hr_msi, hr_msi_frec) * 5.0

            # ADMM ?? (?? INR ???????)
            loss_coupling = L1Loss(Z_final, X_prior) * 1.0

            loss = loss_fhsi + loss_fmsi + loss_coupling
            # ========================================================

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 50 == 0:
                with torch.no_grad():
                    print(f"epoch:{epoch} lr:{self.optimizer.param_groups[0]['lr']:.6f}")
                    print(
                        f"Loss => LrHSI: {loss_fhsi.item():.4f}, HrMSI: {loss_fmsi.item():.4f}, Coupling: {loss_coupling.item():.4f}")

                    hr_hsi_rec_numpy = self.hr_hsi_rec.data.cpu().detach().numpy()[0].transpose(1, 2, 0)

                    # ??????? GT
                    sam, psnr, ergas, cc, rmse, Ssim, Uqi = MetricsCal(
                        self.gt, hr_hsi_rec_numpy, self.args.scale_factor
                    )
                    information = f"Metric vs GT => SAM: {sam:.4f}, PSNR: {psnr:.4f}, ERGAS: {ergas:.4f}, CC: {cc:.4f}"
                    print(information)
                    print('--------------------------------')

                    if sam < flag_best_fhsi[0] and psnr > flag_best_fhsi[1]:
                        flag_best_fhsi[0] = sam
                        flag_best_fhsi[1] = psnr
                        flag_best_fhsi[2] = self.hr_hsi_rec
                        flag_best_fhsi[3] = epoch

        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out_fhsi_S3.mat'),
                         {'Out': flag_best_fhsi[2].data.cpu().numpy()[0].transpose(1, 2, 0)})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Endmember.mat'), {'end': self.endmember.data.cpu().numpy()})
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Abundance.mat'), {'abun': self.abundance.data.cpu().numpy()})

        return flag_best_fhsi[2]