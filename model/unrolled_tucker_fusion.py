# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. ������������ (������������ [-1, 1] ����������)
# =========================================================================
def make_coord_2d(h, w, device=None):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    coord = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (H*W, 2), ���� x,y ���������� grid_sample
    return coord


def make_coord_1d(l, device=None):
    coords = torch.linspace(-1, 1, steps=l, device=device).view(-1, 1)  # (L, 1)
    return coords


# =========================================================================
# 2. ������������������LIIF (UNet + MLP)
# =========================================================================
class SimpleUNet(nn.Module):
    """������������ U-Net������������������������������"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(128 + 64, out_ch, 3, 1, 1), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(
            torch.cat([e1, F.interpolate(e2, size=e1.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        return d1


class Spatial_LIIF_Tucker(nn.Module):
    """
    ���������������������������������������������� U (B, N, Rs)
    """

    def __init__(self, in_channels, Rs, feat_dim=64):
        super().__init__()
        self.unet = SimpleUNet(in_channels, feat_dim)
        # LIIF MLP: ���� [��������(feat_dim) + ��������(2)]
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 2, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, Rs)  # ���������������� Rs
        )

    def forward(self, img_input, coords_2d):
        B = img_input.shape[0]
        N = coords_2d.shape[1]

        # 1. �������������������� (B, feat_dim, H, W)
        feat_map = self.unet(img_input)

        # 2. �������� (Grid Sample)
        grid = coords_2d.unsqueeze(1)  # (B, 1, N, 2)
        local_feat = F.grid_sample(feat_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
        local_feat = local_feat.squeeze(2).permute(0, 2, 1)  # (B, N, feat_dim)

        # 3. �������������������� MLP ������������
        mlp_in = torch.cat([local_feat, coords_2d], dim=-1)
        U_spatial = self.mlp(mlp_in)  # (B, N, Rs)

        return U_spatial


# =========================================================================
# 3. ������������������Spectral-INR
# =========================================================================
class Spectral_INR_Tucker(nn.Module):
    """
    ������������ INR�������������� V (L, Rl)
    """

    def __init__(self, Rl):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, Rl)  # ���������������� Rl
        )

    def forward(self, coords_1d):
        # coords_1d: (L, 1)
        V_spectral = self.mlp(coords_1d)  # (L, Rl)
        return V_spectral


# =========================================================================
# 4. �������������� (Prior Projection Step) - Tucker ����
# =========================================================================
class ContinuousTuckerProjection(nn.Module):
    def __init__(self, in_channels_unet, Rl=10, Rs=32):
        super().__init__()
        self.spatial_net = Spatial_LIIF_Tucker(in_channels_unet, Rs=Rs)
        self.spectral_net = Spectral_INR_Tucker(Rl=Rl)

        # �������� Tucker �������� G (Rl, Rs)
        self.core_G = nn.Parameter(torch.empty(1, Rl, Rs))
        nn.init.xavier_normal_(self.core_G)

    def forward(self, S_k, coords_2d, coords_1d, target_shape):
        B, C, H, W = target_shape

        # 1. �������� U (B, N, Rs)
        U = self.spatial_net(S_k, coords_2d)

        # 2. �������� V (1, L, Rl)
        V = self.spectral_net(coords_1d).unsqueeze(0).expand(B, -1, -1)

        # 3. Tucker ���� Z = V * G * U^T
        # VG = V @ G -> (B, L, Rl) @ (1, Rl, Rs) -> (B, L, Rs)
        VG = torch.matmul(V, self.core_G)

        # Z_flat = VG @ U^T -> (B, L, Rs) @ (B, Rs, N) -> (B, L, N)
        Z_flat = torch.matmul(VG, U.transpose(1, 2))

        # 4. �������������������������� [0, 1] ����
        Z_flat = torch.sigmoid(Z_flat)

        # ��������������������������
        Z_img = Z_flat.view(B, -1, H, W)
        return Z_img


# =========================================================================
# 5. �������������������������� (Unrolled Proximal Gradient Network)
# =========================================================================
class UnrolledTuckerFusion(nn.Module):
    """
    ���� IR&ArF �������������� ��������(Data Consistency) �� ��������(Prior Projection)
    """

    def __init__(self, L_bands, K_iters=3, Rl=10, Rs=32):
        super().__init__()
        self.K = K_iters
        # ������������������ (Learnable)
        self.eta = nn.Parameter(torch.tensor([0.1] * K_iters))

        # ���� K �� Tucker ������ (��������������������������)
        self.projectors = nn.ModuleList([
            ContinuousTuckerProjection(in_channels_unet=L_bands, Rl=Rl, Rs=Rs)
            for _ in range(self.K)
        ])

    def spatial_degradation(self, x, PSF, scale_factor):
        """�������� H(X)������ PSF ������������"""
        # ���������������� PSF ������������������ PSF ���������������� F.conv2d
        # ������������������ F.interpolate ������������������
        blurred = F.conv2d(x, PSF, padding=PSF.shape[-1] // 2, groups=x.shape[1]) if PSF is not None else x
        return F.interpolate(blurred, scale_factor=1 / scale_factor, mode='bilinear', align_corners=False)

    def spatial_transpose(self, x, PSF, target_size):
        """������������ H^T(X)�������������� PSF ��������"""
        up = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        # ��������������������������������������������������������
        return F.conv2d(up, PSF, padding=PSF.shape[-1] // 2, groups=x.shape[1]) if PSF is not None else up

    def spectral_degradation(self, x, SRF):
        """�������� R(X)������ SRF ����������������"""
        # SRF shape: (msi_bands, hsi_bands). x shape: (B, hsi_bands, H, W)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        msi_flat = torch.matmul(x_flat, SRF.T)  # (B, N, msi_bands)
        return msi_flat.permute(0, 2, 1).view(B, -1, H, W)

    def spectral_transpose(self, x, SRF):
        """������������ R^T(X)"""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        hsi_flat = torch.matmul(x_flat, SRF)  # (B, N, hsi_bands)
        return hsi_flat.permute(0, 2, 1).view(B, -1, H, W)

    def forward(self, Y_hsi, Y_msi, PSF=None, SRF=None):
        B, L, H_lr, W_lr = Y_hsi.shape
        B, C_m, H_hr, W_hr = Y_msi.shape
        scale_factor = H_hr // H_lr
        device = Y_hsi.device

        # ������������ (��������������)
        coords_2d = make_coord_2d(H_hr, W_hr, device).unsqueeze(0).expand(B, -1, -1)  # (B, H_hr*W_hr, 2)
        coords_1d = make_coord_1d(L, device)  # (L, 1)

        # ������ X_0 (������������ HSI ��������)
        X_k = F.interpolate(Y_hsi, size=(H_hr, W_hr), mode='bilinear', align_corners=False)

        for k in range(self.K):
            # ---------------------------------------------------------
            # ��������Data Consistency Step (����������������)
            # ---------------------------------------------------------
            # 1. HSI ����
            diff_hsi = self.spatial_degradation(X_k, PSF, scale_factor) - Y_hsi
            grad_hsi = self.spatial_transpose(diff_hsi, PSF, (H_hr, W_hr))

            # 2. MSI ����
            diff_msi = self.spectral_degradation(X_k, SRF) - Y_msi
            grad_msi = self.spectral_transpose(diff_msi, SRF)

            # 3. ������������������ S_k
            # S_k = X_k - eta * (Grad_HSI + Grad_MSI)
            S_k = X_k - self.eta[k] * (grad_hsi + grad_msi)

            # ---------------------------------------------------------
            # ��������Prior Projection Step (Tucker ��������������)
            # ---------------------------------------------------------
            X_k = self.projectors[k](S_k, coords_2d, coords_1d, target_shape=(B, L, H_hr, W_hr))

        return X_k


# =========================================================================
# 6. ���������������� (Test Block)
# =========================================================================
if __name__ == "__main__":
    import time
    from thop import profile, clever_format

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ������������
    B = 1
    L_bands = 93  # ���� Pavia Center ��������
    C_msi = 4  # MSI ������
    H_lr, W_lr = 42, 42  # LrHSI ����
    scale = 4
    H_hr, W_hr = H_lr * scale, W_lr * scale  # 168x168

    print("=" * 50)
    print("Initialize Unrolled Tucker Fusion Network")
    print("=" * 50)

    # ����������: ����3���������� Rl=10�������� Rs=64
    model = UnrolledTuckerFusion(L_bands=L_bands, K_iters=3, Rl=10, Rs=64).to(device)

    # ����������
    Y_hsi = torch.rand(B, L_bands, H_lr, W_lr).to(device)
    Y_msi = torch.rand(B, C_msi, H_hr, W_hr).to(device)

    # ���� SRF ���� (C_msi, L_bands)
    SRF = torch.rand(C_msi, L_bands).to(device)
    SRF = SRF / SRF.sum(dim=1, keepdim=True)  # ������

    # PSF ������������������������������ (L_bands, 1, K, K)
    PSF = None

    # ������������
    start_time = time.time()
    out_hrhsi = model(Y_hsi, Y_msi, PSF, SRF)
    end_time = time.time()

    print(f"Forward Pass Success! Output shape: {out_hrhsi.shape} (Expected: {B, L_bands, H_hr, W_hr})")
    print(f"Physical Constraint Check: Min={out_hrhsi.min().item():.4f}, Max={out_hrhsi.max().item():.4f}")
    print(f"Inference Time: {end_time - start_time:.4f} s")

    try:
        flops, params = profile(model, inputs=(Y_hsi, Y_msi, PSF, SRF), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        print("-" * 50)
        print(f"Total Parameters: {params_str}")
        print(f"Total FLOPs:      {flops_str}")
        print("=" * 50)
    except Exception as e:
        print("Install 'thop' to calculate FLOPs: pip install thop")