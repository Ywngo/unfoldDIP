import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.INR2D import make_coord_2d


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class Grid_INR2D(nn.Module):
    def __init__(self, dim, out_dim, hidden_dim=64):
        super().__init__()
        self.C_in = dim
        self.C_out = out_dim

        # �������������� (4��������)
        # ������������������������������������
        self.grids = nn.ParameterList([
            nn.Parameter(torch.randn(1, 16, 32, 32) * 0.1),  # Coarse
            nn.Parameter(torch.randn(1, 16, 64, 64) * 0.1),
            nn.Parameter(torch.randn(1, 16, 128, 128) * 0.1),
            nn.Parameter(torch.randn(1, 16, 256, 256) * 0.1)  # Fine (���� WaDC 256)
        ])

        # Grid ���������� = 16 * 4 = 64
        # �������� = dim (��������) + 64 (Grid����)
        total_in_dim = dim + 16 * 4

        self.mlp = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        device = feat.device

        # 1. �������� grid [-1, 1] ���� grid_sample
        # (B, H, W, 2)
        coord = make_coord_2d(H, W, device=device).view(H, W, 2).unsqueeze(0).expand(B, -1, -1, -1)

        # 2. ����������������������
        grid_feats = []
        for g in self.grids:
            # g: (1, 16, res, res) -> expand to B
            g_batch = g.expand(B, -1, -1, -1)
            # sample: (B, 16, H, W)
            # align_corners=True ������
            sampled = F.grid_sample(g_batch, coord, align_corners=True, mode='bilinear')
            grid_feats.append(sampled)

        # 3. ���� Grid ����
        # (B, 64, H, W)
        local_feat = torch.cat(grid_feats, dim=1)

        # 4. ���������������� feat
        # (B, C+64, H, W)
        combined = torch.cat([feat, local_feat], dim=1)

        # 5. MLP ����
        # (B, H, W, C+64)
        inp_flat = combined.permute(0, 2, 3, 1)
        out_flat = self.mlp(inp_flat)  # (B, H, W, out_dim)

        return out_flat.permute(0, 3, 1, 2)