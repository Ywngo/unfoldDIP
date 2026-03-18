import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_coord_2d(h, w, device=None):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coord = torch.stack([yy, xx], dim=-1).view(-1, 2)
    return coord


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, hidden_layers=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        orig = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        x = self.net(x)
        return x.view(*orig, -1)


class FactorizedINR2D(nn.Module):
    """
    ���C���������� 2D INR������ abundance / ����������

    ����:
        - ���������������������� B �� R^{M��H0��W0}��������������������������
        - ���������� feat �������� MLP �������������� M ������ ��(y,x, :)��
        - ������ ��_m ��_m(y,x) * B_m(y,x)��
      ������������������ B ���������� learnable �� 2D ���������������� H,W ��������

    ����:
        feat: (B, C_in, H, W)

    ����:
        out:  (B, C_out, H, W)
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        M: int = 32,             # ������
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        base_size: int = 32,     # �������������� (H0=W0)
        use_pe: bool = True,
        pe_dim: int = 16,
    ):
        super().__init__()
        self.C_in = dim
        self.C_out = out_dim
        self.M = M
        self.base_size = base_size
        self.use_pe = use_pe
        self.pe_dim = pe_dim

        # ������: (M, 1, H0, W0)�������������� (H,W)
        self.bases = nn.Parameter(
            torch.randn(M, 1, base_size, base_size) * 0.1
        )

        in_dim = dim
        if use_pe:
            in_dim += pe_dim

        # �� feat(+PE) -> ��(y,x,:)�������������������� M ����������
        self.mlp_alpha = SimpleMLP(in_dim, M, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

        # ���� M ���������� C_out ����������������
        # ��������������������: (��_m) -> (C_out)
        self.lin_out = nn.Linear(M, out_dim)

    def positional_encoding(self, coord):
        if self.pe_dim <= 0:
            return None
        device = coord.device
        D = self.pe_dim // 2
        freq_y = 2 ** torch.arange(D, device=device, dtype=torch.float32) * math.pi
        freq_x = 2 ** torch.arange(D, device=device, dtype=torch.float32) * math.pi
        y, x = coord[:, 0:1], coord[:, 1:2]
        pe_y = torch.cat([torch.sin(y * freq_y), torch.cos(y * freq_y)], dim=-1)
        pe_x = torch.cat([torch.sin(x * freq_x), torch.cos(x * freq_x)], dim=-1)
        pe = torch.cat([pe_y, pe_x], dim=-1)
        return pe[:, : self.pe_dim]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        device = feat.device

        # �������� PE
        coord = make_coord_2d(H, W, device=device).unsqueeze(0).expand(B, -1, -1)  # (B,H*W,2)

        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)                  # (B,H*W,C)

        inputs = [feat_flat]
        if self.use_pe:
            pe = self.positional_encoding(coord.view(-1, 2)).view(B, H * W, -1)
            inputs.append(pe)

        x_in = torch.cat(inputs, dim=-1)                                           # (B,H*W,C+PE)

        alpha = self.mlp_alpha(x_in)                                              # (B,H*W,M)
        alpha = alpha.view(B, H, W, self.M)                                       # (B,H,W,M)

        # ������������ (H,W): bases: (M,1,H0,W0) -> (M,1,H,W)
        bases_resized = F.interpolate(
            self.bases, size=(H, W), mode="bilinear", align_corners=False
        )                                                                         # (M,1,H,W)

        # ����: ���������� (y,x)������ M ���������������� alpha(y,x,:) ����
        # bases: (M,1,H,W) -> (1,M,H,W) -> (1,H,W,M)
        bases_hw_m = bases_resized.permute(0, 2, 3, 1).unsqueeze(0)               # (1,M,H,W,1)
        bases_hw_m = bases_hw_m.squeeze(-1).permute(0, 2, 3, 1)                   # (1,H,W,M)

        # broadcast �� batch
        bases_hw_m = bases_hw_m.expand(B, H, W, self.M)                           # (B,H,W,M)

        # ������ + ������������������ M ����������
        comb = (alpha * bases_hw_m)                                               # (B,H,W,M)

        # ���������������� C_out ��
        comb_flat = comb.view(B * H * W, self.M)                                  # (B*H*W,M)
        out_flat = self.lin_out(comb_flat)                                        # (B*H*W,C_out)
        out = out_flat.view(B, H, W, self.C_out).permute(0, 3, 1, 2)             # (B,C_out,H,W)

        return out