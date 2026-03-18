import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- ������������ ---
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


class FactorizedINR2D_V3_Spectral(nn.Module):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            M: int = 32,  # ���� M ���������� "��������" (Endmembers)
            hidden_dim: int = 128,
            hidden_layers: int = 4,
            base_size: int = 32,  # ����������������������������������������������
            use_pe: bool = True,
            pe_dim: int = 16,
    ):
        super().__init__()
        self.C_in = dim
        self.C_out = out_dim
        self.M = M
        self.use_pe = use_pe
        self.pe_dim = pe_dim

        # 1. �������� (Spectral Endmembers)
        # ����: (M, Out_Dim) -> (������, ������)
        # ����������������������������
        self.spectral_bases = nn.Parameter(torch.randn(M, out_dim))
        nn.init.orthogonal_(self.spectral_bases)

        # 2. ������������ (Abundance Predictor)
        in_dim = dim
        if use_pe:
            in_dim += pe_dim
        self.mlp_abundance = SimpleMLP(in_dim, M, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

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

        # --- �������� ---
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        inputs = [feat_flat]
        if self.use_pe:
            coord = make_coord_2d(H, W, device=device).unsqueeze(0).expand(B, -1, -1)
            pe = self.positional_encoding(coord.view(-1, 2)).view(B, H * W, -1)
            inputs.append(pe)

        x_in = torch.cat(inputs, dim=-1)

        # --- �������� (Abundance) ---
        # ��������: (B, HW, M)
        abundance = self.mlp_abundance(x_in)

        # [����] ��������: ��������������1 (ASC: Abundance Sum-to-one Constraint)
        # ������������ SAM ����
        abundance = F.softmax(abundance, dim=-1)

        # --- �������� (Linear Mixing) ---
        # ����: Y = A * E
        # (B, HW, M) @ (M, Out_Dim) -> (B, HW, Out_Dim)
        out_flat = torch.matmul(abundance, self.spectral_bases)

        # ��������
        out = out_flat.view(B, H, W, self.C_out).permute(0, 3, 1, 2)

        return out