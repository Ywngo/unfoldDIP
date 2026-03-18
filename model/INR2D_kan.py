import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.INR1D_kan import KANLinear
from model.INR2D import make_coord_2d


# --- 2D �������� ---
class KAN_INR2D(nn.Module):
    def __init__(self, dim, out_dim, hidden_dim=64, hidden_layers=2, use_pe=True, pe_dim=16):
        super().__init__()
        self.C_in = dim
        self.C_out = out_dim
        self.use_pe = use_pe
        self.pe_dim = pe_dim

        in_dim = dim
        if use_pe:
            in_dim += pe_dim

        self.norm = nn.LayerNorm(in_dim)

        layers = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(KANLinear(d, hidden_dim))
            d = hidden_dim
        layers.append(KANLinear(d, out_dim))

        self.net = nn.Sequential(*layers)

    # ���������� BasicINR2D �� positional_encoding �� make_coord_2d
    def positional_encoding(self, coord):
        # ... (����������������������������������������) ...
        if self.pe_dim <= 0: return None
        device = coord.device
        D = self.pe_dim // 2
        freq_y = 2 ** torch.arange(D, device=device) * math.pi
        freq_x = 2 ** torch.arange(D, device=device) * math.pi
        y, x = coord[:, 0:1], coord[:, 1:2]
        pe = torch.cat([torch.sin(y * freq_y), torch.cos(y * freq_y),
                        torch.sin(x * freq_x), torch.cos(x * freq_x)], dim=-1)
        return pe[:, :self.pe_dim]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        device = feat.device

        # 1. ��������
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        coord = make_coord_2d(H, W, device=device).unsqueeze(0).expand(B, -1, -1)

        if self.use_pe:
            pe = self.positional_encoding(coord.view(-1, 2)).view(B, H * W, -1)
            inp = torch.cat([feat_flat, pe], dim=-1)
        else:
            inp = feat_flat

        # 2. KAN ����
        inp = self.norm(inp)  # ������
        out_flat = self.net(inp)

        return out_flat.view(B, H, W, self.C_out).permute(0, 3, 1, 2)