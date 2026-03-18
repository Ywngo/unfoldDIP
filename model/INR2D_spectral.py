import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_coord_2d(h, w, device=None):
    ys = torch.linspace(-1, 1, steps=h, device=device)
    xs = torch.linspace(-1, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coord = torch.stack([yy, xx], dim=-1).view(-1, 2)  # (H*W,2)
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
        orig_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        x = self.net(x)
        return x.view(*orig_shape, -1)


class SpectralINR2D(nn.Module):
    """
    ���������� 2D INR������ abundance/������������

    ����:
        feat: (B, C_in, H, W)

    ����:
        out:  (B, C_out, H, W)

    ��������:
        - ���� 1x1+3x3 �������� local feature��
        - ���������� 2D FFT ��������������������������������
        - �� local + ���� pool �������������������������� PE ���� MLP��
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        use_pe: bool = True,
        pe_dim: int = 16,
        use_freq: bool = True,
    ):
        super().__init__()
        self.C_in = dim
        self.C_out = out_dim
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.use_freq = use_freq

        # ������������
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        in_dim = hidden_dim
        if use_freq:
            # ������������������������������ (hidden_dim,) ����������
            in_dim += hidden_dim
        if use_pe:
            in_dim += pe_dim

        self.mlp = SimpleMLP(in_dim, out_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

    def positional_encoding(self, coord):
        """
        coord: (H*W, 2), [-1,1]
        return: (H*W, pe_dim)
        """
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

    def _build_freq_channel(self, local: torch.Tensor) -> torch.Tensor:
        """
        local: (B, Hd, H, W)
        return: (B, Hd)   # ����������������������
        """
        B, Hd, H, W = local.shape
        # ������������ 2D FFT������������������������
        fft2 = torch.fft.fft2(local, dim=(-2, -1))            # (B,Hd,H,W)
        mag = torch.abs(fft2)
        freq_stat = mag.mean(dim=(-2, -1))                    # (B,Hd)
        return freq_stat                                      # (B,Hd)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        device = feat.device

        local = self.local_conv(feat)                         # (B,Hd,H,W)
        Hd = local.shape[1]

        # (B,H*W,Hd)
        local_flat = local.permute(0, 2, 3, 1).reshape(B, H * W, Hd)

        inputs = [local_flat]

        if self.use_freq:
            freq_stat = self._build_freq_channel(local)       # (B,Hd)
            # ������������������ local_flat ����
            freq_broadcast = freq_stat.unsqueeze(1).expand(B, H * W, Hd)
            inputs.append(freq_broadcast)

        coord = make_coord_2d(H, W, device=device).unsqueeze(0).expand(B, -1, -1)
        if self.use_pe:
            pe = self.positional_encoding(coord.view(-1, 2)).view(B, H * W, -1)
            inputs.append(pe)

        x_in = torch.cat(inputs, dim=-1)                      # (B,H*W, in_dim)

        out_flat = self.mlp(x_in)                             # (B,H*W,C_out)
        out = out_flat.view(B, H, W, self.C_out).permute(0, 3, 1, 2)
        return out