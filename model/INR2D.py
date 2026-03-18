# -*- coding: utf-8 -*-
"""
INR2D �� Implicit Neural Representation for 2D Continuous Spatial Super-Resolution.

Provides LIIF-style continuous coordinate querying with:
  - Sinusoidal positional encoding
  - Local ensemble (area-weighted blending from neighboring feature cells)
  - Optional 3��3 feature unfolding for local context
  - Optional cell-decode for arbitrary-scale awareness

Usage:
    model = INR2D(dim=64, out_dim=32, hidden_dim=128, hidden_layers=3, L=4)
    # External query mode (arbitrary scale):
    out = model.query_2D(feat_lr, coord_hr, cell=cell_hr, target_h=H, target_w=W)
    # Convenience forward (fixed 2�� upscale):
    out = model(feat_lr)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def make_coord(shape, ranges=None, flatten=True, device=None):
    """
    Generate a coordinate grid for a given spatial shape.
    Each coordinate is placed at the **center** of its pixel cell.

    Args:
        shape: tuple (H, W)
        ranges: list of (min, max) per dimension; defaults to (-1, 1)
        flatten: if True return (H*W, 2), else (H, W, 2)
        device: torch device

    Returns:
        Tensor of coordinates.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = (-1, 1) if ranges is None else ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class _MLP(nn.Module):
    """Simple fully-connected network used as the implicit decoder."""

    def __init__(self, in_dim, out_dim, hidden_dim=256, hidden_layers=4):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        leading = x.shape[:-1]
        return self.net(x.view(-1, x.shape[-1])).view(*leading, -1)


class INR2D(nn.Module):
    """
    LIIF-style Implicit Neural Representation for 2D spatial super-resolution.

    Given a low-resolution feature map (B, C, h, w), produce a high-resolution
    output at arbitrary target coordinates.

    Args:
        dim:            input feature channel count C
        out_dim:        output channel count per query point
        hidden_dim:     MLP hidden layer width
        hidden_layers:  number of MLP hidden layers
        L:              number of positional-encoding frequency levels
        local_ensemble: use area-weighted blending from 4 neighboring cells
        feat_unfold:    concatenate 3��3 neighborhood features (��9 channels)
        cell_decode:    append relative cell size to MLP input (for multi-scale)
    """

    def __init__(
        self,
        dim,
        out_dim,
        hidden_dim=256,
        hidden_layers=3,
        L=4,
        local_ensemble=True,
        feat_unfold=False,
        cell_decode=True,
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.L = L

        # ---------- compute MLP input dimension ----------
        imnet_in_dim = dim * (9 if self.feat_unfold else 1)
        imnet_in_dim += 2          # relative coordinates
        imnet_in_dim += 4 * L      # positional encoding (sin+cos for 2 dims, L freqs each)
        if self.cell_decode:
            imnet_in_dim += 2      # cell size

        self.imnet = _MLP(imnet_in_dim, out_dim, hidden_dim, hidden_layers)

    # ------------------------------------------------------------------
    # Positional encoding
    # ------------------------------------------------------------------
    def positional_encoding(self, x):
        """
        Sinusoidal positional encoding.

        Args:
            x: (..., D)  �� typically D=2 for 2D coordinates
        Returns:
            (..., D*2*L)
        """
        device = x.device
        shape = x.shape                # (..., D)
        freq = (2 ** torch.arange(self.L, dtype=torch.float32, device=device)) * np.pi   # (L,)
        spectrum = x[..., None] * freq                          # (..., D, L)
        sin_enc = spectrum.sin()                                # (..., D, L)
        cos_enc = spectrum.cos()                                # (..., D, L)
        enc = torch.stack([sin_enc, cos_enc], dim=-2)           # (..., D, 2, L)
        return enc.reshape(*shape[:-1], -1)                     # (..., D*2*L)

    # ------------------------------------------------------------------
    # Core query function �� arbitrary target resolution
    # ------------------------------------------------------------------
    def query_2D(self, feat, coord, cell=None, target_h=None, target_w=None):
        """
        Query the implicit representation at continuous 2D coordinates.

        Args:
            feat:     (B, C, h, w)  �� low-resolution feature map
            coord:    (B, N, 2)     �� query coordinates in [-1, 1]
            cell:     (B, N, 2)     �� relative cell size (required when cell_decode=True)
            target_h: int           �� target output height  (required for reshape)
            target_w: int           �� target output width   (required for reshape)

        Returns:
            (B, out_dim, target_h, target_w)
        """
        B, C, h, w = feat.shape
        N = coord.shape[1]
        device = feat.device

        # --- optional: unfold 3��3 neighbourhood ---
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(B, C * 9, h, w)

        # --- LR feature-centre coordinate map ---
        feat_coord = (
            make_coord((h, w), flatten=False, device=device)       # (h, w, 2)
            .permute(2, 0, 1)                                       # (2, h, w)
            .unsqueeze(0).expand(B, 2, h, w)                        # (B, 2, h, w)
        )

        # --- local-ensemble offsets ---
        vx_lst = [-1, 1] if self.local_ensemble else [0]
        vy_lst = [-1, 1] if self.local_ensemble else [0]
        rx, ry = 1.0 / h, 1.0 / w
        eps = 1e-6

        preds, areas = [], []

        for vx in vx_lst:
            for vy in vy_lst:
                # shifted query coordinates
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps
                coord_[:, :, 1] += vy * ry + eps
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # grid_sample expects (x, y) but coord is (row, col) �� flip
                grid = coord_[:, :, :2].flip(-1).unsqueeze(1)      # (B, 1, N, 2)

                # sample features & LR coordinates at shifted queries
                q_feat  = F.grid_sample(feat,       grid, mode='bilinear', align_corners=True
                                        )[:, :, 0, :].permute(0, 2, 1)       # (B, N, C[*9])
                q_coord = F.grid_sample(feat_coord, grid, mode='bilinear', align_corners=True
                                        )[:, :, 0, :].permute(0, 2, 1)       # (B, N, 2)

                # relative coordinate (scaled to pixel units)
                rel_coord = coord[:, :, :2] - q_coord              # (B, N, 2)
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                # positional encoding of the **original** query coordinate
                pe = self.positional_encoding(coord[:, :, :2])      # (B, N, 4L)

                # assemble MLP input
                inp = torch.cat([q_feat, rel_coord, pe], dim=-1)    # (B, N, C+2+4L)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= h
                    rel_cell[:, :, 1] *= w
                    inp = torch.cat([inp, rel_cell], dim=-1)        # (B, N, ...+2)

                pred = self.imnet(inp)                              # (B, N, out_dim)
                preds.append(pred)

                # area weight for blending
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]) + 1e-9
                areas.append(area)                                  # (B, N)

        # --- weighted blending (cross-swap for correct LIIF weighting) ---
        if self.local_ensemble and len(areas) == 4:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        tot_area = torch.stack(areas, dim=0).sum(dim=0)             # (B, N)
        ret = sum(p * (a / tot_area).unsqueeze(-1) for p, a in zip(preds, areas))

        # reshape to spatial grid
        if target_h is not None and target_w is not None:
            ret = ret.permute(0, 2, 1).view(B, -1, target_h, target_w)
        else:
            ret = ret.permute(0, 2, 1).view(B, -1, N)              # keep flat if shape unknown

        return ret

    # ------------------------------------------------------------------
    # Convenience forward �� generates HR coords for a given scale
    # ------------------------------------------------------------------
    def forward(self, inp, scale=2):
        """
        Convenience entry point: upscale by an integer or float factor.

        Args:
            inp:   (B, C, H, W) �� LR feature map
            scale: float         �� upsampling factor (default 2)

        Returns:
            (B, out_dim, H*scale, W*scale)
        """
        device = inp.device
        B, _, H, W = inp.shape
        target_h = int(round(H * scale))
        target_w = int(round(W * scale))

        # HR coordinate grid
        coord = make_coord((target_h, target_w), device=device)             # (N, 2)
        coord = coord.unsqueeze(0).expand(B, -1, -1)                        # (B, N, 2)

        # cell size (how large each HR pixel is in normalised coords)
        cell = torch.zeros(B, coord.shape[1], 2, device=device)
        cell[:, :, 0] = 2.0 / target_h
        cell[:, :, 1] = 2.0 / target_w

        return self.query_2D(inp, coord, cell, target_h=target_h, target_w=target_w)


# ======================== quick sanity check ========================
if __name__ == "__main__":
    model = INR2D(dim=64, out_dim=3, hidden_dim=128, hidden_layers=3, L=4,
                  local_ensemble=True, cell_decode=True).cuda()
    x = torch.randn(1, 64, 32, 32).cuda()

    # fixed 2�� upscale
    y2 = model(x, scale=2)
    print(f"2�� upscale: {x.shape} -> {y2.shape}")   # (1,3,64,64)

    # arbitrary 3.5�� upscale
    y35 = model(x, scale=3.5)
    print(f"3.5�� upscale: {x.shape} -> {y35.shape}") # (1,3,112,112)

    # external query mode (e.g., from unrolled optimizer)
    coord = make_coord((128, 128), device=x.device).unsqueeze(0)
    cell = torch.zeros(1, 128*128, 2, device=x.device)
    cell[:, :, 0] = 2.0 / 128
    cell[:, :, 1] = 2.0 / 128
    y_ext = model.query_2D(x, coord, cell, target_h=128, target_w=128)
    print(f"External query: {x.shape} -> {y_ext.shape}")  # (1,3,128,128)