# -*- coding: utf-8 -*-
"""
INR1D �� Implicit Neural Representation for 1D Continuous Signal Super-Resolution.

Provides LIIF-style continuous coordinate querying on 1D signals with:
  - Sinusoidal positional encoding
  - Local ensemble (area-weighted blending from neighboring feature positions)
  - Optional cell-decode for arbitrary-scale awareness
  - Support for Q �� N (query count differs from feature count)

Designed for spectral-dimension super-resolution in HSI-MSI fusion:
  Input:  (B, N_in, C)  �� spectral features at N_in band positions
  Query:  (B, Q, 1)     �� continuous spectral coordinates for Q target bands
  Output: (B, Q, out_dim) �� spectral factor at arbitrary spectral resolution

Usage:
    model = INR1D(in_dim=32, out_dim=20, hidden_dim=128, hidden_layers=4, L=4)
    # Arbitrary spectral query:
    out = model.query(feat, coord_target, cell=cell_target)
    # Default forward (Q == N, self-reconstruction):
    out = model(feat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_coord_1d(length, range_=(-1, 1), flatten=True, device=None):
    """
    Generate pixel-centre coordinates in a given range for 1D signal.

    Args:
        length:  number of positions
        range_:  (min, max) coordinate range
        flatten: if True return (length, 1), else (length,)
        device:  torch device

    Returns:
        Tensor of 1D coordinates at pixel centres.
    """
    v0, v1 = range_
    r = (v1 - v0) / (2 * length)
    seq = v0 + r + (2 * r) * torch.arange(length, dtype=torch.float32, device=device)
    if flatten:
        return seq.view(-1, 1)
    else:
        return seq


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


class INR1D(nn.Module):
    """
    LIIF-style Implicit Neural Representation for 1D continuous signal
    super-resolution (e.g., spectral dimension).

    Args:
        in_dim:        input feature channel count C
        out_dim:       output channel count per query point
        hidden_dim:    MLP hidden layer width
        hidden_layers: number of MLP hidden layers
        L:             number of positional-encoding frequency levels
        local_ensemble: use area-weighted blending from neighboring positions
        cell_decode:   append cell size to MLP input (for arbitrary-scale)
    """

    def __init__(self, in_dim, out_dim, hidden_dim=256, hidden_layers=4,
                 L=4, local_ensemble=True, cell_decode=True):
        super().__init__()
        self.L = L
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        # MLP input: feature(C) + rel_coord(1) + pos_enc(2L) [+ cell(1)]
        imnet_in_dim = in_dim + 1 + 2 * L
        if self.cell_decode:
            imnet_in_dim += 1

        self.imnet = _MLP(imnet_in_dim, out_dim, hidden_dim, hidden_layers)

    def positional_encoding(self, x):
        """
        Sinusoidal positional encoding.

        Args:
            x: (..., 1) �� 1D coordinates
        Returns:
            (..., 2L) �� encoded coordinates
        """
        device = x.device
        freq = (2 ** torch.arange(self.L, dtype=torch.float32,
                                  device=device)) * np.pi          # (L,)
        spectrum = x * freq                                         # (..., L)
        return torch.cat([spectrum.sin(), spectrum.cos()], dim=-1)   # (..., 2L)

    def _grid_sample_1d(self, feat, coord):
        """
        1D bilinear interpolation of features at continuous coordinates.
        Analogous to F.grid_sample but for 1D signals.

        Args:
            feat:  (B, N, C)  �� features at N uniformly-spaced positions
            coord: (B, Q, 1)  �� query coordinates in [-1, 1]

        Returns:
            (B, Q, C) �� interpolated features at query positions
        """
        B, N, C = feat.shape
        Q = coord.shape[1]

        # Convert normalised coords [-1, 1] to continuous index [0, N-1]
        # coord = -1 �� index 0, coord = 1 �� index N-1
        idx = (coord[..., 0] + 1) / 2 * (N - 1)          # (B, Q)
        idx = idx.clamp(0, N - 1)

        # Floor and ceil indices
        idx_floor = idx.long().clamp(0, N - 1)             # (B, Q)
        idx_ceil = (idx_floor + 1).clamp(0, N - 1)         # (B, Q)

        # Interpolation weight
        w_ceil = (idx - idx_floor.float()).unsqueeze(-1)    # (B, Q, 1)
        w_floor = 1.0 - w_ceil                              # (B, Q, 1)

        # Gather features
        feat_floor = torch.gather(
            feat, 1,
            idx_floor.unsqueeze(-1).expand(B, Q, C)
        )                                                    # (B, Q, C)
        feat_ceil = torch.gather(
            feat, 1,
            idx_ceil.unsqueeze(-1).expand(B, Q, C)
        )                                                    # (B, Q, C)

        return w_floor * feat_floor + w_ceil * feat_ceil     # (B, Q, C)

    def query(self, feat, coord, cell=None):
        """
        Query the implicit representation at arbitrary 1D coordinates.

        Supports Q �� N: features are interpolated at query positions via
        1D bilinear sampling before being fed to the MLP.

        Args:
            feat:  (B, N, C)    �� features at N source positions
            coord: (B, Q, 1)    �� query coordinates in [-1, 1]
            cell:  (B, Q, 1)    �� cell size at each query (or None)

        Returns:
            (B, Q, out_dim)
        """
        B, N, C = feat.shape
        Q = coord.shape[1]
        device = feat.device

        # Feature-position coordinates
        feat_coord = make_coord_1d(N, device=device)                # (N, 1)
        feat_coord = feat_coord.unsqueeze(0).expand(B, N, 1)       # (B, N, 1)

        # Local ensemble offsets
        vx_list = [-1, 1] if self.local_ensemble else [0]
        rx = 1.0 / N
        eps = 1e-6

        preds = []
        areas = []

        for vx in vx_list:
            # Shift query coordinates
            coord_shift = coord + (vx * rx + eps)                   # (B, Q, 1)
            coord_shift = coord_shift.clamp(-1 + 1e-6, 1 - 1e-6)

            # Interpolate features at shifted query positions
            q_feat = self._grid_sample_1d(feat, coord_shift)        # (B, Q, C)

            # Interpolate feature coordinates at shifted queries
            # to compute accurate relative coordinates
            q_feat_coord = self._grid_sample_1d(
                feat_coord, coord_shift
            )                                                        # (B, Q, 1)

            # Relative coordinate (original query - sampled feature position)
            rel_coord = coord - q_feat_coord                         # (B, Q, 1)
            rel_coord_scaled = rel_coord * N                         # (B, Q, 1)

            # Positional encoding of ORIGINAL query coordinate
            pos_enc = self.positional_encoding(coord)                # (B, Q, 2L)

            # Assemble MLP input
            net_in = torch.cat([q_feat, rel_coord_scaled, pos_enc],
                               dim=-1)                               # (B, Q, C+1+2L)

            if self.cell_decode and cell is not None:
                rel_cell = cell * N                                  # (B, Q, 1)
                net_in = torch.cat([net_in, rel_cell], dim=-1)       # (B, Q, ...+1)

            pred = self.imnet(net_in)                                # (B, Q, out_dim)
            preds.append(pred)

            # Area weight for blending
            area = torch.abs(rel_coord_scaled[..., 0]) + 1e-9       # (B, Q)
            areas.append(area)

        # Cross-swap for correct LIIF weighting (1D: swap left?right)
        if self.local_ensemble and len(areas) == 2:
            areas[0], areas[1] = areas[1], areas[0]

        tot_area = torch.stack(areas, dim=0).sum(dim=0)              # (B, Q)
        out = sum(p * (a / tot_area).unsqueeze(-1)
                  for p, a in zip(preds, areas))

        return out                                                    # (B, Q, out_dim)

    def forward(self, inp, target_length=None):
        """
        Convenience entry point.

        Args:
            inp:           (B, N, C) �� input features at N positions
            target_length: int or None �� number of output positions
                           None �� Q = N (self-reconstruction)

        Returns:
            (B, Q, out_dim)
        """
        B, N, C = inp.shape
        device = inp.device

        Q = target_length if target_length is not None else N

        # Generate query coordinates at target resolution
        coord = make_coord_1d(Q, device=device)                     # (Q, 1)
        coord = coord.unsqueeze(0).expand(B, Q, 1)                  # (B, Q, 1)

        # Cell size: how wide each target position is in normalised coords
        cell = torch.ones(B, Q, 1, device=device) * (2.0 / Q)       # (B, Q, 1)

        return self.query(inp, coord, cell)


# ======================== sanity check ========================
if __name__ == "__main__":
    model = INR1D(in_dim=32, out_dim=20, hidden_dim=128,
                  hidden_layers=4, L=4, local_ensemble=True,
                  cell_decode=True).cuda()

    feat = torch.randn(1, 31, 32).cuda()  # 31 bands, 32-dim features

    # Default: Q == N
    out_same = model(feat)
    print(f"Q==N: {feat.shape} -> {out_same.shape}")      # (1, 31, 20)

    # Spectral super-resolution: 31 �� 64 bands
    out_up = model(feat, target_length=64)
    print(f"31��64: {feat.shape} -> {out_up.shape}")        # (1, 64, 20)

    # Spectral down-sampling: 31 �� 16 bands
    out_down = model(feat, target_length=16)
    print(f"31��16: {feat.shape} -> {out_down.shape}")      # (1, 16, 20)

    # External query at arbitrary positions
    coord = torch.linspace(-1, 1, 50).view(1, 50, 1).cuda()
    cell = torch.ones(1, 50, 1).cuda() * (2.0 / 50)
    out_ext = model.query(feat, coord, cell)
    print(f"External 50 queries: {feat.shape} -> {out_ext.shape}")  # (1, 50, 20)