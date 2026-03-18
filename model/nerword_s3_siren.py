import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def make_coord(shape, ranges=None, flatten=True, device=None):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, hidden_layers):
        super().__init__()
        layers = []
        lastv = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(lastv, hidden_dims))
            layers.append(nn.ReLU(inplace=True))
            lastv = hidden_dims
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class INR2D(nn.Module):
    """
    INR2D with multiple weighting strategies:
    ['area', 'cosine', 'cosine_tau', 'gaussian', 'graph'].
    """

    def __init__(
        self,
        dim,
        out_dim,
        scale,
        hidden_dim=256,
        hidden_layers=3,
        L=4,
        local_ensemble=True,
        feat_unfold=False,
        cell_decode=True,
        weight_mode='gaussian',  # 'area' | 'cosine' | 'cosine_tau' | 'gaussian' | 'graph'
        tau_init=0.1,
        sigma=0.5,
        gat_hidden=64,
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.L = L
        self.weight_mode = weight_mode
        self.sigma = sigma
        self.tau = nn.Parameter(torch.tensor(tau_init)) if weight_mode == 'cosine_tau' else None
        self.scale = scale

        # �������������� (GAT-style)
        if weight_mode == 'graph':
            self.W_graph = nn.Linear(dim, gat_hidden, bias=False)
            self.attn_vec = nn.Linear(gat_hidden * 2, 1, bias=False)
            self.leaky_relu = nn.LeakyReLU(0.2)

        # ����MLP
        imnet_in_dim = dim
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2 + 4 * L
        if self.cell_decode:
            imnet_in_dim += 2

        self.imnet = MLP(imnet_in_dim, out_dim, hidden_dim, hidden_layers)

    # -------------------- Position Encoding --------------------
    def positional_encoding(self, input, L):
        device = input.device
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=device) * np.pi
        spectrum = input[..., None] * freq
        sin, cos = spectrum.sin(), spectrum.cos()
        input_enc = torch.cat([sin, cos], dim=-1).view(input.shape[0], input.shape[1], -1)
        return input_enc

    # -------------------- Weight Calculation --------------------
    def compute_weight(self, q_feat_list, rel_coord_list):
        """
        Compute local ensemble weights based on mode.
        """
        K = len(q_feat_list)
        B, N, C = q_feat_list[0].shape
        q_ref = q_feat_list[0].detach()

        if self.weight_mode == 'area':
            areas = [torch.abs(rc[..., 0] * rc[..., 1]) + 1e-9 for rc in rel_coord_list]
            weights = torch.stack(areas, dim=0)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-9)
            return weights

        elif self.weight_mode == 'cosine':
            sims = [F.cosine_similarity(q, q_ref, dim=-1) for q in q_feat_list[1:6]]
            weights = F.softmax(torch.stack(sims, dim=0), dim=0)
            return weights

        elif self.weight_mode == 'cosine_tau':
            sims = [F.cosine_similarity(q, q_ref, dim=-1) for q in q_feat_list[1:6]]
            weights = F.softmax(torch.stack(sims, dim=0) / self.tau.clamp(min=1e-3), dim=0)
            return weights

        elif self.weight_mode == 'gaussian':

            q_feat_stack = torch.stack(q_feat_list[1:6], dim=0)  # shape: [5, B, D]
            distances = (q_feat_stack - q_ref).pow(2).sum(dim=-1)  # shape: [5, B]
            sigma = distances.mean().sqrt()
            sigma = torch.clamp(sigma, min=1e-6)
            dists = -distances / (2 * sigma ** 2)
            weights = F.softmax(dists, dim=0)  # shape: [5, B]
            return weights

        elif self.weight_mode == 'graph':
            # Graph attention: q_ref as query node, q_feat_list as neighbors
            attn_scores = []
            for q in q_feat_list[1:6]:
                h_q = self.W_graph(q_ref)
                h_i = self.W_graph(q)
                a_input = torch.cat([h_q, h_i], dim=-1)
                e = self.leaky_relu(self.attn_vec(a_input)).squeeze(-1)  # (B,N)
                attn_scores.append(e)
            attn_scores = torch.stack(attn_scores, dim=0)  # (K,B,N)
            weights = F.softmax(attn_scores, dim=0)
            return weights

        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

    # -------------------- Query Operation --------------------
    def query_2D(self, feat, coord, cell=None):
        B, C, h, w = feat.shape
        device = feat.device
        feat_coord = make_coord((h, w), flatten=False).to(device)
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(B, 2, h, w)

        vx_lst = [-1, 1] if self.local_ensemble else [0]
        vy_lst = [-1, 1] if self.local_ensemble else [0]
        rx, ry = 1 / h, 1 / w

        preds, q_feats, rel_coords = [], [], []

        temp_coord = coord.clone()
        q_feat_cloest = F.grid_sample(feat, temp_coord[:, :, :2].flip(-1).unsqueeze(1),
                                      mode='bilinear', align_corners=True)[:, :, 0, :].permute(0, 2, 1)
        q_feats.append(q_feat_cloest)

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                # print(coord_[:, :, :2].flip(-1).unsqueeze(1).shape)
                q_feat = F.grid_sample(feat, coord_[:, :, :2].flip(-1).unsqueeze(1),
                                       mode='bilinear', align_corners=True)[:, :, 0, :].permute(0, 2, 1)

                q_coord = F.grid_sample(feat_coord, coord_[:, :, :2].flip(-1).unsqueeze(1),
                                        mode='bilinear', align_corners=True)[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord[:, :, :2] - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                points_enc = self.positional_encoding(coord[:, :, :2], L=self.L)
                inp_vec = torch.cat([q_feat, points_enc, rel_coord], dim=-1)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= h
                    rel_cell[:, :, 1] *= w
                    inp_vec = torch.cat([inp_vec, rel_cell], dim=-1)

                pred = self.imnet(inp_vec.view(B * coord.shape[1], -1))
                pred = pred.view(B, coord.shape[1], -1)
                preds.append(pred)
                q_feats.append(q_feat)
                rel_coords.append(rel_coord)

        # compute weights
        weights = self.compute_weight(q_feats, rel_coords)  # (K,B,N)

        # weighted combination
        ret = sum(pred * weights[k].unsqueeze(-1) for k, pred in enumerate(preds))
        H = int(np.sqrt(coord.shape[1]))
        W = H
        ret = ret.permute(0, 2, 1).view(B, -1, H, W)
        return ret

    # -------------------- Forward --------------------
    def forward(self, inp):
        B, _, H, W = inp.shape
        h, w = H * self.scale, W * self.scale
        device = inp.device

        coord = make_coord((h, w), device=device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        coord = coord.unsqueeze(0).repeat(B, 1, 1)
        cell = cell.unsqueeze(0).repeat(B, 1, 1)

        points_enc = self.positional_encoding(coord, L=self.L)
        coord = torch.cat([coord, points_enc], dim=-1)
        pred = self.query_2D(inp, coord, cell)
        pred = pred.contiguous().reshape(B, h, w, -1).permute(0, 3, 1, 2)
        return pred


# ---------------- Example ----------------
if __name__ == "__main__":
    model = INR2D(dim=64, out_dim=3, weight_mode='cosine').cuda()
    x = torch.randn(1, 64, 32, 32).cuda()
    y = model(x)
    print(y.shape)
