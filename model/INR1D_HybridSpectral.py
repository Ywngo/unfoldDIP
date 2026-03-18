# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedINR1D(nn.Module):
    """
    ������ FactorizedINR1D ������������������������������������������������
    ��������������FactorizedINR1D(L0, L_real, K, M=32, hidden_dim=128, use_basis_norm=True)
    """
    def __init__(
        self,
        L0: int,
        L_real: int,
        K: int,
        M: int = 32,
        hidden_dim: int = 128,
        use_basis_norm: bool = True,
    ):
        super().__init__()
        self.L0 = L0
        self.L_real = L_real
        self.K = K
        self.M = M
        self.use_basis_norm = use_basis_norm

        # ���������� (L_real, M)
        self.basis = nn.Parameter(torch.empty(L_real, M))
        nn.init.xavier_normal_(self.basis, gain=0.1)

        # �������� -> ���� alpha (B,K,L0) -> (B,K,M)
        self.mlp_alpha = nn.Sequential(
            nn.Linear(L0, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, M),
        )

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        """
        E_base: (B, K, L0)
        return: (B, K, L_real)
        """
        B, K, L0 = E_base.shape
        assert K == self.K and L0 == self.L0

        x = E_base.view(B * K, L0)
        alpha = self.mlp_alpha(x)              # (B*K, M)
        alpha = alpha.view(B, K, self.M)       # (B,K,M)

        Bm = self.basis                        # (L_real, M)
        if self.use_basis_norm:
            Bm = F.normalize(Bm, dim=0)        # ������ L2 ����

        # E_full: (B,K,L_real)
        E_full = torch.einsum("lm,bkm->bkl", Bm, alpha)
        return E_full


class SpectralINR1D(nn.Module):
    """
    �������������� INR����������:
      - ����: E_base ��������
      - ����: ������������ rFFT������������������ L0 ��
      - ��������+���� -> MLP -> ���� (B,K,L_real)
    """
    def __init__(
        self,
        L0: int,
        L_real: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        use_freq: bool = True,
    ):
        super().__init__()
        self.L0 = L0
        self.L_real = L_real
        self.use_freq = use_freq

        in_dim = L0
        if use_freq:
            in_dim += L0  # ���� + ����

        layers = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, L_real))
        self.mlp = nn.Sequential(*layers)

    def _build_freq_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*K, L0) ��������
        return: (B*K, L0) ���������������� L0��
        """
        # rFFT ������
        x_freq = torch.fft.rfft(x, dim=-1).real          # (B*K, L0//2+1)
        # ���������� L0
        x_freq = x_freq.unsqueeze(1)                     # (B*K,1,Lf)
        x_freq = F.interpolate(
            x_freq,
            size=self.L0,
            mode="linear",
            align_corners=False,
        ).squeeze(1)                                     # (B*K,L0)
        return x_freq

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        """
        E_base: (B, K, L0)
        return: (B, K, L_real)
        """
        B, K, L0 = E_base.shape
        assert L0 == self.L0

        x = E_base.view(B * K, L0)                       # (B*K,L0)
        if self.use_freq:
            x_freq = self._build_freq_feature(x)         # (B*K,L0)
            x_in = torch.cat([x, x_freq], dim=-1)
        else:
            x_in = x

        y = self.mlp(x_in)                               # (B*K,L_real)
        y = y.view(B, K, self.L_real)
        return y


class HybridEndmemberINR1D(nn.Module):
    """
    Hybrid ���� INR1D:
      E_full = E_fact (Factorized ������) + s * E_res (������������)

    - E_fact: FactorizedINR1D ����, ����������������������
    - E_res:  �������� SpectralINR1D ����, ������/��������
    - s:      ������������, ������������(��������)

    ����:
        E_base: (B, K, L0)
    ����:
        E_full: (B, K, L_real)
    """

    def __init__(
        self,
        L0: int,
        L_real: int,
        K: int,
        M: int = 32,
        fact_hidden: int = 128,
        res_hidden: int = 128,
        res_layers: int = 2,
        use_freq_res: bool = True,  # ��������������������
    ):
        super().__init__()
        self.L0 = L0
        self.L_real = L_real
        self.K = K

        # ������: Factorized ������
        self.fact_inr = FactorizedINR1D(
            L0=L0,
            L_real=L_real,
            K=K,
            M=M,
            hidden_dim=fact_hidden,
            use_basis_norm=True,
        )

        # ����: SpectralINR1D
        self.res_inr = SpectralINR1D(
            L0=L0,
            L_real=L_real,
            hidden_dim=res_hidden,
            hidden_layers=res_layers,
            use_freq=use_freq_res,
        )

        # ������������������ 0.1 ������
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        """
        E_base: (B, K, L0)
        """
        E_fact = self.fact_inr(E_base)       # (B,K,L_real)
        E_res  = self.res_inr(E_base)        # (B,K,L_real)
        s = self.res_scale
        E_full = E_fact + s * E_res
        return E_full