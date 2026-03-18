import torch
import torch.nn as nn
import torch.nn.functional as F

from model.INR1D_factorized import FactorizedINR1D
from model.INR1D_spectral import SpectralINR1D


class HybridEndmemberINR1D(nn.Module):
    """
    Hybrid ���� INR1D:
      E_full = E_fact (Factorized ������) + s * E_res (���� MLP/���� INR)

    - E_fact: FactorizedINR1D ����, ���������������������� (basis + ��)
    - E_res:  ���������� INR1D, ������������������������
    - s:      ������������, ������������(��������, �� 0.1)

    ����:
        E_base: (B, K, L0)  ��������������
    ����:
        E_full: (B, K, L_real)
    """

    def __init__(
        self,
        L0: int,
        L_real: int,
        K: int,
        M: int = 32,          # Factorized basis ����
        fact_hidden: int = 128,
        res_hidden: int = 128,
        res_layers: int = 2,
        use_freq_res: bool = False,  # �������������� SpectralINR1D
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
            use_basis_norm=True,   # ��������������
        )

        # ����: ������ SpectralINR1D ������ MLP INR1D
        if use_freq_res:
            self.res_inr = SpectralINR1D(
                L0=L0,
                L_real=L_real,
                hidden_dim=res_hidden,
                hidden_layers=res_layers,
                use_freq=True,
            )
        else:
            # ������: ���� MLP, ��������
            self.res_inr = SpectralINR1D(
                L0=L0,
                L_real=L_real,
                hidden_dim=res_hidden,
                hidden_layers=res_layers,
                use_freq=False,
            )

        # ����������������, ����������
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        """
        E_base: (B, K, L0)
        """
        E_fact = self.fact_inr(E_base)                # (B,K,L_real)
        E_res = self.res_inr(E_base)                  # (B,K,L_real)

        # ����: ������������������������ clamp
        s = self.res_scale
        E_full = E_fact + s * E_res
        return E_full