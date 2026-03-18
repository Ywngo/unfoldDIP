import torch
import torch.nn as nn
import torch.nn.functional as F

from model.INR2D_factorized import FactorizedINR2D
from model.INR2D_spectral import SpectralINR2D


class HybridINR2D(nn.Module):
    """
    Hybrid 2D INR:
      Y = Y_fact (Factorized ������������) + s * Y_res (���� INR2D)

    �������� abundance / HR-HSI ��������������:
      - FactorizedINR2D ������������������������������;
      - ���� INR2D (Spectral/basic) ����������������, ������ WaDC ����������������

    ����:
        feat: (B, C_in, H, W)  ������������
    ����:
        out:  (B, C_out, H, W)
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        M: int = 16,          # Factorized ����������
        fact_hidden: int = 128,
        fact_layers: int = 2,
        res_hidden: int = 256,
        res_layers: int = 3,
        use_pe_res: bool = True,
        pe_dim_res: int = 16,
        use_spectral_res: bool = False,  # ���������� SpectralINR2D
    ):
        super().__init__()
        self.C_in = dim
        self.C_out = out_dim

        # ������: FactorizedINR2D
        self.fact_inr2d = FactorizedINR2D(
            dim=dim,
            out_dim=out_dim,
            M=M,
            hidden_dim=fact_hidden,
            hidden_layers=fact_layers,
            base_size=32,      # ��������������������������
            use_pe=True,
            pe_dim=pe_dim_res, # ���������������������� pe_dim
        )

        # ����: basic/Spectral 2D INR
        if use_spectral_res:
            self.res_inr2d = SpectralINR2D(
                dim=dim,
                out_dim=out_dim,
                hidden_dim=res_hidden,
                hidden_layers=res_layers,
                use_pe=use_pe_res,
                pe_dim=pe_dim_res,
                use_freq=True,
            )
        else:
            # ������: ��������, ���� local conv + coord MLP
            self.res_inr2d = SpectralINR2D(
                dim=dim,
                out_dim=out_dim,
                hidden_dim=res_hidden,
                hidden_layers=res_layers,
                use_pe=use_pe_res,
                pe_dim=pe_dim_res,
                use_freq=False,
            )

        # ������������
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C_in, H, W)
        """
        Y_fact = self.fact_inr2d(feat)             # (B,C_out,H,W)
        Y_res = self.res_inr2d(feat)               # (B,C_out,H,W)
        out = Y_fact + self.res_scale * Y_res
        return out