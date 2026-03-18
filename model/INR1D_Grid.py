import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
# ������ 1D ������������������
class MultiScale_INR1D(nn.Module):
    def __init__(self, L0, L_real, hidden_dim=256):
        super().__init__()

        # ��������������������������������
        self.branch_global = nn.Sequential(
            nn.Linear(L0, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.branch_detail = nn.Sequential(
            nn.Linear(L0, hidden_dim),
            Sine(),  # �������� Sirens ������������
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fusion = nn.Linear(hidden_dim * 2, L_real)

    def forward(self, E_base):
        B, K, L0 = E_base.shape
        x = E_base.view(B * K, L0)

        g = self.branch_global(x)
        d = self.branch_detail(x)  # ���������������������� PE

        out = self.fusion(torch.cat([g, d], dim=-1))
        return out.view(B, K, -1)

# �������������� 1D ������������ Basic MLP�������������� Gabor/KAN��
# Grid ���������� 2D ��������