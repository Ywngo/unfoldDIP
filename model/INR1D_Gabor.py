import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega0=10.0, sigma0=10.0):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-math.sqrt(6 / in_features) / omega0,
                                            math.sqrt(6 / in_features) / omega0)

    def forward(self, x):
        lin = self.linear(x)
        # Gabor ����: Gaussian envelope * Sinusoid
        return torch.exp(-(lin ** 2) / (2 * self.scale_0 ** 2)) * torch.sin(self.omega_0 * lin)


class Gabor_INR1D(nn.Module):
    def __init__(self, L0, L_real, hidden_dim=256, hidden_layers=2):
        super().__init__()
        layers = []
        # ������ Gabor
        layers.append(GaborLayer(L0, hidden_dim, is_first=True))

        # ������ Gabor
        for _ in range(hidden_layers):
            layers.append(GaborLayer(hidden_dim, hidden_dim))

        # �������� Linear (������������)
        layers.append(nn.Linear(hidden_dim, L_real))
        self.net = nn.Sequential(*layers)

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        B, K, L0 = E_base.shape
        x = E_base.view(B * K, L0)
        y = self.net(x)
        return y.view(B, K, -1)