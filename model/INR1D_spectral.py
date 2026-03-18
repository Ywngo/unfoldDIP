import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralINR1D(nn.Module):
    """
    �������������� INR��������������������

    ����:
        E_base: (B, K, L0)
            - B: batch size
            - K: ��������
            - L0: ������������ (����������)

    ����:
        E_full: (B, K, L_real)
            - L_real: ����������

    ��������:
        - ���������������������� FFT/DCT ��������������
        - ���������������������������� MLP��
        - MLP ������������������������
    """

    def __init__(
        self,
        L0: int,
        L_real: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        use_freq: bool = True,
        freq_type: str = "fft",  # or "dct"���������� fft ������������
    ):
        super().__init__()
        assert freq_type in ["fft", "dct"]
        self.L0 = L0
        self.L_real = L_real
        self.use_freq = use_freq
        self.freq_type = freq_type

        in_dim = L0
        if use_freq:
            # ���������������������� L0 ������
            in_dim += L0

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
        if self.freq_type == "fft":
            # rfft -> ���������������������� L0
            x_freq = torch.fft.rfft(x, dim=-1).real  # (B*K, L0//2+1)
        else:
            # ������ DCT������������ FFT ����������������
            x_freq = torch.fft.rfft(x, dim=-1).real  # (B*K, L0//2+1)

        # ������ L0 ����
        x_freq = x_freq.unsqueeze(1)  # (B*K,1,Lf)
        x_freq = F.interpolate(
            x_freq,
            size=self.L0,
            mode="linear",
            align_corners=False,
        )  # (B*K,1,L0)
        x_freq = x_freq.squeeze(1)  # (B*K,L0)
        return x_freq

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        """
        E_base: (B, K, L0)
        return: (B, K, L_real)
        """
        B, K, L0 = E_base.shape
        assert L0 == self.L0

        x = E_base.view(B * K, L0)  # (B*K,L0)

        if self.use_freq:
            x_freq = self._build_freq_feature(x)  # (B*K,L0)
            x_in = torch.cat([x, x_freq], dim=-1)  # (B*K,2*L0)
        else:
            x_in = x

        y = self.mlp(x_in)  # (B*K,L_real)
        y = y.view(B, K, self.L_real)
        return y