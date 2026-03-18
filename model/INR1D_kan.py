import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size

        # [FIX] Grid �������������������������� padding ��������
        # ���������������� grid_size + 2 * spline_order + 1
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

        if enable_standalone_scale_spline:
            self.scale_spline = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.scale_spline = nn.Parameter(torch.Tensor(out_features))

        self.scale_base = nn.Parameter(torch.ones(1) * scale_base)

        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters(scale_base, scale_spline, scale_noise, grid_range)

    def reset_parameters(self, scale_base, scale_spline, scale_noise, grid_range):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)

        with torch.no_grad():
            noise = (
                    (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                    * scale_noise
                    / self.grid_size
            )
            # ������������������������ curve2coeff ��������
            self.spline_weight.data.copy_(
                0.1 * torch.randn(self.out_features, self.in_features, self.grid_size + self.spline_order)
            )

            if self.scale_spline.dim() == 2:
                nn.init.constant_(self.scale_spline, scale_spline)
            else:
                nn.init.constant_(self.scale_spline, scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor x.
        Args:
            x: Input tensor of shape (batch_size, in_features).
        Returns:
            bases: B-spline bases of shape (batch_size, in_features, grid_size + spline_order).
        """
        x = x.unsqueeze(-1)  # (batch_size, in_features, 1)

        # ���� Grid (Knots)
        # grid shape: (in_features, grid_size + 2 * spline_order + 1)
        grid = self.grid

        # 0�� B-spline (Step function)
        # x >= grid[:, :-1] check lower bound
        # x <  grid[:, 1:]  check upper bound
        # ���������������� grid ��������������
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # ������������ B-spline
        for k in range(1, self.spline_order + 1):
            # �������������� knot ��������
            # ���������������������������������� term ������ bases[:, :-1] ����

            # t_i: ����������������
            t_i = grid[:, :-(k + 1)]

            # t_i_k: ��������+k������ (��������)
            t_i_k = grid[:, k:-1]

            # t_i_plus_1: ������������������
            t_i_plus_1 = grid[:, 1:-k]

            # t_i_k_plus_1: ����������+k������
            t_i_k_plus_1 = grid[:, k + 1:]

            # ���������� (Term 1)
            # ����������0 (�������������� eps)
            denom1 = t_i_k - t_i
            denom1[denom1 == 0] = 1e-8
            term1 = (x - t_i) / denom1 * bases[:, :, :-1]

            # ���������� (Term 2)
            denom2 = t_i_k_plus_1 - t_i_plus_1
            denom2[denom2 == 0] = 1e-8
            term2 = (t_i_k_plus_1 - x) / denom2 * bases[:, :, 1:]

            # ����
            bases = term1 + term2

        return bases
    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        # [CRITICAL] KAN ������������ [-1, 1] ����
        # ������������ LayerNorm������ KAN_INR1D ����������������������

        # 1. Base activation path
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 2. B-Spline path
        # Clamp to ensure stability within grid range
        x_clamped = x.clamp(-1 + self.grid_eps, 1 - self.grid_eps)
        bases = self.b_splines(x_clamped)  # (batch, in, coeff_num)

        # Spline transformation: (batch, in, coeff) * (out, in, coeff) -> (batch, out)
        # einsum ������������������������
        spline_output = torch.einsum("bij,oij->bo", bases, self.spline_weight)

        if self.scale_spline.dim() == 2:
            # (out, in) -> (out,) broadcasting handled inside sum or via weight calc
            # For simplicity, let's just add them.
            # Actually, standard KAN does scaling *before* sum, but here spline_output is already summed over input?
            # No, einsum "bij,oij->bo" already summed over 'i' (input features)?
            # Wait, "bij,oij->bo" means sum over 'i' (in_features) and 'j' (grid coeffs).
            # This is correct for fully connected layer.
            pass

        # Apply scaling
        # spline_output is (batch, out)
        # self.scale_spline should broadcast properly if we handle dimensions right.
        # But wait, original efficient-kan applies scale_spline to the weights BEFORE sum.
        # Let's stick to the simple formula:

        output = self.scale_base * base_output + self.scale_spline.view(1,
                                                                        -1) * spline_output if self.scale_spline.dim() == 1 else self.scale_base * base_output + spline_output

        return output.view(*original_shape[:-1], self.out_features)


# --- 1D �������� ---
class KAN_INR1D(nn.Module):
    def __init__(self, L0, L_real, hidden_dim=64, hidden_layers=2):
        super().__init__()
        layers = []

        # [CRITICAL] ���� LayerNorm �� Tanh ������������ [-1, 1]
        # ���� KAN �� grid ���������� [-1, 1]
        self.pre_norm = nn.LayerNorm(L0)

        in_dim = L0
        for _ in range(hidden_layers):
            layers.append(KANLinear(in_dim, hidden_dim))
            in_dim = hidden_dim

        layers.append(KANLinear(in_dim, L_real))
        self.net = nn.Sequential(*layers)

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        # E_base: (B, K, L0)
        B, K, L0 = E_base.shape
        x = E_base.view(B * K, L0)

        # 1. Norm: ������������0����1
        x = self.pre_norm(x)
        # 2. Tanh: �������� [-1, 1] ���������� KAN ����������������
        x = torch.tanh(x)

        y = self.net(x)
        return y.view(B, K, -1)
