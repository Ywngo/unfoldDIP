import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedINR1D(nn.Module):
    """
    Factorized 1D Implicit Neural Representation for Spectral Super-Resolution.

    This model decomposes high-resolution spectral signatures into:
    1. A learnable global dictionary (Basis)
    2. Local coefficients predicted by an MLP (Alpha)

    Formula: E_full = Basis @ Alpha
    """

    def __init__(
            self,
            L0: int,
            L_real: int,
            K: int,
            M: int = 48,  # Increased M for better capacity on WaDC
            hidden_dim: int = 256,  # Increased hidden_dim for complex mapping
            use_basis_norm: bool = True,
    ):
        super().__init__()
        self.L0 = L0
        self.L_real = L_real
        self.K = K
        self.M = M
        self.use_basis_norm = use_basis_norm

        # Learnable global basis: (L_real, M)
        # Initialized with random noise, but SHOULD be overwritten by PCA using 'initialize_with_pca'
        self.basis = nn.Parameter(torch.randn(L_real, M) * 0.01)

        # Coefficient predictor: (B, K, L0) -> (B, K, M)
        # Using LeakyReLU and a deeper network for better convergence
        self.mlp_alpha = nn.Sequential(
            nn.Linear(L0, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),  # Added depth
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, M),
        )

    def initialize_with_pca(self, reference_data: torch.Tensor):
        """
        Initialize the global basis using PCA on the reference dataset.
        Automatically handles (B, C, H, W) format by permuting it to (B, H, W, C).
        """
        device = self.basis.device
        data = reference_data.to(device).float()

        # [FIX] 1. �������� PyTorch �� (B, C, H, W) ����
        # ����������4����1����������(191)����permute�� (B, H, W, C)
        if data.dim() == 4 and data.shape[1] == self.L_real:
            print(f"[FactorizedINR1D] Detected (B, C, H, W) input {data.shape}, permuting to channels-last...")
            data = data.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # [FIX] 2. �������������� (C, H, W) ����
        elif data.dim() == 3 and data.shape[0] == self.L_real:
            data = data.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

        # ������ data.shape[-1] ������ 191 ��
        if data.shape[-1] != self.L_real:
            raise ValueError(f"Expected last dim to be {self.L_real} (bands), but got {data.shape[-1]}. "
                             f"Check if input shape is permuted correctly.")

        # Flatten: (..., L_real) -> (N, L_real)
        data = data.reshape(-1, self.L_real)

        print(f"[FactorizedINR1D] Initializing basis with PCA from {data.shape[0]} samples...")

        # Center the data
        mean = torch.mean(data, dim=0)
        data_centered = data - mean

        # Perform SVD
        # V matrix contains eigenvectors (principal components)
        _, _, V = torch.linalg.svd(data_centered, full_matrices=False)

        # Extract top M components
        # torch.linalg.svd returns Vh (transposed V), shape (L_real, L_real) usually
        # We need the top M rows of Vh, then transpose to match basis shape (L_real, M)
        top_components = V[:self.M, :].T

        # Assign to basis
        with torch.no_grad():
            self.basis.data.copy_(top_components)

        print("[FactorizedINR1D] PCA initialization complete.")

    def forward(self, E_base: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_base: (B, K, L0) - Low-resolution endmembers
        Returns:
            E_full: (B, K, L_real) - Super-resolved endmembers
        """
        B, K, L0 = E_base.shape
        # Safety check removed for speed, but ensure L0 matches self.L0

        # Predict coefficients alpha
        x = E_base.view(B * K, L0)  # (B*K, L0)
        alpha = self.mlp_alpha(x)  # (B*K, M)
        alpha = alpha.view(B, K, self.M)  # (B, K, M)

        # Retrieve basis
        Bm = self.basis  # (L_real, M)

        # Normalize basis (optional, but keeps optimization stable)
        if self.use_basis_norm:
            Bm = F.normalize(Bm, dim=0)

        # Linear Combination: E = Basis * Alpha
        # einsum: (L_real, M) and (B, K, M) -> (B, K, L_real)
        E_full = torch.einsum("lm,bkm->bkl", Bm, alpha)

        return E_full