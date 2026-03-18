import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. LayerNorm2d
# -----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

# -----------------------------
# 2. Multi-DConv Head Transposed Attention (MDTA)
# -----------------------------
class MDTA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # �������� Q �� KV ����������
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1, bias=False)

        # ��������������������
        self.q_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2)

        self.project_out = nn.Conv2d(dim, dim, 1)

    def forward(self, q_input, kv_input):
        B, C, H, W = q_input.shape

        # 1?? Q from image A
        q = self.q_dwconv(self.q_proj(q_input))
        # 2?? K, V from image B
        kv = self.kv_dwconv(self.kv_proj(kv_input))
        k, v = kv.chunk(2, dim=1)

        # 3?? reshape for multi-head
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # 4?? cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.view(B, C, H, W)

        return self.project_out(out)

# -----------------------------
# 3. Gated-Dconv Feed Forward Network (GDFN)
# -----------------------------
class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, dim, 1)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

# -----------------------------
# 4. Restormer Block
# -----------------------------
class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, expansion_factor=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expansion_factor)

    def forward(self, x, y ):
        y = y + self.attn(self.norm1(x), self.norm1(y) )
        y = y + self.ffn(self.norm2(y))
        return y

# -----------------------------
# 5. Downsample / Upsample
# -----------------------------
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.conv(x))

# -----------------------------
# 6. Restormer Network
# -----------------------------
class Restormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8]):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # Encoder
        self.encoder1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)
        self.encoder2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim * 2)
        self.encoder3 = nn.Sequential(*[RestormerBlock(dim * 4, num_heads[2]) for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[RestormerBlock(dim * 8, num_heads[3]) for _ in range(num_blocks[3])])

        # Decoder
        self.up3 = Upsample(dim * 8)
        self.decoder3 = nn.Sequential(*[RestormerBlock(dim * 4, num_heads[2]) for _ in range(num_blocks[2])])
        self.up2 = Upsample(dim * 4)
        self.decoder2 = nn.Sequential(*[RestormerBlock(dim * 2, num_heads[1]) for _ in range(num_blocks[1])])
        self.up1 = Upsample(dim * 2)
        self.decoder1 = nn.Sequential(*[RestormerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.encoder1(self.patch_embed(x))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.bottleneck(self.down3(x3))

        x = self.up3(x4) + x3
        x = self.decoder3(x)
        x = self.up2(x) + x2
        x = self.decoder2(x)
        x = self.up1(x) + x1
        x = self.decoder1(x)

        out = self.output(x) + x  # residual connection
        return out

# -----------------------------
# 7. ��������
# -----------------------------
if __name__ == "__main__":
    model = Restormer()
    inp = torch.randn(1, 3, 256, 256)
    out = model(inp)
    print(f"Input: {inp.shape}, Output: {out.shape}")
