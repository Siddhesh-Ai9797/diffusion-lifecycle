import torch
import torch.nn as nn
import torch.nn.functional as F
from src.unet import TimeEmbedding, ResidualBlock


class LinearAttention(nn.Module):
    """Linear attention - O(N) instead of O(N^2) complexity."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        # flatten spatial: (B, C, H, W) -> (B, H*W, C)
        h = h.view(B, C, H * W).transpose(1, 2)

        # Query, Key, Value
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2),
            qkv
        )

        # Linear attention: use ELU feature map instead of softmax
        # phi(x) = ELU(x) + 1  (always positive, approximates softmax behavior)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # KEY TRICK: compute K^T × V first (channels×channels, NOT positions×positions)
        # k: (B, heads, N, head_dim), v: (B, heads, N, head_dim)
        kv = torch.matmul(k.transpose(-2, -1), v)  # (B, heads, head_dim, head_dim)

        # Then Q × (K^T × V)  -- cheap!
        out = torch.matmul(q, kv)  # (B, heads, N, head_dim)

        # Normalize (linear attention needs explicit normalization)
        k_sum = k.sum(dim=-2, keepdim=True)  # (B, heads, 1, head_dim)
        normalizer = torch.matmul(q, k_sum.transpose(-2, -1))  # (B, heads, N, 1)
        out = out / (normalizer + 1e-6)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, H*W, C)
        out = self.proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)

        return x + out


class SimpleUNetLinearAttention(nn.Module):
    """U-Net with LINEAR attention at bottleneck."""
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        t_dim = time_emb_dim * 4

        # Encoder (identical to SimpleUNet)
        self.enc1 = ResidualBlock(in_channels, base_ch, t_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)

        self.enc2 = ResidualBlock(base_ch, base_ch * 2, t_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        # Bottleneck: ResidualBlock + LinearAttention
        self.bottleneck = ResidualBlock(base_ch * 2, base_ch * 2, t_dim)
        self.bottleneck_attn = LinearAttention(base_ch * 2, num_heads=4)

        # Decoder (identical to SimpleUNet)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_ch * 2 + base_ch * 2, base_ch, t_dim)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_ch + base_ch, base_ch, t_dim)

        self.out = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)

        b = self.bottleneck(d2, t_emb)
        b = self.bottleneck_attn(b)      # ← linear attention instead of standard

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out(u1)


if __name__ == "__main__":
    model = SimpleUNetLinearAttention()
    x = torch.randn(8, 3, 32, 32)
    t = torch.randint(0, 1000, (8,))
    out = model(x, t)
    print("Input:", x.shape)
    print("Output:", out.shape)
    print("Params:", sum(p.numel() for p in model.parameters()))
    baseline_params = 2_100_035
    print(f"Extra params vs baseline: {sum(p.numel() for p in model.parameters()) - baseline_params:,}")