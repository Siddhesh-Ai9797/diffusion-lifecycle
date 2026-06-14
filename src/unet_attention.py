import torch
import torch.nn as nn
import math
from src.unet import TimeEmbedding, ResidualBlock  # reuse existing blocks


class SelfAttention(nn.Module):
    """Standard multi-head self-attention at bottleneck (8x8 = 64 positions)."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, channels * 3)  # Q, K, V in one shot
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        # flatten spatial: (B, C, H, W) -> (B, H*W, C)
        h = h.view(B, C, H * W).transpose(1, 2)

        # compute Q, K, V
        qkv = self.qkv(h).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # attention: (B, heads, HW, HW)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # weighted sum of values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H*W, C)
        out = self.proj(out)

        # reshape back: (B, H*W, C) -> (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)
        return x + out  # residual connection


class SimpleUNetAttention(nn.Module):
    """U-Net with standard self-attention at bottleneck. Everything else identical to SimpleUNet."""
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        t_dim = time_emb_dim * 4

        # Encoder (identical to SimpleUNet)
        self.enc1 = ResidualBlock(in_channels, base_ch, t_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)

        self.enc2 = ResidualBlock(base_ch, base_ch * 2, t_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        # Bottleneck: ResidualBlock + SelfAttention
        self.bottleneck = ResidualBlock(base_ch * 2, base_ch * 2, t_dim)
        self.bottleneck_attn = SelfAttention(base_ch * 2, num_heads=4)

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
        b = self.bottleneck_attn(b)      # ← only difference from SimpleUNet

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out(u1)


if __name__ == "__main__":
    model = SimpleUNetAttention()
    x = torch.randn(8, 3, 32, 32)
    t = torch.randint(0, 1000, (8,))
    out = model(x, t)
    print("Input:", x.shape)
    print("Output:", out.shape)
    print("Params:", sum(p.numel() for p in model.parameters()))
    baseline_params = 2_100_035
    print(f"Extra params vs baseline: {sum(p.numel() for p in model.parameters()) - baseline_params:,}")