import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """Converts timestep number (0-999) into a vector the network can use."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        dtype = self.mlp[0].weight.dtype  # match model's dtype (FP32 or FP16)
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device, dtype=dtype) / half_dim)
        args = t[:, None].to(dtype) * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """Conv block that also takes in the timestep embedding."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """Small U-Net for 32x32 images (CIFAR-10), predicts noise."""
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        t_dim = time_emb_dim * 4

        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_ch, t_dim)        # 32x32
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)  # 16x16

        self.enc2 = ResidualBlock(base_ch, base_ch * 2, t_dim)         # 16x16
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 8x8

        # Bottleneck
        self.bottleneck = ResidualBlock(base_ch * 2, base_ch * 2, t_dim)  # 8x8

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)  # 16x16
        self.dec2 = ResidualBlock(base_ch * 2 + base_ch * 2, base_ch, t_dim)  # skip connection

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)  # 32x32
        self.dec1 = ResidualBlock(base_ch + base_ch, base_ch, t_dim)  # skip connection

        self.out = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        e1 = self.enc1(x, t_emb)          # 32x32, base_ch
        d1 = self.down1(e1)               # 16x16, base_ch

        e2 = self.enc2(d1, t_emb)         # 16x16, base_ch*2
        d2 = self.down2(e2)               # 8x8, base_ch*2

        b = self.bottleneck(d2, t_emb)    # 8x8, base_ch*2

        u2 = self.up2(b)                  # 16x16, base_ch*2
        u2 = torch.cat([u2, e2], dim=1)   # skip connection -> base_ch*4
        u2 = self.dec2(u2, t_emb)         # 16x16, base_ch

        u1 = self.up1(u2)                 # 32x32, base_ch
        u1 = torch.cat([u1, e1], dim=1)   # skip connection -> base_ch*2
        u1 = self.dec1(u1, t_emb)         # 32x32, base_ch

        return self.out(u1)               # 32x32, in_channels


if __name__ == "__main__":
    model = SimpleUNet()
    x = torch.randn(8, 3, 32, 32)   # batch of 8 CIFAR images
    t = torch.randint(0, 1000, (8,))  # random timesteps
    out = model(x, t)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Params:", sum(p.numel() for p in model.parameters()))