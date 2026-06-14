import torch
import torch.nn as nn
import torch.nn.functional as F
from src.unet import TimeEmbedding, ResidualBlock


class SSMBlock(nn.Module):
    """
    Simplified Mamba-inspired SSM block.
    Processes spatial positions sequentially with input-dependent gating.
    Not full Mamba (no hardware-optimized scan) but captures the core idea:
    selective state space processing vs attention's all-to-all comparison.
    """
    def __init__(self, channels, d_state=16):
        super().__init__()
        self.channels = channels
        self.d_state = d_state

        self.norm = nn.GroupNorm(8, channels)

        # Input projection
        self.in_proj = nn.Linear(channels, channels * 2)  # split into x and z (gate)

        # SSM parameters
        self.x_proj = nn.Linear(channels, d_state * 2)    # projects to B, C (SSM params)
        self.dt_proj = nn.Linear(channels, channels)       # delta (time step)
        self.A = nn.Parameter(torch.randn(channels, d_state))  # state matrix

        # Output projection
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        # flatten: (B, C, H, W) -> (B, H*W, C)
        h = h.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        N = H * W

        # Split into main path (u) and gate (z)
        xz = self.in_proj(h)                      # (B, N, C*2)
        u, z = xz.chunk(2, dim=-1)                # each (B, N, C)

        # Gate z with SiLU (selective gating — input-dependent)
        z = F.silu(z)

        # SSM parameters (input-dependent — this is the "selective" part)
        x_dbl = self.x_proj(u)                    # (B, N, d_state*2)
        B_ssm, C_ssm = x_dbl.chunk(2, dim=-1)     # each (B, N, d_state)
        dt = F.softplus(self.dt_proj(u))           # (B, N, C) - step size

        # Discretized state update (simplified scan)
        # A: (C, d_state) -> state decay matrix
        A = -torch.exp(self.A)                     # negative = stable decay
        dA = torch.exp(dt.unsqueeze(-1) * A)       # (B, N, C, d_state)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2) # (B, N, C, d_state)

        # Sequential scan (simplified — full Mamba uses parallel scan for speed)
        state = torch.zeros(B, C, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(N):
            state = dA[:, i] * state + dB[:, i] * u[:, i].unsqueeze(-1)
            y = (state * C_ssm[:, i].unsqueeze(1)).sum(dim=-1)  # (B, C)
            ys.append(y)
        out = torch.stack(ys, dim=1)               # (B, N, C)

        # Apply gate
        out = out * z

        # Output projection
        out = self.out_proj(out)

        # reshape back: (B, N, C) -> (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)

        return x + out  # residual connection


class SimpleUNetMamba(nn.Module):
    """U-Net with Mamba-inspired SSM block at bottleneck."""
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        t_dim = time_emb_dim * 4

        # Encoder (identical to SimpleUNet)
        self.enc1 = ResidualBlock(in_channels, base_ch, t_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)

        self.enc2 = ResidualBlock(base_ch, base_ch * 2, t_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)

        # Bottleneck: ResidualBlock + SSMBlock
        self.bottleneck = ResidualBlock(base_ch * 2, base_ch * 2, t_dim)
        self.bottleneck_ssm = SSMBlock(base_ch * 2, d_state=16)

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
        b = self.bottleneck_ssm(b)      # ← Mamba-inspired SSM block

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out(u1)


if __name__ == "__main__":
    model = SimpleUNetMamba()
    x = torch.randn(8, 3, 32, 32)
    t = torch.randint(0, 1000, (8,))
    out = model(x, t)
    print("Input:", x.shape)
    print("Output:", out.shape)
    print("Params:", sum(p.numel() for p in model.parameters()))
    baseline_params = 2_100_035
    print(f"Extra params vs baseline: {sum(p.numel() for p in model.parameters()) - baseline_params:,}")