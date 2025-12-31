# License : CC BY-NC-SA 4.0
import torch
from torch import nn, einsum
from inspect import isfunction
import random
from einops import rearrange
import math
def random_rotate(tensor1, tensor2, tensor3):
    rotation = random.randint(0, 3)
    if rotation == 0:
        return tensor1, tensor2, tensor3  # No rotation
    elif rotation == 1:
        return tensor1.rot90(1, (2, 3)), tensor2.rot90(1, (2, 3)), tensor3.rot90(1, (2, 3))  # 90° clockwise
    elif rotation == 2:
        return tensor1.rot90(3, (2, 3)), tensor2.rot90(3, (2, 3)), tensor3.rot90(3, (2, 3))  # 90° counter-clockwise
    elif rotation == 3:
        return tensor1.rot90(2, (2, 3)), tensor2.rot90(2, (2, 3)), tensor3.rot90(2, (2, 3))  # 180°

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Landmark encoder
class LandEncoder(nn.Module):
    def __init__(self, in_channels: int = 8, embed_dim: int = 256):
        """
        Args:
            in_channels: number of input channels (e.g. 2 maps × 4 channels = 8)
            embed_dim: output embedding size for FiLM MLPs
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),         # 16 -> 8
            nn.ReLU(),
        )
        # final projection
        self.fc = nn.Linear(256 * 8 * 8, embed_dim)

    def forward(self, x):
        """
        x: (B, C, 128, 128)
        returns: (B, embed_dim)
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)   # flatten
        return self.fc(h)

# FiLM module
class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim, feature_dim):
        super().__init__()

        # MLP that turns condition into γ and β
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2)  # outputs γ and β
        )

    def forward(self, cond, features):
        """
        cond: [B, cond_dim]   (flattened condition vector or global pooled map)
        features: [B, C, H, W] (UNet features)
        """
        gamma_beta = self.mlp(cond)  # [B, 2*feature_dim]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # each [B, feature_dim]

        # Reshape to broadcast over spatial dimensions
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]

        return gamma * features + beta
    
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)