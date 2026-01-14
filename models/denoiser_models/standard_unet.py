import torch
from torch import nn, einsum
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.nn.functional import mse_loss
from einops import rearrange
from models.utils_files.nn_utils import *

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True, cond_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim) if norm else nn.Identity()
        self.film = FiLM(cond_dim, hidden_dim=dim, feature_dim=dim) if cond_dim is not None else None
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult) if norm else nn.Identity(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb=None, x_cond=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.norm(h)
        h = self.film(x_cond, h) if exists(self.film) else h
        h = self.net(h)
        return h + self.res_conv(x)
    
class ConvNextBlock_batchnorm(nn.Module):
    """https://arxiv.org/abs/2201.03545
    Using BatchNorm instead of GroupNorm a mistake
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True, cond_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim) if norm else nn.Identity()
        self.film = FiLM(cond_dim, hidden_dim=dim, feature_dim=dim) if cond_dim is not None else None
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb=None, x_cond=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.norm(h)
        h = self.film(x_cond, h) if exists(self.film) else h
        h = self.net(h)
        return h + self.res_conv(x)
    

class Unet(nn.Module):
    def __init__(
        self,
        dim, #dimension of input image
        init_dim=None, #nb of channels in the first conv layer
        out_dim=None, #nb of channels in the last conv layer
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        self_condition_size=0, # nb of conditions for the input 
        GroupNorm=True,
        film_cond_dim=None, # dimension of the condition vector for FiLM layers
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition_size 
        self.film_cond_dim = film_cond_dim
        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)

        self.init_conv = nn.Conv2d((self.self_condition+1)*channels, init_dim, 7, padding=3)
        

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass_film = partial(ConvNextBlock, mult=convnext_mult, cond_dim=film_cond_dim) if GroupNorm else partial(ConvNextBlock_batchnorm, mult=convnext_mult, cond_dim=film_cond_dim)
        block_klass = partial(ConvNextBlock, mult=convnext_mult) if GroupNorm else partial(ConvNextBlock, mult=convnext_mult)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_film(mid_dim, mid_dim, time_emb_dim=time_dim) if film_cond_dim is not None else block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass_film(mid_dim, mid_dim, time_emb_dim=time_dim) if film_cond_dim is not None else block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass_film(dim_out * 2, dim_in, time_emb_dim=time_dim) if film_cond_dim is not None else block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time=None, x_cond_1=None, x_cond_2=None):

        if self.self_condition:
            x_self_cond = default(x_cond_1, lambda: torch.zeros_like(torch.cat((x,x), dim=1)))
            x = torch.cat((x_self_cond.to(x.device), x), dim=1)

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t, x_cond_2) if self.film_cond_dim is not None else self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, x_cond_2) if self.film_cond_dim is not None else self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, x_cond_2) if self.film_cond_dim is not None else block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
    

class Unet_stochinterpolant_1(nn.Module):
    " Unet for stochastic interpolant with 2 cond inputs injected through FiLM layers "
    " Returns two outputs: b and n_z (Stochastic Interpolants: A Unifying Framework for Flows and Diffusions: https://arxiv.org/pdf/2303.08797)"
    " b: probability flow velocity"
    " n_z: denoiser → gives score function "
    def __init__(
        self,
        dim, #dimension of input image
        init_dim=None, #nb of channels in the first conv layer
        out_dim=None, #nb of channels in the last conv layer
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True, 
        GroupNorm=True,
        film_cond_dim=None, # dimension of the condition vector for FiLM layers
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.film_cond_dim = film_cond_dim
        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)

        self.init_conv = nn.Conv2d((self.self_condition+1)*channels, init_dim, 7, padding=3)
        

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass_film = partial(ConvNextBlock, mult=convnext_mult, cond_dim=film_cond_dim) if GroupNorm else partial(ConvNextBlock_batchnorm, mult=convnext_mult, cond_dim=film_cond_dim)
        block_klass = partial(ConvNextBlock, mult=convnext_mult) if GroupNorm else partial(ConvNextBlock, mult=convnext_mult)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        # land encoder
        self.land_encoder = LandEncoder(in_channels= channels, embed_dim = film_cond_dim)
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_film(mid_dim, mid_dim, time_emb_dim=time_dim) if film_cond_dim is not None else block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass_film(mid_dim, mid_dim, time_emb_dim=time_dim) if film_cond_dim is not None else block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass_film(dim_out * 2, dim_in, time_emb_dim=time_dim) if film_cond_dim is not None else block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass_film(dim_in, dim_in, time_emb_dim=time_dim) if film_cond_dim is not None else block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv_b = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )
        self.final_conv_eta = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )
    def forward(self, x, time=None, x_cond_land_1=None, x_cond_land_2=None):

        if self.film_cond_dim is not None and x_cond_land_1.ndim > 2 and x_cond_land_2.ndim > 2:
            x_cond_land_1 = self.land_encoder(x_cond_land_1)
            x_cond_land_2 = self.land_encoder(x_cond_land_2)
            
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t, x_cond_land_1) if self.film_cond_dim is not None else self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, x_cond_land_2) if self.film_cond_dim is not None else self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, x_cond_land_1) if self.film_cond_dim is not None else block1(x, t)
            x = block2(x, t, x_cond_land_2) if self.film_cond_dim is not None else block2(x, t)
            x = attn(x)
            x = upsample(x)

        b = self.final_conv_b(x) 
        eta = self.final_conv_eta(x)
        return b, eta
