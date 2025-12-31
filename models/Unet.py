# License : CC BY-NC-SA 4.0

import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
import pytorch_lightning as pl
from functools import partial
import math
import wandb
from einops import rearrange
from models.forward_diffusion import q_sample
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR,ReduceLROnPlateau
import math
from tqdm import tqdm


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

    
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


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


class Unet(nn.Module):
    def __init__(
        self,
        dim, #dimension of input image
        init_dim=None, #nb of channels in the first conv layer
        out_dim=None, #nb of channels in the last conv layer
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        self_condition=False,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition

        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)
        # if we condition, we need to double the number of channels for conditioning 
        if self.self_condition:
            self.init_conv = nn.Conv2d(2*channels, init_dim, 7, padding=3)
        else:
            self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ConvNextBlock, mult=convnext_mult)

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
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
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

    def forward(self, x, time=None, x_self_cond=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

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
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)    
    

class UnetModel(pl.LightningModule):
    def __init__(
                self, 
                 unet, 
                 lr=[1e-3, 1e-4], 
                 train_criterion=nn.SmoothL1Loss(),
                 validation_criterion=partial(F.mse_loss,reduction='mean'), 
                 T=1000, 
                 T_sampling=1000, 
                 constants_dict=None, 
                 vae=nn.Identity(), 
                 trainer=None
                 ):
        super().__init__()
        self.unet = unet
        self.lr = lr
        self.trainer = trainer
        self.constants_dict = constants_dict
        self.T = T
        self.T_sampling = T_sampling
        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion
        self.vae = vae
        if self.vae is not None:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=["unet", "vae"])
        self.automatic_optimization = True
        
    def forward(self, x, time, x_self_cond=None):
        return self.unet(x, time, x_self_cond)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr[0])
        warmup_epochs = 5
        total_epochs = self.trainer.max_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.lr[0]/self.lr[1],  # = 0.1
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        # Cosine annealing from 1e-3 → 0
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=0.0
        )

        # Combine warmup and cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def shared_step(self, batch, batch_idx):
        condition_data, prediction_data = batch
        if self.vae is not None:
            with torch.no_grad():
                x_cond = self.vae.encode(condition_data.to(self.vae.device)).latent_dist.sample() 
                x = self.vae.encode(prediction_data.to(self.vae.device)).latent_dist.sample() 
                x_cond = x_cond * self.vae.config.scaling_factor
                x = x * self.vae.config.scaling_factor
        else:
            x_cond = condition_data
            x = prediction_data

        batch_size_iter = x.shape[0]
        batch_t = torch.randint(0, self.T, (batch_size_iter,), device=self.device).long()
        # GENERATE GAUSSIAN NOISE
        noise = torch.randn_like(x)
        # CREATE NOISY IMAGE
        x_noisy = q_sample(self.constants_dict, x, batch_t, noise=noise)
        # PREDICT NOISE
        predicted_noise = self.forward(x_noisy, batch_t, x_cond)
        # COMPUTE LOSS
        # print("x std:", x.std().item(), "x_cond std:", x_cond.std().item())
        # print("x_noisy std:", x_noisy.std().item(), "noise std:", noise.std().item())
        # print("predicted_noise std:", predicted_noise.std().item())
        
        return noise, predicted_noise
    def training_step(self, batch, batch_idx):
        noise, predicted_noise = self.shared_step(batch, batch_idx)
        train_loss = self.train_criterion(noise, predicted_noise)       
        self.log("train_loss", train_loss, prog_bar=True, logger=True)
        # Log current LR — this is safe, auto-handles rank zero only
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, logger=True)
        return train_loss
    def validation_step(self, batch, batch_idx):
        noise, predicted_noise = self.shared_step(batch, batch_idx)
        validation_loss = self.validation_criterion(noise, predicted_noise)
        self.log("val_loss", validation_loss, prog_bar=True, sync_dist=True, logger=True, on_epoch=True)    
        return validation_loss
    
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state.items() if not k.startswith("vae.")}
    @torch.no_grad()  
    def test_step(self, batch, batch_idx):
        noise, predicted_noise = self.shared_step(batch, batch_idx)
        test_loss = self.validation_criterion(noise, predicted_noise)
        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True, logger=True)
        return test_loss
    
    @torch.no_grad()
    def sample_step(self, batch, batch_idx):
        """
        DDPM sampling step using precomputed constants from self.constants_dict.
        """
        condition_data, _ = batch
        device = self.device
        const = self.constants_dict
        T = self.T_sampling

        # === Encode condition ===
        if self.vae is not None:
            self.vae.eval()
            with torch.no_grad():
                x_cond = self.vae.encode(condition_data.to(self.vae.device)).latent_dist.sample()
                x_cond = x_cond * self.vae.config.scaling_factor
        else:
            x_cond = condition_data.to(device)

        # === Initialize noise in latent space ===
        shape = x_cond.shape
        x_t = torch.randn(shape, device=device)

        # === Preload constants ===
        betas = const["betas"].to(device)
        sqrt_recip_alphas = const["sqrt_recip_alphas"].to(device)
        sqrt_one_minus_alphas_cumprod = const["sqrt_one_minus_alphas_cumprod"].to(device)
        posterior_variance = const["posterior_variance"].to(device)

        for t in tqdm(reversed(range(T)), desc="Sampling DDPM", total=T):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            eps_theta = self.forward(x_t, t_batch, x_cond)

            # Predict x0
            x0_pred = (
                sqrt_recip_alphas[t] * (x_t - eps_theta * sqrt_one_minus_alphas_cumprod[t])
            )

            # Posterior mean estimate
            posterior_mean = (
                betas[t] / (1.0 - const["sqrt_alphas_cumprod"][t]) * x0_pred +
                (1.0 - const["sqrt_alphas_cumprod"][t] - betas[t]) / (1.0 - const["sqrt_alphas_cumprod"][t]) * x_t
            )

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = posterior_mean + posterior_variance[t].sqrt() * noise
            else:
                x_t = x0_pred  # at t=0, just return final denoised latent
                
        # === Decode final latent ===
        if self.vae is not None:
            x_t = x_t / self.vae.config.scaling_factor
            decoded = self.vae.decode(x_t).sample
            return decoded
        else:
            return x_t


    

### Validation: regarder le résultat dans l'espace d'origine
### Test: faire le débruitage complet 
### On reconstruit en ne tenant compte que du conditionnement