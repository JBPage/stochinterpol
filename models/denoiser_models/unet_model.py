# License : CC BY-NC-SA 4.0

import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
import pytorch_lightning as pl
from functools import partial
import random
from typing import Optional
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.nn.functional import mse_loss
import wandb
from einops import rearrange
from models.forward_diffusion import q_sample, extract
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR,ReduceLROnPlateau, CosineAnnealingWarmRestarts
import math
from tqdm import tqdm
import time

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

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545
    modified to include FiLM conditioning
    """

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
            nn.GroupNorm(1, dim_out * mult),
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
        film_cond_dim=None, # dimension of the condition vector for FiLM layers
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition

        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)
        # if we condition, we need to 4x the number of channels for conditioning 
        if self.self_condition:
            self.init_conv = nn.Conv2d(4*channels, init_dim, 7, padding=3)
        else:
            self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ConvNextBlock, mult=convnext_mult, cond_dim=film_cond_dim)

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
        x = self.mid_block1(x, t, x_cond_2)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, x_cond_2)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, x_cond_2)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)    
    

class DenoisingDiffusionModel(pl.LightningModule):
    def __init__(
                self, 
                 unet: Optional[nn.Module] = None, 
                 lr=[1e-3, 1e-4], 
                 train_criterion=nn.SmoothL1Loss(),
                 validation_criterion=partial(F.mse_loss,reduction='mean'), 
                 T=1000, 
                 constants_dict=None, 
                 prediciton_step=1,
                 vae_pop: Optional[nn.Module] = None,
                 vae_land: Optional[nn.Module] = None, 
                 trainer=None,
                 overfit_mode=False,
                 save_vae=False,
                 scheduler='plateau',
                 clipping_factor=1, # for the clipping technique in q_sample
                 ):
        super().__init__()
        self.overfit_mode = overfit_mode
        self.fixed_noises = {}   # to cache per-sample noise
        self.unet = unet
        self._external_models = {"vae":None}
        self.lr = lr
        self.scheduler = scheduler
        self.trainer = trainer
        self.constants_dict = constants_dict
        self.T = T
        self.prediction_step = prediciton_step
        self.clipping_factor = clipping_factor
        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion
        if vae_pop is not None:
            self._external_models["vae_pop"] = vae_pop
            self._external_models["vae_pop"].eval()
            for param in self._external_models["vae_pop"].parameters():
                param.requires_grad = False
        if vae_land is not None:
            vae_land = vae_land.to(self.device)
            self._external_models["vae_land"] = vae_land
            self._external_models["vae_land"].eval()
            for param in self._external_models["vae_land"].parameters():
                param.requires_grad = False
        if self._external_models["vae_pop"].device != self.device or self._external_models["vae_land"].device != self.device:
            self._external_models["vae_pop"] = self._external_models["vae_pop"].to(self.device)
            self._external_models["vae_land"] = self._external_models["vae_land"].to(self.device)
            print("0:Moved VAE models to device:", self._external_models["vae_pop"].device)
            print("0:Moved VAE landscape model to device:", self._external_models["vae_land"].device)

        self.save_hyperparameters( ignore= ([] if save_vae else ["vae_pop", "vae_land"]) )
        self.automatic_optimization = True
        # self.device = next(self.parameters()).device

        
    def on_fit_start(self):
        print("leaning rate:", self.lr[0])
        if self._external_models["vae"] is not None:
            self._external_models["vae"] = self._external_models["vae"].to(self.device, dtype=self.dtype)
    def forward(self, x, time, x_cond_1=None, x_cond_2=None):
        """ x_cond_1 : condition to be concatenated with x at input
            x_cond_2 : condition to be passed to FiLM layers throughout the network
        """ 
        return self.unet(x, time, x_cond_1, x_cond_2)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr[1])
        
        total_steps = (self.trainer.limit_train_batches * self.trainer.max_epochs) // self.trainer.accumulate_grad_batches
        warmup_steps = total_steps // 10

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            total_iters=warmup_steps, 
            start_factor=self.lr[0]/self.lr[1], 
            end_factor=1.0
        )

        if self.scheduler == "constant":
            # Trivial scheduler: LR stays fixed
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        elif self.scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - warmup_steps, 
                eta_min=0.0
            )

        elif self.scheduler == "cosine_restart":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10, 
                T_mult=2,
                eta_min=0.01*self.lr[1]
            )

        elif self.scheduler == "plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5,
            )
            return [optimizer], [{
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "monitor": "val_loss"   # must log this in validation_step
            }]

        else:
            raise ValueError(f"Unknown scheduler {self.scheduler}")

        # Combine warmup and cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[warmup_steps]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def on_after_backward(self):
        # Compute L2 norm of all gradients
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log to Lightning (and thus WandB)
        self.log("grad_norm", total_norm, on_step=True,on_epoch=True, prog_bar=True, logger=True)
    def shared_step(self, batch, batch_idx):
        const = {k: v.to(self.device) for k, v in self.constants_dict.items()}

        # condition_data_pop, condition_data_landscape, prediction_data = (x.to(self.device) for x in batch)
        condition_data_pop = batch["condition_data_pop"].to(self.device)
        condition_data_k = batch["condition_data_k"].to(self.device)
        condition_data_costhab = batch["condition_data_costhab"].to(self.device)
        prediction_data= batch["prediction_data"].to(self.device)

        ### Optional random rotation ###
        # condition_data_pop, condition_data_landscape, prediction_data = random_rotate(
        #     condition_data_pop, condition_data_landscape, prediction_data
        # )

        if self._external_models["vae_pop"] is not None and self._external_models["vae_land"] is not None:
            ## Sanity check: make sure the VAE models are on the same device as the current model
            if self._external_models["vae_pop"].device != self.device or self._external_models["vae_land"].device != self.device:
                self._external_models["vae_pop"] = self._external_models["vae_pop"].to(self.device)
                self._external_models["vae_land"] = self._external_models["vae_land"].to(self.device)
                # print("1:Moved VAE models to device:", self._external_models["vae_pop"].device)
                # print("1:Moved VAE landscape model to device:", self._external_models["vae_land"].device)

            ## Encode condition and prediction data with VAEs
            with torch.no_grad():
                x_cond_pop = self._external_models["vae_pop"].encode(condition_data_pop.to(self._external_models["vae_pop"].device)).latent_dist.sample()
                x_cond_k = self._external_models["vae_land"].encode(condition_data_k.to(self._external_models["vae_land"].device)).latent_dist.sample() 
                x_cond_costhab = self._external_models["vae_land"].encode(condition_data_costhab.to(self._external_models["vae_land"].device)).latent_dist.sample()
                x = self._external_models["vae_pop"].encode(prediction_data.to(self._external_models["vae_pop"].device)).latent_dist.sample() 
                x_cond_pop = x_cond_pop * self._external_models["vae_pop"].config.scaling_factor
                x_cond_k = x_cond_k * self._external_models["vae_land"].config.scaling_factor
                x_cond_costhab = x_cond_costhab * self._external_models["vae_land"].config.scaling_factor
                x = x * self._external_models["vae_pop"].config.scaling_factor
                # print("Encoded shapes:", x_cond_pop.shape, x_cond_k.shape, x_cond_costhab.shape, x.shape)
                x_cond = torch.cat((x_cond_pop, x_cond_k, x_cond_costhab), dim=1).to(self.device)
                x_cond_film = F.adaptive_avg_pool2d(x_cond_pop, (16,16)).flatten(1)  # condition vector would be 4*16*16
        else:
            x_cond = torch.cat((condition_data_pop, condition_data_k, condition_data_costhab), dim=1).to(self.device)
            x_cond_film  = F.adaptive_avg_pool2d(condition_data_pop, (16,16)).flatten(1)
            x = prediction_data.to(self.device)


        batch_size_iter = x.shape[0]
        batch_t = torch.randint(0, self.T, (batch_size_iter,), device=self.device).long()
        #### DEBUGGING NOISE######
        # if not hasattr(self, "fixed_noise"):
        #     # create it once, same shape as z
        #     self.fixed_noise = torch.randn_like(x).detach()
        # noise = self.fixed_noise.to(x.device)
        # x_noisy = q_sample(x, batch_t, noise)

        #### NORMAL NOISE######
        noise = torch.randn_like(x).to(batch_t.device)
        x_noisy = q_sample(const, x, batch_t, noise=noise, clipping=self.clipping_factor).to(batch_t.device)

        predicted_noise = self.forward(x_noisy, batch_t, x_cond, x_cond_film)
        # print(predicted_noise.shape, noise.shape, x.shape)
        return noise, predicted_noise, x
    def training_step(self, batch, batch_idx):
        noise, predicted_noise, x = self.shared_step(batch, batch_idx)
        # train_loss = self.train_criterion(noise, predicted_noise) 
        train_loss = self.train_criterion(noise, predicted_noise)      
        self.log("train_loss", 
                 train_loss, 
                 prog_bar=True, 
                 sync_dist=True,
                 on_epoch=True,
                 on_step=True,
                 logger=True
                 )
        # Log current LR — this is safe, auto-handles rank zero only
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", 
                 current_lr, 
                 prog_bar=True,
                 sync_dist=True, 
                 on_step=True,
                 on_epoch=True,
                 logger=True
                 )
        return train_loss
    def validation_step(self, batch, batch_idx):
        noise, predicted_noise, x = self.shared_step(batch, batch_idx)
        validation_loss = self.validation_criterion(noise, predicted_noise)
        print("Validation loss:", round(validation_loss.item(), 3))
        self.log("val_loss", 
                 validation_loss,
                 prog_bar=True, 
                 sync_dist=True, 
                 logger=True,
                 on_step=True,
                 on_epoch=True
                 )    
        return validation_loss
    
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state.items() if not k.startswith("vae.")}
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        condition_data_pop, condition_data_landscape, prediction_data = (x.to(self.device) for x in batch)
        condition_data_pop, condition_data_landscape, prediction_data = random_rotate(
            condition_data_pop, condition_data_landscape, prediction_data
        )

        device = self.device

        # === Run full DDPM sampling ===
        generated = self.sample_step(batch, batch_idx,decode=False).to(device)  # shape: (B, C, H, W)

        # === Compare with ground truth ===
        # Ground truth should be in [0,1] range as well
        if self._external_models["vae_pop"] is not None:
            target = prediction_data.to(self._external_models["vae_pop"].device)
            target = self._external_models["vae_pop"].encode(target).latent_dist.sample() 
            target = target / self._external_models["vae_pop"].config.scaling_factor
            # target = self._external_models["vae"].decode(target).sample
            # target = torch.clamp(target, 0, 1).to(device)
        else:
            target = prediction_data.to(device)

        # === Metrics ===
        mse = mse_loss(generated, target)
        ssim_val = ssim(generated, target)

        self.log("test_mse", mse, prog_bar=True, sync_dist=True)
        self.log("test_ssim", ssim_val, prog_bar=True, sync_dist=True)

        return {"mse": mse, "ssim": ssim_val}
    


    @torch.no_grad()
    def ddim_sample_step(self, batch, batch_idx, ddim_steps=100, eta=0.0):
        condition_data, _ = batch
        device = self.device
        vae = self._external_models["vae"]

        # === Encode condition ===
        if vae is not None:
            vae = vae.to(device=device, dtype=self.dtype)
            x_cond = vae.encode(condition_data.to(device)).latent_dist.sample()
            # x_cond = x_cond * vae.config.scaling_factor
        else:
            x_cond = condition_data.to(device)

        shape = x_cond.shape
        x_t = torch.randn(shape, device=device)

        # === Constants ===
        const = {k: v.to(device) for k, v in self.constants_dict.items()}
        betas = const["betas"]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # === Choose DDIM steps ===
        ddim_timesteps = torch.linspace(0, self.T - 1, steps=ddim_steps, dtype=torch.long, device=device)
        ddim_timesteps = ddim_timesteps.flip(0).long()

        # === Sampling loop with tqdm ===
        for i, t in enumerate(tqdm(ddim_timesteps, desc="DDIM Sampling", total=ddim_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            alpha_t = alphas_cumprod[t]
            alpha_prev = alphas_cumprod_prev[t]
            beta_t = betas[t]

            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            # Predict noise
            eps_theta = self.forward(x_t, t_batch, x_cond)

            # Predict x0
            x0_pred = (x_t - sqrt_one_minus_alpha_t * eps_theta) / sqrt_alpha_t

            # DDIM update
            sigma_t = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * beta_t)
            noise = torch.randn_like(x_t) if i < ddim_steps - 1 else torch.zeros_like(x_t)
            x_t = (
                torch.sqrt(alpha_prev) * x0_pred +
                torch.sqrt(1.0 - alpha_prev - sigma_t**2) * eps_theta +
                sigma_t * noise
            )

        # === Decode to original space ===
        if vae is not None:
            x_t = vae.decode(x_t).sample
            x_t = torch.clamp(x_t, 0, 1)

        return x_t
    @torch.no_grad()
    def p_sample(self, constants_dict, batch_xt, predicted_noise, batch_t, fixed_noise=None):
        # We first get every constants needed
        # `extract` will get the output in the same device as batch_t
        betas_t = extract(constants_dict['betas'], batch_t, batch_xt.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            constants_dict['sqrt_one_minus_alphas_cumprod'], batch_t, batch_xt.shape
        )
        sqrt_recip_alphas_t = extract(
            constants_dict['sqrt_recip_alphas'], batch_t, batch_xt.shape
        )
        
        # Equation 11 in the ddpm paper
        # Use predicted noise to predict the mean (mu theta)
        model_mean = sqrt_recip_alphas_t * (
            batch_xt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # We have to be careful to not add noise if we want to predict the final image
        predicted_image = torch.zeros(batch_xt.shape).to(batch_xt.device)
        t_zero_index = (batch_t == torch.zeros(batch_t.shape).to(batch_xt.device))
        
        # Algorithm 2 line 4, we add noise when timestep is not 1:
        posterior_variance_t = extract(constants_dict['posterior_variance'], batch_t, batch_xt.shape)
        if fixed_noise is not None:
            noise = fixed_noise
        else:
            noise = torch.randn_like(batch_xt)        
        # If t>0, we add noise to mu 
        predicted_image[~t_zero_index] = model_mean[~t_zero_index] + (
            torch.sqrt(posterior_variance_t[~t_zero_index]) * noise[~t_zero_index]
        ) 
        # If t=0, we don't add noise to mu
        predicted_image[t_zero_index] = model_mean[t_zero_index]
        
        return predicted_image

    @torch.no_grad()
    def p_sample_ode(self, constants_dict, batch_xt, predicted_noise, batch_t, clipping_factor=1.0):
        """
        Deterministic DDIM-style update (ODE sampling).
        constants_dict must contain:
        - 'sqrt_alphas_cumprod'  = sqrt(alpha_bar_t)
        - 'sqrt_one_minus_alphas_cumprod' = sqrt(1 - alpha_bar_t)
        """
        sqrt_ab = constants_dict['sqrt_alphas_cumprod']                # sqrt(alpha_bar_t) sequence
        sqrt_1m_ab = constants_dict['sqrt_one_minus_alphas_cumprod']   # sqrt(1 - alpha_bar_t)

        # a_t = sqrt(alpha_bar_t)
        a_t = extract(sqrt_ab, batch_t, batch_xt.shape)

        # a_prev = sqrt(alpha_bar_{t-1}), with a_prev for t=0 set to 1.0
        a_prev_full = F.pad(sqrt_ab[:-1], (1, 0), value=1.0)
        a_prev = extract(a_prev_full, batch_t, batch_xt.shape)

        sqrt_1_minus_a_t = extract(sqrt_1m_ab, batch_t, batch_xt.shape)

    # 🔥 Apply clipping correction on xt
        xt_scaled = clipping_factor * batch_xt
    
        # reconstruct x0_hat
        x0_pred = (xt_scaled - sqrt_1_minus_a_t * predicted_noise) / a_t

        # deterministic DDIM/ODE update: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat
        x_t_prev = a_prev * x0_pred

        return x_t_prev

    @torch.no_grad()
    def sampling(self, batch, batch_idx, T_sampling=None, constant_dict_sampling=None, decode=True,
                intermediate_samples_period=None, fixed_noise=None, deterministic=False, clipping_factor=1):
        """
        DDPM sampling step using precomputed constants from self.constants_dict.
        - deterministic=False: standard DDPM (stochastic)
        - deterministic=True: ODE / DDIM-style deterministic sampling (no noise)
        """
        condition_data_pop, condition_data_k, condition_data_costhab,prediction_data = (x.to(self.device) for x in batch)


        # Set sampling timesteps
        T_sampling = T_sampling or self.T
        constants_dict = {k: v.to(self.device) for k, v in (constant_dict_sampling.items() if constant_dict_sampling else self.constants_dict.items())}

        # Encode conditioning with VAEs if available
        if self._external_models["vae_pop"] and self._external_models["vae_land"]:
            for key in ["vae_pop", "vae_land"]:
                if self._external_models[key].device != self.device:
                    self._external_models[key] = self._external_models[key].to(self.device)
                ## Encode condition and prediction data with VAEs
            with torch.no_grad():
                x_cond_pop = self._external_models["vae_pop"].encode(condition_data_pop.to(self._external_models["vae_pop"].device)).latent_dist.sample()
                x_cond_k = self._external_models["vae_land"].encode(condition_data_k.to(self._external_models["vae_land"].device)).latent_dist.sample() 
                x_cond_costhab = self._external_models["vae_land"].encode(condition_data_costhab.to(self._external_models["vae_land"].device)).latent_dist.sample()
                x = self._external_models["vae_pop"].encode(prediction_data.to(self._external_models["vae_pop"].device)).latent_dist.sample() 
                x_cond_pop = x_cond_pop * self._external_models["vae_pop"].config.scaling_factor
                x_cond_k = x_cond_k * self._external_models["vae_land"].config.scaling_factor
                x_cond_costhab = x_cond_costhab * self._external_models["vae_land"].config.scaling_factor
                x_pred = x * self._external_models["vae_pop"].config.scaling_factor
                # print("Encoded shapes:", x_cond_pop.shape, x_cond_k.shape, x_cond_costhab.shape, x.shape)
                batch_x_cond = torch.cat((x_cond_pop, x_cond_k, x_cond_costhab), dim=1).to(self.device)
                x_cond_film = F.adaptive_avg_pool2d(x_cond_pop, (16,16)).flatten(1)  # condition vector would be 4*16*16
        else:
            batch_x_cond = torch.cat((condition_data_pop, condition_data_k, condition_data_costhab), dim=1).to(self.device)
            x_cond_film  = F.adaptive_avg_pool2d(condition_data_pop, (16,16)).flatten(1)
            x_pred = prediction_data.to(self.device)


        # Initialize latent noise
        shape = x_pred.shape
        batch_xt = torch.randn(shape, device=self.device)
        intermediate_samples = [batch_xt]

        # Sampling loop
        for t in tqdm(reversed(range(T_sampling)), desc="Sampling DDPM", total=T_sampling):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.forward(batch_xt, t_batch, batch_x_cond, x_cond_film)

            if deterministic:
                # DDIM/ODE: remove noise injection
                batch_xt = self.p_sample_ode(constants_dict, batch_xt, predicted_noise, t_batch, clipping_factor=1)
            else:
                # Standard DDPM: add noise
                batch_xt = self.p_sample(constants_dict, batch_xt, predicted_noise, t_batch, fixed_noise=fixed_noise)

            if intermediate_samples_period is not None and t % intermediate_samples_period == 0:
                intermediate_samples.append(batch_xt)

        if intermediate_samples_period and (len(intermediate_samples) == 0 or intermediate_samples[-1] is not batch_xt):
            intermediate_samples.append(batch_xt)
        if intermediate_samples_period:
            return intermediate_samples

        # Decode final latent if requested
        if decode and self._external_models["vae_pop"]:
            x_t = self._external_models["vae_pop"].decode(batch_xt / self._external_models["vae_pop"].config.scaling_factor).sample
            x_t = torch.clamp(x_t, 0, 1)
            return x_t

        return batch_xt

    @torch.no_grad()
    def iterated_sample_step(self, batch, batch_idx, prediction_steps=1, decode=True):
        """
        Iterative DDPM sampling with progress bars for both steps and timesteps.
        """
        condition_data, _ = batch
        device = self.device
        const = self.constants_dict
        T = self.T_sampling
        vae = self._external_models["vae"]
        vae = vae.to(device=self.device, dtype=self.dtype)

        # Preload constants
        betas = const["betas"].to(device)
        sqrt_recip_alphas = const["sqrt_recip_alphas"].to(device)
        sqrt_one_minus_alphas_cumprod = const["sqrt_one_minus_alphas_cumprod"].to(device)
        posterior_variance = const["posterior_variance"].to(device)
        sqrt_alphas_cumprod = const["sqrt_alphas_cumprod"].to(device)

        preds = []

        # === Encode current condition ===
        if vae is not None:
            vae.eval()
            x_cond = vae.encode(condition_data.to(vae.device)).latent_dist.sample()
            # x_cond = x_cond * vae.config.scaling_factor
        else:
            x_cond = condition_data.to(device)

        outer_bar = tqdm(range(prediction_steps), desc="Predicting Steps", position=0)
        for step in outer_bar:
            shape = x_cond.shape
            x_t = torch.randn(shape, device=device)

            # === DDPM reverse process ===
            inner_bar = tqdm(reversed(range(T)), desc=f"DDPM Step {step+1}", position=1, leave=False)
            for t in inner_bar:
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

                # Predict noise
                eps_theta = self.forward(x_t, t_batch, x_cond)

                # Predict x0
                x0_pred = sqrt_recip_alphas[t] * (x_t - eps_theta * sqrt_one_minus_alphas_cumprod[t])

                # Posterior mean
                posterior_mean = (
                    betas[t] / (1.0 - sqrt_alphas_cumprod[t]) * x0_pred +
                    (1.0 - sqrt_alphas_cumprod[t] - betas[t]) / (1.0 - sqrt_alphas_cumprod[t]) * x_t
                )

                if t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = posterior_mean + posterior_variance[t].sqrt() * noise
                else:
                    x_t = x0_pred
                    
            x_cond = x_t
            # === Decode and store ===
            if vae is not None and decode:
                prediction = vae.decode(x_t).sample
                prediction = torch.clamp(prediction, 0, 1)
            else:
                prediction = x_t

            preds.append(prediction)

        return preds  # list of predictions


class DenoisingDiffusionModel_2conds(pl.LightningModule):
    def __init__(
                self, 
                 unet: Optional[nn.Module] = None, 
                 lr=[1e-3, 1e-4], 
                 train_criterion=nn.SmoothL1Loss(),
                 validation_criterion=partial(F.mse_loss,reduction='mean'), 
                 T=1000, 
                 constants_dict=None, 
                 prediciton_step=1,
                 vae_pop: Optional[nn.Module] = None,
                 vae_land: Optional[nn.Module] = None, 
                 trainer=None,
                 overfit_mode=False,
                 save_vae=False,
                 scheduler='plateau',
                 clipping_factor=1, # for the clipping technique in q_sample
                 ):
        super().__init__()
        self.overfit_mode = overfit_mode
        self.fixed_noises = {}   # to cache per-sample noise
        self.unet = unet
        self._external_models = {"vae":None}
        self.lr = lr
        self.scheduler = scheduler
        self.trainer = trainer
        self.constants_dict = constants_dict
        self.T = T
        self.prediction_step = prediciton_step
        self.clipping_factor = clipping_factor
        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion
        if vae_pop is not None:
            self._external_models["vae_pop"] = vae_pop
            self._external_models["vae_pop"].eval()
            for param in self._external_models["vae_pop"].parameters():
                param.requires_grad = False
        if vae_land is not None:
            vae_land = vae_land.to(self.device)
            self._external_models["vae_land"] = vae_land
            self._external_models["vae_land"].eval()
            for param in self._external_models["vae_land"].parameters():
                param.requires_grad = False
        if self._external_models["vae_pop"].device != self.device or self._external_models["vae_land"].device != self.device:
            self._external_models["vae_pop"] = self._external_models["vae_pop"].to(self.device)
            self._external_models["vae_land"] = self._external_models["vae_land"].to(self.device)
            print("0:Moved VAE models to device:", self._external_models["vae_pop"].device)
            print("0:Moved VAE landscape model to device:", self._external_models["vae_land"].device)

        self.save_hyperparameters( ignore= ([] if save_vae else ["vae_pop", "vae_land"]) )
        self.automatic_optimization = True
        # self.device = next(self.parameters()).device

        
    def on_fit_start(self):
        print("leaning rate:", self.lr[0])
        if self._external_models["vae"] is not None:
            self._external_models["vae"] = self._external_models["vae"].to(self.device, dtype=self.dtype)
    def forward(self, x, time, x_cond_1=None):
        """ x_cond_1 : condition to be concatenated with x at input
            x_cond_2 : condition to be passed to FiLM layers throughout the network
        """ 
        return self.unet(x, time, x_cond_1)
    

    @torch.no_grad()
    def p_sample(self, constants_dict, batch_xt, predicted_noise, batch_t, fixed_noise=None):
        # We first get every constants needed
        # `extract` will get the output in the same device as batch_t
        betas_t = extract(constants_dict['betas'], batch_t, batch_xt.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            constants_dict['sqrt_one_minus_alphas_cumprod'], batch_t, batch_xt.shape
        )
        sqrt_recip_alphas_t = extract(
            constants_dict['sqrt_recip_alphas'], batch_t, batch_xt.shape
        )
        
        # Equation 11 in the ddpm paper
        # Use predicted noise to predict the mean (mu theta)
        model_mean = sqrt_recip_alphas_t * (
            batch_xt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # We have to be careful to not add noise if we want to predict the final image
        predicted_image = torch.zeros(batch_xt.shape).to(batch_xt.device)
        t_zero_index = (batch_t == torch.zeros(batch_t.shape).to(batch_xt.device))
        
        # Algorithm 2 line 4, we add noise when timestep is not 1:
        posterior_variance_t = extract(constants_dict['posterior_variance'], batch_t, batch_xt.shape)
        if fixed_noise is not None:
            noise = fixed_noise
        else:
            noise = torch.randn_like(batch_xt)        
        # If t>0, we add noise to mu 
        predicted_image[~t_zero_index] = model_mean[~t_zero_index] + (
            torch.sqrt(posterior_variance_t[~t_zero_index]) * noise[~t_zero_index]
        ) 
        # If t=0, we don't add noise to mu
        predicted_image[t_zero_index] = model_mean[t_zero_index]
        
        return predicted_image

    @torch.no_grad()
    def p_sample_ode(self, constants_dict, batch_xt, predicted_noise, batch_t, clipping_factor=1.0):
        """
        Deterministic DDIM-style update (ODE sampling).
        constants_dict must contain:
        - 'sqrt_alphas_cumprod'  = sqrt(alpha_bar_t)
        - 'sqrt_one_minus_alphas_cumprod' = sqrt(1 - alpha_bar_t)
        """
        sqrt_ab = constants_dict['sqrt_alphas_cumprod']                # sqrt(alpha_bar_t) sequence
        sqrt_1m_ab = constants_dict['sqrt_one_minus_alphas_cumprod']   # sqrt(1 - alpha_bar_t)

        # a_t = sqrt(alpha_bar_t)
        a_t = extract(sqrt_ab, batch_t, batch_xt.shape)

        # a_prev = sqrt(alpha_bar_{t-1}), with a_prev for t=0 set to 1.0
        a_prev_full = F.pad(sqrt_ab[:-1], (1, 0), value=1.0)
        a_prev = extract(a_prev_full, batch_t, batch_xt.shape)

        sqrt_1_minus_a_t = extract(sqrt_1m_ab, batch_t, batch_xt.shape)

    # 🔥 Apply clipping correction on xt
        xt_scaled = clipping_factor * batch_xt
    
        # reconstruct x0_hat
        x0_pred = (xt_scaled - sqrt_1_minus_a_t * predicted_noise) / a_t

        # deterministic DDIM/ODE update: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat
        x_t_prev = a_prev * x0_pred

        return x_t_prev

    @torch.no_grad()
    def sampling(self, batch, batch_idx, T_sampling=None, constant_dict_sampling=None, decode=True,
                intermediate_samples_period=None, fixed_noise=None, deterministic=False, clipping_factor=1):
        """
        DDPM sampling step using precomputed constants from self.constants_dict.
        - deterministic=False: standard DDPM (stochastic)
        - deterministic=True: ODE / DDIM-style deterministic sampling (no noise)
        """
        condition_data_pop, condition_data_land ,prediction_data = (x.to(self.device) for x in batch)


        # Set sampling timesteps
        T_sampling = T_sampling or self.T
        constants_dict = {k: v.to(self.device) for k, v in (constant_dict_sampling.items() if constant_dict_sampling else self.constants_dict.items())}

        # Encode conditioning with VAEs if available
        if self._external_models["vae_pop"] and self._external_models["vae_land"]:
            for key in ["vae_pop", "vae_land"]:
                if self._external_models[key].device != self.device:
                    self._external_models[key] = self._external_models[key].to(self.device)
                ## Encode condition and prediction data with VAEs
            with torch.no_grad():
                x_cond_pop = self._external_models["vae_pop"].encode(condition_data_pop.to(self._external_models["vae_pop"].device)).latent_dist.sample()
                x_cond_land = self._external_models["vae_land"].encode(condition_data_land.to(self._external_models["vae_land"].device)).latent_dist.sample() 
                x = self._external_models["vae_pop"].encode(prediction_data.to(self._external_models["vae_pop"].device)).latent_dist.sample() 
                x_cond_pop = x_cond_pop * self._external_models["vae_pop"].config.scaling_factor
                x_cond_land = x_cond_land * self._external_models["vae_land"].config.scaling_factor
                x_pred = x * self._external_models["vae_pop"].config.scaling_factor
                batch_x_cond = torch.cat((x_cond_pop, x_cond_land), dim=1).to(self.device)
        else:
            batch_x_cond = torch.cat((condition_data_pop, condition_data_land), dim=1).to(self.device)
            x_pred = prediction_data.to(self.device)


        # Initialize latent noise
        shape = x_pred.shape
        batch_xt = torch.randn(shape, device=self.device)
        intermediate_samples = [batch_xt]

        # Sampling loop
        for t in tqdm(reversed(range(T_sampling)), desc="Sampling DDPM", total=T_sampling):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.forward(batch_xt, t_batch, batch_x_cond)

            if deterministic:
                # DDIM/ODE: remove noise injection
                batch_xt = self.p_sample_ode(constants_dict, batch_xt, predicted_noise, t_batch, clipping_factor=1)
            else:
                # Standard DDPM: add noise
                batch_xt = self.p_sample(constants_dict, batch_xt, predicted_noise, t_batch, fixed_noise=fixed_noise)

            if intermediate_samples_period is not None and t % intermediate_samples_period == 0:
                intermediate_samples.append(batch_xt)

        if intermediate_samples_period and (len(intermediate_samples) == 0 or intermediate_samples[-1] is not batch_xt):
            intermediate_samples.append(batch_xt)
        if intermediate_samples_period:
            return intermediate_samples

        # Decode final latent if requested
        if decode and self._external_models["vae_pop"]:
            x_t = self._external_models["vae_pop"].decode(batch_xt / self._external_models["vae_pop"].config.scaling_factor).sample
            x_t = torch.clamp(x_t, 0, 1)
            return x_t

        return batch_xt

    

### Validation: regarder le résultat dans l'espace d'origine
### Test: faire le débruitage complet 
### On reconstruit en ne tenant compte que du conditionnement

