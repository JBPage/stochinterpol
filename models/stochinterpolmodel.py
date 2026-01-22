# License : CC BY-NC-SA 4.0

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial
from typing import Optional
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.nn.functional import mse_loss
import bitsandbytes as bnb
from models.utils_files.nn_utils import *
from models.denoiser_models.standard_unet import Unet
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, LinearLR, CosineAnnealingLR,ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm

class StochasticInterpolentModel(pl.LightningModule):
    def __init__(
                self, 
                 denoiser: Optional[nn.Module] = None, # if None, will create a standard Unet
                 lr=[1e-3, 1e-4], 
                 train_criterion=nn.SmoothL1Loss(),
                 validation_criterion=partial(F.mse_loss,reduction='mean'), 
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

        if denoiser is None:
            denoiser = Unet(
                dim=64,
                init_dim=128,
                out_dim=64,
                dim_mults=(1, 2, 4, 8),
                channels=3,
                with_time_emb=True,
                self_condition_size=0, # no self-conditioning
                GroupNorm=True,
                film_cond_dim=256, # dimension of the condition vector for FiLM layers
                convnext_mult=2,
            )
            print("Created standard Unet as denoiser model")
        self.denoiser = denoiser        
        self._external_models = {"vae":None} # to store external models like VAE so they are not considered as parameters of this model
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

        self.lr = lr
        self.scheduler = scheduler
        self.trainer = trainer
        self.prediction_step = prediciton_step
        self.clipping_factor = clipping_factor
        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion
        self.validation_step_outputs = []
        # CORRECTION : ignorer les nn.Module et le trainer
        self.save_hyperparameters(ignore=[
            'denoiser', 
            'train_criterion', 
            'vae_pop', 
            'vae_land', 
            'trainer'
        ])
        self.automatic_optimization = True
        # self.device = next(self.parameters()).device

        
    def on_fit_start(self):
        print("leaning rate:", self.lr[0])
        if self._external_models["vae"] is not None:
            self._external_models["vae"] = self._external_models["vae"].to(self.device, dtype=self.dtype)
    def forward(self, x, time, x_cond_1=None, x_condfilm_1=None, x_condfilm_2=None):
        """ x_cond_1 : condition to be concatenated with x at input
            x_cond_2 : condition to be passed to FiLM layers throughout the network
        """ 
        return self.denoiser(x, time, x_cond_1, x_condfilm_1, x_condfilm_2)
    
    def configure_optimizers(self):
        optimizer = bnb.optim.Adam8bit(self.denoiser.parameters(), lr=self.lr[1]) #torch.optim.AdamW(self.denoiser.parameters(), lr=self.lr[1])
        
        total_steps = (self.trainer.limit_train_batches * self.trainer.max_epochs) // self.trainer.accumulate_grad_batches
        warmup_steps = total_steps // 10 if total_steps < 1000 else 1000

        warmup_scheduler = LinearLR(
            optimizer, 
            total_iters=warmup_steps, 
            start_factor=self.lr[0]/self.lr[1], 
            end_factor=1.0
        )

        if self.scheduler == "constant":
            # Trivial scheduler: LR stays fixed
            lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        elif self.scheduler == "cosine":
            lr_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - warmup_steps, 
                eta_min=0.0
            )

        elif self.scheduler == "cosine_restart":
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=50, 
                T_mult=2,
                eta_min=1e-6, #0.01*self.lr[1]
            )

        elif self.scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
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
        scheduler = SequentialLR(
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
    def shared_step(self, batch, batch_idx, gamma_func=None):
        """
        z0: latent at time t
        z1: latent at time t+1
        cond: conditioning tuple for model (e.g. landscape latents)
        """
        # condition_data_pop, condition_data_landscape, prediction_data = (x.to(self.device) for x in batch)
        condition_data_pop = batch["condition_data_pop"].to(self.device)
        condition_data_k = batch["condition_data_k"].to(self.device)
        condition_data_costhab = batch["condition_data_costhab"].to(self.device)
        prediction_data= batch["prediction_data"].to(self.device)

        const = 1e-4
        
        if gamma_func is None:
            gamma_func = lambda t: const * torch.sin(np.pi * t)

        # derivative of s(t)
        def s_prime(t):
            return const * np.pi * torch.cos(np.pi * t)

        pop_scaling_factor = self._external_models["vae_pop"].config.scaling_factor
        land_scaling_factor = self._external_models["vae_land"].config.scaling_factor

        if self._external_models["vae_pop"].device != self.device or self._external_models["vae_land"].device != self.device:
            self._external_models["vae_pop"] = self._external_models["vae_pop"].to(self.device)
            self._external_models["vae_land"] = self._external_models["vae_land"].to(self.device)
            # print("1:Moved VAE models to device:", self._external_models["vae_pop"].device)
            # print("1:Moved VAE landscape model to device:", self._external_models["vae_land"].device)

        ## Encode condition and prediction data with VAEs
        with torch.no_grad():
            x_cond_pop = self._external_models["vae_pop"].encode(condition_data_pop.to(self._external_models["vae_pop"].device)).latent_dist.sample() * pop_scaling_factor
            x = self._external_models["vae_pop"].encode(prediction_data.to(self._external_models["vae_pop"].device)).latent_dist.sample() * pop_scaling_factor
            x_condfilm_1 = condition_data_k* land_scaling_factor #we pre-encoded and stored the landscape latents to save time
            x_condfilm_2 = condition_data_costhab * land_scaling_factor #we pre-encoded and stored the landscape latents to save time
            x_cond_1 = (x_condfilm_1, x_condfilm_2)
            # print("Encoded shapes:", x_cond_pop.shape, x_cond_k.shape, x_cond_costhab.shape, x.shape)

        # sample t ~ Uniform(0,1)
        batch_t = torch.rand(x.shape[0], device=x.device)

        # sample Gaussian noise
        z = torch.randn_like(x).to(self.device)

        # build x(t) = linear interpolant + noise
        gamma_t = gamma_func(batch_t)[:, None, None, None]
        x_t = (1 - batch_t)[:, None, None, None] * x_cond_pop \
            + batch_t[:, None, None, None] * x \
            + gamma_t * z

        b_true = (x - x_cond_pop) + s_prime(batch_t)[:, None, None, None] * z # true velocity from paper notation

        b_pred, eta_pred = self.forward(x_t, batch_t, x_cond_1, x_condfilm_1, x_condfilm_2)

        return (b_pred, b_true), (eta_pred, z), gamma_t

    def training_step(self, batch, batch_idx):
        (b_pred, b_true), (eta_pred, z), gamma_t = self.shared_step(batch, batch_idx)
        lambda_eta = gamma_t.mean()**2

        L_b = self.train_criterion(b_pred, b_true)   
        L_nz = self.train_criterion(eta_pred, z)
        train_loss = L_b + lambda_eta * L_nz
        self.log("L_b", 
                 L_b, 
                 prog_bar=True, 
                 sync_dist=True,
                #  on_epoch=True,
                 on_step=True,
                 logger=True
                 )
        self.log("L_nz", 
                 L_nz, 
                 prog_bar=True, 
                 sync_dist=True,
                #  on_epoch=True,
                 on_step=True,
                 logger=True
                 )
        self.log("train_loss", 
                 train_loss, 
                 prog_bar=True, 
                 sync_dist=True,
                 on_epoch=False,
                 on_step=True,
                 logger=True
                 )
        # Log current LR — this is safe, auto-handles rank zero only
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", 
                 current_lr, 
                 prog_bar=True,
                 sync_dist=True, 
                 on_step=False,
                 on_epoch=True,
                 logger=True
                 )
        return train_loss
    def validation_step(self, batch, batch_idx):
        (b_pred, b_true), (eta_pred, z), gamma_t = self.shared_step(batch, batch_idx)
        lambda_eta = gamma_t.mean()**2

        Lb = self.validation_criterion(b_pred, b_true)   
        Ln = self.validation_criterion(eta_pred, z)

        validation_loss = Lb + lambda_eta * Ln
        # print("Validation loss:", round(validation_loss.item(), 3))
        self.log("L_b_val", Lb, prog_bar=True, sync_dist=True,
            logger=True, on_step=False, on_epoch=True)
        self.log("L_nz_val", Ln, prog_bar=True, sync_dist=True,
            logger=True, on_step=False, on_epoch=True)
        self.log("val_loss", validation_loss, prog_bar=True, sync_dist=True,
            logger=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.append(validation_loss)
        return validation_loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        # Compute mean validation loss
        val_losses = [x.item() for x in self.validation_step_outputs]
        epoch_mean_loss = sum(val_losses) / len(val_losses)

        # Only global rank 0 writes to disk
        if self.trainer.is_global_zero:
            import os

            # Folder
            out_dir = "./val_losses"
            os.makedirs(out_dir, exist_ok=True)

            # Job ID (SLURM first, fallback otherwise)
            job_id = (
                os.environ.get("RUN_ID")
                or os.environ.get("SLURM_JOB_ID")
                or os.environ.get("JOB_ID")
                or "unknown"
            )

            # File path
            txt_path = os.path.join(
                out_dir,
                f"val_loss_epoch_{job_id}.txt"
            )
            print(f"Writing validation losses to {txt_path}")
            # Append epoch + loss
            with open(txt_path, "a") as f:
                f.write(f"{self.current_epoch}\t{epoch_mean_loss:.6f}\n")

        # Clear stored outputs
        self.validation_step_outputs.clear()


    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state.items() if not k.startswith("vae.")}

