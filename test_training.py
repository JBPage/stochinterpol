import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from diffusers import AutoencoderKL
import argparse
from models.sdxl_vae import SDXLAELightning
from models.forward_diffusion import linear_beta_schedule, cosine_beta_schedule, get_alph_bet
# from Python.DDPM.models.denoiser_models.unet_model import Unet, DenoisingDiffusionModel
from models.denoiser_models.standard_unet import Unet, Unet_stochinterpolant_1, Unet_filmconcat_cond
from models.stochinterpolmodel import StochasticInterpolentModel
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler
import torch.nn.functional as F
from functools import partial
import ast

import functools
from torch.serialization import add_safe_globals


from datetime import date

add_safe_globals([functools.partial])
from torch.utils.data import DataLoader, Dataset

class RandomDictDataset(Dataset):
    def __init__(self, num_samples, batch_size):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retourne un dictionnaire avec des tenseurs aléatoires
        return {
            "condition_data_pop": torch.rand(3, 1024, 1024),
            "condition_data_k": torch.rand(3, 1024, 1024),
            "condition_data_costhab": torch.rand(3, 1024, 1024),
            "prediction_data": torch.rand(3, 1024, 1024),
        }

class RandomDictDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, num_samples_train=100, num_samples_val=20, num_samples_test=20, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.num_samples_test = num_samples_test
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Initialiser les datasets
        self.train_dataset = RandomDictDataset(self.num_samples_train, self.batch_size)
        self.val_dataset = RandomDictDataset(self.num_samples_val, self.batch_size)
        self.test_dataset = RandomDictDataset(self.num_samples_test, self.batch_size)

    def collate_fn(self, batch):
        # Fonction pour combiner les échantillons en un batch
        result = {
            "condition_data_pop": torch.stack([item["condition_data_pop"] for item in batch], dim=0),
            "condition_data_k": torch.stack([item["condition_data_k"] for item in batch], dim=0),
            "condition_data_costhab": torch.stack([item["condition_data_costhab"] for item in batch], dim=0),
            "prediction_data": torch.stack([item["prediction_data"] for item in batch], dim=0),
        }
        return result

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

if __name__ == '__main__':

    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device, torch.cuda.get_device_name() if torch.cuda.is_available() else "")
    if device == 'cuda':
        if "A100" in torch.cuda.get_device_name().split('-'):
            print("Running on A100")
            torch.set_float32_matmul_precision('high')  # Enable Tensor Core acceleration
    print("A100" in torch.cuda.get_device_name() if torch.cuda.is_available() else "")
    print(torch.get_float32_matmul_precision())



    today = date.today()
    formatted = today.strftime("%Y_%m_%d")
    
    # Hyperparameters
        # load the vae for encoding 
    vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            cache_dir = os.getenv("CACHE_DIR") if os.getenv("CACHE_DIR") is not None else None,
            in_channels = 3,
            out_channels = 3,
            torch_dtype=torch.float32
            )

    random_data_module = RandomDictDataModule(batch_size=batch_size)
    

    unet = Unet_filmconcat_cond(
            dim=128, #for conditioning 
            init_dim=None,
            out_dim=None,
            dim_mults=(1,2,4,8),
            channels=4,
            self_condition_size=2,
            with_time_emb=True,
            convnext_mult=2,
            GroupNorm=True,
        )



    wandb_logger = None
    trainer = Trainer(
        limit_train_batches=10, 
        limit_val_batches=5,
        limit_test_batches=5,
        precision="32-true",#"bf16-mixed", #if args.mixed_precision else "16-mixed",
        strategy="ddp", #args.trainer_strategy,
        num_nodes=1,
        gradient_clip_val=0.0, #5.0
        gradient_clip_algorithm="norm",
        accelerator="gpu", 
        devices="auto",
        accumulate_grad_batches=1,
        max_epochs=10,
        )
    model = StochasticInterpolentModel(
        denoiser=unet,
        # criterion=criterion,
        train_criterion=partial(F.mse_loss,reduction='mean'),
        trainer=trainer,
        lr=1e-4,
        save_vae=False,
        vae_pop=vae,
        vae_land=vae,
        scheduler='constant',
        lambda_eta=0.0,
        noise_scale=0.0,
        )

    print("Starting training...")
    trainer.fit(
        model=model,
        datamodule=random_data_module,
    )

