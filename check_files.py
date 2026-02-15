import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from models.utils_files.dataloading_utils import MyDataModule, log_wandb_config, MyDistributedIterableDataset, EMACallback
import argparse
from models.sdxl_vae import SDXLAELightning
from models.forward_diffusion import linear_beta_schedule, cosine_beta_schedule, get_alph_bet
# from Python.DDPM.models.denoiser_models.unet_model import Unet, DenoisingDiffusionModel
from models.denoiser_models.standard_unet import Unet, Unet_stochinterpolant_1, Unet_filmconcat_cond
from models.stochinterpolmodel import UnetModel
import os
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler
import torch.nn.functional as F
from functools import partial
import ast

import wandb
from collections import OrderedDict
import functools
from torch.serialization import add_safe_globals

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from types import SimpleNamespace
from datetime import date
from peft import LoraConfig

if __name__ == '__main__':
    folders = [os.path.join(os.getenv("DATA_DIR"), "landscape_{i}".format(i=i)) for i in range(1, 301)]
    valid_files = []
    for folder in folders:
        file_path = os.path.join(folder, "Output_Maps", "Population_maps_latent.pt")
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                torch.load(file_path)  # Teste le chargement
                valid_files.append(folder)
            except:
                print(f"Fichier invalide : {file_path}")