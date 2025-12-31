import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
# from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

class SDXLAELightning(pl.LightningModule):
    def __init__(self, 
                 model, 
                 training_criterion=nn.SmoothL1Loss(),
                 validation_criterion=nn.MSELoss(),
                 trainer_strategy=None,
                 trainer=None,
                 lr=[1e-5, 1e-4]  # [min_lr, max_lr]
                 ):
        super().__init__()
        self.model = model
        self.training_criterion = training_criterion
        self.validation_criterion = validation_criterion
        self.trainer_strategy = trainer_strategy
        self.trainer = trainer
        self.lr_min = lr[0]
        self.lr_max = lr[1]

        # self.save_hyperparameters({
        #     'lr': lr,
        #     'trainer_strategy': trainer.trainer_strategy
        # })
    def forward(self, x):
        return self.model(x).sample  # Assuming we need the reconstructed output

    def shared_step(self, batch):
        inputs = batch
        outputs = self.model(inputs).sample
        outputs = torch.clamp(outputs, 0, 1)
        
        return inputs, outputs

    def training_step(self, batch, batch_idx):
        inputs, outputs = self.shared_step(batch)
        loss = self.training_criterion(outputs, inputs)
        self.log("train_loss", loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)
        # Log current LR — this is safe, auto-handles rank zero only
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, outputs = self.shared_step(batch)
        loss = self.validation_criterion(outputs, inputs)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        population_loss = self.validation_criterion(outputs[:, 2:3, :, :], inputs[:, 2:3, :, :])
        self.log("population_loss", population_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return {"val_loss": loss}  

    def test_step(self, batch, batch_idx):
        inputs, outputs = self.shared_step(batch)
        loss = self.validation_criterion(outputs, inputs)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True, logger=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.lr_min
        )
        warmup_epochs = 1
        total_epochs = self.trainer.max_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.lr_min/self.lr_max,  # = 0.1
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
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss","frequency": 1}]
