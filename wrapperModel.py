from typing import Any
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from model import S3D
from utils import load_parameters


class WrapperModel(pl.LightningModule):
    def __init__(self, num_class: int):
        super().__init__()
        self.model = S3D(num_class=num_class)
        load_parameters('./S3D_kinetics400.pt', self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        # log val loss
        # accuracy
        return loss

    def configure_optimizers(self) -> Any:
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-03)
        return optim

