from abc import ABC, abstractmethod
import tempfile

# Torch
import torch
from torch.nn import functional as F
import torchvision
import pytorch_lightning as pl


class CVAE2D(pl.LightningModule):
    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def tng_dataloader(self):
        pass

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.sigmoid(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        return {"loss": loss}

    def train_dataloader(self):
        pass
