import os

import torch
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader


class CVAE2d(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)  # TODO: check if this is safe
        self.encoder = torch.nn.Conv2d(
            self.image_dims[-1], self.latent_dim, 3, stride=2, padding=2
        )
        self.decoder = torch.nn.ConvTranspose2d(
            self.latent_dim,
            self.image_dims[-1],
            3,
            stride=2,
            padding=2,
            output_padding=1,
        )

    def forward(self, x):
        latent = self.encoder(x)
        return torch.sigmoid(self.decoder(latent))

    def training_step(self, batch, batch_nb):
        x, _ = batch
        pred = self.forward(x)
        loss = F.mse_loss(pred, x)
        return {"loss": loss}

    def train_dataloader(self):
        pass

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            torchvision.datasets.MNIST(
                os.getcwd(),
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            ),
            batch_size=self.batch_size,
        )
