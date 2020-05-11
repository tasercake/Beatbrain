import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from cached_property import cached_property

from ..utils.config import Config


class MNISTAutoencoder(pl.LightningModule):
    def __init__(self, hparams: Config):
        super().__init__()
        self.hparams = hparams
        self.latent_dim = hparams.latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        return {"loss": loss}

    @cached_property
    def default_train_transform(self):
        return transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        """
        Split train dataset into train and val (80-20)
        """
        train_ratio = 0.8
        train_dataset = torchvision.datasets.MNIST(
            self.hparams.data_root,
            train=True,
            download=True,
            transform=self.default_train_transform,
        )
        num_train_samples = len(train_dataset)
        train_indices = list(range(0, int(train_ratio * num_train_samples)))
        val_indices = list(
            range(int(train_ratio * num_train_samples), num_train_samples)
        )
        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(train_dataset, val_indices)
        self.test_dataset = torchvision.datasets.MNIST(
            self.hparams.data_root,
            train=False,
            download=True,
            transform=self.default_train_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
