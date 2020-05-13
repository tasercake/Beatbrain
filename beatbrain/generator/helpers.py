import numpy as np
import torch


def reparameterize(mean, logvar, training=True):
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    else:
        return mean


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = torch.log(2.0 * np.pi)
    return torch.sum(
        -0.5 * ((sample - mean) ** 2.0 * torch.exp(-logvar) + logvar + log2pi),
        dim=raxis,
    )


def sample(latent_dim, decoder, eps=None):
    if eps is None:
        eps = torch.normal(0, 1, (100, latent_dim))
    return decode(decoder, eps, apply_sigmoid=True)


def encode(encoder, x):
    inference = encoder(x)
    mean, logvar = torch.split(inference, 2, dim=1)
    return mean, logvar


def decode(decoder, z, apply_sigmoid=False):
    logits = decoder(z)
    if apply_sigmoid:
        probs = torch.sigmoid(logits)
        return probs
    return logits
