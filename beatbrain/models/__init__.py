"""
Defines model architectures and provides methods to find
models by name or register new ones.
"""
from ._registry import models, get_model, register_model
from .mnist import MNISTAutoencoder

register_model("MNISTAutoencoder", MNISTAutoencoder)
