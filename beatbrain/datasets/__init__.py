"""
Defines datasets and provides methods to find
datasets by name or register new ones.
"""
from ._registry import datasets, get_dataset, register_dataset

from .audio import AudioClipDataset

register_dataset("AudioClipDataset", AudioClipDataset)
