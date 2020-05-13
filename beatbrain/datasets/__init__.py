"""
Defines datasets and provides methods to find
datasets by name or register new ones.
"""
from ._registry import datasets, get_dataset, register_dataset

from .fma import FMADataset

register_dataset("FMADataset", FMADataset)
