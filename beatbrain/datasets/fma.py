import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms


class FMADataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def default_transform(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
