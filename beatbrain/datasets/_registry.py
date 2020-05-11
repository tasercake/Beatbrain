import logging
from typing import Type

logger = logging.getLogger(__name__)
datasets = {}


def get_dataset(name: str) -> Type:
    """
    Get a dataset class by its registered name.
    """
    if not isinstance(name, str):
        raise ValueError(f"Invalid dataset name: expected string, got {type(name)}")
    try:
        return datasets[name]
    except KeyError:
        print(f"No such dataset: '{name}'")
        raise


def register_dataset(name: str, cls: Type, alias=False):
    """
    Add a dataset to the list of available datasets.

    Args:
        name: The name to register the dataset with
        cls: The dataset class to register
        alias: if True, allow the same dataset to be registered with different names
    """
    if not isinstance(name, str):
        raise ValueError(f"Invalid dataset name: expected string, got {type(name)}")
    # Check if dataset class is already registered
    registered_names = [name for name, dataset in datasets.items() if dataset == cls]
    if len(registered_names) == 1:
        registered_names = registered_names[0]
    if not alias and registered_names:
        raise ValueError(f"{cls} is already registered as {registered_names}")
    # Check for naming conflicts
    if name in datasets:
        raise ValueError(
            f"A dataset with name {name} is already registered: {datasets[name]}"
        )
    datasets[name] = cls
