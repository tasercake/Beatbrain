import logging
from pathlib import Path
from colorama import Fore, Style
from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .. import models
from .. import datasets
from .. import utils
from ..utils.config import Config, get_default_config
from ..utils import registry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TODO: refactor out the internally computed stuff so it can be imported from outside
def train_model(config: Config):
    """
    Train a model based on a Config.

    Args:
        config (Config): Either a path to a YAML file, or a dict-like
        object defining the training configuration.
    """
    raise NotImplementedError
    if isinstance(config, (str, Path)):
        config = Config.load(config)
    config = config or get_default_config()
    print(f"{Fore.GREEN}{Style.BRIGHT}Starting training...{Style.RESET_ALL}")
    logger.info(f"Training config: {config}")

    model_class = models.get_model(config.model.architecture)
    train_transform = config.data.train.transform or model_class.default_train_transform
    model = model_class(**config.hparams)

    train_dataset = datasets.get_dataset(config.data.train.dataset)(
        **config.data.train.options
    )
    train_dataloader = DataLoader(train_dataset)

    val_dataset = datasets.get_dataset(config.data.val.dataset)(
        **config.data.val.options
    )
    val_dataloader = DataLoader(val_dataset)

    trainer = get_trainer(**config.trainer)
    trainer.fit(model, train_dataloader=train_dataloader)
    return model


def get_trainer(**kwargs):
    """
    Creates a PyTorch-Lightning Trainer based on the given config.

    The value passed in the `logger` field (if any) is converted to PyTorch Lightning Logger instances.

    Args:
        **kwargs: Arguments to pass to `pytorch_lightning.Trainer`
    """
    config = Config(kwargs)
    config.logger = (
        get_pl_loggers(config.logger) if "logger" in config else [pl.loggers.TestTubeLogger(save_dir="experiments/")]
    )
    if "weights_save_path" not in config:
        config.weights_save_path = config.logger[0].save_dir
    return Trainer(**config)


def get_pl_loggers(loggers_config):
    """
    Returns a list of PyTorch-Lightning loggers based on the given config.

    Args:
        loggers_config: A collection of any combination of dictionaries and PyTorch Lightning Logger instances.
        Dictionaries must be of the format {"LoggerName": {**logger_kwargs}}.
    """
    def create_logger(logger_config):
        if isinstance(logger_config, Mapping):
            assert len(logger_config) == 1, f"Each logger config must be a single-key dictionary like {'name': {''}}"
            name, options = list(logger_config.items())[0]
            return getattr(pl.loggers, name)(**options)
        elif isinstance(logger_config, pl.loggers.LightningLoggerBase):
            return logger_config
        else:
            raise ValueError(f" Got an unrecognized logger config type. Got object of type {type(logger_config)}")

    return list(map(create_logger, loggers_config))


def get_dataset(dataset_config):
    dataset_class = registry.get("dataset", dataset_config["dataset_class"])
    return dataset_class(**{k: v for k, v in dataset_config.items() if k != "dataset_class"})
