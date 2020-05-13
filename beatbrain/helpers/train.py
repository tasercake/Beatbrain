import logging
from pathlib import Path
from colorama import Fore, Style

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .. import models
from .. import datasets
from .. import utils
from ..utils import Config, default_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

default_pl_logger = [pl.loggers.TestTubeLogger(save_dir="experiments/")]


# TODO: refactor out the internally computed stuff so it can be imported from outside
def train_model(config: Config):
    """
    Train a model based on a Config.

    Args:
        config (Config): Either a path to a YAML file, or a dict-like
        object defining the training configuration.
    """
    if isinstance(config, (str, Path)):
        config = Config.load(config)
    config = config or default_config.deepcopy()
    print(f"{Fore.GREEN}{Style.BRIGHT}Starting training...{Style.RESET_ALL}")
    logger.info(f"Training config: {config}")

    model_class = models.get_model(config.model.architecture)
    model = model_class(config.hparams)
    trainer = get_trainer(**config.trainer)
    trainer.fit(model)
    return model


def get_trainer(**kwargs):
    """
    Creates a PyTorch-Lightning Trainer based on the given config.

    Args:
        **kwargs: Arguments to pass to `Trainer`
    """
    trainer_config = Config(kwargs)
    trainer_config.logger = (
        get_pl_loggers(**trainer_config.logger)
        if "logger" in trainer_config
        else default_pl_logger
    )
    return pl.Trainer(**trainer_config)


def get_pl_loggers(**kwargs):
    """
    Returns a list of PyTorch-Lightning loggers based on the given config.

    Args:
        **kwargs: Logger class names and their arguments
    """
    loggers = [getattr(pl.loggers, l)(**kwargs[l]) for l in kwargs]
    if len(loggers) == 1:
        loggers = loggers[0]
    elif len(loggers) == 0:
        raise ValueError("No matching PyTorch-Lightning loggers found :(")
    return loggers
