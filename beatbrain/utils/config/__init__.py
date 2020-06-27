__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "get_default_config",
]

from pathlib import Path
from .config import Config

# TODO: define config schema
DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("default_config.yaml")


def get_default_config():
    return Config.load(DEFAULT_CONFIG_PATH)
