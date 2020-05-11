__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "default_config",
]

from pathlib import Path
from .config import Config

# TODO: define config schema
DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("default_config.yaml")
default_config = Config.load(DEFAULT_CONFIG_PATH)
