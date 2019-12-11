from pathlib import Path
import yaml


def load_yaml(path):
    with Path(path).open("r") as file:
        docs = yaml.load_all(file, Loader=yaml.Loader)
        return {k: v for doc in docs for k, v in doc.items()}


class Config(dict):
    DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("default.yaml")

    def __init__(self, data=None):
        data = data or {}
        super().__init__(data)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)

    def merge(self, override):
        """
        Merge the override config into this one
        """
        for k in override:
            if k in self:
                if isinstance(self[k], dict) and isinstance(override[k], dict):
                    self[k].merge(override[k])
                else:
                    self[k] = override[k]
            else:
                self[k] = override[k]

    def add_defaults(self, default_config=None):
        if default_config is None:
            default_config = self.from_yaml(self.DEFAULT_CONFIG_PATH)
        elif isinstance(default_config, str):
            default_config = self.__cls__.from_yaml(default_config)
        elif isinstance(default_config, dict):
            default_config = self.__cls__(default_config)
        else:
            raise ValueError(
                f"Expected dict, str, or None. Got {type(default_config)} instead."
            )
        default_config.merge(self)
        self.update(default_config)

    @classmethod
    def from_yaml(cls, file):
        file = Path(file)
        if file.exists():
            config = load_yaml(file)
            return cls(config)
        raise FileNotFoundError(file)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


default = Config()
default.add_defaults()
