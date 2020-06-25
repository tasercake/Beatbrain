import yaml
import json
import copy
from pathlib import Path
from pprint import pformat
from collections import abc
from copy import deepcopy
from addict import Dict
from io import StringIO
from typing import Union, Mapping, Any


class Config(Dict):
    """
    Dict-like class that allows access via dot-notation.
    Extension of `addict.Dict`: https://github.com/mewwts/addict/blob/master/addict/addict.py
    """

    _autoreload_compat_keys = [
        "__class__",
    ]  # Hack to make this class work with Ipython's autoreload

    # yaml = YAML()
    # yaml.default_flow_style = None

    @classmethod
    def load(cls, path, format: str = "yaml"):
        """
        Instantiate a Config from a YAML or JSON file.

        Args:
            path: Path to a YAML or JSON file
            format: one of ["yaml", "json"]
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if format.lower() == "json":
            with open(path, "r") as f:
                data = json.load(f)
        elif format.lower() == "yaml":
            # data = cls.yaml.load(path)
            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError(f"Unknown format: {format}. Expected json or yaml")
        return cls(data)

    def dump(self, path = None, format="yaml"):
        """
        Get the string representation of the Config object in the
        specified format, and (optionally) write it to a file.

        Args:
            path: Path to save the serialized Config to
            format: one of ["yaml", "json"]
        """
        stream = StringIO()
        if format.lower() == "json":
            json.dump(self.to_dict(), stream, indent=2)
        elif format.lower() == "yaml":
            # self.yaml.dump(self.to_dict(), stream=stream)
            yaml.dump(self.to_dict(), stream, default_flow_style=False)
        dump = stream.getvalue()
        if path:
            with open(path, "w") as f:
                f.write(dump)
        return dump

    def copy(self, other=None):
        """
        Create a shallow copy of the Config object and (optionally)
        merge another mapping into the copy.
        """
        new = super().copy()
        if other:
            new.update(other)
        return new

    def deepcopy(self, other=None):
        """
        Create a deep copy of the Config object and (optionally)
        merge another mapping into the copy.
        """
        new = super().deepcopy()
        if other:
            new.update(other)
        return new

    def __setattr__(self, key, value):
        if key in self._autoreload_compat_keys:
            object.__setattr__(self, key, value)
        else:
            value = self._hook(value)
            super().__setattr__(key, value)

    def __setitem__(self, key, value):
        # Apply hook only if `value` is a dict or one of [list, tuple]
        # containing at least one dict at the top level
        hook_condition = (
            isinstance(value, (list, tuple))
            and len(value)
            and any(isinstance(el, dict) for el in value)
        ) or isinstance(value, dict)
        if hook_condition:
            value = self._hook(value)
        super().__setitem__(key, value)

    def __str__(self):
        return pformat(self, indent=1, compact=True, width=120)
