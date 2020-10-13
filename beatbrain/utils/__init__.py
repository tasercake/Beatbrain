"""
Low-level utility classes and functions.

NOTE: Modules in `utils` shouldn't import from other Pantheon-AI packages.
Try to limit imports to within this package.
"""
from . import data, config, visualization, misc, core

from typing import Type


class registry:
    registries = {}

    @classmethod
    def register(cls, registry_name: str, key: str):
        """
        Decorator for adding an entry to a registry.

        Args:
            registry_name: Name of the registry to add the entry to
            key: Name to file the entry under
        """
        def inner(obj):
            reg = cls.registries.setdefault(registry_name, {})
            reg[key] = obj
            return obj
        return inner

    @classmethod
    def get(cls, registry_name: str, key: str, allow_passthrough=True):
        """
        Get an element from a registry

        Args:
            registry_name: Name of the registry.
            key: Entry key in the specified registry to retrieve.
            allow_passthrough: If True, then if `key` is not a key in the specified registry but is present as a value in the registry, `key` is returned.
        """
        try:
            reg = cls.registries[registry_name]
        except KeyError as e:
            raise KeyError(f"No such registry: '{registry_name}'") from e
        try:
            return reg[key]
        except KeyError as e:
            if key in reg.values() and allow_passthrough:
                return key
            raise KeyError(f"Couldn't find '{key}' in registry '{registry_name}'") from e

    @classmethod
    def unique(cls, registry_name: str):
        reverse_registry = {}
        for key, value in cls.registries[registry_name].items():
            reverse_registry.setdefault(value, []).append(key)
        return {names[0]: names[1:] for names in reverse_registry.values()}

    @classmethod
    def get_dataset(cls, name: str) -> Type:
        return cls.get("dataset", name)

    @classmethod
    def get_model(cls, name: str) -> Type:
        return cls.get("model", name)
