import logging
from typing import Type

logger = logging.getLogger(__name__)
models = {}


# TODO: create `utils` module for this stuff to reduce code duplication
def get_model(name: str) -> Type:
    """
    Get a model class by its registered name.
    """
    if not isinstance(name, str):
        raise ValueError(f"Invalid dataset name: expected string, got {type(name)}")
    try:
        return models[name]
    except KeyError:
        print(f"No such model architecture: '{name}'")
        raise


# TODO: create registration decorator: https://blog.miguelgrinberg.com/post/the-ultimate-guide-to-python-decorators-part-i-function-registration
def register_model(name: str, cls: Type, alias=False):
    """
    Add a model to the list of available models.

    Args:
        name: The name to register the model with
        cls: The model class to register
        alias: if True, allow the same model to be registered with different names
    """
    if not isinstance(name, str):
        raise ValueError(f"Invalid dataset name: expected string, got {type(name)}")
    # Check if model class is already registered
    registered_names = [name for name, model in models.items() if model == cls]
    if len(registered_names) == 1:
        registered_names = registered_names[0]
    if not alias and registered_names:
        raise ValueError(f"{cls} is already registered as {registered_names}")
    # Check for naming conflicts
    if name in models:
        raise ValueError(
            f"A model with name {name} is already registered: {models[name]}"
        )
    models[name] = cls
