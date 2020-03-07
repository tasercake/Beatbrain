import logging
from inspect import isclass, ismodule
from functools import lru_cache

import torch
from . import models, helpers, metrics

logger = logging.getLogger(__name__)
# MODULES = (mnist,)  # Must be a hashable container


@lru_cache()
def discover_models():
    modules = dict(filter(lambda e: ismodule(e[1]), models.__dict__.items()))
    _models = {}
    for module_name, module in modules.items():
        for obj_name, obj in module.__dict__.items():
            if isclass(obj) and issubclass(obj, torch.nn.Module):
                _models[f"{module_name}/{obj_name}"] = obj
    logger.info(f"Discovered {len(_models)} models: {list(_models.keys())}")
    return _models


def get_model(architecture):
    try:
        _models = discover_models()
    except:
        logger.exception("Model discovery failed. This is probably the developer's fault. Please report this by creating a github issue.")
        raise
    if isinstance(architecture, str):
        try:
            return _models[architecture]
        except KeyError:
            print(f"No such model architecture: '{architecture}'")
            raise
    raise ValueError(f"Invalid architecture: expected string, got {type(architecture)}")
