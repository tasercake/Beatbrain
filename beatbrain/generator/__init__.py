from beatbrain.generator import models, layers, helpers, callbacks


def get_module(architecture):
    if isinstance(architecture, str):
        try:
            return getattr(models, architecture)
        except AttributeError:
            print(f"No such model architecture: '{architecture}'")
            raise
    raise ValueError(f"Invalid architecture: expected string, got {type(architecture)}")
