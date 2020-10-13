import click
import logging
from pyfiglet import Figlet

from .. import helpers
from ..utils import registry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.group(invoke_without_command=True, short_help="Model Utilities")
@click.pass_context
def models_group(ctx):
    """
    BeatBrain/models: View, train, evaluate, and run inference on ML models.
    """
    click.echo(
        click.style(
            "----------------\n"
            "BeatBrain Models\n"
            "----------------\n",
            fg="green",
            bold=True,
        )
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Click commands - these serve as wrappers to expose functions on the CLI
@models_group.command(name="train", short_help="Train a model")
@click.option(
    "-c", "--config", help="Path to config YAML file", show_default=True,
)
def train_model(*args, **kwargs):
    """
    Train a model based on a YAML config.
    """
    return helpers.train.train_model(*args, **kwargs)


@models_group.command(name="list", short_help="List available models")
def list_models():
    """
    Prints a list of registered model classes.
    """
    unique = registry.unique("model")
    print("Available models:")
    for i, (name, aliases) in enumerate(unique.items()):
        if aliases:
            print(f"{i + 1}. {name} | Aliases: {aliases}")
        else:
            print(f"{i + 1}. {name}")
