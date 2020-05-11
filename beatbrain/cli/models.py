import click
import logging
from pprint import pprint
from pyfiglet import Figlet
from typing import Union

from .. import helpers
from .. import models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.group(invoke_without_command=True, short_help="Model Utilities")
@click.pass_context
def models_group(ctx):
    """
    Pantheon-AI/models: View, train, evaluate, and run inference on ML models.
    """
    f = Figlet(font="doom")
    click.echo(click.style(f.renderText("models"), fg="bright_blue", bold=True,))
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
def list_models(*args, **kwargs):
    """
    Prints a list of registered model classes.
    """
    print(f"Found {len(models.models)} model(s):")
    pprint(list(models.models))
