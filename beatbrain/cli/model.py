import click

import beatbrain.utils.data
from beatbrain.config import Config
from beatbrain import models


@click.group(invoke_without_command=True, short_help="Model Utilities")
@click.pass_context
def model(ctx):
    click.echo(
        click.style(
            "----------------\n" "BeatBrain Models\n" "----------------\n",
            fg="green",
            bold=True,
        )
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@model.command(name="train", short_help="Train a model")
@click.option(
    "-c", "--config", help="Config file that defines the model", show_default=True,
)
def train(config, **kwargs):
    config = Config(config, add_defaults=True)
    print(config)
