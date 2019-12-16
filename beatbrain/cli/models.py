import click

import beatbrain.utils.data
from beatbrain.config import Config
from beatbrain import generator

from pytorch_lightning import Trainer


@click.group(invoke_without_command=True, short_help="Model Utilities")
@click.pass_context
def models(ctx):
    click.echo(
        click.style(
            "----------------\n" "BeatBrain Models\n" "----------------\n",
            fg="green",
            bold=True,
        )
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@models.command(name="train", short_help="Train a model")
@click.option(
    "-c", "--config", help="Config file that defines the model", show_default=True,
)
def train(config, **kwargs):
    config = Config(config, add_defaults=True)
    print(config)
    model = generator.get_module(config.model.architecture)
    trainer = Trainer()
    trainer.fit(model)
