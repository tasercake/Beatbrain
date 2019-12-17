import click
import logging

from pytorch_lightning import Trainer

import beatbrain.utils.data
from beatbrain.config import Config
from beatbrain import generator

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True, short_help="Model Utilities")
@click.pass_context
def models(ctx):
    click.echo(
        click.style(
            "----------------\nBeatBrain Models\n----------------\n",
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
    logger.debug(f"Training config: {config}")
    logger.info(click.style("Starting training...", fg="green"))
    model = generator.get_module(config.system.architecture)(
        **config.system.hyperparameters
    )
    trainer = Trainer(gpus=config.train.gpus)
    trainer.fit(model)
    return model
