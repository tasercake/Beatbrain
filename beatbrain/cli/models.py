from pyfiglet import Figlet
import click
import logging

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import beatbrain.utils.data
from beatbrain.config import Config
from beatbrain import generator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.group(invoke_without_command=True, short_help="Model Utilities")
@click.pass_context
def models(ctx):
    f = Figlet(font="big")
    click.echo(click.style(f.renderText("/ models"), fg="bright_blue", bold=True,))
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@models.command(name="train", short_help="Train a model")
@click.option(
    "-c", "--config", help="Config file that defines the model", show_default=True,
)
def train(config, **kwargs):
    config = Config(config, add_defaults=True)
    logger.info(f"Training config: {config}")
    logger.info(click.style("Starting training...", fg="bright_green"))
    model = generator.get_model(config.model.architecture)(
        **config.model.options  # TODO: maybe don't unpack?
    )
    pl_loggers = [getattr(pl.loggers, l)(**config.loggers[l]) for l in config.loggers]
    if len(pl_loggers) == 1:
        pl_loggers = pl_loggers[0]
    trainer = Trainer(**config.trainer, logger=pl_loggers)
    trainer.fit(model)
    return model
