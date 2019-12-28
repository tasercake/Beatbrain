from pyfiglet import Figlet
import logging
import click

from beatbrain.cli import convert
from beatbrain.cli import models

logger = logging.getLogger(__name__)
logging.basicConfig()


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    f = Figlet(font="slant")
    click.echo(click.style(f.renderText("BeatBrain"), fg="bright_blue", bold=True))
    # click.echo(
    #     click.style(
    #         "BeatBrain is distributed under the MIT License\n",
    #         fg="cyan",
    #     )
    # )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


main.add_command(convert.convert)
main.add_command(models.models)
