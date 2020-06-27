from pyfiglet import Figlet
from loguru import logger
import click

from . import convert, models


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    f = Figlet(font="doom")
    click.echo(click.style(f.renderText("BeatBrain"), fg="bright_blue", bold=True))
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


main.add_command(convert.convert)
main.add_command(models.models_group, name="models")
