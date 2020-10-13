import click
from loguru import logger

from ..utils import data as data_utils


@click.group(invoke_without_command=True, short_help="Data Conversion Utilities")
@click.pass_context
def convert(ctx):
    click.echo(
        click.style(
            "------------------------\n"
            "BeatBrain Data Converter\n"
            "------------------------\n",
            fg="green",
            bold=True,
        )
    )
    logger.warning("The BeatBrain Data Converter is deprecated. Use on-the-fly conversion (and caching) during training instead.",)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@convert.command(
    name="audio",
    short_help="Convert audio to the .wav format",
)
@click.argument("path")
@click.argument("output")
@click.option("--sr", default=22050, show_default=True)
@click.option("--format", default="flac", show_default=True)
@click.option("--split", is_flag=True)
@click.option("--chunk_duration", help="Maximum length of output audio chunks", default=10, show_default=True)
@click.option("--discard_shorter", help="Discard audio chunks shorter than this many seconds", default=4, show_default=True)
def convert_audio(path, output, **kwargs):
    data_utils.convert_audio(path, output, **kwargs)
