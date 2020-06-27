import click
from loguru import logger
from deprecation import deprecated

from ..utils import data as data_utils
from ..utils.config import get_default_config

default_config = get_default_config()


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
    name="wav",
    short_help="Convert audio to the .wav format",
)
@click.argument("path")
@click.argument("output")
@click.option("--sr", default=22050)
def to_wav(path, output, **kwargs):
    data_utils.convert_audio_to_wav(path, output, **kwargs)


@convert.command(
    name="split",
    short_help="Convert audio to the .wav format",
)
@click.argument("path")
@click.argument("output")
@click.option("--chunk_duration", help="Maximum length of output audio chunks", default=10, show_default=True)
@click.option("--discard_shorter", help="Discard audio chunks shorter than this many seconds", default=4, show_default=True)
@click.option("--sr", default=22050)
def split_audio(path, output, **kwargs):
    data_utils.split_audio(path, output, **kwargs)

# ==========
# DEPRECATED
# ==========
_converter_options = [
    click.argument("path"),
    click.argument("output"),
    click.option(
        "--sr",
        help="Rate at which to resample audio",
        default=default_config.SAMPLE_RATE,
        show_default=True,
    ),
    click.option(
        "--offset",
        help="Audio start timestamp (seconds)",
        default=default_config.AUDIO_OFFSET,
        show_default=True,
    ),
    click.option(
        "--duration",
        help="Audio duration (seconds)",
        default=default_config.AUDIO_DURATION,
        type=float,
        show_default=True,
    ),
    click.option(
        "--n_fft",
        help="Size of FFT window to use",
        default=default_config.N_FFT,
        show_default=True,
    ),
    click.option(
        "--hop_length",
        help="Short-time Fourier Transform hop length",
        default=default_config.HOP_LENGTH,
        show_default=True,
    ),
    click.option(
        "--n_mels",
        help="Number of frequency bins to use",
        default=default_config.N_MELS,
        show_default=True,
    ),
    click.option(
        "--chunk_size",
        help="Number of frames per spectrogram chunk",
        default=default_config.CHUNK_SIZE,
        show_default=True,
    ),
    click.option(
        "--flip",
        help="Whether to flip images veritcally",
        default=default_config.IMAGE_FLIP,
        show_default=True,
    ),
    click.option(
        "--truncate/--pad",
        help="Whether to truncate or pad the last chunk",
        default=True,
        show_default=True,
    ),
    click.option(
        "--skip",
        help="Number of samples to skip. Useful when restarting a failed job.",
        default=0,
        show_default=True,
    ),
]


def converter_options(func):
    for option in reversed(_converter_options):
        func = option(func)
    return func


@deprecated()
@convert.command(
    name="numpy",
    short_help="Convert audio files or EXR images to numpy arrays",
    deprecated=True,
)
@converter_options
def to_numpy(path, output, **kwargs):
    return data_utils.convert_to_numpy(path, output, **kwargs)


@convert.command(
    name="image",
    short_help="Convert audio or .npz files to EXR images",
    deprecated=True,
)
@converter_options
@click.option(
    "--flip",
    help="Whether to flip images vertically",
    default=default_config.IMAGE_FLIP,
    show_default=True,
)
def to_image(path, output, **kwargs):
    return data_utils.convert_to_image(path, output, **kwargs)


@convert.command(
    name="audio",
    short_help="Convert .npz files or EXR images to audio files",
    deprecated=True,
)
@click.argument("path")
@click.argument("output")
@click.option(
    "--sr",
    help="Rate at which to resample audio",
    default=default_config.SAMPLE_RATE,
    show_default=True,
)
@click.option(
    "--n_fft",
    help="Size of FFT window to use",
    default=default_config.N_FFT,
    show_default=True,
)
@click.option(
    "--hop_length",
    help="Short-time Fourier Transform hop length",
    default=default_config.HOP_LENGTH,
    show_default=True,
)
@click.option(
    "--offset",
    help="Start point (in seconds) of reconstructed audio",
    default=default_config.AUDIO_OFFSET,
    show_default=True,
)
@click.option(
    "--duration",
    help="Maximum seconds of audio to convert",
    default=default_config.AUDIO_DURATION,
    type=float,
    show_default=True,
)
@click.option(
    "--flip",
    help="Whether to flip images veritcally",
    default=default_config.IMAGE_FLIP,
    show_default=True,
)
@click.option(
    "--skip",
    help="Number of samples to skip. Useful when restarting a failed job.",
    default=0,
    show_default=True,
)
def to_audio(path, output, **kwargs):
    return data_utils.convert_to_audio(path, output, **kwargs)
