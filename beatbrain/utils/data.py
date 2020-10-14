import os
import errno
import warnings
from loguru import logger
import librosa
import soundfile as sf
from tqdm.auto import tqdm
from boltons.pathutils import augpath
from joblib import Parallel, delayed
from pathlib import Path
from colorama import Fore
from audioread import DecodeError, NoBackendError

from .config import get_default_config

default_config = get_default_config()


def convert_audio(inp, out, format="flac", split=False, chunk_duration=10, discard_shorter=4, **kwargs):
    """
    Convert an audio file or directory of audio files to a different format.
    Input files can be of any format supported by Librosa.

    Args:
        inp: An audio file or a directory of audio files (format must be supported by librosa)
        out: Destination directory
        format (str): The format to convert to (must be supported by PySoundFile)
        split (bool): Whether to split the audio file(s) into smaller chunks
        discard_shorter (float): Minimum duration of output wav files. Shorter segments are discarded.
        chunk_duration (float): Maximum duration of each output wav file (in seconds)
        **kwargs: Passed to librosa.load()
    """
    inp = Path(inp)
    out = Path(out)
    format = format.lower()
    if not inp.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(inp))
    out.mkdir(exist_ok=True, parents=True)
    input_files = [str(f) for f in inp.iterdir() if f.is_file()] if inp.is_dir() else [inp]
    output_files = [augpath(f, ext=f".{format}", dpath=out) for f in input_files]
    subtype = {"flac": "PCM_24", "wav": "PCM_24", "ogg": "VORBIS"}.get(format)

    def convert_one(src, dst):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="librosa")
            try:
                audio, sr = librosa.load(src, mono=True, **kwargs)
            except (NoBackendError, DecodeError):
                return
            sf.write(dst, audio, sr, subtype=subtype)

    def split_one(src, dst):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="librosa")
            try:
                audio, sr = librosa.load(src, mono=True, **kwargs)
            except (NoBackendError, DecodeError):
                return
        chunk_samples = chunk_duration * sr
        for i, start in enumerate(range(0, len(audio), chunk_samples)):
            chunk = audio[start: start + chunk_samples]
            if discard_shorter and len(chunk) < discard_shorter * sr:
                break
            dst_i = augpath(dst, suffix=f"_{i + 1}")
            sf.write(dst_i, chunk, sr, subtype=subtype)

    if split:
        logger.info(f"Splitting {len(input_files)} audio file(s): {Fore.YELLOW}{inp}{Fore.RESET} -> {Fore.YELLOW}{out}{Fore.RESET}")
        Parallel(n_jobs=-1, backend="threading")(delayed(split_one)(i, o) for i, o in tqdm(zip(input_files, output_files), total=len(input_files)))
    else:
        logger.info(f"Converting {len(input_files)} file(s) to {format.upper()}: {Fore.YELLOW}{inp}{Fore.RESET} -> {Fore.YELLOW}{out}{Fore.RESET}")
        Parallel(n_jobs=-1, backend="threading")(delayed(convert_one)(i, o) for i, o in tqdm(zip(input_files, output_files), total=len(input_files)))
