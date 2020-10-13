import os
import errno
import warnings
from loguru import logger
import librosa
import soundfile as sf
from tqdm.auto import tqdm, trange
from boltons.pathutils import augpath
from joblib import Parallel, delayed
from pathlib import Path
from colorama import Fore
from natsort import natsorted
from audioread import DecodeError, NoBackendError
from deprecation import deprecated

import torch

from .config import Config, get_default_config
from .core import (
    save_images,
    load_image,
    load_images,
    load_arrays,
    audio_to_spectrogram,
    split_spectrogram,
    save_arrays,
    spectrogram_to_audio,
)

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
    # subtype = dict(flac="PCM_24", wav="PCM_24", ogg="VORBIS").get(format)
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
        Parallel(n_jobs=-1)(delayed(split_one)(i, o) for i, o in tqdm(zip(input_files, output_files), total=len(input_files)))
    else:
        logger.info(f"Converting {len(input_files)} file(s) to {format.upper()}: {Fore.YELLOW}{inp}{Fore.RESET} -> {Fore.YELLOW}{out}{Fore.RESET}")
        Parallel(n_jobs=-1)(delayed(convert_one)(i, o) for i, o in tqdm(zip(input_files, output_files), total=len(input_files)))


def create_fb_matrix(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int):
    r"""Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * create_fb_matrix(A.size(-1), ...)``.
    """
    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb


class InverseMelScale(torch.nn.Module):
    """
    Taken from torchaudio: https://pytorch.org/audio/transforms.html#inversemelscale
    Solve for a normal STFT from a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    It minimizes the euclidian norm between the input mel-spectrogram and the product between
    the estimated spectrogram and the filter banks using SGD.

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        max_iter (int, optional): Maximum number of optimization iterations. (Default: ``100000``)
        tolerance_loss (float, optional): Value of loss to stop optimization at. (Default: ``1e-5``)
        tolerance_change (float, optional): Difference in losses to stop optimization at. (Default: ``1e-8``)
        sgdargs (dict or None, optional): Arguments for the SGD optimizer. (Default: ``None``)
    """
    __constants__ = [
        "n_stft",
        "n_mels",
        "sample_rate",
        "f_min",
        "f_max",
        "max_iter",
        "tolerance_loss",
        "tolerance_change",
        "sgdargs",
    ]

    def __init__(
        self,
        n_stft: int,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float = None,
        max_iter: int = 100000,
        tolerance_loss: float = 1e-5,
        tolerance_change: float = 1e-8,
        sgdargs: dict = None,
    ) -> None:
        super(InverseMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min
        self.max_iter = max_iter
        self.tolerance_loss = tolerance_loss
        self.tolerance_change = tolerance_change
        self.sgdargs = {
            "lr": 0.3,
            "momentum": 0.9,
        }
        self.sgdargs.update(sgdargs or {})

        assert f_min <= self.f_max, "Require f_min: %f < f_max: %f" % (
            f_min,
            self.f_max,
        )

        fb = create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate
        )
        self.register_buffer("fb", fb)

    def forward(self, melspec):
        r"""
        Args:
            melspec (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)

        Returns:
            Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        # pack batch
        shape = melspec.size()
        melspec = melspec.view(-1, shape[-2], shape[-1])

        n_mels, time = shape[-2], shape[-1]
        freq, _ = self.fb.size()  # (freq, n_mels)
        melspec = melspec.transpose(-1, -2)
        assert self.n_mels == n_mels

        specgram = torch.rand(
            melspec.size()[0],
            time,
            freq,
            requires_grad=True,
            dtype=melspec.dtype,
            device=melspec.device,
        )

        optim = torch.optim.SGD([specgram], **self.sgdargs)

        loss = float("inf")
        with trange(self.max_iter, dynamic_ncols=True, mininterval=0.5) as pbar:
            for i in pbar:
                optim.zero_grad()
                diff = melspec - specgram.matmul(self.fb)
                new_loss = diff.pow(2).sum(axis=-1).mean()
                # take sum over mel-frequency then average over other dimensions
                # so that loss threshold is applied par unit timeframe
                new_loss.backward()
                optim.step()
                specgram.data = specgram.data.clamp(min=0)

                new_loss = new_loss.item()
                if (
                    new_loss < self.tolerance_loss
                    or abs(loss - new_loss) < self.tolerance_change
                ):
                    break
                loss = new_loss
                pbar.set_postfix(loss=loss)

        specgram.requires_grad_(False)
        specgram = specgram.clamp(min=0).transpose(-1, -2)

        # unpack batch
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram

# ==========
# DEPRECATED
# ==========
@deprecated()
def get_data_type(path, raise_exception=False):
    """
    Given a file or directory, return the (homogeneous) data type contained in that path.

    Args:
        path: Path at which to check the data type.
        raise_exception: Whether to raise an exception on unknown or ambiguous data types.

    Returns:
        DataType: The type of data contained at the given path (Audio, Numpy, or Image)

    Raises:
        ValueError: If `raise_exception` is True, the number of matched data types is either 0 or >1.
    """
    print(f"Checking input type(s) in {Fore.YELLOW}'{path}'{Fore.RESET}...")
    found_types = set()
    path = Path(path)
    files = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = filter(Path.is_file, path.rglob("*"))
    for f in files:
        for dtype, exts in EXTENSIONS.items():
            suffix = f.suffix[1:]
            if suffix in exts:
                found_types.add(dtype)
    if len(found_types) == 0:
        dtype = DataType.UNKNOWN
        if raise_exception:
            raise ValueError(
                f"Unknown source data type. No known file types we matched."
            )
    elif len(found_types) == 1:
        dtype = found_types.pop()
    else:
        dtype = DataType.AMBIGUOUS
        if raise_exception:
            raise ValueError(
                f"Ambiguous source data type. The following types were matched: {found_types}"
            )
    print(f"Determined input type to be {Fore.CYAN}'{dtype.name}'{Fore.RESET}")
    return dtype


@deprecated()
def get_paths(inp, directories=False, sort=True):
    """
    Recursively get the filenames under a given path

    Args:
        inp (str): The path to search for files under
        directories (bool): If True, return the unique parent directories of the found files
        sort (bool): Whether to sort the paths
    """
    inp = Path(inp)
    if not inp.exists():
        raise ValueError(f"Input must be a valid file or directory. Got '{inp}'")
    elif inp.is_dir():
        paths = filter(Path.is_file, inp.rglob("*"))
        if directories:
            paths = {p.parent for p in paths}  # Unique parent directories
        paths = natsorted(paths) if sort else list(paths)
    else:
        paths = [inp]
    return paths


# TODO: Consolidate these functions into one
@deprecated()
def get_numpy_output_path(path, out_dir, inp):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


@deprecated()
def get_image_output_path(path, out_dir, inp):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.stem)
    output.mkdir(parents=True, exist_ok=True)
    return output


@deprecated()
def get_audio_output_path(path, out_dir, inp, fmt):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.name).with_suffix(f".{fmt}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


@deprecated()
def load_dataset(
    path,
    flip=default_config.hparams.spec.flip,
    batch_size=default_config.hparams.batch_size,
    shuffle_buffer=50000,
    prefetch=32,
    parallel=True,
    subset=None,
):
    """
    Loads a one or more images or .np{y,z} files as a `tf.data.Dataset` instance.

    Args:
        path (str): The file or directory to load data from
        flip (bool): Whether to flip loaded images
        batch_size (int):
        shuffle_buffer (int):
        prefetch (int):
        parallel (bool):
        subset (int):

    Returns:
        A `tf.data.Dataset` instance
    """
    num_parallel = tf.data.experimental.AUTOTUNE if parallel else None
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"Could not find '{path}'")
    dtype = get_data_type(path)
    supported_dtypes = (DataType.NUMPY, DataType.IMAGE)
    if dtype not in supported_dtypes:
        raise TypeError(
            f"Unsupported or ambiguous data type: {dtype}."
            f"Must be one of {supported_dtypes}."
        )
    files = []
    if path.is_file():
        files.append(path)
    elif path.is_dir():
        files.extend(get_paths(path, directories=False))
    if subset:
        files = files[:subset]
    files = [str(f) for f in files]
    dataset = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    if dtype == DataType.IMAGE:
        dataset = dataset.map(
            lambda file: tf.py_function(load_image, [file, flip], Tout=tf.float32),
            num_parallel_calls=num_parallel,
        )
    elif dtype == DataType.NUMPY:
        dataset = dataset.map(
            lambda file: tf.py_function(
                load_arrays, [file, False, True], Tout=tf.float32
            ),
            num_parallel_calls=num_parallel,
        )
        dataset = dataset.unbatch()
    dataset = dataset.map(
        lambda x: tf.py_function(tf.expand_dims, [x, -1], Tout=tf.float32),
        num_parallel_calls=num_parallel,
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if prefetch:
        dataset = dataset.prefetch(prefetch)
    #     train = dataset.skip(math.floor(len(files) * test_split))
    #     test = dataset.take(math.floor(len(files) * test_split))
    return dataset


def convert_audio_to_numpy(
    inp,
    out_dir,
    sr=default_config.hparams.audio.sample_rate,
    offset=default_config.hparams.audio.offset,
    duration=default_config.hparams.audio.duration,
    res_type=default_config.hparams.audio.resample_type,
    n_fft=default_config.hparams.spec.n_fft,
    hop_length=default_config.hparams.spec.hop_length,
    n_mels=default_config.hparams.spec.n_mels,
    chunk_size=default_config.hparams.spec.n_frames,
    truncate=default_config.hparams.spec.truncate,
    skip=0,
):
    paths = get_paths(inp, directories=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to Numpy arrays...")
    print(f"Arrays will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        try:
            audio, sr = librosa.load(
                str(path), sr=sr, offset=offset, duration=duration, res_type=res_type
            )
        except DecodeError as e:
            print(f"Error decoding {path}: {e}")
            continue
        spec = audio_to_spectrogram(
            audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalize=True,
        )
        chunks = split_spectrogram(spec, chunk_size, truncate=truncate)
        if truncate and chunks[0].shape[1] < chunk_size:
            print(f"Skipping {path}: not enough time frames")
            continue
        output = get_numpy_output_path(path, out_dir, inp)
        save_arrays(chunks, output)


def convert_image_to_numpy(inp, out_dir, flip=default_config.spec.flip, skip=0):
    paths = get_paths(inp, directories=True)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to Numpy arrays...")
    print(f"Arrays will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        chunks = load_images(path, flip=flip)
        output = get_numpy_output_path(path, out_dir, inp)
        save_arrays(chunks, output)


def convert_audio_to_image(
    inp,
    out_dir,
    sr=default_config.hparams.audio.sample_rate,
    offset=default_config.hparams.audio.offset,
    duration=default_config.hparams.audio.duration,
    res_type=default_config.hparams.audio.resample_type,
    n_fft=default_config.hparams.spec.n_fft,
    hop_length=default_config.spec.hop_length,
    n_mels=default_config.hparams.spec.n_mels,
    chunk_size=default_config.hparams.spec.n_frames,
    truncate=default_config.hparams.spec.truncate,
    flip=default_config.hparams.spec.flip,
    skip=0,
):
    paths = get_paths(inp, directories=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to images...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        try:
            audio, sr = librosa.load(
                str(path), sr=sr, offset=offset, duration=duration, res_type=res_type
            )
        except DecodeError as e:
            print(f"Error decoding {path}: {e}")
            continue
        spec = audio_to_spectrogram(
            audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalize=True,
        )
        chunks = split_spectrogram(spec, chunk_size, truncate=truncate)
        if truncate and chunks[0].shape[1] < chunk_size:
            print(f"Skipping {path}: not enough time frames")
            continue
        output = get_image_output_path(path, out_dir, inp)
        save_images(chunks, output, flip=flip)


def convert_numpy_to_image(inp, out_dir, flip=default_config.hparams.spec.flip, skip=0):
    paths = get_paths(inp, directories=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to images...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        chunks = load_arrays(path)
        output = get_image_output_path(path, out_dir, inp)
        save_images(chunks, output, flip=flip)


def convert_numpy_to_audio(
    inp,
    out_dir,
    sr=default_config.hparams.audio.sample_rate,
    n_fft=default_config.hparams.spec.n_fft,
    hop_length=default_config.hparams.spec.hop_length,
    fmt=default_config.hparams.audio.format,
    offset=default_config.hparams.audio.offset,
    duration=default_config.hparams.audio.duration,
    skip=0,
):
    paths = get_paths(inp, directories=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to audio...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        spec = load_arrays(path, concatenate=True)
        audio = spectrogram_to_audio(
            spec, sr=sr, n_fft=n_fft, hop_length=hop_length, denormalize=True,
        )
        output = get_audio_output_path(path, out_dir, inp, fmt)
        sf.write(output, audio, sr)


def convert_image_to_audio(
    inp,
    out_dir,
    sr=default_config.hparams.audio.sample_rate,
    n_fft=default_config.hparams.spec.n_fft,
    hop_length=default_config.hparams.spec.hop_length,
    fmt=default_config.hparams.audio.format,
    offset=default_config.hparams.audio.offset,
    duration=default_config.hparams.audio.duration,
    flip=default_config.hparams.spec.flip,
    skip=0,
):
    paths = get_paths(inp, directories=True)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to audio...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        spec = load_images(path, flip=flip, concatenate=True)
        audio = spectrogram_to_audio(
            spec, sr=sr, n_fft=n_fft, hop_length=hop_length, denormalize=True,
        )
        output = get_audio_output_path(path, out_dir, inp, fmt)
        sf.write(output, audio, sr)


def convert_to_numpy(inp, out_dir, **kwargs):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.AUDIO:
        return convert_audio_to_numpy(
            inp,
            out_dir,
            sr=kwargs.get("sr"),
            offset=kwargs.get("offset"),
            duration=kwargs.get("duration"),
            res_type=kwargs.get("res_type"),
            n_fft=kwargs.get("n_fft"),
            hop_length=kwargs.get("hop_length"),
            n_mels=kwargs.get("n_mels"),
            chunk_size=kwargs.get("chunk_size"),
            truncate=kwargs.get("truncate"),
            skip=kwargs.get("skip"),
        )
    elif dtype == DataType.IMAGE:
        return convert_image_to_numpy(
            inp, out_dir, flip=kwargs.get("flip"), skip=kwargs.get("skip")
        )


def convert_to_image(
    inp, out_dir, **kwargs,
):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.AUDIO:
        return convert_audio_to_image(
            inp,
            out_dir,
            sr=kwargs.get("sr"),
            offset=kwargs.get("offset"),
            duration=kwargs.get("duration"),
            res_type=kwargs.get("res_type"),
            n_fft=kwargs.get("n_fft"),
            hop_length=kwargs.get("hop_length"),
            chunk_size=kwargs.get("chunk_size"),
            n_mels=kwargs.get("n_mels"),
            truncate=kwargs.get("truncate"),
            flip=kwargs.get("flip"),
            skip=kwargs.get("skip"),
        )
    elif dtype == DataType.NUMPY:
        return convert_numpy_to_image(
            inp, out_dir, flip=kwargs.get("flip"), skip=kwargs.get("skip")
        )


def convert_to_audio(inp, out_dir, **kwargs):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.NUMPY:
        return convert_numpy_to_audio(
            inp,
            out_dir,
            sr=kwargs.get("sr"),
            n_fft=kwargs.get("n_fft"),
            hop_length=kwargs.get("hop_length"),
            fmt=kwargs.get("fmt"),
            offset=kwargs.get("offset"),
            duration=kwargs.get("duration"),
            skip=kwargs.get("skip"),
        )
    elif dtype == DataType.IMAGE:
        return convert_image_to_audio(
            inp,
            out_dir,
            sr=kwargs.get("sr"),
            n_fft=kwargs.get("n_fft"),
            hop_length=kwargs.get("hop_length"),
            fmt=kwargs.get("fmt"),
            offset=kwargs.get("offset"),
            duration=kwargs.get("duration"),
            skip=kwargs.get("skip"),
            flip=kwargs.get("flip"),
        )
