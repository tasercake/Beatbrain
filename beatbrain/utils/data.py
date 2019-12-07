import librosa
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from colorama import Fore
from natsort import natsorted
from audioread import DecodeError

from .. import defaults
from .misc import DataType, EXTENSIONS
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


# TODO: Consolidate these functions into one
def get_numpy_output_path(path, out_dir, inp):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def get_image_output_path(path, out_dir, inp):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.stem)
    output.mkdir(parents=True, exist_ok=True)
    return output


def get_audio_output_path(path, out_dir, inp, fmt):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.name).with_suffix(f".{fmt}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def load_dataset(
    path,
    flip=defaults.IMAGE_FLIP,
    batch_size=defaults.BATCH_SIZE,
    shuffle_buffer=defaults.SHUFFLE_BUFFER,
    prefetch=defaults.DATA_PREFETCH,
    parallel=defaults.DATA_PARALLEL,
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
    sr=defaults.SAMPLE_RATE,
    offset=defaults.AUDIO_OFFSET,
    duration=defaults.AUDIO_DURATION,
    res_type=defaults.RESAMPLE_TYPE,
    n_fft=defaults.N_FFT,
    hop_length=defaults.HOP_LENGTH,
    n_mels=defaults.N_MELS,
    chunk_size=defaults.CHUNK_SIZE,
    truncate=defaults.TRUNCATE,
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


def convert_image_to_numpy(inp, out_dir, flip=defaults.IMAGE_FLIP, skip=0):
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
    sr=defaults.SAMPLE_RATE,
    offset=defaults.AUDIO_OFFSET,
    duration=defaults.AUDIO_DURATION,
    res_type=defaults.RESAMPLE_TYPE,
    n_fft=defaults.N_FFT,
    hop_length=defaults.HOP_LENGTH,
    n_mels=defaults.N_MELS,
    chunk_size=defaults.CHUNK_SIZE,
    truncate=defaults.TRUNCATE,
    flip=defaults.IMAGE_FLIP,
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


def convert_numpy_to_image(inp, out_dir, flip=defaults.IMAGE_FLIP, skip=0):
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
    sr=defaults.SAMPLE_RATE,
    n_fft=defaults.N_FFT,
    hop_length=defaults.HOP_LENGTH,
    fmt=defaults.AUDIO_FORMAT,
    offset=defaults.AUDIO_OFFSET,
    duration=defaults.AUDIO_DURATION,
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
    sr=defaults.SAMPLE_RATE,
    n_fft=defaults.N_FFT,
    hop_length=defaults.HOP_LENGTH,
    fmt=defaults.AUDIO_FORMAT,
    offset=defaults.AUDIO_OFFSET,
    duration=defaults.AUDIO_DURATION,
    flip=defaults.IMAGE_FLIP,
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


def convert_to_numpy(
    inp,
    out_dir,
    sr=defaults.SAMPLE_RATE,
    offset=defaults.AUDIO_OFFSET,
    duration=defaults.AUDIO_DURATION,
    res_type=defaults.RESAMPLE_TYPE,
    n_fft=defaults.N_FFT,
    hop_length=defaults.HOP_LENGTH,
    n_mels=defaults.N_MELS,
    chunk_size=defaults.CHUNK_SIZE,
    truncate=defaults.TRUNCATE,
    flip=defaults.IMAGE_FLIP,
    skip=0,
):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.AUDIO:
        return convert_audio_to_numpy(
            inp,
            out_dir,
            sr=sr,
            offset=offset,
            duration=duration,
            res_type=res_type,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            chunk_size=chunk_size,
            truncate=truncate,
            skip=skip,
        )
    elif dtype == DataType.IMAGE:
        return convert_image_to_numpy(inp, out_dir, flip=flip, skip=skip)


def convert_to_image(
    inp,
    out_dir,
    sr=defaults.SAMPLE_RATE,
    offset=defaults.AUDIO_OFFSET,
    duration=defaults.AUDIO_DURATION,
    res_type=defaults.RESAMPLE_TYPE,
    n_fft=defaults.N_FFT,
    hop_length=defaults.HOP_LENGTH,
    chunk_size=defaults.CHUNK_SIZE,
    truncate=defaults.TRUNCATE,
    flip=defaults.IMAGE_FLIP,
    skip=0,
):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.AUDIO:
        return convert_audio_to_image(
            inp,
            out_dir,
            sr=sr,
            offset=offset,
            duration=duration,
            res_type=res_type,
            n_fft=n_fft,
            hop_length=hop_length,
            chunk_size=chunk_size,
            truncate=truncate,
            flip=flip,
            skip=skip,
        )
    elif dtype == DataType.NUMPY:
        return convert_numpy_to_image(inp, out_dir, flip=flip, skip=skip)


def convert_to_audio(
    inp,
    out_dir,
    sr=defaults.SAMPLE_RATE,
    n_fft=defaults.N_FFT,
    hop_length=defaults.HOP_LENGTH,
    fmt=defaults.AUDIO_FORMAT,
    offset=defaults.AUDIO_OFFSET,
    duration=defaults.AUDIO_DURATION,
    flip=defaults.IMAGE_FLIP,
    skip=0,
):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.NUMPY:
        return convert_numpy_to_audio(
            inp,
            out_dir,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            fmt=fmt,
            offset=offset,
            duration=duration,
            skip=skip,
        )
    elif dtype == DataType.IMAGE:
        return convert_image_to_audio(
            inp,
            out_dir,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            fmt=fmt,
            offset=offset,
            duration=duration,
            flip=flip,
            skip=skip,
        )


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
