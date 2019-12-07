from pathlib import Path

import librosa
import imageio
import numpy as np
from natsort import natsorted

from .misc import DataType, EXTENSIONS
from .. import defaults


def split_spectrogram(spec, chunk_size, truncate=True, axis=1):
    """
    Split a numpy array along the chosen axis into fixed-length chunks

    Args:
        spec (np.ndarray): The array to split along the chosen axis
        chunk_size (int): The number of elements along the chosen axis in each chunk
        truncate (bool): If True, the array is truncated such that the number of elements
                         along the chosen axis is a multiple of `chunk_size`.
                         Otherwise, the array is zero-padded to a multiple of `chunk_size`.
        axis (int): The axis along which to split the array

    Returns:
        list: A list of arrays of equal size
    """
    if spec.shape[axis] >= chunk_size:
        remainder = spec.shape[axis] % chunk_size
        if truncate:
            spec = spec[:, :-remainder]
        else:
            spec = np.pad(spec, ((0, 0), (0, chunk_size - remainder)), mode="constant")
        chunks = np.split(spec, spec.shape[axis] // chunk_size, axis=axis)
    else:
        chunks = [spec]
    return chunks


def load_image(path, flip=True, **kwargs):
    """
    Load an image as an array

    Args:
        path: The file to load image from
        flip (bool): Whether to flip the image vertically
    """
    path = _decode_tensor_string(path)
    kwargs["format"] = kwargs.get("format") or "exr"
    spec = imageio.imread(path, **kwargs)
    if flip:
        spec = spec[::-1]
    return spec


def load_arrays(path, concatenate=False, stack=False):
    """
    Load a sequence of spectrogram arrays from a npy or npz file

    Args:
        path: The file to load arrays from
        concatenate (bool): Whether to concatenate the loaded arrays (along axis 1)
        stack (bool): Whether to stack the loaded arrays
    """
    if concatenate and stack:
        raise ValueError(
            "Cannot do both concatenation and stacking: choose one or neither."
        )
    path = _decode_tensor_string(path)
    with np.load(path) as npz:
        keys = natsorted(npz.keys())
        chunks = [npz[k] for k in keys]
    if concatenate:
        return np.concatenate(chunks, axis=1)
    elif stack:
        return np.stack(chunks)
    return chunks


def audio_to_spectrogram(audio, normalize=False, norm_kwargs=None, **kwargs):
    """
    Convert an array of audio samples to a mel spectrogram

    Args:
        audio (np.ndarray): The array of audio samples to convert
        normalize (bool): Whether to log and normalize the spectrogram to [0, 1] after conversion
        norm_kwargs (dict): Additional keyword arguments to pass to the spectrogram normalization function
    """
    norm_kwargs = norm_kwargs or {}
    spec = librosa.feature.melspectrogram(audio, **kwargs)
    if normalize:
        spec = normalize_spectrogram(spec, **norm_kwargs)
    return spec


def spectrogram_to_audio(spec, denormalize=False, norm_kwargs=None, **kwargs):
    """
    Convert a mel spectrogram to audio

    Args:
        spec (np.ndarray): The mel spectrogram to convert to audio
        denormalize (bool): Whether to exp and denormalize the spectrogram before conversion
        norm_kwargs (dict): Additional keyword arguments to pass to the spectrogram denormalization function
    """
    norm_kwargs = norm_kwargs or {}
    if denormalize:
        spec = denormalize_spectrogram(spec, **norm_kwargs)
    audio = librosa.feature.inverse.mel_to_audio(spec, **kwargs)
    return audio


# TODO: Remove dependency on settings.TOP_DB
def normalize_spectrogram(
    spec, scale_fn=None, top_db=defaults.TOP_DB, ref=np.max, **kwargs
):
    """
    Log and normalize a mel spectrogram using `librosa.power_to_db()`
    """
    scale_fn = scale_fn or librosa.power_to_db
    return (scale_fn(spec, top_db=top_db, ref=ref, **kwargs) / top_db) + 1


def denormalize_spectrogram(
    spec, scale_fn=None, top_db=defaults.TOP_DB, ref=32768, **kwargs
):
    """
    Exp and denormalize a mel spectrogram using `librosa.db_to_power()`
    """
    scale_fn = scale_fn or librosa.db_to_power
    return scale_fn((spec - 1) * top_db, ref=ref, **kwargs)


def save_arrays(chunks, output, compress=True):
    """
    Save a sequence of arrays to a npy or npz file.

    Args:
        chunks (list): A sequence of arrays to save
        output (str): The file to save the arrays to'
        compress (bool): Whether to use `np.savez` to compress the output file
    """
    save = np.savez_compressed if compress else np.savez
    save(str(output), *chunks)


def save_image(spec, output, flip=True, **kwargs):
    """
    Save an array as an image.

    Args:
        spec (np.ndarray): A array to save as an image
        output (str): The path to save the image to
        flip (bool): Whether to flip the array vertically
    """
    if flip:
        spec = spec[::-1]
    kwargs["format"] = kwargs.get("format") or "exr"
    imageio.imwrite(output, spec, **kwargs)


def save_images(chunks, output: str, flip=True, **kwargs):
    """
    Save a sequence of arrays as images.

    Args:
        chunks (list): A sequence of arrays to save as images
        output (str): The directory to save the images to
        flip (bool): Whether to flip the images vertically
    """
    output = Path(output)
    for j, chunk in enumerate(chunks):
        save_image(chunk, output.joinpath(f"{j}.exr"), flip=flip, **kwargs)


def load_images(path, flip=True, concatenate=False, stack=False, **kwargs):
    """
    Load a sequence of spectrogram images from a directory as arrays

    Args:
        path: The directory to load images from
        flip (bool): Whether to flip the images vertically
        concatenate (bool): Whether to concatenate the loaded arrays (along axis 1)
        stack (bool): Whether to stack the loaded arrays
    """
    if concatenate and stack:
        raise ValueError(
            "Cannot do both concatenation and stacking: choose one or neither."
        )
    path = _decode_tensor_string(path)
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = []
        for ext in EXTENSIONS[DataType.IMAGE]:
            files.extend(path.glob(f"*.{ext}"))
        files = natsorted(files)
    chunks = [load_image(file, flip=flip, **kwargs) for file in files]
    if concatenate:
        return np.concatenate(chunks, axis=1)
    elif stack:
        return np.stack(chunks)
    return chunks


def _decode_tensor_string(tensor):
    try:
        return tensor.numpy().decode("utf8")
    except:
        return tensor
