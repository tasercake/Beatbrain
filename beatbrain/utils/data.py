from pathlib import Path
import math
import librosa
import imageio
import numpy as np
from natsort import natsorted
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import default_config
from .misc import DataType, EXTENSIONS

__all__ = [
    "InverseMelScale",
    "split_spectrogram",
    "audio_to_spectrogram",
    "spectrogram_to_audio",
    "normalize_spectrogram",
    "denormalize_spectrogram",
    "load_arrays",
    "load_image",
    "load_images",
    "save_arrays",
    "save_image",
    "save_images",
]


def create_fb_matrix(
    n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int
):
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
    r"""Solve for a normal STFT from a mel frequency STFT, using a conversion
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
    spec, scale_fn=None, top_db=default_config.hparams.spec.top_db, ref=np.max, **kwargs
):
    """
    Log and normalize a mel spectrogram using `librosa.power_to_db()`
    """
    scale_fn = scale_fn or librosa.power_to_db
    return (scale_fn(spec, top_db=top_db, ref=ref, **kwargs) / top_db) + 1


def denormalize_spectrogram(
    spec, scale_fn=None, top_db=default_config.hparams.spec.top_db, ref=32768, **kwargs
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
