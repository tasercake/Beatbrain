import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd

from . import defaults


def show_heatmap(
    spec,
    normalize=True,
    scale_fn=None,
    title=None,
    labels=False,
    flip=True,
    cmap=None,
    **kwargs
):
    """
    Display a spectrogram

    Args:
        spec (np.ndarray): The spectrogram to display
        normalize (bool): Whether to normalize the spectrogram using `scale_fn`
        scale_fn (function): A function that scales the spectrogram. If false, no scaling is performed
        title (str): The title of the plot
        labels (bool): If True, removes axis labels, colorbar, etc.
        flip (bool):
        cmap (str):
        **kwargs (dict): Additional keyword arguments passed to `sns.heatmap`
    """
    scale_fn = scale_fn or librosa.power_to_db
    if flip:
        spec = spec[::-1]
    if normalize:
        spec = scale_fn(spec, ref=np.max)
    sns.heatmap(
        spec,
        vmin=-80,
        vmax=0,
        cbar_kws={"format": "%+2.0f dB"},
        cmap=cmap,
        xticklabels="auto" if labels else False,
        yticklabels="auto" if labels else False,
        **kwargs
    )
    plt.title(title)


def show_spec(
    spec,
    title=None,
    normalize=True,
    labels=True,
    cbar=True,
    flip=True,
    cmap=None,
    mel=True,
    log=False,
    sr=defaults.SAMPLE_RATE,
    hop_length=defaults.HOP_LENGTH,
    **kwargs
):
    """
    spec (np.ndarray): The spectrogram to display
    title (str): Figure title
    normalize (bool): Whether to normalize the spectrogram
    mel (bool): Whether the spectrogram is a mel spectrogram
    labels (bool): Whether to label the plot axes
    cbar (bool): Whether to draw the color bar
    flip (bool): Whether to flip the spectrogram
    cmap: The colormap to use
    sr (int): Spectrogram sample rate
    hop_length (int): Spectrogram hop length
    **kwargs: Keyword arguments passed to `librosa.display.specshow()`
    """
    if not flip:  # Librosa flips by default!
        spec = spec[::-1]
    if normalize:
        scale_fn = librosa.power_to_db if mel else librosa.amplitude_to_db
        spec = scale_fn(spec, ref=np.max)
    kwargs["x_axis"] = "time" if labels else None
    kwargs["y_axis"] = ("mel" if mel else ("log" if log else "hz")) if labels else None
    librosa.display.specshow(
        spec, cmap=cmap, sr=sr, hop_length=hop_length, vmin=-80, vmax=0, **kwargs
    )
    if cbar:
        plt.colorbar(format="%+2.0f dB")
    plt.title(title)


def show_audio(audio, **kwargs):
    ipd.display(ipd.Audio(audio, **kwargs))
