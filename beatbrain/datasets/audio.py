from pathlib import Path
import librosa
from joblib import Parallel, delayed
from itertools import tee
import numpy as np
import os
import audioread
from natsort import natsorted
import mutagen

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from nnAudio import Spectrogram
from ..utils import registry


@registry.register("dataset", "AudioClipDataset")
class AudioClipDataset(Dataset):
    def __init__(
        self,
        directory,
        audio_load_options=None,
        clip_duration=4,
        transform=None,
    ):
        super().__init__()
        self.directory = Path(directory)
        self.audio_load_options = {"sr": 44100, "res_type": "kaiser_fast"}
        if audio_load_options:
            self.audio_load_options.update(audio_load_options)
        self.transform = transform or self.default_transform
        self.clip_duration = clip_duration

        self.tracks = self._get_files()
        self.track_durations = self._get_durations(self.tracks)
        self.cumulative_track_durations = np.cumsum(self.track_durations)
        self.track_starts = np.roll(self.cumulative_track_durations, 1)
        self.track_starts[0] = 0
        print(f"Found {self.cumulative_track_durations[-1]} seconds of audio")

    def _get_files(self):
        dir_contents = natsorted(map(str, self.directory.rglob("*")))
        files = list(filter(os.path.isfile, dir_contents))
        return files

    def _get_durations(self, files):
        def get_duration(f):
            try:
                # Loads metadata only, but unsure of filetype support
                return mutagen.File(f).info.length
            except mutagen.MutagenError:
                # Probably loads the entire file before length can be calculated
                with audioread.audio_open(f) as audio:
                    return audio.duration

        durations = Parallel(n_jobs=-1, backend="threading")(
            delayed(get_duration)(f) for f in files
        )
        # Truncate track lengths to multiples of clip duration
        durations = [(d // self.clip_duration) * self.clip_duration for d in durations]
        return durations

    def __getitem__(self, key):
        if key < 0:
            raise IndexError(f"Negative index not supported (got {key})")
        clip_start = key * self.clip_duration
        # print("Clip start point:", clip_start)
        if clip_start >= self.cumulative_track_durations[-1]:
            raise IndexError(f"Clip index {key} is out of bounds in clip list with {len(self)} elements.")
        track_index = np.argmax(self.cumulative_track_durations > clip_start)
        # print("Track index:", track_index)
        clip_index = int((clip_start - self.track_starts[track_index]) / self.clip_duration)
        # print("Clip index in track:", clip_index)
        track, sr = librosa.load(self.tracks[track_index], **self.audio_load_options)
        clip = track[sr * clip_index: sr * (clip_index + self.clip_duration)]
        # print("Clip length:", len(clip))
        if self.transform:
            clip = self.transform(clip)
        return clip

    def __len__(self):
        return int(self.cumulative_track_durations[-1] / self.clip_duration)

    @property
    def default_transform(self):
        return transforms.Compose([transforms.Lambda(torch.from_numpy),])
