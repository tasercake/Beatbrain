from pathlib import Path
from joblib import Parallel, delayed
from natsort import natsorted
from loguru import logger
import soundfile as sf
import resampy
import numpy as np

from torch.utils.data import Dataset
from ..utils import registry


def get_num_segments(path, max_segment_length, min_segment_length):
    """
    Calculate the number of audio segments of sufficient length contained within an audio file.

    Args:
        path: Path to a single audio file
        max_segment_length (float): The maximum length (in seconds) of each audio segment. If `None`, 1 segment is assumed.
        min_segment_length (float): The minimum length (in seconds) of each audio segment.
    """
    try:
        file_info = sf.info(str(path))  # Load file info and check its validity
        # Return 1 if segmenting is disabled
        if max_segment_length is None:
            return 1
        # Compute the number of segments otherwise
        else:
            sr = file_info.samplerate
            samples = file_info.frames
            duration = samples / sr
            num_segments = int(duration / max_segment_length)
            if duration % max_segment_length >= min_segment_length:
                num_segments += 1
            return num_segments
    except RuntimeError:  # SoundFile raises a `RuntimeError` when it fails to read a file :(
        return 0


@registry.register("dataset", "AudioClipDataset")
class AudioClipDataset(Dataset):
    def __init__(self, paths, recursive=True, max_segment_length=5, min_segment_length=1, sample_rate=22050, mono=True, pad=True):
        """
        Args:
            paths: A path (file or directory) or a collection of file paths.
            recursive (bool): Whether to recursively search for audio files when a directory is provided
            max_segment_length (float): The maximum length (in seconds) of each audio segment.
            min_segment_length (float): The minimum length (in seconds) of each audio segment. Shorter segments are discarded.
            sample_rate (int): The rate at which to resample audio. If `None`, no resampling is performed.
            mono (bool): Whether to downmix multichannel audio clips to a single channel.
        """
        super().__init__()
        # Store params
        self.recursive = recursive
        self.max_segment_length = max_segment_length
        self.min_segment_length = min_segment_length
        self.sample_rate = sample_rate
        self.mono = mono
        self.pad = pad

        # Scan for files
        try:  # Single file or directory
            paths = Path(paths)
            if paths.is_dir():
                self.paths = list(filter(lambda f: f.is_file(), paths.rglob("*") if self.recursive else paths.iterdir()))
            else:
                self.paths = [paths]
        except TypeError:  # Collection of files
            self.paths = list(map(Path, paths))
        self.paths = np.asarray(natsorted(self.paths))
        if len(self.paths) == 0:
            raise ValueError(f"Couldn't find any valid audio files in {paths}")

        # Count the number of segments in each audio file
        self.num_track_segments = np.array(Parallel(n_jobs=-1, backend="threading")(delayed(get_num_segments)(str(path), self.max_segment_length, self.min_segment_length) for path in self.paths))
        # Find and exclude unusable tracks (either unreadable or too short)
        valid_tracks_mask = self.num_track_segments > 0
        invalid_tracks_mask = ~valid_tracks_mask
        if invalid_tracks_mask.sum() > 0:
            logger.warning(f"Failed to load {invalid_tracks_mask.sum()} tracks.")
            for invalid_index in invalid_tracks_mask.nonzero()[0]:
                logger.debug(f"Failed to load {str(self.paths[invalid_index])}")
            self.paths = self.paths[valid_tracks_mask]
            self.num_track_segments = self.num_track_segments[valid_tracks_mask]
        self.cumulative_num_track_segments = np.cumsum(self.num_track_segments)
        self.num_total_segments = self.cumulative_num_track_segments[-1]

    def __getitem__(self, index):
        """
        Fetches an audio segment as a numpy array.
        Reads a segment of audio, performs resampling/downmixing to mono

        Args:
            index (int): The index of the segment to fetch.

        Returns:
            tuple: A 2D `np.float32` array of raw audio, and the audio's sample rate
        """
        # TODO: Fix crash when using a DataLoader with several workers
        if index < 0:
            index = self.num_total_segments + index
        if index >= len(self):
            raise IndexError(f"Sample index out of range. Max index is {len(self) - 1}")
        track_index = np.min(np.where(self.cumulative_num_track_segments > index))
        if track_index == 0:
            index_remainder = index
        else:
            index_remainder = index - self.cumulative_num_track_segments[track_index - 1]
        track_path = self.paths[track_index]
        with sf.SoundFile(str(track_path)) as file:
            # Get track info
            track_sample_rate = file.samplerate
            start_pos = track_sample_rate * self.max_segment_length * index_remainder
            num_samples = track_sample_rate * self.max_segment_length

            # Load raw audio
            file.seek(start_pos)
            audio = file.read(num_samples, dtype=np.float32, fill_value=0 if self.pad else None, always_2d=True)

        # Channel-first index order
        audio = audio.T

        # Resample if required
        if self.sample_rate is None or self.sample_rate == track_sample_rate:
            output_sr = track_sample_rate
        else:
            output_sr = self.sample_rate
            audio = resampy.resample(audio, track_sample_rate, output_sr, filter="kaiser_fast")

        # Optionally downmix to mono
        if self.mono and audio.ndim > 1:
            audio = audio.mean(0, keepdims=True)
        return audio, output_sr

    def __len__(self):
        return self.num_total_segments
