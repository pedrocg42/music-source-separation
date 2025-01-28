from einops import rearrange
import torch
import numpy as np
import musdb
import random
from torch.utils.data import IterableDataset, DataLoader

from src import MUSDB_PATH


class MusDB18BufferDataset(IterableDataset):
    def __init__(
        self,
        target: str,
        split: str = "train",
        segment_length: int = 2,  # 4 seconds
        segment_overlap: int = 2,
        sample_rate: int = 44100,
        num_channels: int = 2,
        buffer_size: int = 512,  # Number of segments to keep in buffer
        drop_stem_prob: float = 0.1,
        stereo: bool = False,
        musdb_path: str = MUSDB_PATH,
    ):
        self.target = target
        self.split = split
        self.segment_length = segment_length * sample_rate
        self.segment_overlap = segment_overlap * sample_rate
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.drop_stem_prob = drop_stem_prob
        self.stereo = stereo

        # Initialize buffers for each stem
        self.stems = ["drums", "bass", "vocals", "other"]
        self.buffers = {stem: [] for stem in self.stems}

        self.mus = musdb.DB(
            root=musdb_path,
            is_wav=True,
            subsets="train" if self.split in ["train", "valid"] else self.split,
            split=self.split,
        )
        if not self.mus.tracks:
            raise ValueError(f"The dataset for split '{self.split}' is empty or not loaded properly.")

        self.current_track_idx = 0

    def _fill_buffers(self):
        """Fill buffers with segments from tracks."""
        while any(len(buffer) < self.buffer_size for buffer in self.buffers.values()):
            track = self.mus.tracks[self.current_track_idx]

            # Load segments for all stems
            for stem in self.stems:
                stem_audio = track.targets[stem].audio
                torch_stem_audio = torch.tensor(stem_audio).to(torch.float32)
                segments = rearrange(
                    torch_stem_audio.unfold(dimension=0, size=self.segment_length, step=self.segment_overlap),
                    "s c d -> s d c",
                )

                # Convert to mono if needed
                if self.num_channels == 1:
                    segments = segments.sum(dim=2)

                self.buffers[stem].extend([segment for segment in segments])

            self.current_track_idx += 1

    def _get_random_segments(self) -> dict[str, torch.Tensor]:
        """Get random segments from buffers."""
        segments = {}
        for stem in self.stems:
            # Randomly select a segment
            idx = random.randrange(len(self.buffers[stem]))
            segments[stem] = self.buffers[stem][idx]

        return segments

    def _create_mix(self, segments: dict[str, torch.Tensor]) -> torch.Tensor:
        """Create a mix from segments with random weights."""
        weights = 0.3 + torch.rand(len(self.stems), dtype=torch.float32)
        mix = torch.zeros_like(segments[self.stems[0]])
        for stem, weight in zip(self.stems, weights, strict=True):
            if stem == self.target or torch.rand(1) > self.drop_stem_prob:
                mix += weight * segments[stem]
        return mix

    def __iter__(self):
        self.current_track_idx = 0
        random.shuffle(self.mus.tracks)
        # Pre-fill buffers
        self._fill_buffers()
        return self

    def __next__(self):
        """Get a training example with random segments."""

        # Refill if running low unless we have used all tracks
        if any(
            len(self.buffers[stem]) < (0.9 * self.buffer_size) for stem in self.stems
        ) and self.current_track_idx < len(self.mus.tracks):
            self._fill_buffers()

        segments = self._get_random_segments()
        for buffer in self.buffers.values():
            buffer.pop(-1)
            if len(buffer) == 0:
                raise StopIteration

        # Create input mix
        mix = self._create_mix(segments)
        target = segments[self.target]

        if not self.stereo:
            mix = mix.sum(-1)
            target = target.sum(-1)

        # Return dictionary with mix and all stems
        return mix, target
