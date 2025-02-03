from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import musdb
import torch
from einops import rearrange
from torch.utils.data import IterableDataset

from src import MUSDB_PATH


# Implementation from https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_sourceseparation.html
class MusDBDataset(IterableDataset):
    """
    A dataset for MusDB tracks with segment processing and multiprocessing
    loading.

    Attributes:
    split (str): The dataset split ('train', 'valid', etc.).
    targets (list): List of target audios ('vocals', 'drums', etc.).
    segment_dur (float): Duration of each segment in seconds.
    segment_overlap (float): Overlap between segments in seconds.
    num_workers (int): Number of threads for parallel processing.
    mus (musdb.DB): MusDB dataset.
    """

    def __init__(
        self,
        split: str,
        targets: list[str],
        segment_dur: float,
        segment_overlap: float,
        num_workers: int,
        musdb_path: str = MUSDB_PATH,
    ) -> None:
        """
        Initialize the MusDBDataset.

        Parameters:
        split (str): The dataset split ('train', 'valid', etc.).
        targets (list): List of target audios ('vocals', 'drums', etc.).
        segment_dur (float): Duration of each segment in seconds.
        segment_overlap (float): Overlap between segments in seconds.
        num_workers (int): Number of threads for parallel processing.
        """

        valid_targets = {"vocals", "drums", "bass", "other"}
        assert all(target in valid_targets for target in targets)

        self.split = split
        self.targets = targets
        self.segment_dur = segment_dur
        self.segment_overlap = segment_overlap
        self.num_workers = num_workers
        self.mus = musdb.DB(
            root=musdb_path,
            subsets="train" if self.split in ["train", "valid"] else self.split,
            split=self.split,
            is_wav=True,
        )

        if not self.mus.tracks:
            raise ValueError(f"The dataset for split '{self.split}' is empty or not loaded properly.")

    def _process_track(self, track: musdb.MultiTrack) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        """
        Process a track by unfolding and rearranging its audio data.

        Parameters:
        track (musdb.Track): A musdb track object.

        Returns:
        list[tuple[torch.Tensor, torch.Tensor, str]]: Processed input and target audio segments.
        """
        size = int(track.rate * self.segment_dur)
        step = int(track.rate * self.segment_overlap)
        name = track.name
        rate = track.rate
        mix = torch.as_tensor(track.audio, dtype=torch.float32)
        mix = rearrange(mix.unfold(dimension=0, size=size, step=step), "s c d -> s d c")
        segments = []
        for target in self.targets:
            y = torch.as_tensor(track.targets[target].audio, dtype=torch.float32)
            y = rearrange(y.unfold(dimension=0, size=size, step=step), "s c d -> s d c")
            segments.extend(list(zip(mix, y, [target] * mix.shape[0], strict=True)))
        del (mix, y, track)
        return (segments, name, rate)

    def _process_test_track(self, track: musdb.MultiTrack) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        """
        Process a track by unfolding and rearranging its audio data.

        Parameters:
        track (musdb.Track): A musdb track object.

        Returns:
        list[tuple[torch.Tensor, torch.Tensor, str]]: Processed input and target audio segments.
        """
        name = track.name
        rate = track.rate
        mix = torch.tensor(track.audio).to(torch.float32)
        pairs = [None] * len(self.targets)
        for i, target in enumerate(self.targets):
            y = torch.tensor(track.targets[target].audio).to(torch.float32)
            pairs[i] = (mix, y, target)
        del (mix, y, track)
        return (pairs, name, rate)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, str]]:
        """
        Iterate over the dataset with parallel track processing.

        Yields:
        tuple[torch.Tensor, torch.Tensor, str]: Segmented input and target audio.
        """

        if self.split == "train":
            # Continuous processing for training data
            while True:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    tracks = [self.mus.tracks[i] for i in torch.randperm(len(self.mus.tracks)).tolist()][
                        : self.num_workers
                    ]
                    futures = [executor.submit(self._process_track, track) for track in tracks]
                    for future in as_completed(futures):
                        segments = future.result()[0]
                        yield from segments
        elif self.split in ["valid", "test"]:
            # One-time processing for test data
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._process_test_track, track) for track in self.mus.tracks]
                for future in as_completed(futures):
                    yield future.result()
        else:
            raise ValueError(f"Unknown split: {self.split}")
