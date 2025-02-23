from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import musdb
import torch
from einops import rearrange
from loguru import logger
from torch.utils.data import IterableDataset

from src import MUSDB_PATH


# Implementation from https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_sourceseparation.html
class MusDBStemDataset(IterableDataset):
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

        self.valid_targets = {"vocals", "drums", "bass", "other"}
        assert all(target in self.valid_targets for target in targets)

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

        # Test purposes
        if split == "valid":
            self.mus.tracks = self.mus.tracks[:1]

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
        sources = [None] * len(self.valid_targets)
        for i, source in enumerate(self.valid_targets):
            y = torch.as_tensor(track.sources[source].audio, dtype=torch.float32)
            # Unfold into segments
            segments = y.unfold(dimension=0, size=size, step=step)
            segments = rearrange(segments, "s c d -> s d c")

            # Vectorized dB calculation
            # Calculate RMS across time dimension (dim=1) and channels (dim=2)
            rms = torch.sqrt(torch.mean(segments**2, dim=(1, 2)))
            db_levels = 20 * torch.log10(rms + 1e-10)

            # Create mask for segments above threshold
            valid_mask = db_levels > -50

            # Apply mask to keep valid segments
            valid_segments = segments[valid_mask]

            # Handle empty case
            if len(valid_segments) == 0:
                valid_segments = torch.zeros((0, size, segments.shape[-1]), dtype=torch.float32)

            sources[i] = zip(
                valid_segments,
                [source] * len(valid_segments),
                [rate] * len(valid_segments),
                strict=True,
            )

        sources = list(zip(*sources, strict=False))
        del (y, track)
        return (sources, name, rate)

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
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    try:
                        tracks = [self.mus.tracks[i] for i in torch.randperm(len(self.mus.tracks)).tolist()][
                            : self.num_workers
                        ]
                        futures = [executor.submit(self._process_track, track) for track in tracks]
                        for future in as_completed(futures):
                            yield from future.result()[0]
                    except BrokenProcessPool as err:
                        logger.warning(err)
        elif self.split in ["valid", "test"]:
            try:
                # One-time processing for test data
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(self._process_test_track, track) for track in self.mus.tracks]
                    for future in as_completed(futures):
                        yield future.result()[0]
            except BrokenProcessPool as err:
                logger.warning(err)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self) -> int:
        return len(self.mus.tracks) if self.split in ["valid", "test"] else None
