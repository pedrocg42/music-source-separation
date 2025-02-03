import random
from collections import defaultdict

import torch


class SourceSeparationStemDataloader:
    """
    A custom data loader with shuffle buffer functionality.

    Attributes:
    dataset (torch.utils.data.Dataset): The dataset to shuffle.
    buffer_size (int): Size of the shuffle buffer.
        The bigger the buffer size is more audio segments  will be include.
        Since we are getting all the segments of a song to speed up the process
        we want this to be big to mix segments from different songs
    batch_size (int): Size of each batch.
    buffer (list): The buffer holding dataset elements.
    dataset_iter (iterator): An iterator over the dataset.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        buffer_size: int,
        batch_size: int,
        num_steps_per_epoch: int,
        target: str,
    ) -> None:
        """
        Initialize a shuffle buffer for the dataset.

        Parameters:
        dataset (torch.utils.data.Dataset): The dataset to shuffle.
        buffer_size (int): Size of the shuffle buffer.
        batch_size (int): Size of each batch.
        """
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_steps_per_epoch = num_steps_per_epoch
        self.target = target

        self.current_step = 0

        self.buffer = defaultdict(list)
        self.dataset_iter = iter(dataset)

        for i in range(buffer_size):
            if (i + 1) % (buffer_size // 10) == 0:
                print(f"Buffer {int(((i + 1) / buffer_size) * 100)}% filled.")
            sources = next(self.dataset_iter)
            for source_segment, source in sources:
                self.buffer[source].append(source_segment)

    def get_next(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Get the next batch of shuffled elements from the buffer.

        Returns:
        tuple[torch.Tensor, torch.Tensor, List[str]]: A batch of randomly shuffled elements.
        """
        batch_x = []
        batch_y = []
        batch_label = []

        for _ in range(self.batch_size):
            if any(len(buffer) == 0 for buffer in self.buffer.values()):
                break  # If buffer is empty, stop forming the batch

            new_segments = next(self.dataset_iter)
            new_segments = {source: segment for segment, source in new_segments}

            mix = None
            target = None
            for source in self.buffer:
                idx = random.randint(0, len(self.buffer[source]) - 1)

                if random.uniform(0, 1) > 0.1:
                    # Augment the mix +- 3dB
                    stem = self.buffer[source][idx] * random.uniform(0.7, 1.4)
                    if source == self.target:
                        target = stem

                    if mix is None:
                        mix = stem
                    else:
                        mix += stem

                if source in new_segments:
                    self.buffer[source][idx] = new_segments[source]

            batch_x.append(mix if mix is not None else torch.zeros_like(self.buffer[source][0]))
            batch_y.append(target if target is not None else torch.zeros_like(self.buffer[source][0]))
            batch_label.append(self.target)

        return torch.stack(batch_x), torch.stack(batch_y), batch_label

    def __len__(self) -> int:
        return self.num_steps_per_epoch

    def __iter__(self) -> "SourceSeparationStemDataloader":
        self.current_step = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if self.current_step < self.num_steps_per_epoch:
            self.current_step += 1
            return self.get_next()
        else:
            raise StopIteration
