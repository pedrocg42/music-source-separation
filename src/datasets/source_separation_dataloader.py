import random

import torch


class SourceSeparationDataloader:
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
        self, dataset: torch.utils.data.Dataset, buffer_size: int, batch_size: int, num_steps_per_epoch: int
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
        self.current_step = 0

        self.buffer = []
        self.dataset_iter = iter(dataset)

        for i in range(buffer_size):
            if (i + 1) % (buffer_size // 10) == 0:
                print(f"Buffer {int(((i + 1) / buffer_size) * 100)}% filled.")
            self.buffer.append(next(self.dataset_iter))

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
            if len(self.buffer) == 0:
                break  # If buffer is empty, stop forming the batch

            idx = random.randint(0, len(self.buffer) - 1)
            batch_x.append(self.buffer[idx][0])
            batch_y.append(self.buffer[idx][1])
            batch_label.append(self.buffer[idx][2])

            try:
                self.buffer[idx] = next(self.dataset_iter)
            except StopIteration:
                self.buffer.pop(idx)

        return torch.stack(batch_x), torch.stack(batch_y), batch_label

    def __len__(self) -> int:
        return self.num_steps_per_epoch

    def __iter__(self) -> "SourceSeparationDataloader":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if self.current_step < self.num_steps_per_epoch:
            self.current_step += 1
            return self.get_next()
        else:
            raise StopIteration
