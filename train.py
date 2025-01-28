import torch
from tqdm import tqdm
from src import DEVICE
from src.datasets.musdb18_buffer_dataset import MusDB18BufferDataset
from src.models.model import CustomMusic2Latent
from src.models.music2latent import Music2Latent2Source


def train():
    duration = 4
    dataset = MusDB18BufferDataset(
        split="train",
        target="vocals",
        segment_length=duration,
        segment_overlap=duration // 2,
        stereo=False,
    )
    # Example of how to initialize the data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True,
        collate_fn=None,
    )

    model = CustomMusic2Latent().to(DEVICE)

    for x, y in tqdm(loader, desc="Training..."):
        x = x.to(DEVICE)
        # x and y are the input and target audio segments, respectively
        pred = model(x)
        pass


if __name__ == "__main__":
    train()
