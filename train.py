import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from src import DEVICE
from src.datasets.musdb18_dataset import MusDBDataset
from src.datasets.musdb18_stem_dataset import MusDBStemDataset
from src.datasets.source_separation_stem_dataloader import SourceSeparationStemDataloader
from src.models.custom_demucs import CustomDemucs
from src.source_separation_trainer import SourceSeparationTrainer

seed_everything(42, workers=True)


def train():
    duration = 4
    dataset = MusDBStemDataset(
        split="train", targets=["vocals"], segment_dur=duration, segment_overlap=duration / 2, num_workers=8
    )

    # Example of how to initialize the data loader
    loader = SourceSeparationStemDataloader(
        dataset, buffer_size=512, batch_size=32, num_steps_per_epoch=1000, target="vocals"
    )

    val_dataset = MusDBDataset(
        split="valid", targets=["vocals"], segment_dur=duration, segment_overlap=duration / 2, num_workers=8
    )

    model = CustomDemucs(duration=duration).to(DEVICE)
    pl_model = SourceSeparationTrainer(
        model=model,
        loss_fn=torch.nn.L1Loss(),
        segment_length=duration,
        segment_overlap=duration / 2,
        sample_rate=44100,
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=100,
        logger=TensorBoardLogger("logs", name="vocal_custom-demucs_l1-es-sdr"),
        callbacks=[
            ModelSummary(),
            ModelCheckpoint(verbose=True, save_top_k=3, auto_insert_metric_name=True, monitor="SDR", mode="max"),
            EarlyStopping(monitor="SDR", patience=10, mode="max"),
        ],
    )
    trainer.fit(pl_model, loader, val_dataloaders=val_dataset)


if __name__ == "__main__":
    train()
