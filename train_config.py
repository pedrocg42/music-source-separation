import torch
from fire import Fire
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import DEVICE
from src.configs.train_config import TrainConfig
from src.datasets.dataset_factory import DatasetFactory
from src.datasets.source_separation_stem_dataloader import SourceSeparationStemDataloader
from src.models.loss_factory import LossFactory
from src.models.model_factory import ModelFactory
from src.source_separation_trainer import SourceSeparationTrainer

seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")


def train(config: str = None):
    train_config = TrainConfig.from_file(config)

    dataset = DatasetFactory.build(train_config.dataset)
    loader = SourceSeparationStemDataloader(dataset, **train_config.loader)

    val_dataset = DatasetFactory.build(train_config.val_dataset)

    model = ModelFactory.build(train_config.model).to(DEVICE)

    sstrainer = train_config.source_separation_trainer
    pl_model = SourceSeparationTrainer(
        model=model,
        loss_fn=LossFactory.build(sstrainer.criteria),
        segment_length=sstrainer.segment_length,
        segment_overlap=sstrainer.segment_overlap,
        sample_rate=sstrainer.sample_rate,
    )

    trainer = Trainer(
        accelerator=train_config.trainer.accelerator,
        max_epochs=train_config.trainer.accelerator,
        logger=TensorBoardLogger("logs", name=train_config.experiment_name),
        callbacks=[
            ModelCheckpoint(
                verbose=True,
                save_top_k=3,
                auto_insert_metric_name=True,
                monitor=train_config.trainer.monitor,
                mode=train_config.trainer.monitor_mode,
            ),
            EarlyStopping(
                monitor=train_config.trainer.monitor,
                patience=train_config.trainer.patience,
                mode=train_config.trainer.monitor_mode,
            ),
        ],
        precision=train_config.trainer.precision,
        accumulate_grad_batches=train_config.trainer.accumulate_grad_batches,
    )
    trainer.fit(pl_model, loader, val_dataloaders=val_dataset)


if __name__ == "__main__":
    Fire(train)
