import yaml
from pydantic import BaseModel

from src.configs.implementation_configs import DatasetConfig, ModelConfig


class DataLoaderConfig(BaseModel):
    buffer_size: int
    batch_size: int
    steps_per_epoch: int
    target: str


class SourceSeparationTrainerConfig(BaseModel):
    criteria: str
    segment_length: float
    segment_overlap: float
    sample_rate: int


class TrainerConfig(BaseModel):
    accelerator: str
    max_epochs: int
    monitor: str
    monitor_mode: str
    patience: int
    precision: str
    accumulate_grad_batches: int


class TrainConfig(BaseModel):
    experiment_name: str
    dataset: DatasetConfig
    loader: DataLoaderConfig
    val_dataset: DatasetConfig
    model: ModelConfig
    source_separation_trainer: SourceSeparationTrainerConfig
    trainer: TrainerConfig

    @staticmethod
    def from_file(path: str) -> "TrainConfig":
        TrainConfig.model_validate(yaml.safe_load(path))
