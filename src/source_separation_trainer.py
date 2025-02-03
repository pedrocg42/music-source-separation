from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange

from src.utils.inference import reconstruct_from_segments
from src.utils.metrics import compute_metrics


class SourceSeparationTrainer(pl.LightningModule):
    """
    A Siamese Network implemented using PyTorch Lightning, designed to work with
    a backbone neural network and a loss function. The network projects two input
    samples into a latent space and optimizes their relationship via the provided loss function.

    Args:
        encoder (nn.Module): The feature extractor model that projects inputs into a latent space.
        loss_fn (nn.Module): The loss function used to optimize the network based on the similarity
                                   or dissimilarity between the two inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        segment_length: int = 4,  # 4 seconds
        segment_overlap: int = 2,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.segment_length = int(segment_length * sample_rate)
        self.segment_overlap = int(segment_overlap * sample_rate)
        self.sample_rate = sample_rate

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        mix, target, _target_name = batch

        # Project x1 and x2 into the latent space
        pred_target = self.model(mix)

        # Compute the loss based on z1 and z2
        loss = self.loss_fn(pred_target, target).mean()

        # Log the loss for visualization and monitoring
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pairs, track_name, sample_rate = batch

        for mix, target, target_name in pairs:
            segments = rearrange(
                mix.unfold(dimension=0, size=self.segment_length, step=self.segment_overlap),
                "s c d -> s d c",
            )

            pred_segments = self.model(segments)
            pred_target = reconstruct_from_segments(pred_segments)

            self.logger.experiment.add_audio(
                f"validation/epoch{self.current_epoch}/{track_name}-mix",
                mix.cpu().numpy().sum(axis=1),
                self.current_epoch,
            )
            self.logger.experiment.add_audio(
                f"validation/epoch{self.current_epoch}/{track_name}-{target_name}",
                target.cpu().numpy().sum(axis=1),
                self.current_epoch,
            )
            self.logger.experiment.add_audio(
                f"validation/epoch{self.current_epoch}/{track_name}-{target_name}_pred",
                pred_target.cpu().numpy().sum(axis=1),
                self.current_epoch,
            )

            target = target[: len(pred_target)]
            loss = self.loss_fn(pred_target, target).mean()

            # Log validation loss
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)

            metrics = compute_metrics(
                target.cpu().numpy()[np.newaxis],
                pred_target.cpu().numpy()[np.newaxis],
                sr=sample_rate,
            )

            # Log other metrics if needed
            self.log("SDR", np.nanmedian(metrics["sdr"]), on_step=False, on_epoch=True, logger=True, batch_size=1)
            self.log("ISR", np.nanmedian(metrics["isr"]), on_step=False, on_epoch=True, logger=True, batch_size=1)
            self.log("SIR", np.nanmedian(metrics["sir"]), on_step=False, on_epoch=True, logger=True, batch_size=1)
            self.log("SAR", np.nanmedian(metrics["sar"]), on_step=False, on_epoch=True, logger=True, batch_size=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RAdam(self.parameters(), lr=1e-4, weight_decay=1e-5)
