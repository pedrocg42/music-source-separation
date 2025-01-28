from typing import Any
import torch
from music2latent import 
import pytorch_lightning as pl

class SourceSeparationModel(pl.LightningModule):
    

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
        in_channels: int = 512,
        out_channels: int = 64,
    ):
        super().__init__()
        
        self.model 

        

        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone network.

        Args:
            x (torch.Tensor): Input tensor to be projected into the latent space.

        Returns:
            torch.Tensor: Latent representation of the input.
        """
        y = self.model(x)
        return self.projector(y)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines a single training step, including the forward pass and loss computation.

        Args:
            batch (Any): A batch of data, expected to contain two input tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for this step.
        """
        x1, x2 = batch

        # Project x1 and x2 into the latent space
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        # Compute the loss based on z1 and z2
        loss = self.loss_fn(z1, z2)

        # Log the loss for visualization and monitoring
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with a learning rate of 1e-4.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)
