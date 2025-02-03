# Implementation from https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_sourceseparation.html
from typing import Any

import torch
import torch.nn as nn

from src.models.layers import ConvBlock, FreqGain, UpConvBlock
from src.models.stft import STFTModule, iSTFTModule


class Encoder(nn.Module):
    """
    Encoder module consisting of multiple convolutional blocks.

    Attributes:
    n_layers (int): Number of layers in the encoder.
    ops (nn.ModuleList): List of convolutional operations.
    block_shapes (List[List[int]]): List of shapes of the blocks in the encoder.
    """

    def __init__(
        self, block_input_shape: list[int], n_layers: int = 6, n_filters: int = 16, max_n_filters: int = 512
    ) -> None:
        """
        Initialize the Encoder module.

        Parameters:
        block_input_shape (List[int]): Shape of the input block.
        n_layers (int): Number of layers in the encoder. Default is 6.
        n_filters (int): Initial number of filters for the convolutions. Default is 16.
        max_n_filters (int): Maximum number of filters for the convolutions. Default is 512.
        """
        super().__init__()
        self.n_layers = n_layers
        self.ops = nn.ModuleList([])
        self.block_shapes = [block_input_shape]
        kargs = {"out_channels": n_filters, "stride": 2, "in_channels": block_input_shape[0]}
        for _ in range(self.n_layers):
            self.ops.append(ConvBlock(**kargs))
            block_input_shape = self.ops[-1].get_out_shape(block_input_shape)
            self.block_shapes.append(block_input_shape)
            kargs["in_channels"] = self.block_shapes[-1][0]
            kargs["out_channels"] = min(self.block_shapes[-1][0] * 2, max_n_filters)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor | Any]:
        """
        Forward pass through the encoder.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        list[Union[torch.Tensor, Any]]: List of outputs from each layer of the encoder.
        """
        outputs = [x]
        for op in self.ops:
            x = op(x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    """
    Decoder module for upsampling and reconstructing the input.

    Attributes:
    mask_act (str): Activation function for the mask.
    encoder_block_shapes (List[List[Any]]): Shapes of the encoder blocks.
    ops (nn.ModuleList): List of upsampling convolutional operations.
    """

    def __init__(
        self,
        encoder_block_shapes: list[list[Any]],
        mask_act: str,
    ) -> None:
        """
        Initialize the Decoder module.

        Parameters:
        encoder_block_shapes (List[List[Any]]): Shapes of the encoder blocks.
        mask_act (str): Activation function for the mask.
        """
        super().__init__()
        self.mask_act = mask_act
        self.encoder_block_shapes = encoder_block_shapes
        self.ops = nn.ModuleList([])

        for i in range(len(self.encoder_block_shapes) - 1):
            in_channels = self.encoder_block_shapes[i][0] * 2 if i != 0 else self.encoder_block_shapes[i][0]
            kargs = {
                "out_shape": self.encoder_block_shapes[i + 1],
                "in_channels": in_channels,
                "out_channels": self.encoder_block_shapes[i + 1][0],
                "padding": "same",
            }
            if i == len(self.encoder_block_shapes) - 2:
                kargs["activation"] = mask_act
            self.ops.append(UpConvBlock(**kargs))

    def forward(self, *x: Any) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters:
        x (Any): Input tensors.

        Returns:
        torch.Tensor: Output tensor after upsampling and reconstruction.
        """
        x, encoder_outputs = x[0], x[1:]
        for i in range(len(encoder_outputs)):
            if i < len(encoder_outputs) - 1:
                x = torch.cat((self.ops[i](x), encoder_outputs[i]), dim=1)
            else:
                x = self.ops[i](x)
        return x


class SourceSeparation(nn.Module):
    """
    Source separation module using STFT, Encoder, Decoder, and iSTFT.

    Attributes:
    duration (float): Duration of the audio.
    behavior (str): Behavior for the output ("masking" or "mapping").
    stft (STFTModule): Short-Time Fourier Transform module.
    encoder (Encoder): Encoder module.
    decoder (Decoder): Decoder module.
    istft (iSTFTModule): Inverse Short-Time Fourier Transform module.
    """

    def __init__(
        self,
        duration: float,
        mask_act: str = "tanh",
        behavior: str = "masking",
    ) -> None:
        """
        Initialize the SourceSeparation module.

        Parameters:
        duration (float): Duration of the audio.
        mask_act (str): Activation function for the mask. Default is "tanh".
        behavior (str): Behavior for the output. Either "masking" or "mapping". Default is "masking".
        """
        super().__init__()
        assert behavior in ["masking", "mapping"]
        self.duration = duration
        self.behavior = behavior
        self.stft = STFTModule(duration)
        self.freq_gain = FreqGain(self.stft.get_out_shape()[2])
        self.encoder = Encoder(self.stft.get_out_shape())
        self.decoder = Decoder(encoder_block_shapes=self.encoder.block_shapes[::-1], mask_act=mask_act)
        self.istft = iSTFTModule(duration)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the source separation module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after source separation.
        """

        # we transform the waveform into a time-frequency representation
        x = self.stft(x)

        # Automatic Frequency Gain Control
        x = self.freq_gain(x)

        """
        The encoder takes spec with the real/img info on each channel and processes
        it through multiple convolutional layers. This step essentially extracts features
        and compresses the audio representation.
        """
        x_encoder = self.encoder(x)

        """
        The decoder takes the output from the encoder and reconstructs the separated
        audio signal. It uses upsampling and skip connections to recover fine details.
        """
        y = self.decoder(*x_encoder[::-1])

        if self.behavior == "masking":
            y = torch.multiply(x, y)

        return self.istft(y)
