# Implementation from https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_sourceseparation.html
from typing import Any

import torch
import torch.nn as nn
from einops import pack, rearrange, unpack
from x_transformers import Encoder as TransformerEncoder

from src.models.layers import ConvBlock, FreqGain, get_activation
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
        self, block_input_shape: list[int], n_layers: int = 4, n_filters: int = 48, max_n_filters: int = 512
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
        kargs = {
            "in_channels": block_input_shape[0],
            "out_channels": n_filters,
            "kernel_size": (1, 8),
            "stride": (1, 4),
            "padding": (0, 2),
        }
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


class UpConvBlock(nn.Module):
    """
    UpConvBlock module for upsampling and convolutional operations.

    Attributes:
    out_shape (List[int]): Shape of the output tensor.
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (int): Size of the kernel.
    stride (int): Stride of the convolution.
    padding (str): Padding type for the convolution.
    activation (str): Activation function for the output.
    """

    def __init__(
        self,
        out_shape: list[int],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: str = "same",
        activation: str = "gelu",
    ) -> None:
        """
        Initialize the UpConvBlock module.

        Parameters:
        out_shape (List[int]): Shape of the output tensor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel. Default is 4.
        stride (int): Stride of the convolution. Default is 2.
        padding (str): Padding type for the convolution. Default is "same".
        activation (str): Activation function for the output. Default is "gelu".
        """
        super().__init__()
        self.out_shape = out_shape
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UpConvBlock.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after upsampling and convolution.
        """
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


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
            kargs = {
                "out_shape": self.encoder_block_shapes[i + 1],
                "in_channels": self.encoder_block_shapes[i][0],
                "out_channels": self.encoder_block_shapes[i + 1][0],
                "kernel_size": (1, 8),
                "stride": (1, 4),
                "padding": (0, 2),
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
            x = self.ops[i](x) + encoder_outputs[i] if i < len(encoder_outputs) - 1 else self.ops[i](x)
        return x


class CustomDemucs(nn.Module):
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
        n_layers: int = 4,
        n_filters: int = 48,
        num_att_layers: int = 4,
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
        self.encoder = Encoder(self.stft.get_out_shape(), n_layers=n_layers, n_filters=n_filters)
        self.attention_layers = nn.ModuleList([])
        for _i in range(num_att_layers):
            self.attention_layers.append(
                nn.ModuleList(
                    [
                        TransformerEncoder(
                            dim=self.encoder.block_shapes[-1][0],
                            depth=1,
                            heads=8,
                            rotary_pos_emb=True,
                            use_rmsnorm=True,
                        ),
                        TransformerEncoder(
                            dim=self.encoder.block_shapes[-1][0],
                            depth=1,
                            heads=8,
                            rotary_pos_emb=True,
                            use_rmsnorm=True,
                        ),
                    ]
                )
            )
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

        x_att = x_encoder[-1]
        x_att = rearrange(x_att, "b d t f -> b f t d")
        for time_transformer, freq_transformer in self.attention_layers:
            x_att, ps = pack([x_att], "* t d")

            x_att = time_transformer(x_att)

            (x_att,) = unpack(x_att, ps, "* t d")
            x_att = rearrange(x_att, "b f t d -> b t f d")
            x_att, ps = pack([x_att], "* f d")

            x_att = freq_transformer(x_att)

            (x_att,) = unpack(x_att, ps, "* f d")
            x_att = rearrange(x_att, "b t f d -> b f t d")

        x_encoder[-1] = rearrange(x_att, "b f t d -> b d t f")

        """
        The decoder takes the output from the encoder and reconstructs the separated
        audio signal. It uses upsampling and skip connections to recover fine details.
        """
        y = self.decoder(*x_encoder[::-1])

        if self.behavior == "masking":
            y = torch.multiply(x[..., : y.shape[-1]], y)

        out = x.detach().clone()
        out[..., : y.shape[-1]] = y

        return self.istft(out)
