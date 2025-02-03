# Implementation from https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_sourceseparation.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqGain(nn.Module):
    def __init__(self, freq_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((1, 1, 1, freq_dim)))

    def forward(self, input):
        return input * self.scale


def interpolate_tensor(x: torch.Tensor, shape: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """
    Interpolate a tensor to a given shape using specified interpolation mode.

    Parameters:
    x (torch.Tensor): Input tensor.
    shape (tuple[int, int]): Target shape for interpolation.
    mode (str): Interpolation mode. Default is "bilinear".

    Returns:
    torch.Tensor: Interpolated tensor.
    """
    return F.interpolate(x, size=shape, mode=mode, align_corners=False, antialias=False)


def get_activation(name: str) -> nn.Module:
    """
    Retrieve activation function by name.

    Parameters:
    name (str): Name of the activation function.

    Returns:
    nn.Module: Activation function module.

    Raises:
    ValueError: If the activation function name is not found.
    """
    activation_map = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
        "prelu": torch.nn.PReLU(),
        "tanh": nn.Tanh(),
        "identity": nn.Identity(),
    }
    if name in activation_map:
        return activation_map[name]
    else:
        raise ValueError(f"Activation function {name} not found.")


class ConvBlock(nn.Module):
    """
    Convolutional block with optional dropout and activation.

    Attributes:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    ops (nn.ModuleList): List of operations in the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int = 3,
        dropout: bool = False,
        activation: str = "relu",
        **kargs: dict,
    ) -> None:
        """
        Initialize the ConvBlock.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[tuple[int, int], int]): Size of the convolving kernel. Default is 3.
        dropout (bool): If True, includes a dropout layer. Default is True.
        activation (str): Activation function name. Default is 'relu'.
        **kargs (dict): Additional keyword arguments for Conv2d.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ops = nn.ModuleList()

        self.ops.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kargs))
        self.ops.append(nn.BatchNorm2d(out_channels))
        if dropout:
            self.ops.append(nn.Dropout(p=0.25))
        self.ops.append(get_activation(activation))

    def get_out_shape(self, shape: tuple[int, int]) -> list[int]:
        """
        Compute the output shape of the block given input shape.

        Parameters:
        shape (tuple[int, int]): Input shape.

        Returns:
        list[int]: Output shape.
        """

        def to_dim_fn(v: int, p: int, d: int, k: int, s: int) -> int:
            return int(1 + ((v + 2 * p - d * (k - 1) - 1) / s))

        if self.ops[0].padding != "same":
            p = self.ops[0].padding
            d = self.ops[0].dilation
            k = self.ops[0].kernel_size
            s = self.ops[0].stride
            a = to_dim_fn(shape[1], p[0], d[0], k[0], s[0])
            b = to_dim_fn(shape[2], p[1], d[1], k[1], s[1])
        else:
            a, b = shape[1:]
        return [self.out_channels, a, b]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        for op in self.ops:
            x = op(x)
        return x


class UpConvBlock(ConvBlock):
    """
    Upsampling convolutional block with interpolation.

    Attributes:
    out_shape (list[int]): Target shape after upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, out_shape: list[int], **kargs: dict) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            **kargs,
        )
        self.out_shape = out_shape

    def get_out_shape(self) -> list[int]:
        """
        Get the output shape for the block.

        Returns:
        list[int]: Output shape.
        """
        return [self.out_channels, *self.out_shape]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = interpolate_tensor(x, self.out_shape[1:])
        return super().forward(x)
