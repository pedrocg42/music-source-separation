import torch
import torch.nn as nn
import torch.nn.functional as F
from bs_roformer.bs_roformer import Transformer
from einops import pack, rearrange, unpack
from music2latent.audio import realimag2wv, wv2realimag

from src import DEVICE


def upsample_2d(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


def downsample_2d(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


class FreqGain(nn.Module):
    def __init__(self, freq_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((1, 1, freq_dim, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class UpsampleFreqConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(4, 1), mode="nearest")
        x = self.conv(x)
        return x


class DownsampleFreqConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=(4, 1), padding=(2, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


class TimeFreqAttention(nn.Module):
    def __init__(self, dim: int, depth: int = 1, heads: int = 8, attn_dropout: float = 0.0, ff_dropout: float = 0.0):
        super().__init__()
        self.time_att = nn.ModuleList(
            Transformer(dim=dim, depth=1, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            for _ in range(depth)
        )
        self.freq_att = nn.ModuleList(
            Transformer(dim=dim, depth=1, heads=heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b d f t -> b t f d")
        for time_transformer, freq_transformer in zip(self.time_att, self.freq_att, strict=True):
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            x, _ = time_transformer(x)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            x, _ = freq_transformer(x)

            (x,) = unpack(x, ps, "* f d")

        x = rearrange(x, "b t f d -> b d f t")
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        attention: bool = False,
        attention_depth: int = 1,
        heads: int = 4,
        dropout_rate: float = 0.0,
        min_res_dropout: int = 16,
    ):
        super().__init__()

        self.attention = attention
        self.kernel_size = kernel_size
        self.min_res_dropout = min_res_dropout

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same")

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = nn.Identity()

        self.norm1 = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.norm2 = nn.GroupNorm(min(out_channels // 4, 32), out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        if self.attention:
            self.att = TimeFreqAttention(out_channels, depth=attention_depth, heads=heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)

        if x.shape[-1] <= self.min_res_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        y = self.res_conv(y)
        x = x + y
        if self.attention:
            x = self.att(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        input_channels: int = 4,
        hop: int = 1024,
        heads: int = 4,
    ):
        super().__init__()

        attention_list = [False, False, True, True, True]
        self.multipliers_list = [1, 2, 4, 4, 4]

        self.gain = FreqGain(freq_dim=hop * 2)

        output_channels = base_channels * self.multipliers_list[0]
        self.conv_inp = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)

        # DOWNSAMPLING
        self.down_blocks = nn.ModuleList()
        input_channels = output_channels
        for i, multiplier in enumerate(self.multipliers_list):
            down_block = []
            output_channels = base_channels * multiplier
            down_block.append(
                ResBlock(
                    input_channels,
                    output_channels,
                    attention=attention_list[i],
                    heads=heads,
                )
            )
            input_channels = output_channels
            if (i + 1) < len(self.multipliers_list):
                down_block.append(DownsampleFreqConv(input_channels))
            self.down_blocks.append(nn.Sequential(*down_block))

    def forward(self, x):
        x = self.conv_inp(x)
        x = self.gain(x)

        # DOWNSAMPLING
        outputs = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            outputs.append(x)

        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        data_channels: int = 4,
        heads: int = 4,
    ):
        super().__init__()

        self.attention_list = [1, 1, 1, 0, 0]
        self.multipliers_list = [4, 4, 2, 1, 1]

        input_channels = base_channels * self.multipliers_list[0]

        # UPSAMPLING
        self.up_blocks = nn.ModuleList()
        for i, multiplier in enumerate(self.multipliers_list):
            upsampling_block = []
            upsampling_block.append(
                ResBlock(
                    input_channels,
                    input_channels,
                    attention=list(reversed(self.attention_list))[i] == 1,
                    heads=heads,
                )
            )
            if i != 0:
                output_channels = base_channels * multiplier
                upsampling_block.append(UpsampleFreqConv(input_channels, output_channels))
                input_channels = output_channels
            self.up_blocks.append(nn.Sequential(*upsampling_block))

        self.head = nn.Sequential(
            nn.GroupNorm(min(input_channels // 4, 32), input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, data_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        x, encoder_outputs = latents[0], latents[1:]
        for i, (up_block, encoder_output) in enumerate(zip(self.up_blocks, encoder_outputs, strict=True)):
            x = up_block(x)
            if i < len(encoder_output):
                x = torch.add(x, encoder_output)
        x = self.head(x)
        return x


class CustomMusic2Latent(nn.Module):
    def __init__(self, hop: int = 1024):
        super().__init__()

        self.hop = hop

        self.encoder = Encoder(hop=self.hop)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=str(DEVICE)):
            if x.dim() == 3:  # stereo
                spec_l = wv2realimag(x[..., 0], hop_size=self.hop)
                spec_r = wv2realimag(x[..., 1], hop_size=self.hop)
                spec = torch.cat([spec_l, spec_r], dim=1)
            elif x.dim() == 2:  # mono
                spec = wv2realimag(x, hop_size=self.hop)

        latents = self.encoder(spec)
        decoded_spec = self.decoder(latents[::-1])

        rec_spec = torch.multiply(spec, decoded_spec)

        with torch.autocast(enabled=False, device_type=str(DEVICE)):
            if x.dim() == 3:
                rec_l = realimag2wv(rec_spec[:, :2], hop_size=self.hop)
                rec_r = realimag2wv(rec_spec[:, 2:], hop_size=self.hop)
                y = torch.stack([rec_l, rec_r], dim=-1)
            elif x.dim() == 2:
                y = realimag2wv(rec_spec, hop_size=self.hop)

        return y
