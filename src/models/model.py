import torch
import torch.nn as nn
import torch.nn.functional as F
from music2latent.audio import wv2realimag, realimag2wv


def upsample_1d(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def downsample_1d(x):
    return F.avg_pool1d(x, kernel_size=2, stride=2)


def upsample_2d(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def downsample_2d(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)


class FreqGain(nn.Module):
    def __init__(self, freq_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((1, 1, freq_dim, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class UpsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")

    def forward(self, x):
        x = upsample_2d(x)
        x = self.conv(x)
        return x


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class UpsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=1, padding="same")

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(4, 1), mode="nearest")
        x = self.conv(x)
        return x


class DownsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.norm = nn.GroupNorm(min(in_channels // 4, 32), in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=(4, 1), padding=(2, 0))

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=0.0, add_zero_attn=False, batch_first=True
        )
        self.norm = nn.GroupNorm(min(dim // 4, 32), dim)

    def forward(self, x):
        inp = x
        x = self.norm(x)

        x = x.permute(0, 3, 2, 1)  # shape: [bs,len,freq,channels]
        bs, len, freq, channels = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(bs * len, freq, channels)  # shape: [bs*len,freq,channels]

        x = self.mha(x, x, x, need_weights=False)[0]

        x = x.reshape(bs, len, freq, channels).permute(0, 3, 2, 1)

        x = x + inp
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        attention=False,
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
        if attention:
            self.att = Attention(out_channels, heads)

    def forward(self, x):
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
        input_channels: int = 2,
        hop: int = 512,
        heads: int = 4,
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        bottleneck_channels: int = 64,
    ):
        super().__init__()

        attention_list = [False, False, True, True, True]
        self.multipliers_list = [1, 2, 4, 4, 4]
        self.freq_downsample_list = [True, False, False, False]

        self.gain = FreqGain(freq_dim=hop * 2)

        output_channels = base_channels * self.multipliers_list[0]
        self.conv_inp = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)

        self.freq_dim = (hop * 2) // (4 ** self.freq_downsample_list.count(True))
        self.freq_dim = self.freq_dim // (2 ** self.freq_downsample_list.count(False))

        # DOWNSAMPLING
        down_layers = []
        input_channels = output_channels
        for i, multiplier in enumerate(self.multipliers_list):
            output_channels = base_channels * multiplier
            down_layers.append(
                ResBlock(
                    input_channels,
                    output_channels,
                    attention=attention_list[i] == 1,
                    heads=heads,
                )
            )
            input_channels = output_channels
            if (i + 1) < len(self.multipliers_list):
                if self.freq_downsample_list[i]:
                    down_layers.append(DownsampleFreqConv(input_channels))
                else:
                    down_layers.append(DownsampleConv(input_channels))
        self.down_layers = nn.Sequential(*down_layers)

        self.prenorm_1d_to_2d = nn.GroupNorm(min(input_channels // 4, 32), input_channels)

        bottleneck_layers = []
        output_channels = bottleneck_base_channels
        bottleneck_layers.append(
            nn.Conv1d(input_channels * self.freq_dim, output_channels, kernel_size=1, stride=1, padding="same")
        )
        for _i in range(num_bottleneck_layers):
            bottleneck_layers.append(ResBlock(output_channels, output_channels))
        self.bottleneck_layers = nn.Sequential(*bottleneck_layers)

        self.norm_out = nn.GroupNorm(min(output_channels // 4, 32), output_channels)
        self.activation_out = nn.GELU()
        self.conv_out = nn.Conv1d(output_channels, bottleneck_channels, kernel_size=1, stride=1, padding="same")
        self.activation_bottleneck = nn.Tanh()

    def forward(self, x, extract_features=False):
        x = self.conv_inp(x)
        x = self.gain(x)

        # DOWNSAMPLING
        x = self.down_layers(x)

        x = self.prenorm_1d_to_2d(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
        if extract_features:
            return x

        x = self.bottleneck_layers(x)

        x = self.norm_out(x)
        x = self.activation_out(x)
        x = self.conv_out(x)
        x = self.activation_bottleneck(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        input_channels: int = 2,
        hop: int = 512,
        heads: int = 4,
        bottleneck_base_channels: int = 512,
        num_bottleneck_layers: int = 4,
        bottleneck_channels: int = 64,
    ):
        super().__init__()

        self.attention_list = [1, 1, 1, 0, 0]
        self.multipliers_list = [4, 4, 4, 2, 1]
        self.freq_upsample_list = [False, False, False, False, True]

        input_channels = base_channels * self.multipliers_list[0]

        output_channels = bottleneck_base_channels
        self.conv_inp = nn.Conv1d(bottleneck_channels, output_channels, kernel_size=1, stride=1, padding="same")

        self.freq_dim = (hop * 2) // (4 ** self.freq_upsample_list.count(True))
        self.freq_dim = self.freq_dim // (2 ** self.freq_upsample_list.count(False))

        bottleneck_layers = []
        for _i in range(num_bottleneck_layers):
            bottleneck_layers.append(ResBlock(output_channels, output_channels))
        bottleneck_layers.append(
            nn.Conv1d(output_channels, input_channels * self.freq_dim, kernel_size=1, stride=1, padding="same")
        )
        self.bottleneck_layers = nn.Sequential(*bottleneck_layers)

        # UPSAMPLING
        up_layers = []
        for i, multiplier in enumerate(self.multipliers_list):
            up_layers.append(
                ResBlock(
                    input_channels,
                    input_channels,
                    attention=list(reversed(self.attention_list))[i] == 1,
                    heads=heads,
                )
            )
            if (i + 1) < len(self.multipliers_list):
                output_channels = base_channels * multiplier
                if self.freq_upsample_list[i]:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(UpsampleConv(input_channels, output_channels))
                input_channels = output_channels

        self.up_layers = nn.Sequential(*up_layers)

    def forward(self, x):
        x = self.conv_inp(x)

        x = self.bottleneck_layers(x)

        x_ls = torch.chunk(x.unsqueeze(-2), self.freq_dim, -3)
        x = torch.cat(x_ls, -2)

        # UPSAMPLING
        x = self.up_layers(x)

        return x


class CustomMusic2Latent(nn.Module):
    def __init__(self, hop: int = 256):
        super().__init__()

        self.hop = hop

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            spec = wv2realimag(x, hop_size=self.hop)

        latent = self.encoder(spec)
        rec_spec = self.decoder(latent)

        with torch.no_grad():
            y = realimag2wv(rec_spec, hop_size=self.hop)

        return y
