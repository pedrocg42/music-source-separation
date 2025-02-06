# Code extracted from https://geoffroypeeters.github.io/deeplearning-101-audiomir_book/task_sourceseparation.html
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def get_audio_prepro_args(
    dur: float,
    window: str = "hanning",
    n_fft: int = 4096,
    sr: int = 44100,
    hop_factor: float = 0.5,
    stereo: bool = True,
) -> tuple[torch.Tensor, int, int, int, int, int, int, bool]:
    """
    Prepare audio preprocessing arguments.

    Parameters:
    dur (float): Duration in seconds.
    window (str): Window type. Default is "hanning".
    n_fft (int): Number of FFT points. Default is 2048.
    sr (int): Sample rate. Default is 44100.
    hop_factor (float): Factor to calculate hop length. Default is 0.5.
    stereo (bool): If True, stereo audio is used. Default is True.

    Returns:
    tuple: window tensor, number of FFT points, hop length, sample rate, number of frames,
           number of bins, length in samples, stereo flag.
    """
    hop_fft = np.round(n_fft * hop_factor).astype(np.int32)
    length_in_samples = int(np.ceil(dur * sr))
    n_frames = int(np.ceil(length_in_samples / hop_fft))
    n_bins = n_fft // 2 + 1
    if window == "hanning":
        w = torch.hann_window(n_fft)
    return w, n_fft, hop_fft, sr, n_frames, n_bins, length_in_samples, stereo


def view_as_real_img(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor to a real image tensor.

    Parameters:
    x (torch.Tensor): Complex input tensor.

    Returns:
    torch.Tensor: Real image tensor with separated real and imaginary parts.
    """
    return torch.cat((x.real.unsqueeze(-1), x.imag.unsqueeze(-1)), dim=-1)


def waveform2spec(x: torch.Tensor, window: torch.Tensor, n_fft_audio: int, fft_hop: int) -> torch.Tensor:
    """
    Convert waveform tensor to spectrogram tensor using Short-Time Fourier Transform (STFT).

    Parameters:
    x (torch.Tensor): Input waveform tensor.
    window (torch.Tensor): Window function tensor.
    n_fft_audio (int): Number of FFT points.
    fft_hop (int): Hop length for STFT.

    Returns:
    torch.Tensor: Spectrogram tensor with separated real and imaginary parts.
    """
    return view_as_real_img(
        x.stft(n_fft=n_fft_audio, window=window, hop_length=fft_hop, return_complex=True).type(torch.complex64)
    )


class STFTModule(nn.Module):
    """
    Module for Short-Time Fourier Transform (STFT) on audio signals.

    Attributes:
    duration (float): Duration of audio signal.
    window_fft (torch.Tensor): Window function tensor.
    n_fft (int): Number of FFT points.
    hop_fft (int): Hop length.
    sr (int): Sample rate.
    n_frames (int): Number of frames.
    n_bins (int): Number of frequency bins.
    length_in_samples (int): Length of audio signal in samples.
    stereo (bool): If True, stereo audio is used.
    stft_fn (function): Function for STFT computation.
    """

    def __init__(self, duration: float) -> None:
        super().__init__()
        (
            self.window_fft,
            self.n_fft,
            self.hop_fft,
            self.sr,
            self.n_frames,
            self.n_bins,
            self.length_in_samples,
            self.stereo,
        ) = get_audio_prepro_args(duration)
        self.duration = duration
        self.stft_fn = waveform2spec

    def get_input_shape(self) -> tuple[int, int]:
        """
        Get the input shape for the module.

        Returns:
        tuple: Shape of input tensor.
        """
        return self.length_in_samples, 2 if self.stereo else 1

    def get_out_shape(self) -> list[int]:
        """
        Get the output shape for the module.

        Returns:
        list: Shape of output tensor.
        """
        return [4 if self.stereo else 2, int(self.n_frames), int(self.n_bins)]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        tuple: Output tensor after STFT.
        """
        shape = x.shape
        x = rearrange(x, "b s c -> (b c) s")
        x = self.stft_fn(x, self.window_fft.to(x.device), self.n_fft, self.hop_fft)
        x = rearrange(x, "(b c) f t r -> b (c r) t f", b=shape[0], c=shape[-1])
        return x


class iSTFTModule(nn.Module):
    """
    Module for Inverse Short-Time Fourier Transform (iSTFT) on audio signals.

    Attributes:
    duration (float): Duration of audio signal.
    window_fft (torch.Tensor): Window function tensor.
    n_fft (int): Number of FFT points.
    hop_fft (int): Hop length.
    sr (int): Sample rate.
    n_frames (int): Number of frames.
    n_bins (int): Number of frequency bins.
    length_in_samples (int): Length of audio signal in samples.
    stereo (bool): If True, stereo audio is used.
    """

    def __init__(self, duration: float) -> None:
        super().__init__()
        (
            self.window_fft,
            self.n_fft,
            self.hop_fft,
            self.sr,
            self.n_frames,
            self.n_bins,
            self.length_in_samples,
            self.stereo,
        ) = get_audio_prepro_args(duration)
        self.duration = duration

    def istft_fn(
        self, x_fft: torch.Tensor, window: torch.Tensor, n_fft_audio: int, fft_hop: int, length: int
    ) -> torch.Tensor:
        """
        Compute Inverse Short-Time Fourier Transform (iSTFT).

        Parameters:
        x_fft (torch.Tensor): Input spectrogram tensor.
        window (torch.Tensor): Window function tensor.
        n_fft_audio (int): Number of FFT points.
        fft_hop (int): Hop length for iSTFT.
        length (int): Length of the output waveform.

        Returns:
        torch.Tensor: Reconstructed waveform tensor.
        """
        return x_fft.istft(
            n_fft=n_fft_audio,
            window=window,
            hop_length=fft_hop,
            return_complex=False,
            length=length,
        )

    def get_input_shape(self) -> list[int]:
        """
        Get the input shape for the module.

        Returns:
        list: Shape of input tensor.
        """
        return [4 if self.stereo else 2, self.n_frames, self.n_bins]

    def get_out_shape(self) -> tuple[int, int]:
        """
        Get the output shape for the module.

        Returns:
        tuple: Shape of output tensor.
        """
        return self.length_in_samples, 2 if self.stereo else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the module.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after iSTFT.
        """
        shape = x.shape
        x = rearrange(x, "b (c r) t f -> (b c) f t r", r=2, c=shape[1] // 2)
        x = torch.view_as_complex(x.contiguous())
        x = self.istft_fn(x, self.window_fft.to(x.device), self.n_fft, self.hop_fft, self.length_in_samples)
        return rearrange(x, "(b c) s -> b s c", b=shape[0], c=shape[1] // 2)
