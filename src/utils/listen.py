import torch
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
import librosa.display


def visualize_and_play_audio(
    audio_tensor, sample_rate=44100, title="Audio Waveform and Spectrogram", play=True, fig_size=(15, 8)
):
    """
    Visualize and optionally play an audio tensor.

    Args:
        audio_tensor (torch.Tensor): Audio tensor of shape (channels, samples) or (samples,)
        sample_rate (int): Audio sample rate
        title (str): Plot title
        play (bool): Whether to play the audio
        fig_size (tuple): Figure size for the plot
    """
    # Convert to numpy and handle channels
    if audio_tensor.ndim == 2:
        # If stereo, average channels for visualization
        audio_np = audio_tensor.detach().cpu().numpy()
        audio_mono = audio_np.mean(axis=0)
    else:
        audio_mono = audio_tensor.detach().cpu().numpy()

    # Play audio if requested
    if play:
        # Ensure audio is in [-1, 1] range
        max_val = np.abs(audio_np).max()
        if max_val > 1:
            audio_np = audio_np / max_val
        sd.play(audio_np.T, sample_rate)

    # Create plot
    plt.figure(figsize=fig_size)

    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.title(f"{title} - Waveform")
    times = np.arange(len(audio_mono)) / sample_rate
    plt.plot(times, audio_mono)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.title(f"{title} - Spectrogram")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_mono, n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis="time", y_axis="hz", hop_length=512)
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.show()


def visualize_source_separation(mix_tensor, sources_dict, sample_rate=44100, play_audio=True, fig_size=(15, 10)):
    """
    Visualize and optionally play mixture and separated sources.

    Args:
        mix_tensor (torch.Tensor): Mixed audio tensor
        sources_dict (dict): Dictionary of source tensors
        sample_rate (int): Audio sample rate
        play_audio (bool): Whether to play each audio
        fig_size (tuple): Figure size for the plot
    """
    n_sources = len(sources_dict) + 1  # +1 for the mixture

    plt.figure(figsize=fig_size)

    # Plot mixture
    plt.subplot(n_sources, 1, 1)
    visualize_and_play_audio(mix_tensor, sample_rate=sample_rate, title="Mixture", play=play_audio, fig_size=None)

    # Plot each source
    for idx, (source_name, source_tensor) in enumerate(sources_dict.items(), 2):
        plt.subplot(n_sources, 1, idx)
        visualize_and_play_audio(
            source_tensor, sample_rate=sample_rate, title=f"Source: {source_name}", play=play_audio, fig_size=None
        )

    plt.tight_layout()
    plt.show()


class AudioPlayer:
    """Class to handle audio playback with play/stop control."""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.stream = None

    def play(self, audio_tensor):
        """Play audio tensor."""
        if self.stream is not None:
            self.stop()

        audio_np = audio_tensor.detach().cpu().numpy()
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)

        # Normalize audio
        max_val = np.abs(audio_np).max()
        if max_val > 1:
            audio_np = audio_np / max_val

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=audio_np.shape[0], callback=lambda *args: None
        )
        self.stream.start()
        sd.play(audio_np.T, self.sample_rate)

    def stop(self):
        """Stop audio playback."""
        if self.stream is not None:
            sd.stop()
            self.stream.stop()
            self.stream.close()
            self.stream = None
