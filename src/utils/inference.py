import torch


def reconstruct_from_segments(segments: torch.Tensor, hop_size: int = None, stereo: bool = True):
    """
    Reconstruct signal from overlapping segments using overlap-add method.

    Parameters:
    -----------
    segments : torch.Tensor or numpy.ndarray
        Tensor/array of shape (num_segments, segment_size) containing the predicted segments
    hop_size : int
        Number of samples between consecutive segments (hop_size < segment_size for overlap)

    Returns:
    --------
    reconstructed : torch.Tensor or numpy.ndarray
        Reconstructed signal
    """
    hop_size = hop_size if hop_size else segments.shape[1] // 2
    num_segments, segment_size = segments.shape[:2]

    # Calculate the expected length of the reconstructed signal
    expected_length = (num_segments + 1) * hop_size

    # Initialize the reconstructed signal and normalization buffer
    reconstructed = torch.zeros((expected_length, 2 if stereo else 1), dtype=torch.float32).to(segments.device)

    # Overlap-average
    for i in range(num_segments):
        start = i * hop_size
        end = start + segment_size
        half = segment_size // 2

        if i == 0:
            middle = half
            reconstructed[start:middle] = segments[i, :middle]
            reconstructed[middle:end] = 0.5 * segments[i, middle:]
        elif i == num_segments - 1:
            middle = start + half
            reconstructed[start:middle] = segments[i, :half]
            reconstructed[middle:end] += 0.5 * segments[i, half:]
        else:
            reconstructed[start:end] += 0.5 * segments[i]

    return reconstructed
