import numpy as np
import numpy.typing as npt
from loguru import logger
from museval.metrics import bss_eval


def compute_metrics(
    ref: npt.NDArray[np.float32],
    est: npt.NDArray[np.float32],
    sr: int = 44100,
    window: int = 1,
    hop: int = 1,
) -> dict[str, dict[str, float]]:
    """
    Computes and logs various audio separation metrics.

    Parameters:
    ref (npt.NDArray[np.float32]): Reference signal.
    est (npt.NDArray[np.float32]): Estimated signal.
    sr (int): Sampling rate.
    window (int): Window size in seconds. Defaults to 1.
    hop (int): Hop size in seconds. Defaults to 1.

    Returns:
    Dict[str, Dict[str, int]]: A dictionary containing metrics for vocals and accompaniment (acc).
    """
    logger.info("Computing the metrics...")
    sdr, isr, sir, sar, _ = bss_eval(
        ref,
        est,
        window=window * sr,
        hop=hop * sr,
        framewise_filters=False,
        bsseval_sources_version=False,
        compute_permutation=False,
    )
    logger.info("Done!")

    output = {"sdr": sdr[0], "isr": isr[0], "sir": sir[0], "sar": sar[0]}

    # The median is the most standard way of comparing metrics on source separation
    logger.info(
        f"Results target: \n\tSDR: {np.nanmedian(sdr[0])}, \n\tSIR: {np.nanmedian(sir[0])}, \n\tSAR: {np.nanmedian(sar[0])}"
    )
    return output
