import numpy as np


def normalize(audio: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + 1e-9)
    return (audio / peak) * target_peak


def trim_silence(audio: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """Very simple amplitude-based silence trimming."""
    idx = np.where(np.abs(audio) > threshold)[0]
    if len(idx) == 0:
        return audio
    return audio[idx[0] : idx[-1] + 1]
