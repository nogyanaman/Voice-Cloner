from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa


def load_mono_audio(path: str | Path, sr: int) -> Tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at the given sample rate."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, out_sr = librosa.load(path, sr=sr, mono=True)
    return audio.astype("float32"), out_sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Save mono float32 audio to WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)
