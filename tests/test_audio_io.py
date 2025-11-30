from pathlib import Path
import numpy as np

from voice_conv.audio_io import save_audio, load_mono_audio
from voice_conv.config import DATA_DIR


def test_audio_roundtrip(tmp_path: Path):
    sr = 16000
    audio = np.random.randn(sr).astype("float32") * 0.01
    out_file = tmp_path / "test.wav"

    save_audio(out_file, audio, sr)
    loaded, loaded_sr = load_mono_audio(out_file, sr=sr)

    assert loaded_sr == sr
    assert loaded.shape == audio.shape
