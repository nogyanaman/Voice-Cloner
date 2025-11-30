import pytest
import numpy as np

from voice_conv.preprocess import normalize, trim_silence


def test_normalize():
    audio = np.array([0.1, -0.2, 0.3], dtype="float32")
    out = normalize(audio, target_peak=0.5)
    assert np.max(np.abs(out)) == pytest.approx(0.5, rel=1e-3)


def test_trim_silence():
    audio = np.concatenate(
        [np.zeros(100, dtype="float32"), np.ones(50, dtype="float32"), np.zeros(100, dtype="float32")]
    )
    trimmed = trim_silence(audio, threshold=0.01)
    assert len(trimmed) == 50
