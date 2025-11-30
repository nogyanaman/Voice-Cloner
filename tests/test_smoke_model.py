import pytest

from voice_conv.embedding import SpeakerEncoder
from voice_conv.content_encoder import ContentEncoder
from voice_conv.config import VCConfig


@pytest.mark.skip(reason="Downloads models; enable manually if desired")
def test_speaker_encoder_init():
    enc = SpeakerEncoder(device="cpu", cfg=VCConfig())
    assert enc is not None


@pytest.mark.skip(reason="Downloads models; enable manually if desired")
def test_content_encoder_init():
    enc = ContentEncoder(device="cpu", cfg=VCConfig())
    assert enc is not None
