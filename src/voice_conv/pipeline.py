from pathlib import Path
from typing import Optional

from .config import VCConfig
from .audio_io import load_mono_audio, save_audio
from .preprocess import normalize, trim_silence
from .embedding import SpeakerEncoder
from .content_encoder import ContentEncoder
from .vc_model import VoiceConversionModel
from .utils.logging_utils import get_logger

log = get_logger("voice_conv.pipeline")


def run_conversion(
    ref_path: str | Path,
    src_path: str | Path,
    out_path: str | Path,
    config: Optional[VCConfig] = None,
) -> None:
    """
    High-level A+B -> C pipeline:
      ref_path: audio A (target voice)
      src_path: audio B (content)
      out_path: output audio C

    We:
      - Use feature_sample_rate (16k) for SpeechBrain + Wav2Vec2
      - Use vc_sample_rate (32k or 48k) for the RVC model
    """
    cfg = config or VCConfig()

    feat_sr = cfg.feature_sample_rate
    vc_sr = cfg.vc_sample_rate

    log.info(f"Running conversion: ref={ref_path}, src={src_path}, out={out_path}")
    log.info(f"Feature SR={feat_sr}, VC SR={vc_sr}")

    # 1. Load audio for features (16k) and VC (32/48k)
    ref_audio, _ = load_mono_audio(ref_path, sr=feat_sr)
    log.debug(f"Loaded ref_audio len={len(ref_audio)} @ {feat_sr}")

    src_audio_feat, _ = load_mono_audio(src_path, sr=feat_sr)
    log.debug(f"Loaded src_audio_feat len={len(src_audio_feat)} @ {feat_sr}")

    src_audio_vc, _ = load_mono_audio(src_path, sr=vc_sr)
    log.debug(f"Loaded src_audio_vc len={len(src_audio_vc)} @ {vc_sr}")

    ref_audio = trim_silence(normalize(ref_audio))
    src_audio_feat = trim_silence(normalize(src_audio_feat))
    src_audio_vc = trim_silence(normalize(src_audio_vc))

    # 2. Initialize models
    speaker_encoder = SpeakerEncoder(device=cfg.device, cfg=cfg)
    content_encoder = ContentEncoder(device=cfg.device, cfg=cfg)
    vc_model = VoiceConversionModel(device=cfg.device)

    # 3. Extract features (using 16k audio)
    log.info("Extracting speaker embedding and content features...")
    speaker_emb = speaker_encoder.extract_embedding(ref_audio, feat_sr)
    content_feats = content_encoder.extract_content(src_audio_feat, feat_sr)
    f0 = content_encoder.extract_f0(src_audio_feat, feat_sr)

    # 4. Voice conversion (RVC backend, using 32/48k audio)
    log.info("Running RVC model...")
    out_audio = vc_model.convert(
        src_audio=src_audio_vc,
        sr=vc_sr,
        content_features=content_feats,
        f0=f0,
        speaker_embedding=speaker_emb,
    )

    # 5. Normalize & save at VC sample rate
    out_audio = normalize(out_audio)
    log.info(f"Saving output to {out_path}")
    save_audio(out_path, out_audio, vc_sr)
