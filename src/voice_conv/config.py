from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REF_DIR = DATA_DIR / "ref"
SRC_DIR = DATA_DIR / "src"
OUT_DIR = DATA_DIR / "out"

WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base-960h"
SPEAKER_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


def load_yaml_config(path: Path | None = None) -> Dict[str, Any]:
    """
    Load config.yaml from project root if present.
    Returns {} if file is missing or invalid.
    """
    if path is None:
        path = BASE_DIR / "config.yaml"

    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


@dataclass
class VCConfig:
    # For SpeechBrain + Wav2Vec2 (both expect 16k)
    feature_sample_rate: int = 16000

    # For RVC model (32k or 48k typically)
    vc_sample_rate: int = 48000

    device: str = "cuda"
    wav2vec2_model_id: str = WAV2VEC2_MODEL_ID
    speaker_model_id: str = SPEAKER_MODEL_ID

    @classmethod
    def from_yaml(cls, overrides: Dict[str, Any] | None = None) -> "VCConfig":
        """
        Build VCConfig from YAML + optional overrides dict.
        """
        yaml_cfg = load_yaml_config()
        data: Dict[str, Any] = {
            "feature_sample_rate": yaml_cfg.get("feature_sample_rate", 16000),
            "vc_sample_rate": yaml_cfg.get("vc_sample_rate", 48000),
            "device": yaml_cfg.get("device", "cuda"),
        }
        if overrides:
            data.update(overrides)
        return cls(**data)
