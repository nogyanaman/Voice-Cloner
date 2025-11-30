from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa

# ---- SHIM: allow fairseq Dictionary in torch.load (PyTorch >= 2.6) ----
import torch.serialization as torch_serialization
from fairseq.data.dictionary import Dictionary

# Allow-list this class so torch.load(weights_only=True) can unpickle it
torch_serialization.add_safe_globals([Dictionary])
# ---- END SHIM ----

from infer_rvc_python import BaseLoader  # RVC fast inference backend


class VoiceConversionModel:
    """
    Voice conversion wrapper using RVC via infer_rvc_python.

    This currently uses RVC as a black box:
    - It takes your source audio (B)
    - Feeds it to the RVC model
    - Returns converted audio in the target voice (from model.pth)

    content_features, f0, speaker_embedding are accepted but not used yet.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        tag: str = "vc_model",
        device: str = "cuda",
    ) -> None:
        # Decide device and mode
        if torch.cuda.is_available() and device.startswith("cuda"):
            only_cpu = False
        else:
            only_cpu = True

        # >>> IMPORTANT: default model location <<<
        if model_path is None:
            model_path = Path("models/vc/model.pth")

        if index_path is None:
            index_path = None  # example: Path("models/vc/model.index")

        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path is not None else None
        self.tag = tag

        if not self.model_path.is_file():
            raise FileNotFoundError(f"RVC model not found: {self.model_path}")

        # Initialize RVC loader backend
        self.loader = BaseLoader(
            only_cpu=only_cpu,
            hubert_path=None,
            rmvpe_path=None,
        )

        # Apply model config
        self.loader.apply_conf(
            tag=self.tag,
            file_model=str(self.model_path),
            pitch_algo="rmvpe+",
            pitch_lvl=0,
            file_index=str(self.index_path) if self.index_path is not None else None,
            index_influence=0.66,
            respiration_median_filtering=3,
            envelope_ratio=0.25,
            consonant_breath_protection=0.33,
        )

    def convert(
        self,
        src_audio: np.ndarray,
        sr: int,
        content_features: Optional[np.ndarray] = None,
        f0: Optional[np.ndarray] = None,
        speaker_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert source audio into the RVC model's target voice.

        src_audio: mono float32 numpy array
        sr: sample rate (32k or 48k recommended)
        """

        audio_data = (src_audio, sr)

        # Generate using cached RVC model
        result_array, result_sr = self.loader.generate_from_cache(
            audio_data=audio_data,
            tag=self.tag,
        )

        # Resample if needed
        if result_sr != sr:
            try:
                result_array = librosa.resample(
                    result_array.astype("float32"),
                    orig_sr=result_sr,
                    target_sr=sr,
                )
            except Exception:
                pass  # fallback: leave audio unchanged

        return result_array.astype("float32")
