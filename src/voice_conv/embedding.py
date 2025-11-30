from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

# --- SHIM 1: torchaudio.list_audio_backends for new torchaudio versions ---
if not hasattr(torchaudio, "list_audio_backends"):
    # Newer torchaudio removed this API, but SpeechBrain still calls it.
    # We define a minimal fake implementation so SpeechBrain's check passes.
    def list_audio_backends():
        # Pretend "soundfile" backend is available.
        return ["soundfile"]

    torchaudio.list_audio_backends = list_audio_backends
# --- END SHIM 1 ---

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy  # <-- important

from .config import VCConfig


class SpeakerEncoder:
    """SpeechBrain ECAPA-based speaker embedding extractor."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        cfg: Optional[VCConfig] = None,
    ) -> None:
        self.cfg = cfg or VCConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_id = model_id or self.cfg.speaker_model_id

        cache_dir = Path.home() / ".cache" / "speechbrain" / "spkrec"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # KEY CHANGE: local_strategy=LocalStrategy.COPY to avoid symlinks
        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_id,
            run_opts={"device": str(self.device)},
            savedir=str(cache_dir),
            local_strategy=LocalStrategy.COPY,
        )

    def extract_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Given mono audio at sr=16kHz, return a speaker embedding vector.
        """
        # SpeechBrain ECAPA expects [batch, time]
        signal = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.classifier.encode_batch(signal)  # [batch, emb_dim]
        emb = emb.squeeze(0).cpu().numpy().astype("float32")

        # L2-normalize for safe downstream use
        norm = float(np.linalg.norm(emb) + 1e-9)
        emb /= norm
        return emb
