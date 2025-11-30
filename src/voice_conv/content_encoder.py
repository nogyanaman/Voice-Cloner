from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from .config import VCConfig


class ContentEncoder:
    """
    Content encoder based on Wav2Vec2 (Hugging Face).

    It extracts high-level features (roughly 'what is being said')
    from the source audio B.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        cfg: Optional[VCConfig] = None,
    ) -> None:
        self.cfg = cfg or VCConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_id = model_id or self.cfg.wav2vec2_model_id

        # Processor handles resampling / feature extraction,
        # model outputs hidden states.
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def extract_content(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Return content features as a [time, feature_dim] numpy array.
        """
        # Wav2Vec2 expects 16kHz; our VCConfig sets sr=16000.
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(
                input_values=inputs["input_values"].to(self.device),
                attention_mask=inputs.get("attention_mask", None),
            )

        # last_hidden_state: [batch, time, dim]
        hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy().astype("float32")
        return hidden  # [time, dim]

    def extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Placeholder F0 extractor. For now, returns a flat contour.

        You can later replace this with a real F0 method (e.g. via torchaudio or a dedicated F0 model).
        """
        # Simple hack: one F0 per 20 ms frame
        frame_len = int(0.02 * sr)
        n_frames = max(1, len(audio) // frame_len)
        f0 = np.ones(n_frames, dtype="float32") * 120.0  # flat 120 Hz
        return f0
