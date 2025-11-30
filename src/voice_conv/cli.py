import argparse
from pathlib import Path

from .config import VCConfig, load_yaml_config
from .pipeline import run_conversion
from .utils.logging_utils import setup_logging, get_logger



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GPU voice conversion:\n"
            "Audio A (target voice) + Audio B (content) -> Audio C (B in A's voice)"
        )
    )
    parser.add_argument("--ref", required=True, help="Path to reference audio A (target voice)")
    parser.add_argument("--src", required=True, help="Path to source audio B (content)")
    parser.add_argument("--out", required=True, help="Path to output audio C")

    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Feature sample rate for encoders (SpeechBrain/Wav2Vec2, default: 16000)",
    )
    parser.add_argument(
        "--vc-sr",
        type=int,
        default=48000,
        help="Sample rate expected by the VC (RVC) model, e.g. 32000 or 48000",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu' (auto-falls back if CUDA not available)",
    )

    args = parser.parse_args()

    # --- logging + config.yaml ---
    # Load config.yaml if present
    yaml_cfg = load_yaml_config()

    log_level = yaml_cfg.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)
    log = get_logger("voice_conv.cli")

    log.info("Starting VoiceConv")
    log.debug(f"Args: {args}")
    log.debug(f"YAML config: {yaml_cfg}")
    
    cfg = VCConfig(
        feature_sample_rate=args.sr,
        vc_sample_rate=args.vc_sr,
        device=args.device,
    )
    run_conversion(Path(args.ref), Path(args.src), Path(args.out), config=cfg)


if __name__ == "__main__":
    main()
