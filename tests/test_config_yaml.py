from pathlib import Path
import textwrap

from voice_conv.config import load_yaml_config, VCConfig


def test_load_yaml(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            feature_sample_rate: 22050
            vc_sample_rate: 48000
            device: cpu
            logging:
              level: DEBUG
            """
        ),
        encoding="utf-8",
    )

    from voice_conv import config as cfg_module

    # Monkeypatch BASE_DIR to our tmp dir
    monkeypatch.setattr(cfg_module, "BASE_DIR", tmp_path)

    data = cfg_module.load_yaml_config()
    assert data["feature_sample_rate"] == 22050

    vc_cfg = cfg_module.VCConfig.from_yaml()
    assert vc_cfg.feature_sample_rate == 22050
    assert vc_cfg.vc_sample_rate == 48000
    assert vc_cfg.device == "cpu"
