def test_imports():
    import voice_conv  # noqa: F401
    from voice_conv import cli, pipeline, config, audio_io, preprocess, embedding, content_encoder, vc_model  # noqa: F401

    assert cli is not None
    assert pipeline is not None
    assert config is not None
    assert audio_io is not None
    assert preprocess is not None
    assert embedding is not None
    assert content_encoder is not None
    assert vc_model is not None
