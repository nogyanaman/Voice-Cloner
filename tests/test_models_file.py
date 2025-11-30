import os

def test_model_file_exists():
    model_path = "models/vc/model.pth"
    assert os.path.exists(model_path), f"Missing model file: {model_path}"
