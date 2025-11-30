Write-Host "Creating venv..."
python -m venv .venv

Write-Host "Activating..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing PyTorch CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing project requirements..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

Write-Host "Environment ready."

# Run CLI
python -m voice_conv.cli --ref data/ref/A.wav --src data/src/B.wav --out data/out/C.wav --vc-sr 48000
