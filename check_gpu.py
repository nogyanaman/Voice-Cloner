import torch
import torchaudio

print("====== GPU STATUS ======")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

print("\n====== TORCHAUDIO ======")
print("Torchaudio version:", torchaudio.__version__)

# Small GPU test
if torch.cuda.is_available():
    x = torch.randn(2000, 2000, device="cuda")
    y = torch.matmul(x, x)
    print("\nGPU test OK, result:", y[0][0].item())
else:
    print("GPU not available.")
