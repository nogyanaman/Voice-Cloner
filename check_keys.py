
import torch
checkpoint = torch.load(r".pth file path here", map_location="cpu")
print(checkpoint.keys())
