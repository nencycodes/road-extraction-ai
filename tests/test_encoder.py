import torch
from src.models.encoder import EncoderBlock
encoder = EncoderBlock(3, 64)
x = torch.randn(1, 3, 256, 256)
features, pooled = encoder(x)
print("=" * 50)
print("Input Shape    :", x.shape)
print("Features Shape :", features.shape)
print("Pooled Shape   :", pooled.shape)
print("=" * 50)