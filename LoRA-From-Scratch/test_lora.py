from lora.lora_layer import LoRALinear
import torch

model = LoRALinear(in_features=512, out_features=512, r=4)
x = torch.randn(2, 512)
out = model(x)

print("Output shape:", out.shape)
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
