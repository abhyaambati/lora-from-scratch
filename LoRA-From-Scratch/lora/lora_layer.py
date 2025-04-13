import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = alpha / r

        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        result = nn.functional.linear(x, self.weight)
        lora_adjustment = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_adjustment
