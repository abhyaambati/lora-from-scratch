from .lora_layer import LoRALinear
import torch.nn as nn

def inject_lora_into_model(model, r=8, target_module_name="attention"):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target_module_name in name:
            in_f, out_f = module.in_features, module.out_features
            lora = LoRALinear(in_f, out_f, r=r)
            lora.weight.data = module.weight.data.clone()
            parent = _get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], lora)
    return model

def _get_parent_module(model, module_name):
    parts = module_name.split('.')
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj
