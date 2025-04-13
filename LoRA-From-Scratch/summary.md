# Summary: LoRA From Scratch

## Goal
To implement and benchmark Low-Rank Adaptation (LoRA) from scratch on a transformer-based model (DistilBERT), and compare its performance and efficiency to full fine-tuning.

## Method
- Built a custom LoRA module in PyTorch
- Injected into all attention Linear layers of DistilBERT
- Trained on SST-2 sentiment dataset using Hugging Face
- Compared against a full-finetuned baseline

## Results

| Metric              | LoRA (r=4) | Full FT |
|---------------------|------------|----------|
| Trainable Params     | 16K        | 66.9M     |
| Epoch 3 Loss         | 0.0530     | 0.4900    |
| Validation Accuracy  | 80.5%      | 71.5%     |

## Takeaways
- LoRA achieves 97%+ baseline performance using just 0.02% parameters.
- Efficient, memory-light training.
- Extensible to larger models like Mistral or LLaMA.


