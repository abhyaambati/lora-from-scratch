# 🔬 LoRA From Scratch: A Minimal and Modular Reimplementation in PyTorch

This repository contains a **from-scratch PyTorch reimplementation of LoRA (Low-Rank Adaptation)** and its application to transformer fine-tuning. Unlike plug-and-play libraries like PEFT, this project manually injects low-rank matrices into attention layers of a transformer model — offering transparency, control, and extensibility.

📝 **Paper submitted to arXiv — awaiting publication.**  
---

## 🚀 Highlights

- ✅ Rebuilt LoRA from scratch — no PEFT dependencies
- ✅ Injected LoRA modules directly into DistilBERT attention layers
- ✅ Trained on SST-2 with Hugging Face Transformers
- ✅ Compared against full fine-tuning
- ✅ Evaluation metrics: Accuracy, BLEU, ROUGE, Exact Match
- ✅ Plot generation included (loss, accuracy)
- ✅ Extensible to Mistral, LLaMA, QLoRA (next steps)

---

## 📊 Results

| Metric              | LoRA (r=4) | Full Fine-Tune |
|---------------------|------------|----------------|
| Trainable Params     | 16K        | 66.9M          |
| Validation Accuracy  | 80.5%      | 71.5%          |
| Epoch 3 Loss         | 0.053      | 0.490          |

---

## 📁 Project Structure

```
lora-from-scratch/
├── lora/                       # Custom LoRA layer + injector
│   ├── lora_layer.py
│   └── injector.py
├── scripts/                    # Training scripts
│   ├── train_lora_distilbert.py
│   └── train_baseline_distilbert.py
├── eval_metrics.py            # BLEU, ROUGE, EM evaluation
├── plot_metrics.py            # Accuracy/loss plotting
├── plots/                     # Output figures
└── README.md
```

---


## 🛠 Usage

### ✅ Install dependencies
```bash
pip install -r requirements.txt
```

### ✅ Train with LoRA
```bash
python scripts/train_lora_distilbert.py --epochs 3 --r 4 --batch_size 16
```

### ✅ Train baseline
```bash
python scripts/train_baseline_distilbert.py
```

### ✅ Evaluate predictions
```bash
python eval_metrics.py
```

---

## 🧪 Further goals

- Apply LoRA to Mistral-7B / LLaMA-2
- Add QLoRA and BitFit comparisons
- Extend to multi-task instruction tuning

---

## 🤝 Contributors

- **Abhya Ambati** — Emory University  
- **Rajiv Kalyan Chilla** — Independent Contributor


---

## 🙏 Acknowledgments

This project was implemented independently using Hugging Face Transformers, Datasets, and Evaluate libraries.  
Some formatting and organization were assisted by AI tools, with all results and content verified by the listed contributors.

---

## ⭐ Citation (coming soon)

This work has been submitted to arXiv. A citation BibTeX entry will be added here after approval.
