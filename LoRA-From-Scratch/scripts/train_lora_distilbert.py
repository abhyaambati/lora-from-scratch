# scripts/train_lora_distilbert.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from lora.injector import inject_lora_into_model
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def main():
    # Load dataset
    dataset = load_dataset("glue", "sst2")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

    encoded = dataset.map(tokenize, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Reduce dataset size for testing
    small_train = encoded["train"].select(range(1000))
    small_val = encoded["validation"].select(range(200))
    train_loader = DataLoader(small_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(small_val, batch_size=32)

    # Load model and inject LoRA
    base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = inject_lora_into_model(base_model, r=4)

    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, labels = [], []
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        preds += torch.argmax(logits, axis=-1).cpu().tolist()
        labels += batch["labels"].cpu().tolist()

    acc = accuracy_score(labels, preds)
    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
