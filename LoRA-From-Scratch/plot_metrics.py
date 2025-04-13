import matplotlib.pyplot as plt

epochs = [1, 2, 3]

lora_loss = [0.5252, 0.2249, 0.0530]
lora_acc = [0.61, 0.72, 0.805]

baseline_loss = [0.7049, 0.6420, 0.4900]
baseline_acc = [0.58, 0.65, 0.715]

# Loss Curve
plt.figure()
plt.plot(epochs, lora_loss, label='LoRA Loss', marker='o')
plt.plot(epochs, baseline_loss, label='Baseline Loss', marker='x')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plots/loss_comparison.png")
plt.close()

# Accuracy Curve
plt.figure()
plt.plot(epochs, lora_acc, label='LoRA Accuracy', marker='o')
plt.plot(epochs, baseline_acc, label='Baseline Accuracy', marker='x')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("plots/accuracy_comparison.png")
plt.close()

print(" Plots saved to /plots/")

