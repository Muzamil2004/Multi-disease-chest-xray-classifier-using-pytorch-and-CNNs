import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["val_auc"], label="Val AUC")
    plt.legend()
    plt.title("Training History")
    plt.show()
