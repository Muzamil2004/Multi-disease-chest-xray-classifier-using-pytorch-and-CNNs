import torch
from sklearn.metrics import roc_auc_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_targets, all_probs, average="macro")
    return auc
