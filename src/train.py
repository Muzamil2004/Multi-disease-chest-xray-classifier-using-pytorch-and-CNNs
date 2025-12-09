import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    save_path="best_model.pth"
):
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_targets, all_probs = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(all_targets, all_probs, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch+1}: "
            f"TrainLoss={train_loss:.4f}, "
            f"ValLoss={val_loss:.4f}, "
            f"ValAUC={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved")

    return history, best_val_auc
