import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
transform = transforms.Compose(
    [
        transforms.Resize((1300, 1300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4629, 0.4317, 0.4781], std=[0.4031, 0.3864, 0.4044]
        ),
    ]
)
train_dataset = ImageFolderWithPaths(
    root="/home/chris/Documents/chagas/dataset_split/train", transform=transform
)
val_dataset = ImageFolderWithPaths(
    root="/home/chris/Documents/chagas/dataset_split/val", transform=transform
)
test_dataset = ImageFolderWithPaths(
    root="/home/chris/Documents/chagas/dataset_split/test", transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
)
print(
    f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
)
print(f"Classes: {train_dataset.classes}")


class NeuralNetwork(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.conv1_3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(self.dropout_rate)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(self.dropout_rate)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(256)
        self.dropout5 = nn.Dropout(self.dropout_rate)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6_2 = nn.BatchNorm2d(512)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6_3 = nn.BatchNorm2d(512)
        self.dropout6 = nn.Dropout(self.dropout_rate)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.conv7_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn7_1 = nn.BatchNorm2d(1024)
        self.conv7_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn7_2 = nn.BatchNorm2d(1024)
        self.conv7_3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.bn7_3 = nn.BatchNorm2d(1024)
        self.dropout7 = nn.Dropout(self.dropout_rate)
        self.pool7 = nn.MaxPool2d(2, 2)
        self.lat7 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lat6 = nn.Conv2d(512, 256, kernel_size=1)
        self.lat5 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat4 = nn.Conv2d(128, 256, kernel_size=1)
        self.lat3 = nn.Conv2d(64, 256, kernel_size=1)
        self.lat2 = nn.Conv2d(32, 256, kernel_size=1)
        self.lat1 = nn.Conv2d(16, 256, kernel_size=1)
        self.prediction = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        outputs = []
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        outputs.append(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        outputs.append(x)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        outputs.append(x)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        outputs.append(x)
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool5(x)
        x = self.dropout5(x)
        outputs.append(x)
        x = F.relu(self.bn6_1(self.conv6_1(x)))
        x = F.relu(self.bn6_2(self.conv6_2(x)))
        x = F.relu(self.bn6_3(self.conv6_3(x)))
        x = self.pool6(x)
        x = self.dropout6(x)
        outputs.append(x)
        x = F.relu(self.bn7_1(self.conv7_1(x)))
        x = F.relu(self.bn7_2(self.conv7_2(x)))
        x = F.relu(self.bn7_3(self.conv7_3(x)))
        x = self.pool7(x)
        x = self.dropout7(x)
        outputs.append(x)
        p7 = self.lat7(outputs[6])
        p6 = F.interpolate(p7, size=outputs[5].shape[2:], mode="nearest") + self.lat6(
            outputs[5]
        )
        p5 = F.interpolate(p6, size=outputs[4].shape[2:], mode="nearest") + self.lat5(
            outputs[4]
        )
        p4 = F.interpolate(p5, size=outputs[3].shape[2:], mode="nearest") + self.lat4(
            outputs[3]
        )
        p3 = F.interpolate(p4, size=outputs[2].shape[2:], mode="nearest") + self.lat3(
            outputs[2]
        )
        p2 = F.interpolate(p3, size=outputs[1].shape[2:], mode="nearest") + self.lat2(
            outputs[1]
        )
        p1 = F.interpolate(p2, size=outputs[0].shape[2:], mode="nearest") + self.lat1(
            outputs[0]
        )
        out = self.prediction(p5)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = torch.sigmoid(self.fc(out))
        return out


model = NeuralNetwork().to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
params = sum((p.numel() for p in model.parameters()))
print(f"Model parameters: {params:,}")
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=7, threshold=0.001, threshold_mode="rel"
)
num_epochs = 300
checkpoint_dir = "/home/chris/Documents/chagas/checkpoints/presplit_fpn_v2"
os.makedirs(checkpoint_dir, exist_ok=True)
log_file_path = os.path.join(checkpoint_dir, "training_log.txt")
train_losses = []
val_losses = []
best_val_loss = float("inf")
print(f"\nStarting training for {num_epochs} epochs...")
total_start = time.time()
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    total_train_loss = 0.0
    for images, labels, paths in train_loader:
        images, labels = (images.to(device), labels.to(device))
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).type_as(outputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images, labels = (images.to(device), labels.to(device))
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).type_as(outputs))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    epoch_min = (time.time() - start_time) / 60
    marker = ""
    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint["best_val_loss"] = best_val_loss
        torch.save(
            checkpoint, os.path.join(checkpoint_dir, "best_model_checkpoint.pth")
        )
        marker = " *BEST*"
    torch.save(checkpoint, os.path.join(checkpoint_dir, "recent_model_checkpoint.pth"))
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] {epoch_min:.1f}min | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}{marker}"
    )
    with open(log_file_path, "a") as f:
        f.write(
            f"Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_min:.2f}min, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}{marker}\n"
        )
    scheduler.step(avg_val_loss)
total_min = (time.time() - total_start) / 60
print(
    f"\nTraining complete in {total_min:.1f} minutes. Best val loss: {best_val_loss:.4f}"
)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(train_losses, label="Train Loss", linewidth=1.5)
axes[0].plot(val_losses, label="Val Loss", linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
skip = min(10, len(train_losses) // 5)
axes[1].plot(
    range(skip, len(train_losses)),
    train_losses[skip:],
    label="Train Loss",
    linewidth=1.5,
)
axes[1].plot(
    range(skip, len(val_losses)), val_losses[skip:], label="Val Loss", linewidth=1.5
)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title(f"Loss Curves (epoch {skip}+, zoomed)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "loss_curves.png"), dpi=150)
print("Saved loss_curves.png")
best_ckpt = torch.load(os.path.join(checkpoint_dir, "best_model_checkpoint.pth"))
model.load_state_dict(best_ckpt["state_dict"])
print(
    f"\nLoaded best model from epoch {best_ckpt['epoch']} (val loss: {best_ckpt['best_val_loss']:.4f})"
)
model.eval()
true_labels = []
predictions = []
prob_scores = []
with torch.no_grad():
    for images, labels, paths in test_loader:
        images, labels = (images.to(device), labels.to(device))
        outputs = model(images)
        probs = outputs.cpu().numpy().squeeze()
        preds = outputs.round().cpu().numpy()
        true_labels.extend(labels.cpu().numpy())
        predictions.extend(preds)
        prob_scores.extend(probs if probs.ndim > 0 else [probs.item()])
true_labels = np.array(true_labels)
predictions = np.array(predictions).flatten()
cm = confusion_matrix(true_labels, predictions)
TN, FP, FN, TP = cm.ravel()
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
specificity = TN / (TN + FP)
roc_auc = roc_auc_score(true_labels, prob_scores)
print(f"\n{'=' * 40}")
print(f"TEST SET RESULTS")
print(f"{'=' * 40}")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC:         {roc_auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    square=True,
    ax=axes[0],
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].set_title("Confusion Matrix")
fpr, tpr, _ = roc_curve(true_labels, prob_scores)
axes[1].plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
)
axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "eval_plots.png"), dpi=150)
print("Saved eval_plots.png")
print("\nDone!")
