import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset
class VinBigDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def preprocess(self, image):
        # Convert PIL Image to NumPy array
        img = np.array(image)

        # Apply OpenCV preprocessing steps
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply histogram equalization to each channel independently
        channels = cv2.split(img)
        equalized_channels = []
        for channel in channels:
            equalized_channels.append(cv2.equalizeHist(channel))
        img = cv2.merge(equalized_channels)

        # Stack the grayscale image to have 3 channels if it's not already
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 1:
            img = np.concatenate([img]*3, axis=-1)

        # Convert back to PIL Image
        img = Image.fromarray(img)
        return img

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     path = os.path.join(self.img_dir, row['image_id'] + ".dicom")
    #     image = self.preprocess(path)
    #     if self.transform:
    #         image = self.transform(image)
    #     label = torch.tensor(row[class_names].values.astype(np.float32))
    #     return image, label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        # Add .png only if it's not already there
        if not image_id.endswith(".png"):
            image_id += ".png"

        image_path = os.path.join(self.img_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[class_names].values.astype(np.float32))
        return image, label

def get_sampler(df):
    targets = df[class_names].values
    class_sample_counts = targets.sum(axis=0)
    class_weights = 1. / (class_sample_counts + 1e-6)
    weights = np.dot(targets, class_weights)
    return WeightedRandomSampler(weights, len(weights))

train_set = VinBigDataset(df_train, train_dir, train_transform)
val_set = VinBigDataset(df_val, train_dir, val_transform)

train_loader = DataLoader(train_set, batch_size=32, sampler=get_sampler(df_train))
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for param in self.base.parameters():
            param.requires_grad = True  # Unfreeze all
        self.base.heads = nn.Sequential(
            nn.LayerNorm(self.base.heads.head.in_features),
            nn.Linear(self.base.heads.head.in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)

model = ViTClassifier(num_classes).to(device)

class PrecisionRecallLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha  # how much weight to give to Dice

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6

        # Dice Loss = 1 - F1 score
        intersection = (probs * targets).sum(dim=0)
        dice_score = (2. * intersection + smooth) / (
            probs.sum(dim=0) + targets.sum(dim=0) + smooth)
        dice_loss = 1 - dice_score.mean()

        return 3*(1 - self.alpha) * bce_loss + 3*self.alpha * dice_loss

criterion = PrecisionRecallLoss().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5) #slow down learning rate?
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

from tqdm import tqdm, trange
from sklearn.metrics import classification_report, average_precision_score, precision_score, recall_score, precision_recall_curve
import numpy as np

def get_optimal_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_thresh = thresh[np.argmax(f1)] if len(thresh) > 0 else 0.5
        thresholds.append(best_thresh)
    return np.array(thresholds)

EPOCHS = 5
for epoch in trange(EPOCHS, desc="Epochs"):
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss, all_outputs, all_targets = 0, [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    optimal_thresholds = get_optimal_thresholds(all_targets, all_outputs)
    preds = (all_outputs > optimal_thresholds).astype(int)

    print(f"\nEpoch {epoch+1} Classification Report:")
    print(classification_report(all_targets, preds, target_names=class_names, zero_division=0))

    aupr = average_precision_score(all_targets, all_outputs, average='macro')
    prec = precision_score(all_targets, preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, preds, average='macro', zero_division=0)

    print(f"Train Loss: {train_loss / len(train_loader):.4f}")
    print(f"Val Loss: {val_loss / len(val_loader):.4f} | AUC-PR: {aupr:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f}")

    scheduler.step(val_loss / len(val_loader))


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_outputs, all_targets = [], []

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    # Calculate ROC AUC, Precision, and Recall (example - adapt as needed)
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    roc = roc_auc_score(all_targets, all_outputs, average='macro')  # Adjust 'average' if needed
    prec = precision_score(all_targets, (all_outputs > 0.5).astype(int), average='macro', zero_division=0)
    recall = recall_score(all_targets, (all_outputs > 0.5).astype(int), average='macro', zero_division=0)

    return total_loss / len(loader), roc, prec, recall


EPOCHS = 5
for epoch in range(EPOCHS):
    tqdm.write(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_roc, val_prec, val_recall = validate(model, val_loader, criterion)
    scheduler.step(val_loss)
    tqdm.write(f"Train Loss: {train_loss:.4f}")
    tqdm.write(f"Val Loss: {val_loss:.4f} | ROC AUC: {val_roc:.4f} | Precision: {val_prec:.4f} | Recall: {val_recall:.4f}")


model.eval()
all_outputs, all_targets = [], []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Collecting predictions for confusion matrix", leave=False):
        images = images.to(device)
        outputs = model(images)
        all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

all_outputs = np.vstack(all_outputs)
all_targets = np.vstack(all_targets)

true_labels = np.argmax(all_targets, axis=1)
pred_labels = np.argmax(all_outputs, axis=1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("15x15 Confusion Matrix (Argmax of Predictions)")
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# confusion matrices per class
fig_cm, axes_cm = plt.subplots(3, 5, figsize=(18, 10))
axes_cm = axes_cm.flatten()

for i, cls in enumerate(class_names):
    preds_bin = (all_outputs[:, i] > 0.5).astype(int)
    targets_bin = all_targets[:, i].astype(int)

    cm = confusion_matrix(targets_bin, preds_bin)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
    disp.plot(ax=axes_cm[i], colorbar=False)
    axes_cm[i].set_title(cls)
    axes_cm[i].set_xlabel('')
    axes_cm[i].set_ylabel('')

fig_cm.suptitle("Per-Class Binary Confusion Matrices", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# roc curves
plt.figure(figsize=(12, 8))

for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(all_targets[:, i], all_outputs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Classes')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import f1_score

metrics_dict = {
    'Class': [],
    'Accuracy': [],
    'Sensitivity (Recall)': [],
    'Specificity': [],
    'Precision': [],
    'F1 Score': [],
    'AUC': []
}

for i, cls in enumerate(class_names):
    preds = (all_outputs[:, i] > 0.5).astype(int)
    targets = all_targets[:, i].astype(int)

    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0,1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    f1 = f1_score(targets, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(targets, all_outputs[:, i])
    auc_score = auc(fpr, tpr)

    metrics_dict['Class'].append(cls)
    metrics_dict['Accuracy'].append(accuracy)
    metrics_dict['Sensitivity (Recall)'].append(sensitivity)
    metrics_dict['Specificity'].append(specificity)
    metrics_dict['Precision'].append(precision)
    metrics_dict['F1 Score'].append(f1)
    metrics_dict['AUC'].append(auc_score)

metrics_df = pd.DataFrame(metrics_dict)

metrics_df

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Plot confusion matrix for each class
for i, cls in enumerate(class_names):
    preds = (all_outputs[:, i] > 0.5).astype(int)
    targets = all_targets[:, i].astype(int)

    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0,1]).ravel()

    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'], cbar=False, annot_kws={'size': 16})
    plt.title(f'Confusion Matrix for {cls}', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.show()

from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

# plot ROC curves for each class
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(all_targets[:, i], all_outputs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')

# Plot the diagonal line (no discrimination)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.title('ROC Curves for All Classes', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate (Recall)', fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_recall_curve

plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(all_targets[:, i], all_outputs[:, i])
    plt.plot(recall, precision, label=f'{cls}')

plt.title('Precision-Recall Curves for All Classes', fontsize=16)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_radar(ax, data, labels, title):
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data += data[:1]
    angles += angles[:1]
    ax.fill(angles, data, color='blue', alpha=0.25)
    ax.plot(angles, data, color='blue', linewidth=2)

    ax.set_yticklabels([])  
    ax.set_xticks(angles[:-1])  
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, size=10, color='black')

class_names = mlb.classes_  # Get the class names from MultiLabelBinarizer

metrics_df = pd.DataFrame({
    'Accuracy': np.random.rand(len(class_names)),
    'Sensitivity (Recall)': np.random.rand(len(class_names)),
    'Specificity': np.random.rand(len(class_names)),
    'Precision': np.random.rand(len(class_names)),
    'F1 Score': np.random.rand(len(class_names)),
    'AUC': np.random.rand(len(class_names))
})

fig, axes = plt.subplots(3, 5, figsize=(18, 12), subplot_kw=dict(polar=True))
axes = axes.flatten()

for i, cls in enumerate(class_names):
    data = [
        metrics_df.loc[i, 'Accuracy'],
        metrics_df.loc[i, 'Sensitivity (Recall)'],
        metrics_df.loc[i, 'Specificity'],
        metrics_df.loc[i, 'Precision'],
        metrics_df.loc[i, 'F1 Score'],
        metrics_df.loc[i, 'AUC']
    ]
    plot_radar(axes[i], data, ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'AUC'], cls)

plt.tight_layout()
plt.show()


