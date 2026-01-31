import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import nibabel as nib
import torchvision.transforms as transforms

class CTMultiLabelDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def preprocess(self, image):
        img = np.array(image)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])
        return Image.fromarray(img)

    def load_middle_slice(self, filepath):
        volume = nib.load(filepath).get_fdata()
        mid_slice = volume.shape[2] // 2
        slice_img = volume[:, :, mid_slice]
        slice_img = np.clip(slice_img, np.percentile(slice_img, 1), np.percentile(slice_img, 99))  # Normalize contrast
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255
        slice_img = slice_img.astype(np.uint8)
        return Image.fromarray(slice_img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']

        try:
            image = self.load_middle_slice(filepath).convert("L")  # Grayscale
        except Exception as e:
            print(f"Error loading volume {filepath}: {e}")
            image = Image.new("L", (224, 224))  # Dummy image

        image = self.preprocess(image)
        image = image.convert("RGB")  # Convert to 3-channel if needed by transforms

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row[label_columns].values.astype(np.float32))
        return image, label

    def __len__(self):
        return len(self.df)


# Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Class weights and sample weights
class_counts = df_train[label_columns].sum().to_dict()
total = sum(class_counts.values())
class_weights = {cls: total / count for cls, count in class_counts.items()}

def get_sample_weights(df, class_weights, class_names):
    weights = []
    for _, row in df.iterrows():
        labels = row[class_names].values
        sample_weight = sum(class_weights[class_names[i]] for i in range(len(class_names)) if labels[i] == 1)
        weights.append(sample_weight)
    return weights

sample_weights = get_sample_weights(df_train, class_weights, label_columns)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Datasets and Dataloaders
train_dataset = CTMultiLabelDataset(df_train, transform=train_transform)
val_dataset = CTMultiLabelDataset(df_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


def build_model(num_classes):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

model = build_model(len(label_columns))


# compute positive weights for each class
pos_counts = df_train[label_columns].sum()
neg_counts = len(df_train) - pos_counts
pos_weight = (neg_counts / (pos_counts + 1e-5)).values
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)

class PrecisionRecallLoss(torch.nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha  # how much weight to give to Dice

    def forward(self, logits, targets):
        # BCE loss
        bce_loss = self.bce(logits, targets)

        # Sigmoid for probabilities
        probs = torch.sigmoid(logits)
        smooth = 1e-6

        # Dice Loss = 1 - F1 score
        intersection = (probs * targets).sum(dim=0)
        dice_score = (2. * intersection + smooth) / (
            probs.sum(dim=0) + targets.sum(dim=0) + smooth)
        dice_loss = 1 - dice_score.mean()

        # Combined loss
        return 3*(1 - self.alpha) * bce_loss + 3*self.alpha * dice_loss

criterion = PrecisionRecallLoss(alpha=0.6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    loop = tqdm(loader, desc="Training", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        print(f"Min output: {torch.min(outputs).item():.4f}, Max output: {torch.max(outputs).item():.4f}") # Inspect outputs

        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

        pred_bin = (preds > 0.5).astype(int)
        precision = precision_score(all_labels[-1], pred_bin, average='macro', zero_division=0)
        recall = recall_score(all_labels[-1], pred_bin, average='macro', zero_division=0)
        print(all_labels[-1])
        try:
            roc = roc_auc_score(all_labels[-1], preds, average='macro')
        except:
            roc = 0
        acc = (pred_bin == all_labels[-1]).mean()

        loop.set_postfix(loss=loss.item(), acc=f"{acc:.3f}", precision=f"{precision:.3f}", recall=f"{recall:.3f}", roc=f"{roc:.3f}")

    return total_loss / len(loader)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds_all, targets_all = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds_all.append(torch.sigmoid(outputs).cpu().numpy())
            targets_all.append(labels.cpu().numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)
    pred_bin = (preds_all > 0.5).astype(int)
    acc = (pred_bin == targets_all).mean()
    precision = precision_score(targets_all, pred_bin, average='macro', zero_division=0)
    recall = recall_score(targets_all, pred_bin, average='macro', zero_division=0)

    roc_scores = []
    for i in range(targets_all.shape[1]):
        try:
            score = roc_auc_score(targets_all[:, i], preds_all[:, i])
        except ValueError:
            score = 0.0
        roc_scores.append(score)

    return total_loss / len(loader), acc, np.mean(roc_scores), precision, recall

def plot_conf_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
  
#TRAINING
EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, acc, roc, prec, rec = validate(model, val_loader, criterion)
    scheduler.step(val_loss)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Accuracy: {acc:.4f} | ROC AUC: {roc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

model.eval()
all_outputs, all_targets = [], []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Collecting predictions for confusion matrix", leave=False):
        images = images.to(device)
        outputs = model(images)
        all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
        all_targets.append(labels.cpu().numpy())


#output processing
all_outputs = np.vstack(all_outputs)
all_targets = np.vstack(all_targets)

true_labels = np.argmax(all_targets, axis=1)
pred_labels = np.argmax(all_outputs, axis=1)

true_labels = np.argmax(all_targets, axis=1)
pred_labels = np.argmax(all_outputs, axis=1)

unique_labels = np.unique(np.concatenate([true_labels, pred_labels]))

display_labels = [label_columns[i] for i in unique_labels]

# plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels) # Use filtered labels here
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Argmax of Predictions)")
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Confusion Matrices for each class
fig_cm, axes_cm = plt.subplots(4, 5, figsize=(14, 8))
axes_cm = axes_cm.flatten()

for i, cls in enumerate(label_columns):
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

# ROC curves
plt.figure(figsize=(10, 6))

for i, cls in enumerate(label_columns):
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

for i, cls in enumerate(label_columns):
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

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# plot confusion matrix for each class
for i, cls in enumerate(label_columns):
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

# Plot ROC curves for each class
for i, cls in enumerate(label_columns):
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

# Plot Precision-Recall curves for each class
for i, cls in enumerate(label_columns):
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

    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticks(angles[:-1])  # Remove angular ticks
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, size=10, color='black')

mlb = MultiLabelBinarizer()
mlb.fit([label_columns]) # Fit the binarizer with the label columns

class_names = mlb.classes_  # Get the class names from MultiLabelBinarizer

metrics_df = pd.DataFrame({
    'Accuracy': np.random.rand(len(class_names)),
    'Sensitivity (Recall)': np.random.rand(len(class_names)),
    'Specificity': np.random.rand(len(class_names)),
    'Precision': np.random.rand(len(class_names)),
    'F1 Score': np.random.rand(len(class_names)),
    'AUC': np.random.rand(len(class_names))
})

fig, axes = plt.subplots(4, 5, figsize=(18, 12), subplot_kw=dict(polar=True))
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





