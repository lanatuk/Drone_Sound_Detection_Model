from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchaudio
import matplotlib.pyplot as plt
from audiodataset import AudioDataset
from model import AudioEfficientNet
import torch.nn.functional as F

# Load dataset
dataset = load_dataset("geronimobasso/drone-audio-detection-samples")
train_data = dataset["train"]

# Create DataFrame
df = pd.DataFrame({
    'label': train_data['label'],
    'audio_index': range(len(train_data))
})

# Split dataset
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

# DataLoaders
train_dataset = AudioDataset(train_df, train_data, target_seconds=1.0)
val_dataset = AudioDataset(val_df, train_data, target_seconds=1.0)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)


device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

model = AudioEfficientNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training progress bar
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                      leave=False, ncols=100)
    
    for batch_idx, (images, labels) in enumerate(train_pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Batch': f'{batch_idx+1}/{len(train_loader)}'
        })

    # Validation progress bar
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                    leave=False, ncols=100)
    
    with torch.no_grad():
        for images, labels in val_pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            probs = F.softmax(outputs, dim=1)[:,1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            acc = 100.0 * correct / max(1, total)
            val_pbar.set_postfix({
                'Acc': f'{acc:.2f}%',
                'Batch': f'{len(val_pbar)}/{len(val_loader)}'
            })
    
    final_acc = 100.0 * correct / max(1, total)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Acc: {final_acc:.2f}%")

    # --------- ROC & PR Curves ---------
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels_np, all_probs_np)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Validation ROC Curve - Epoch {epoch+1}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_epoch{epoch+1}.png')
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_labels_np, all_probs_np)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Validation Precision-Recall Curve - Epoch {epoch+1}')
    plt.legend(loc='lower left')
    plt.savefig(f'pr_epoch{epoch+1}.png')
    plt.close()

    print(f"Saved ROC and PR curves for epoch {epoch+1}")

    # After the last epoch, save misclassified validation samples once
    if epoch == num_epochs - 1:
        val_mis_dir = os.path.join("misclassified", "val_final")
        os.makedirs(val_mis_dir, exist_ok=True)
        val_mis_rows = []
        val_global_idx = 0

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Saving val misclassified", ncols=100, leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                batch_size_now = labels_np.shape[0]
                for bi in range(batch_size_now):
                    if preds_np[bi] != labels_np[bi]:
                        row = val_df.iloc[val_global_idx + bi]
                        audio_index = int(row['audio_index'])
                        true_label = int(row['label'])
                        # Fetch original audio from HF dataset
                        audio_data = train_data[audio_index]['audio']
                        wav = torch.tensor(audio_data['array'], dtype=torch.float32).unsqueeze(0)
                        sr = int(audio_data['sampling_rate'])
                        # Filename
                        fname = f"val_idx{val_global_idx+bi}_audio{audio_index}_true{true_label}_pred{int(preds_np[bi])}.wav"
                        fpath = os.path.join(val_mis_dir, fname)
                        # Save wav
                        try:
                            torchaudio.save(fpath, wav, sr)
                        except Exception:
                            if wav.abs().max() > 1.0:
                                wav = wav / wav.abs().max()
                            torchaudio.save(fpath, wav, sr)
                        # Collect row
                        val_mis_rows.append({
                            'epoch': int(epoch+1),
                            'val_global_idx': int(val_global_idx+bi),
                            'audio_index': audio_index,
                            'true_label': true_label,
                            'pred_label': int(preds_np[bi]),
                            'filepath': fpath,
                            'sampling_rate': sr
                        })
                val_global_idx += batch_size_now

        if len(val_mis_rows) > 0:
            val_csv = os.path.join(val_mis_dir, "misclassified_val_summary.csv")
            pd.DataFrame(val_mis_rows).to_csv(val_csv, index=False)
            print(f"Saved {len(val_mis_rows)} validation misclassified samples to {val_mis_dir}")
        else:
            print("No misclassified samples in validation on final epoch.")

torch.save(model.state_dict(), "trained_model_weights.pth")
print("Saved final model weights: trained_model_weights.pth")

# Test evaluation after final epoch
print("\n" + "="*50)
print("FINAL TEST EVALUATION")
print("="*50)

# Create test dataset and loader
test_dataset = AudioDataset(test_df, train_data, target_seconds=1.0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Test evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    test_pbar = tqdm(test_loader, desc="Testing", ncols=100)
    for images, labels in test_pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_accuracy = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, average='weighted')
test_recall = recall_score(all_labels, all_preds, average='weighted')
test_f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"\nConfusion Matrix:")
print(conf_matrix)
