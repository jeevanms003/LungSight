!pip install -q gradio optuna scikit-learn torchvision matplotlib pillow pandas

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import glob
from PIL import Image
import json
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import optuna
except ImportError:
    optuna = None

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/Mini Project"
CSV_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ALL_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 
              'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

print("Loading image paths...")
all_image_paths = {}
for i in range(1, 13):
    folder = f"images_{i:03d}"
    imgs = glob.glob(os.path.join(DATA_DIR, folder, 'images', '*.png'))
    for p in imgs:
        all_image_paths[os.path.basename(p)] = p
print(f"Loaded {len(all_image_paths)} image paths.")

# ---------------------------------------------------------------------------
# Stronger, medically-reasonable augmentation (prevents overfitting on small sets)
# ---------------------------------------------------------------------------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),  # mild Cutout-style
])

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_compat_state_dict(model, state_dict):
    model_state = model.state_dict()
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("fc."):
            if k == "fc.weight" and "fc.1.weight" in model_state:
                new_state["fc.1.weight"] = v
            elif k == "fc.bias" and "fc.1.bias" in model_state:
                new_state["fc.1.bias"] = v
            elif k == "fc.1.weight" and "fc.weight" in model_state:
                new_state["fc.weight"] = v
            elif k == "fc.1.bias" and "fc.bias" in model_state:
                new_state["fc.bias"] = v
            else:
                new_state[k] = v
        elif k.startswith("classifier."):
            if k == "classifier.weight" and "classifier.1.weight" in model_state:
                new_state["classifier.1.weight"] = v
            elif k == "classifier.bias" and "classifier.1.bias" in model_state:
                new_state["classifier.1.bias"] = v
            elif k == "classifier.1.weight" and "classifier.weight" in model_state:
                new_state["classifier.weight"] = v
            elif k == "classifier.1.bias" and "classifier.bias" in model_state:
                new_state["classifier.bias"] = v
            else:
                new_state[k] = v
        else:
            new_state[k] = v
    result = model.load_state_dict(new_state, strict=False)
    if result.missing_keys:
        print(f"[load_compat] Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[load_compat] Unexpected keys: {result.unexpected_keys}")

# ---------------------------------------------------------------------------
# Models – higher default dropout for small-data regime
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class HybridModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet_features = self.mobilenet.features
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.densenet_features = self.densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512 + 1280 + 1024, num_classes))
    def forward(self, x):
        r = torch.flatten(self.resnet_features(x), 1)
        m = torch.flatten(self.pool(self.mobilenet_features(x)), 1)
        d = torch.flatten(self.pool(self.densenet_features(x)), 1)
        return self.classifier(torch.cat((r, m, d), dim=1))

def get_model(model_type="ResNet-18", num_classes=len(ALL_LABELS), dropout=0.5):
    if model_type == "MobileNet-V2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif model_type == "DenseNet-121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif model_type == "Simple CNN":
        model = SimpleCNN(num_classes, dropout=dropout)
    elif model_type == "Hybrid Model":
        model = HybridModel(num_classes, dropout=dropout)
    else:  # ResNet-18
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return model

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NIHDataset(Dataset):
    def __init__(self, df, transform=None, target_labels=ALL_LABELS):
        self.df = df
        self.transform = transform
        self.target_labels = target_labels
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = all_image_paths.get(img_name)
        if not img_path:
            img = Image.new('RGB', (224, 224))
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224))
        labels = row['Finding Labels'].split('|')
        label_tensor = torch.zeros(len(self.target_labels))
        for i, l in enumerate(self.target_labels):
            if l in labels:
                label_tensor[i] = 1.0
        if self.transform:
            img = self.transform(img)
        return img, label_tensor

def compute_pos_weight(df_labels_series, target_labels, device, max_weight=10.0):
    n = len(df_labels_series)
    pos_counts = torch.zeros(len(target_labels))
    for labels_str in df_labels_series:
        for i, l in enumerate(target_labels):
            if l in labels_str.split('|'):
                pos_counts[i] += 1
    pos_weight = torch.ones(len(target_labels))
    for i in range(len(target_labels)):
        if pos_counts[i] > 0:
            pos_weight[i] = min((n - pos_counts[i]) / pos_counts[i], max_weight)
        else:
            pos_weight[i] = max_weight
    return pos_weight.to(device)

def patient_split(df, val_frac=0.2, seed=42):
    patients = df['Patient ID'].unique().tolist()
    random.seed(seed)
    random.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])
    train_df = df[~df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    val_df   = df[ df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    return train_df, val_df

def build_optimizer(model, architecture, opt_name="AdamW", backbone_lr=1e-5, classifier_lr=1e-3, weight_decay=1e-4):
    backbone_params, classifier_params = [], []
    classifier_layer_name = "fc" if architecture == "ResNet-18" else "classifier"
    if architecture == "Simple CNN":
        return torch.optim.AdamW(model.parameters(), lr=classifier_lr, weight_decay=weight_decay)
    for name, param in model.named_parameters():
        if classifier_layer_name in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    pg = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": classifier_params, "lr": classifier_lr}
    ]
    if opt_name == "AdamW":
        return torch.optim.AdamW(pg, weight_decay=weight_decay)
    elif opt_name == "Adam":
        return torch.optim.Adam(pg, weight_decay=weight_decay)
    elif opt_name == "SGD":
        return torch.optim.SGD(pg, momentum=0.9, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(pg, weight_decay=weight_decay)

# ---------------------------------------------------------------------------
# Evaluation – now returns F1, loss, Acc, mean AUC
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader, device, criterion=None, target_labels=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item()
                n_batches += 1
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Micro F1 / Acc at 0.5
    preds_bin = (all_preds > 0.5).astype(np.float32)
    tp = ((preds_bin == 1) & (all_labels == 1)).sum()
    fp = ((preds_bin == 1) & (all_labels == 0)).sum()
    fn = ((preds_bin == 0) & (all_labels == 1)).sum()
    tn = ((preds_bin == 0) & (all_labels == 0)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    # Mean AUC (macro) – the metric that actually matters
    aucs = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            try:
                aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
            except Exception:
                pass
    mean_auc = float(np.mean(aucs)) if aucs else 0.0

    avg_loss = (total_loss / n_batches) if n_batches > 0 else None
    return f1, avg_loss, acc, mean_auc

# ---------------------------------------------------------------------------
# TRAIN MODEL – fixed against overfitting
# ---------------------------------------------------------------------------
def train_model(model_name, architecture, epochs, batch_size, selected_diseases, balanced_sampling, balanced_size, progress=gr.Progress()):
    if not model_name:
        model_name = f"model_{int(time.time())}"
        
    log_text = f"Initializing training for model: {model_name} (Arch: {architecture})...\n"
    yield log_text, gr.update(), gr.update()
    
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint.pth")
    checkpoint = None
    start_epoch = 0
    target_labels = selected_diseases if (selected_diseases and len(selected_diseases) > 0) else ALL_LABELS
    
    history = {
        "loss": [], "val_loss": [],
        "accuracy": [], "val_accuracy": [],
        "f1": [], "val_f1": [],
        "auc": [], "val_auc": [],
        "architecture": architecture,
        "target_labels": target_labels
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                architecture     = cfg.get('architecture', architecture)
                batch_size       = int(cfg.get('batch_size', batch_size))
                selected_diseases = cfg.get('selected_diseases', selected_diseases)
                balanced_sampling = cfg.get('balanced_sampling', balanced_sampling)
                balanced_size    = cfg.get('balanced_size', balanced_size)
            log_text += f"Found existing checkpoint. Loaded config: Arch={architecture}, BS={batch_size}.\n"
            yield log_text, gr.update(), gr.update()
        except Exception as e:
            log_text += f"Failed to load checkpoint config ({e}). Starting from scratch.\n"
            yield log_text, gr.update(), gr.update()
        
    df = pd.read_csv(CSV_PATH)
    df = df[df['Image Index'].isin(all_image_paths.keys())].reset_index(drop=True)
    
    if selected_diseases:
        if balanced_sampling:
            sample_size = int(balanced_size)
            log_text += f"Using Balanced Sampling: {sample_size} images for each of {len(selected_diseases)} diseases...\n"
            yield log_text, gr.update(), gr.update()
            dfs = []
            for d in selected_diseases:
                d_df = df[df['Finding Labels'].str.contains(d, regex=False)].copy()
                sample_n = min(len(d_df), sample_size)
                if sample_n > 0:
                    dfs.append(d_df.sample(n=sample_n, random_state=42))
            if dfs:
                df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
            else:
                log_text += "No images found for selected diseases.\n"
                yield log_text, update_model_dropdown(), update_checkpoint_dropdown()
                return
        else:
            log_text += "Using all matching images...\n"
            yield log_text, gr.update(), gr.update()
            mask = df['Finding Labels'].apply(lambda x: any(d in x for d in selected_diseases))
            df = df[mask].reset_index(drop=True)
    else:
        log_text += f"Training on full available set of {len(df)} images...\n"
        yield log_text, gr.update(), gr.update()

    if 'Patient ID' in df.columns and len(df) >= 10:
        train_df, val_df = patient_split(df, val_frac=0.2, seed=42)
    else:
        train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        val_df   = df.drop(train_df.index).reset_index(drop=True)

    n_train = len(train_df)
    msg = (f"Training on {n_train} images | Validating on {len(val_df)} images | "
           f"Target Diseases: {len(target_labels)} | {epochs} epochs | batch size {int(batch_size)}.")
    log_text += msg + "\n"
    if n_train < 2000:
        log_text += "⚠️ Small dataset detected → strong regularization + staged unfreezing activated.\n"
    yield log_text, gr.update(), gr.update()
    
    log_text += f"Using device: {device}\n"
    yield log_text, gr.update(), gr.update()
    
    # Higher dropout for small data
    dropout = 0.5 if n_train < 3000 else 0.3
    model = get_model(architecture, num_classes=len(target_labels), dropout=dropout).to(device)
    
    # Staged training: freeze backbone first
    freeze_epochs = 4 if n_train < 3000 else 2
    classifier_layer_name = "fc" if architecture == "ResNet-18" else "classifier"
    for name, param in model.named_parameters():
        if classifier_layer_name not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Conservative LRs for small data
    b_lr = 1e-5 if n_train < 2000 else 3e-5
    c_lr = 1e-3 if n_train < 2000 else 5e-4
    weight_decay = 5e-4 if n_train < 2000 else 1e-4
    
    optimizer = build_optimizer(model, architecture, opt_name="AdamW",
                                backbone_lr=b_lr, classifier_lr=c_lr, weight_decay=weight_decay)
    total_epochs = int(epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)

    pos_weight = compute_pos_weight(train_df['Finding Labels'], target_labels, device, max_weight=10.0)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            log_text += f"Resumed weights from epoch {start_epoch}...\n"
            yield log_text, gr.update(), gr.update()
        except Exception as e:
            log_text += f"Failed to load checkpoint weights ({e}). Starting fresh.\n"
            yield log_text, gr.update(), gr.update()

    train_dataset = NIHDataset(train_df, transform=TRAIN_TRANSFORM, target_labels=target_labels)
    val_dataset   = NIHDataset(val_df,   transform=INFER_TRANSFORM, target_labels=target_labels)
    train_loader  = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True,  num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=int(batch_size), shuffle=False, num_workers=2, pin_memory=True)

    best_val_auc  = -1.0
    best_val_f1   = -1.0
    epochs_no_improve = 0
    best_model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    patience = 7 if n_train < 2000 else 10

    for epoch in progress.tqdm(range(start_epoch, total_epochs), desc="Epochs"):
        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs:
            log_text += f"\n🔓 Unfreezing backbone at epoch {epoch+1} (differential LR continues)...\n"
            yield log_text, gr.update(), gr.update()
            for param in model.parameters():
                param.requires_grad = True
            # Rebuild optimizer with backbone params now trainable
            optimizer = build_optimizer(model, architecture, opt_name="AdamW",
                                        backbone_lr=b_lr, classifier_lr=c_lr, weight_decay=weight_decay)
            # Re-attach scheduler state roughly
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-epoch, eta_min=1e-7)

        model.train()
        running_loss = 0.0
        tp = fp = fn = tn = 0
        total_samples = 0
        
        for inputs, labels in progress.tqdm(train_loader, desc="Batches"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).float().sum().item()
            fp += ((preds == 1) & (labels == 0)).float().sum().item()
            fn += ((preds == 0) & (labels == 1)).float().sum().item()
            tn += ((preds == 0) & (labels == 0)).float().sum().item()
            
        scheduler.step()

        epoch_loss = running_loss / (total_samples + 1e-8)
        precision  = tp / (tp + fp + 1e-8)
        recall     = tp / (tp + fn + 1e-8)
        epoch_f1   = 2 * precision * recall / (precision + recall + 1e-8)
        epoch_acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        val_f1, val_loss, val_acc, val_auc = evaluate_model(model, val_loader, device, criterion, target_labels)

        history["loss"].append(epoch_loss)
        history["val_loss"].append(val_loss if val_loss is not None else 0.0)
        history["accuracy"].append(epoch_acc)
        history["val_accuracy"].append(val_acc)
        history["f1"].append(epoch_f1)
        history["val_f1"].append(val_f1)
        history["auc"].append(0.0)  # train AUC expensive; skip
        history["val_auc"].append(val_auc)
        
        msg  = f"Epoch [{epoch+1}/{total_epochs}]\n"
        msg += f"  -> Train Loss: {epoch_loss:.4f}  |  Train Acc: {epoch_acc*100:.2f}%  |  Train F1: {epoch_f1*100:.2f}%\n"
        msg += f"  -> Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc*100:.2f}%  |  Val   F1: {val_f1*100:.2f}%  |  Val AUC: {val_auc*100:.2f}%\n"
        print(msg)
        log_text += msg + "\n"
        yield log_text, gr.update(), gr.update()

        # Save on best Val AUC (primary) or F1
        improved = False
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_val_f1 = val_f1
            improved = True
        elif abs(val_auc - best_val_auc) < 1e-4 and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            improved = True

        if improved:
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            log_text += f"  ✅ New best Val AUC: {best_val_auc*100:.2f}% (F1: {best_val_f1*100:.2f}%) — model saved.\n"
            yield log_text, gr.update(), gr.update()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_text += f"\n🛑 Early stopping at epoch {epoch+1} (no Val AUC improvement for {patience} epochs).\n"
                log_text += f"Best Val AUC: {best_val_auc*100:.2f}% | Best Val F1: {best_val_f1*100:.2f}%.\n"
                yield log_text, gr.update(), gr.update()
                break
        
        # Checkpoint
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'config': {
                    'architecture': architecture,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'selected_diseases': selected_diseases,
                    'target_labels': target_labels,
                    'balanced_sampling': balanced_sampling,
                    'balanced_size': balanced_size
                }
            }, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        
    # Final metrics
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f)
        
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass
            
    log_text += f"\nTraining complete! Best Val AUC: {best_val_auc*100:.2f}% | Best Val F1: {best_val_f1*100:.2f}%. Model saved as {model_name}.pth\n"
    yield log_text, update_model_dropdown(), update_checkpoint_dropdown()

def update_model_dropdown():
    models_list = [
        f.replace(".pth", "")
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".pth") and not f.endswith("_checkpoint.pth")
    ]
    return gr.Dropdown(choices=models_list, label="Select Model")

def get_performance(model_name):
    if not model_name:
        return None, "No model selected."
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(metrics_path):
        return None, "No metrics found for this model."
    with open(metrics_path, "r") as f:
        history = json.load(f)
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs_range = range(1, len(history["loss"]) + 1)
    
    # Loss
    axes[0].plot(epochs_range, history["loss"], marker='o', color='royalblue', linewidth=2, label="Train Loss")
    if history.get("val_loss"):
        axes[0].plot(epochs_range, history["val_loss"], marker='s', color='tomato', linewidth=2, linestyle='--', label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # F1
    if history.get("f1"):
        axes[1].plot(epochs_range, [f*100 for f in history["f1"]], marker='o', color='royalblue', linewidth=2, label="Train F1")
    if history.get("val_f1"):
        axes[1].plot(range(1, len(history["val_f1"])+1), [f*100 for f in history["val_f1"]], marker='s', color='tomato', linewidth=2, linestyle='--', label="Val F1")
    axes[1].set_title("F1-Score (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # AUC
    if history.get("val_auc"):
        axes[2].plot(range(1, len(history["val_auc"])+1), [a*100 for a in history["val_auc"]], marker='s', color='green', linewidth=2, label="Val AUC")
    axes[2].set_title("Val Mean AUC (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    final_loss = history["loss"][-1]
    best_val_f1 = max(history.get("val_f1", [0]))
    best_val_auc = max(history.get("val_auc", [0]))
    arch = history.get("architecture", "Unknown")
    
    stats  = f"Architecture: {arch}\n"
    stats += f"Final Train Loss: {final_loss:.4f}\n"
    stats += f"Best Val F1: {best_val_f1*100:.2f}%\n"
    stats += f"Best Val AUC: {best_val_auc*100:.2f}%\n"
    return fig, stats

# ---------------------------------------------------------------------------
# OPTUNA – fixed + patient split + AUC objective
# ---------------------------------------------------------------------------
def optuna_tune_model(study_name, num_trials, epochs_per_trial, selected_architectures, selected_optimizers, selected_batch_sizes, selected_diseases, balanced_sampling, balanced_size, progress=gr.Progress()):
    if optuna is None:
        yield "Error: Optuna not installed.", gr.update(), None, gr.update()
        return
    if not study_name:
        yield "Error: Please specify a Study Name.", gr.update(), None, gr.update()
        return
    if not selected_architectures or not selected_optimizers or not selected_batch_sizes:
        yield "Error: Select at least one architecture, optimizer and batch size.", gr.update(), None, gr.update()
        return

    log_text  = f"Starting Optuna Study '{study_name}'...\n"
    log_text += f"Trials={num_trials}, Epochs/trial={epochs_per_trial}\n"
    yield log_text, gr.update(), None, gr.update()
    
    df = pd.read_csv(CSV_PATH)
    df = df[df['Image Index'].isin(all_image_paths.keys())].reset_index(drop=True)
    
    if selected_diseases:
        if balanced_sampling:
            sample_size = int(balanced_size)
            log_text += f"Balanced Sampling: {sample_size} per disease...\n"
            yield log_text, gr.update(), None, gr.update()
            dfs = []
            for d in selected_diseases:
                d_df = df[df['Finding Labels'].str.contains(d, regex=False)].copy()
                sample_n = min(len(d_df), sample_size)
                if sample_n > 0:
                    dfs.append(d_df.sample(n=sample_n, random_state=42))
            if dfs:
                df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
            else:
                yield log_text + "No images found.\n", gr.update(), None, gr.update()
                return
        else:
            mask = df['Finding Labels'].apply(lambda x: any(d in x for d in selected_diseases))
            df = df[mask].reset_index(drop=True)
    else:
        log_text += f"Using full set of {len(df)} images...\n"
        yield log_text, gr.update(), None, gr.update()
        
    if len(df) < 10:
        yield log_text + "Dataset too small.\n", gr.update(), None, gr.update()
        return

    if 'Patient ID' in df.columns:
        train_df, val_df = patient_split(df, val_frac=0.2, seed=42)
    else:
        train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        val_df   = df.drop(train_df.index).reset_index(drop=True)
    
    log_text += f"Split: {len(train_df)} train | {len(val_df)} val (patient-level).\n"
    yield log_text, gr.update(), None, gr.update()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_labels = selected_diseases if selected_diseases else ALL_LABELS
    pos_weight = compute_pos_weight(train_df['Finding Labels'], target_labels, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    db_path = os.path.abspath(os.path.join(MODELS_DIR, "optuna_studies.db"))
    storage_url = f"sqlite:///{db_path.replace(os.sep, '/')}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    )
    
    completed_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    log_text += f"Loaded study. Completed trials so far: {completed_before}.\n"
    yield log_text, gr.update(), None, gr.update()
    
    for trial_idx in range(int(num_trials)):
        trial = study.ask()
        trial_num = trial.number
        
        arch           = trial.suggest_categorical("architecture", selected_architectures)
        opt_name       = trial.suggest_categorical("optimizer", selected_optimizers)
        b_size         = int(trial.suggest_categorical("batch_size", selected_batch_sizes))
        dropout_val    = trial.suggest_float("dropout", 0.3, 0.6)
        classifier_lr  = trial.suggest_float("classifier_lr", 1e-4, 5e-3, log=True)
        backbone_lr    = trial.suggest_float("backbone_lr", 1e-6, 5e-5, log=True) if arch != "Simple CNN" else 0.0
            
        t_log  = f"\n--- [Trial {trial_num+1}] ---\n"
        t_log += f"  Arch: {arch} | Opt: {opt_name} | BS: {b_size} | Drop: {dropout_val:.2f} | LR cls: {classifier_lr:.2e}\n"
        log_text += t_log
        yield log_text, gr.update(), None, gr.update()
        
        train_dataset = NIHDataset(train_df, transform=TRAIN_TRANSFORM, target_labels=target_labels)
        val_dataset   = NIHDataset(val_df,   transform=INFER_TRANSFORM, target_labels=target_labels)
        train_loader  = DataLoader(train_dataset, batch_size=b_size, shuffle=True,  num_workers=2)
        val_loader    = DataLoader(val_dataset,   batch_size=b_size, shuffle=False, num_workers=2)
        
        model = get_model(arch, num_classes=len(target_labels), dropout=dropout_val)
        # Force the suggested dropout
        if arch == "ResNet-18":
            in_f = model.fc[1].in_features
            model.fc = nn.Sequential(nn.Dropout(dropout_val), nn.Linear(in_f, len(target_labels)))
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            model.classifier[0] = nn.Dropout(dropout_val)
            
        model.to(device)
        for param in model.parameters():
            param.requires_grad = True

        optimizer = build_optimizer(model, arch, opt_name, backbone_lr, classifier_lr, weight_decay=5e-4)
            
        best_val_auc = 0.0
        for epoch in progress.tqdm(range(int(epochs_per_trial)), desc=f"Trial {trial_num+1}"):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            _, _, _, val_auc = evaluate_model(model, val_loader, device, criterion, target_labels)
            if val_auc > best_val_auc:
                best_val_auc = val_auc

            trial.report(val_auc, epoch)
            if trial.should_prune():
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                log_text += f"  Trial {trial_num+1} pruned at epoch {epoch+1}.\n"
                yield log_text, gr.update(), None, gr.update()
                break
        else:
            study.tell(trial, best_val_auc)
            log_text += f"  Trial {trial_num+1} done. Best Val AUC: {best_val_auc*100:.2f}%\n"
            yield log_text, gr.update(), None, gr.update()
        
    best_trial = study.best_trial
    log_text += f"\n=====================================\n"
    log_text += f"OPTIMIZATION COMPLETE!\n"
    log_text += f"Best Trial: {best_trial.number+1}  |  Best Val AUC: {best_trial.value*100:.2f}%\n"
    log_text += "Best Parameters:\n"
    for k, v in best_trial.params.items():
        log_text += f"  - {k}: {v}\n"
    log_text += "=====================================\n"
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    trial_nums = [t.number + 1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    aucs = [t.value * 100 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ax.plot(trial_nums, aucs, marker='o', color='purple', linewidth=2, label="Trial AUC")
    ax.set_title("Optuna History (Val AUC)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Val Mean AUC (%)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    
    best_params_path = os.path.join(MODELS_DIR, "best_optuna_params.json")
    with open(best_params_path, "w") as f:
        json.dump({"best_trial_number": best_trial.number+1, "best_value": best_trial.value, "best_params": best_trial.params}, f, indent=4)
        
    results_summary  = f"Best Validation AUC: {best_trial.value*100:.2f}%\n\nBest Hyperparameters:\n"
    for k, v in best_trial.params.items():
        results_summary += f"{k}: {v}\n" if not isinstance(v, float) else (f"{k}: {v:.2e}\n" if v < 1e-3 else f"{k}: {v:.4f}\n")
            
    yield log_text, fig, results_summary, gr.update(choices=get_optuna_studies(), value=study_name)

def predict_image(image, model_name):
    if image is None:
        return "Please upload an image."
    if not model_name:
        return "Please select a trained model."
        
    model_path   = os.path.join(MODELS_DIR, f"{model_name}.pth")
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(model_path):
        return "Model file not found."
    
    arch = "ResNet-18"
    target_labels = ALL_LABELS
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                meta = json.load(f)
                arch = meta.get("architecture", "ResNet-18")
                target_labels = meta.get("target_labels", ALL_LABELS)
        except Exception:
            pass
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = get_model(arch, num_classes=len(target_labels), dropout=0.5)
    load_compat_state_dict(model, torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    image = image.convert('RGB')
    img_t = INFER_TRANSFORM(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs   = torch.sigmoid(outputs).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = [probs.item()]
        
    results = {label: float(prob) for label, prob in zip(target_labels, probs)}
    return results

def get_optuna_studies():
    if optuna is None:
        return []
    db_path = os.path.abspath(os.path.join(MODELS_DIR, "optuna_studies.db"))
    if not os.path.exists(db_path):
        return []
    try:
        storage_url = f"sqlite:///{db_path.replace(os.sep, '/')}"
        summaries = optuna.get_all_study_summaries(storage=storage_url)
        return [s.study_name for s in summaries]
    except Exception as e:
        print(f"Error loading studies: {e}")
        return []

def get_optuna_study_details(study_name):
    if optuna is None or not study_name:
        return "No study selected.", None, None
    db_path = os.path.abspath(os.path.join(MODELS_DIR, "optuna_studies.db"))
    storage_url = f"sqlite:///{db_path.replace(os.sep, '/')}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        return f"Error loading study: {e}", None, None
        
    stats  = f"Study Name: {study_name}\nTotal Trials: {len(study.trials)}\n"
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    stats += f"Completed: {len(completed)}\n"
    if completed:
        best = study.best_trial
        stats += f"Best Trial: {best.number+1}\nBest Val AUC: {best.value*100:.2f}%\n\nBest Hyperparameters:\n"
        for k, v in best.params.items():
            stats += f"  - {k}: {v}\n" if not isinstance(v, float) else (f"  - {k}: {v:.2e}\n" if v < 1e-3 else f"  - {k}: {v:.4f}\n")
    else:
        stats += "\nNo completed trials yet."

    trial_data = []
    for t in study.trials:
        row = {"Trial": t.number+1, "State": t.state.name, "AUC (%)": round(t.value*100, 2) if t.value is not None else None}
        for k, v in t.params.items():
            row[k] = round(v, 6) if isinstance(v, float) else v
        trial_data.append(row)
    df = pd.DataFrame(trial_data) if trial_data else pd.DataFrame()
    
    fig = None
    if completed:
        fig, ax = plt.subplots(figsize=(7, 4))
        trial_nums = [t.number+1 for t in completed]
        aucs = [t.value*100 for t in completed]
        ax.plot(trial_nums, aucs, marker='o', color='purple', linewidth=2, label="Trial AUC")
        running_max = np.maximum.accumulate(aucs)
        ax.plot(trial_nums, running_max, linestyle='--', color='darkorange', linewidth=2, label="Best so far")
        ax.set_title(f"Optimization History – {study_name}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Val Mean AUC (%)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
    return stats, df, fig

def get_active_checkpoints():
    if not os.path.exists(MODELS_DIR):
        return []
    checkpoints = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_checkpoint.pth"):
            model_name = f.replace("_checkpoint.pth", "")
            path = os.path.join(MODELS_DIR, f)
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                epoch = ckpt.get('epoch', 0)
                config = ckpt.get('config', {})
                arch = config.get('architecture', 'Unknown')
                epochs = config.get('epochs', '?')
                checkpoints.append((model_name, f"{model_name} (Arch: {arch}, Epoch: {epoch+1}/{epochs})"))
            except Exception:
                checkpoints.append((model_name, f"{model_name} (Unknown state)"))
    return checkpoints

def load_checkpoint_info_to_ui(selected_checkpoint_display):
    if not selected_checkpoint_display or selected_checkpoint_display == "None (Start New)":
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    model_name = selected_checkpoint_display.split(" ")[0]
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        return (model_name,
                cfg.get('architecture', gr.update()),
                cfg.get('epochs', gr.update()),
                cfg.get('batch_size', gr.update()),
                cfg.get('balanced_sampling', gr.update()),
                cfg.get('balanced_size', gr.update()),
                cfg.get('selected_diseases', gr.update()))
    except Exception as e:
        print(f"Error loading checkpoint metadata: {e}")
        return model_name, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def update_checkpoint_dropdown():
    ckpt_list = ["None (Start New)"] + [disp for _, disp in get_active_checkpoints()]
    return gr.Dropdown(choices=ckpt_list, value="None (Start New)", label="Resume Training from Checkpoint")

# ---------------------------------------------------------------------------
# Gradio UI (unchanged structure)
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# NIH Chest X-ray Model Trainer & Predictor (Overfitting-Fixed)")
    gr.Markdown("Stronger augmentation • Staged unfreezing • AdamW + higher WD • AUC monitoring • Gradient clipping")
    
    with gr.Tab("1. Train Model"):
        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(choices=["None (Start New)"], value="None (Start New)", label="Resume Training from Checkpoint")
            refresh_checkpoint_btn = gr.Button("🔄 Refresh Checkpoints", size="sm")
        with gr.Row():
            model_name_input    = gr.Textbox(label="Model Name (optional)", placeholder="my_resnet")
            architecture_input  = gr.Dropdown(choices=["ResNet-18", "MobileNet-V2", "DenseNet-121", "Simple CNN", "Hybrid Model"], value="ResNet-18", label="Base Architecture")
        with gr.Row():
            epochs_input     = gr.Slider(minimum=1, maximum=50, value=15, step=1, label="Epochs")
            batch_size_input = gr.Slider(minimum=4, maximum=128, value=16, step=4, label="Batch Size")
        with gr.Row():
            balanced_input      = gr.Checkbox(label="Enable Balanced Sampling", value=True)
            balanced_size_input = gr.Dropdown(choices=["50", "100", "150", "200", "500", "1000", "2000", "5000"], value="200", label="Balanced Sample Size (images per selected disease)", visible=True)
        with gr.Row():
            with gr.Column():
                diseases_input = gr.CheckboxGroup(choices=ALL_LABELS, label="Target Diseases", info="Select diseases (recommend ≥200 images each for better results)")
                with gr.Row():
                    select_all_btn   = gr.Button("Select All")
                    deselect_all_btn = gr.Button("Clear Selection")
            
        train_btn    = gr.Button("Train Model", variant="primary")
        train_output = gr.Textbox(label="Live Terminal Output", lines=12, max_lines=25)
        
    with gr.Tab("2. Performance"):
        refresh_btn    = gr.Button("Refresh Models")
        model_dropdown = gr.Dropdown(choices=[], label="Select Model")
        with gr.Row():
            perf_plot  = gr.Plot(label="Performance Metrics (Train vs Val + AUC)")
            perf_stats = gr.Textbox(label="Final Stats")
            
    with gr.Tab("3. Inference"):
        infer_model_dropdown = gr.Dropdown(choices=[], label="Select Model for Inference")
        with gr.Row():
            image_input       = gr.Image(type="pil", label="Upload X-ray Image")
            prediction_output = gr.Label(num_top_classes=5, label="Disease Predictions")
        predict_btn = gr.Button("Predict Disease", variant="primary")
        
    with gr.Tab("4. Hyperparameter Tuning (Optuna)"):
        gr.Markdown("### Optimize using Optuna (patient-level split, AUC objective).")
        with gr.Row():
            with gr.Column():
                optuna_study_name_input = gr.Textbox(label="Study Name", value="optuna_study")
                optuna_trials_input     = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Number of Trials")
                optuna_epochs_input     = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Epochs per Trial")
                optuna_archs_input      = gr.CheckboxGroup(choices=["ResNet-18", "MobileNet-V2", "DenseNet-121", "Simple CNN", "Hybrid Model"], value=["ResNet-18", "DenseNet-121"], label="Architectures")
                optuna_opts_input       = gr.CheckboxGroup(choices=["Adam", "AdamW", "SGD"], value=["AdamW", "Adam"], label="Optimizers")
                optuna_batch_input      = gr.CheckboxGroup(choices=["8", "16", "32"], value=["16", "32"], label="Batch Sizes")
            with gr.Column():
                optuna_balanced_input      = gr.Checkbox(label="Enable Balanced Sampling", value=True)
                optuna_balanced_size_input = gr.Dropdown(choices=["50", "100", "150", "200", "500", "1000", "2000", "5000"], value="200", label="Balanced Sample Size")
                optuna_diseases_input      = gr.CheckboxGroup(choices=ALL_LABELS, label="Target Diseases")
                with gr.Row():
                    optuna_select_all_btn   = gr.Button("Select All")
                    optuna_deselect_all_btn = gr.Button("Clear Selection")
                    
        optuna_tune_btn = gr.Button("Start Hyperparameter Optimization", variant="primary")
        with gr.Row():
            optuna_log_output = gr.Textbox(label="Optimization Log", lines=10, max_lines=20)
            with gr.Column():
                optuna_plot_output   = gr.Plot(label="Optimization History")
                optuna_params_output = gr.Textbox(label="Best Hyperparameters", lines=8)

    with gr.Tab("5. Hyperparameter Tuning Results"):
        with gr.Row():
            results_study_dropdown = gr.Dropdown(choices=[], label="Select Study")
            refresh_studies_btn    = gr.Button("🔄 Refresh Studies", size="sm")
        with gr.Row():
            with gr.Column(scale=1):
                results_stats_output = gr.Textbox(label="Study Statistics", lines=12)
            with gr.Column(scale=2):
                results_plot_output = gr.Plot(label="Study History")
        gr.Markdown("### All Trial History")
        results_trials_df = gr.Dataframe(label="Trials", interactive=False)
                
    # Event wiring
    def toggle_balanced_size(balanced):
        return gr.update(visible=balanced)
        
    balanced_input.change(fn=toggle_balanced_size, inputs=[balanced_input], outputs=[balanced_size_input])
    select_all_btn.click(fn=lambda: gr.update(value=ALL_LABELS), outputs=diseases_input)
    deselect_all_btn.click(fn=lambda: gr.update(value=[]), outputs=diseases_input)
    
    train_btn.click(
        fn=train_model, 
        inputs=[model_name_input, architecture_input, epochs_input, batch_size_input, diseases_input, balanced_input, balanced_size_input], 
        outputs=[train_output, model_dropdown, checkpoint_dropdown]
    )
    
    checkpoint_dropdown.change(
        fn=load_checkpoint_info_to_ui,
        inputs=[checkpoint_dropdown],
        outputs=[model_name_input, architecture_input, epochs_input, batch_size_input, balanced_input, balanced_size_input, diseases_input]
    )
    refresh_checkpoint_btn.click(fn=update_checkpoint_dropdown, inputs=None, outputs=[checkpoint_dropdown])
    
    def on_refresh():
        models_dd = update_model_dropdown()
        ckpt_dd   = update_checkpoint_dropdown()
        studies_dd = gr.Dropdown(choices=get_optuna_studies(), label="Select Study")
        return models_dd, models_dd, ckpt_dd, studies_dd
        
    refresh_btn.click(fn=on_refresh, inputs=None, outputs=[model_dropdown, infer_model_dropdown, checkpoint_dropdown, results_study_dropdown])
    demo.load(fn=on_refresh, inputs=None, outputs=[model_dropdown, infer_model_dropdown, checkpoint_dropdown, results_study_dropdown])
    
    model_dropdown.change(fn=get_performance, inputs=[model_dropdown], outputs=[perf_plot, perf_stats])
    infer_model_dropdown.change(fn=lambda x: x, inputs=[infer_model_dropdown], outputs=[model_dropdown])
    predict_btn.click(fn=predict_image, inputs=[image_input, infer_model_dropdown], outputs=[prediction_output])
    
    optuna_balanced_input.change(fn=toggle_balanced_size, inputs=[optuna_balanced_input], outputs=[optuna_balanced_size_input])
    optuna_select_all_btn.click(fn=lambda: gr.update(value=ALL_LABELS), outputs=optuna_diseases_input)
    optuna_deselect_all_btn.click(fn=lambda: gr.update(value=[]), outputs=optuna_diseases_input)
    
    optuna_tune_btn.click(
        fn=optuna_tune_model,
        inputs=[optuna_study_name_input, optuna_trials_input, optuna_epochs_input, optuna_archs_input, optuna_opts_input, optuna_batch_input, optuna_diseases_input, optuna_balanced_input, optuna_balanced_size_input],
        outputs=[optuna_log_output, optuna_plot_output, optuna_params_output, results_study_dropdown]
    )
    
    refresh_studies_btn.click(fn=lambda: gr.Dropdown(choices=get_optuna_studies(), label="Select Study"), inputs=None, outputs=[results_study_dropdown])
    results_study_dropdown.change(fn=get_optuna_study_details, inputs=[results_study_dropdown], outputs=[results_stats_output, results_trials_df, results_plot_output])

if __name__ == "__main__":
    try:
        gr.close_all()
    except Exception:
        pass
    demo.launch(share=True)
