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
import io
import time
import random

try:
    import optuna
except ImportError:
    optuna = None

DATA_DIR = os.environ.get("DATA_DIR", r"d:\NIH_Chest_Xray_Project")
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
# Preprocessing transforms
# ---------------------------------------------------------------------------
# Training: includes augmentation to improve generalization on unseen data
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation / Inference: deterministic — no augmentation
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------------------------
# Utility: load state dict with key-name compatibility
# ---------------------------------------------------------------------------
def load_compat_state_dict(model, state_dict):
    """
    Loads state_dict into model, dynamically adapting between older format (nn.Linear)
    and newer format (nn.Sequential(nn.Dropout, nn.Linear)).
    Logs any keys that could not be matched so failures are visible.
    """
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
        print(f"[load_compat_state_dict] Missing keys (will use random init): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[load_compat_state_dict] Unexpected keys (ignored): {result.unexpected_keys}")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14x14
            
            nn.AdaptiveAvgPool2d((1, 1)) # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet_features = self.mobilenet.features
        
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.densenet_features = self.densenet.features
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 + 1280 + 1024, num_classes)
        )
        
    def forward(self, x):
        r_feat = self.resnet_features(x)
        r_feat = torch.flatten(r_feat, 1)
        
        m_feat = self.mobilenet_features(x)
        m_feat = self.pool(m_feat)
        m_feat = torch.flatten(m_feat, 1)
        
        d_feat = self.densenet_features(x)
        d_feat = self.pool(d_feat)
        d_feat = torch.flatten(d_feat, 1)
        
        combined = torch.cat((r_feat, m_feat, d_feat), dim=1)
        return self.classifier(combined)

def get_model(model_type="ResNet-18"):
    num_classes = len(ALL_LABELS)
    if model_type == "MobileNet-V2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == "DenseNet-121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier.in_features, num_classes)
        )
    elif model_type == "Simple CNN":
        model = SimpleCNN(num_classes)
    elif model_type == "Hybrid Model":
        model = HybridModel(num_classes)
    else: # Default ResNet-18
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, num_classes)
        )
    return model

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NIHDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
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
            except:
                img = Image.new('RGB', (224, 224))
            
        labels = row['Finding Labels'].split('|')
        label_tensor = torch.zeros(len(ALL_LABELS))
        for i, l in enumerate(ALL_LABELS):
            if l in labels:
                label_tensor[i] = 1.0
                
        if self.transform:
            img = self.transform(img)
            
        return img, label_tensor

# ---------------------------------------------------------------------------
# Helper: compute pos_weight for BCEWithLogitsLoss
# ---------------------------------------------------------------------------
def compute_pos_weight(df_labels_series, device):
    n = len(df_labels_series)
    pos_counts = torch.zeros(len(ALL_LABELS))
    for labels_str in df_labels_series:
        for i, l in enumerate(ALL_LABELS):
            if l in labels_str.split('|'):
                pos_counts[i] += 1
    pos_weight = torch.ones(len(ALL_LABELS))
    for i in range(len(ALL_LABELS)):
        if pos_counts[i] > 0:
            pos_weight[i] = min((n - pos_counts[i]) / pos_counts[i], 10.0)
    return pos_weight.to(device)

# ---------------------------------------------------------------------------
# Helper: patient-aware train/val split
# ---------------------------------------------------------------------------
def patient_split(df, val_frac=0.2, seed=42):
    """
    Splits df at patient level so no patient appears in both train and val.
    This prevents data leakage when the same patient has multiple X-rays.
    """
    patients = df['Patient ID'].unique().tolist()
    random.seed(seed)
    random.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])
    train_df = df[~df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    val_df   = df[ df['Patient ID'].isin(val_patients)].reset_index(drop=True)
    return train_df, val_df

# ---------------------------------------------------------------------------
# Helper: build optimizer with differential LR
# ---------------------------------------------------------------------------
def build_optimizer(model, architecture, opt_name="Adam", backbone_lr=1e-5, classifier_lr=1e-3):
    backbone_params = []
    classifier_params = []
    classifier_layer_name = "fc" if architecture == "ResNet-18" else "classifier"
    if architecture == "Simple CNN":
        # Simple CNN has no separate backbone — train everything at classifier_lr
        params = list(model.parameters())
        pg = [{"params": params, "lr": classifier_lr}]
    else:
        for name, param in model.named_parameters():
            if classifier_layer_name in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        pg = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": classifier_params, "lr": classifier_lr}
        ]

    opt_map = {
        "Adam":    torch.optim.Adam,
        "AdamW":   torch.optim.AdamW,
        "RMSprop": torch.optim.RMSprop,
        "Adagrad": torch.optim.Adagrad,
    }
    if opt_name in opt_map:
        return opt_map[opt_name](pg)
    elif opt_name == "SGD":
        return torch.optim.SGD(pg, momentum=0.9)
    else:
        return torch.optim.Adam(pg)

# ---------------------------------------------------------------------------
# evaluate_model  (used by Optuna and train loop)
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader, device, criterion=None):
    """Returns (f1, avg_loss). Loss is None if criterion not provided."""
    model.eval()
    tp, fp, fn = 0, 0, 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item()
                n_batches += 1
            preds = (torch.sigmoid(outputs) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).float().sum().item()
            fp += ((preds == 1) & (labels == 0)).float().sum().item()
            fn += ((preds == 0) & (labels == 1)).float().sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_loss = (total_loss / n_batches) if n_batches > 0 else None
    return f1, avg_loss

# ---------------------------------------------------------------------------
# TRAIN MODEL  (main fix: val split, augmentation, best-val save, LR scheduler)
# ---------------------------------------------------------------------------
def train_model(model_name, architecture, epochs, batch_size, selected_diseases, balanced_sampling, balanced_size, progress=gr.Progress()):
    if not model_name:
        model_name = f"model_{int(time.time())}"
        
    log_text = f"Initializing training for model: {model_name} (Arch: {architecture})...\n"
    yield log_text, gr.update(), gr.update()
    
    # Check for existing checkpoint to override config parameters early
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint.pth")
    checkpoint = None
    start_epoch = 0
    history = {
        "loss": [], "val_loss": [],
        "accuracy": [], "val_accuracy": [],
        "architecture": architecture
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                architecture    = cfg.get('architecture', architecture)
                batch_size      = int(cfg.get('batch_size', batch_size))
                selected_diseases = cfg.get('selected_diseases', selected_diseases)
                balanced_sampling = cfg.get('balanced_sampling', balanced_sampling)
                balanced_size   = cfg.get('balanced_size', balanced_size)
            log_text += f"Found existing checkpoint for '{model_name}'. Loaded config: Arch={architecture}, Batch Size={batch_size}.\n"
            yield log_text, gr.update(), gr.update()
        except Exception as e:
            log_text += f"Failed to load checkpoint config ({e}). Starting training from scratch.\n"
            yield log_text, gr.update(), gr.update()
        
    df = pd.read_csv(CSV_PATH)
    
    # Filter dataset to only include images that actually exist on disk
    df = df[df['Image Index'].isin(all_image_paths.keys())].reset_index(drop=True)
    
    if selected_diseases:
        if balanced_sampling:
            sample_size = int(balanced_size)
            log_text += f"Using Balanced Sampling: {sample_size} images for each of the {len(selected_diseases)} selected diseases...\n"
            yield log_text, gr.update(), gr.update()
            dfs = []
            for d in selected_diseases:
                d_df = df[df['Finding Labels'].str.contains(d, regex=False)].copy()
                sample_n = min(len(d_df), sample_size)
                if sample_n > 0:
                    # FIX: random sample instead of head() to avoid always picking same patients
                    dfs.append(d_df.sample(n=sample_n, random_state=42))
            if dfs:
                df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
            else:
                log_text += "No images found for the selected diseases.\n"
                yield log_text, update_model_dropdown(), update_checkpoint_dropdown()
                return
        else:
            log_text += "Using all matching images for the selected diseases...\n"
            yield log_text, gr.update(), gr.update()
            mask = df['Finding Labels'].apply(lambda x: any(d in x for d in selected_diseases))
            df = df[mask].reset_index(drop=True)
    else:
        log_text += f"No diseases selected. Training on the entire dataset of {len(df)} images...\n"
        yield log_text, gr.update(), gr.update()
        df = df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # FIX: Patient-level train/val split (prevents data leakage)
    # -----------------------------------------------------------------------
    if 'Patient ID' in df.columns and len(df) >= 10:
        train_df, val_df = patient_split(df, val_frac=0.2, seed=42)
    else:
        # Fallback: random split when Patient ID is not available
        train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        val_df   = df.drop(train_df.index).reset_index(drop=True)

    msg = (f"Training on {len(train_df)} images | Validating on {len(val_df)} images | "
           f"{epochs} epochs | batch size {int(batch_size)}.")
    print(msg)
    log_text += msg + "\n"
    yield log_text, gr.update(), gr.update()
    
    log_text += f"Using device: {device}\n"
    yield log_text, gr.update(), gr.update()
    
    model = get_model(architecture).to(device)
    
    # All parameters trainable so the backbone adapts to X-ray details
    for param in model.parameters():
        param.requires_grad = True
        
    # -----------------------------------------------------------------------
    # Build optimizer with differential learning rates
    # -----------------------------------------------------------------------
    if architecture == "Simple CNN":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = build_optimizer(model, architecture, opt_name="Adam",
                                    backbone_lr=1e-5, classifier_lr=1e-3)

    # -----------------------------------------------------------------------
    # FIX: LR Scheduler — CosineAnnealingLR for smooth convergence
    # -----------------------------------------------------------------------
    total_epochs = int(epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # -----------------------------------------------------------------------
    # Weighted loss to handle class imbalance
    # -----------------------------------------------------------------------
    pos_weight = compute_pos_weight(train_df['Finding Labels'], device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            log_text += f"Resumed training weights from epoch {start_epoch}...\n"
            yield log_text, gr.update(), gr.update()
        except Exception as e:
            log_text += f"Failed to load checkpoint weights ({e}). Starting from scratch.\n"
            yield log_text, gr.update(), gr.update()

    # -----------------------------------------------------------------------
    # FIX: Use TRAIN_TRANSFORM (with augmentation) for training
    #       Use INFER_TRANSFORM (no augmentation) for validation
    # -----------------------------------------------------------------------
    train_dataset = NIHDataset(train_df, transform=TRAIN_TRANSFORM)
    val_dataset   = NIHDataset(val_df,   transform=INFER_TRANSFORM)
    train_loader  = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True,  num_workers=0, pin_memory=False)
    val_loader    = DataLoader(val_dataset,   batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=False)

    # -----------------------------------------------------------------------
    # FIX: Save best val model, not just the last epoch
    # -----------------------------------------------------------------------
    best_val_f1   = -1.0
    best_model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")

    for epoch in progress.tqdm(range(start_epoch, total_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        tp = fp = fn = 0
        total_samples = 0
        
        for inputs, labels in progress.tqdm(train_loader, desc="Batches"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).float().sum().item()
            fp += ((preds == 1) & (labels == 0)).float().sum().item()
            fn += ((preds == 0) & (labels == 1)).float().sum().item()
            
        scheduler.step()

        epoch_loss = running_loss / (total_samples + 1e-8)
        precision  = tp / (tp + fp + 1e-8)
        recall     = tp / (tp + fn + 1e-8)
        epoch_f1   = 2 * precision * recall / (precision + recall + 1e-8)

        # -----------------------------------------------------------------------
        # FIX: Compute validation metrics every epoch
        # -----------------------------------------------------------------------
        val_f1, val_loss = evaluate_model(model, val_loader, device, criterion)

        history["loss"].append(epoch_loss)
        history["val_loss"].append(val_loss if val_loss is not None else 0.0)
        history["accuracy"].append(epoch_f1)
        history["val_accuracy"].append(val_f1)
        
        msg  = f"Epoch [{epoch+1}/{total_epochs}]\n"
        msg += f"  -> Train Loss: {epoch_loss:.4f}  |  Train F1: {epoch_f1 * 100:.2f}%\n"
        msg += f"  -> Val   Loss: {val_loss:.4f}  |  Val   F1: {val_f1 * 100:.2f}%\n"
        print(msg)
        log_text += msg + "\n"
        yield log_text, gr.update(), gr.update()

        # -----------------------------------------------------------------------
        # FIX: Save best-validation model weights
        # -----------------------------------------------------------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            log_text += f"  ✅ New best val F1: {best_val_f1 * 100:.2f}% — model saved.\n"
            yield log_text, gr.update(), gr.update()
        
        # Save epoch checkpoint (for crash recovery)
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
                    'balanced_sampling': balanced_sampling,
                    'balanced_size': balanced_size
                }
            }, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        
    # Save final metrics JSON (best weights were already saved above)
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f)
        
    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            print(f"Error removing checkpoint: {e}")
            
    log_text += f"\nTraining complete! Best val F1: {best_val_f1*100:.2f}%. Model saved as {model_name}.pth\n"
    yield log_text, update_model_dropdown(), update_checkpoint_dropdown()

# ---------------------------------------------------------------------------
# FIX: update_model_dropdown — exclude checkpoint files
# ---------------------------------------------------------------------------
def update_model_dropdown():
    models_list = [
        f.replace(".pth", "")
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".pth") and not f.endswith("_checkpoint.pth")
    ]
    return gr.Dropdown(choices=models_list, label="Select Model")

# ---------------------------------------------------------------------------
# Performance plot — now shows train AND val curves
# ---------------------------------------------------------------------------
def get_performance(model_name):
    if not model_name:
        return None, "No model selected."
        
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(metrics_path):
        return None, "No metrics found for this model."
        
    with open(metrics_path, "r") as f:
        history = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_range = range(1, len(history["loss"]) + 1)
    
    # Loss subplot
    ax1.plot(epochs_range, history["loss"], marker='o', color='royalblue', linewidth=2, label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs_range, history["val_loss"], marker='s', color='tomato', linewidth=2, linestyle='--', label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # F1-Score subplot
    train_f1_key = "f1_score" if "f1_score" in history else "accuracy"
    val_f1_key   = "val_f1_score" if "val_f1_score" in history else "val_accuracy"
    train_f1 = history.get(train_f1_key, [])
    val_f1   = history.get(val_f1_key, [])

    if train_f1:
        ax2.plot(epochs_range, [f * 100 for f in train_f1], marker='o', color='royalblue', linewidth=2, label="Train F1")
    if val_f1:
        ax2.plot(range(1, len(val_f1) + 1), [f * 100 for f in val_f1], marker='s', color='tomato', linewidth=2, linestyle='--', label="Val F1")
    ax2.set_title("F1-Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1-Score (%)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    final_loss    = history["loss"][-1]
    final_val_f1  = val_f1[-1] if val_f1 else None
    best_val_f1   = max(val_f1) if val_f1 else None
    arch = history.get("architecture", "Unknown")
    
    stats  = f"Architecture: {arch}\n"
    stats += f"Final Train Loss: {final_loss:.4f}\n"
    if best_val_f1 is not None:
        stats += f"Best Val F1: {best_val_f1*100:.2f}%  |  Final Val F1: {final_val_f1*100:.2f}%\n"
    
    return fig, stats

# ---------------------------------------------------------------------------
# OPTUNA TUNING  (added patient-level split, random sampling, pruner)
# ---------------------------------------------------------------------------
def optuna_tune_model(study_name, num_trials, epochs_per_trial, selected_architectures, selected_optimizers, selected_batch_sizes, selected_diseases, balanced_sampling, balanced_size, progress=gr.Progress()):
    if optuna is None:
        yield "Error: Optuna library is not installed. Please run 'pip install optuna' to enable hyperparameter tuning.", gr.update(), None, gr.update()
        return
        
    if not study_name:
        yield "Error: Please specify a Study Name to identify and save/resume this optimization.", gr.update(), None, gr.update()
        return
        
    if not selected_architectures:
        yield "Error: Please select at least one base architecture to search over.", gr.update(), None, gr.update()
        return
        
    if not selected_optimizers:
        yield "Error: Please select at least one optimizer to search over.", gr.update(), None, gr.update()
        return
        
    if not selected_batch_sizes:
        yield "Error: Please select at least one batch size to search over.", gr.update(), None, gr.update()
        return

    log_text  = f"Starting Optuna Hyperparameter Optimization Study '{study_name}'...\n"
    log_text += f"Config: Trials={num_trials}, Epochs per Trial={epochs_per_trial}\n"
    log_text += f"Architectures: {selected_architectures}\n"
    log_text += f"Optimizers: {selected_optimizers}\n"
    log_text += f"Batch Sizes: {selected_batch_sizes}\n"
    yield log_text, gr.update(), None, gr.update()
    
    df = pd.read_csv(CSV_PATH)
    df = df[df['Image Index'].isin(all_image_paths.keys())].reset_index(drop=True)
    
    if selected_diseases:
        if balanced_sampling:
            sample_size = int(balanced_size)
            log_text += f"Using Balanced Sampling: {sample_size} images per disease...\n"
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
                yield log_text + "No images found for the selected diseases.\n", gr.update(), None, gr.update()
                return
        else:
            log_text += "Using all matching images for target diseases...\n"
            yield log_text, gr.update(), None, gr.update()
            mask = df['Finding Labels'].apply(lambda x: any(d in x for d in selected_diseases))
            df = df[mask].reset_index(drop=True)
    else:
        log_text += f"No diseases specified. Using full dataset of {len(df)} images...\n"
        yield log_text, gr.update(), None, gr.update()
        
    if len(df) < 10:
        yield log_text + f"Error: Dataset too small ({len(df)} images).\n", gr.update(), None, gr.update()
        return

    # FIX: Patient-level split for Optuna too
    if 'Patient ID' in df.columns:
        train_df, val_df = patient_split(df, val_frac=0.2, seed=42)
    else:
        train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        val_df   = df.drop(train_df.index).reset_index(drop=True)
    
    log_text += f"Split: {len(train_df)} train | {len(val_df)} val (patient-level).\n"
    yield log_text, gr.update(), None, gr.update()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_text += f"Running on: {device}\n"
    yield log_text, gr.update(), None, gr.update()
    
    pos_weight = compute_pos_weight(train_df['Finding Labels'], device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optuna study setup with median pruner
    db_path = os.path.abspath(os.path.join(MODELS_DIR, "optuna_studies.db"))
    db_path_url = db_path.replace(os.sep, '/')
    storage_url = f"sqlite:///{db_path_url}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    )
    
    completed_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    log_text += f"Loaded study '{study_name}'. Completed trials so far: {completed_before}.\n"
    yield log_text, gr.update(), None, gr.update()
    
    for trial_idx in range(int(num_trials)):
        trial = study.ask()
        trial_num = trial.number
        
        arch           = trial.suggest_categorical("architecture", selected_architectures)
        opt_name       = trial.suggest_categorical("optimizer", selected_optimizers)
        b_size         = int(trial.suggest_categorical("batch_size", selected_batch_sizes))
        dropout_val    = trial.suggest_float("dropout", 0.1, 0.5)
        classifier_lr  = trial.suggest_float("classifier_lr", 1e-4, 1e-2, log=True)
        backbone_lr    = trial.suggest_float("backbone_lr", 1e-6, 1e-4, log=True) if arch != "Simple CNN" else 0.0
            
        t_log  = f"\n--- [Trial {trial_num+1}/{completed_before + num_trials}] ---\n"
        t_log += f"  Arch: {arch} | Opt: {opt_name} | BS: {b_size} | Dropout: {dropout_val:.2f} | LR cls: {classifier_lr:.2e}\n"
        log_text += t_log
        yield log_text, gr.update(), None, gr.update()
        
        train_dataset = NIHDataset(train_df, transform=TRAIN_TRANSFORM)
        val_dataset   = NIHDataset(val_df,   transform=INFER_TRANSFORM)
        train_loader  = DataLoader(train_dataset, batch_size=b_size, shuffle=True,  num_workers=0)
        val_loader    = DataLoader(val_dataset,   batch_size=b_size, shuffle=False, num_workers=0)
        
        model = get_model(arch)
        
        # Inject trial dropout
        if arch == "ResNet-18":
            model.fc = nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Linear(model.fc[1].in_features, len(ALL_LABELS))
            )
        elif arch in ("MobileNet-V2", "DenseNet-121", "Simple CNN", "Hybrid Model"):
            model.classifier[0] = nn.Dropout(dropout_val)
            
        model.to(device)
        for param in model.parameters():
            param.requires_grad = True

        optimizer = build_optimizer(model, arch, opt_name, backbone_lr, classifier_lr)
            
        best_val_f1 = 0.0
        for epoch in progress.tqdm(range(int(epochs_per_trial)), desc=f"Trial {trial_num+1} Epochs"):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            val_f1, _ = evaluate_model(model, val_loader, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            # Report intermediate for pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                log_text += f"  Trial {trial_num+1} pruned at epoch {epoch+1}.\n"
                yield log_text, gr.update(), None, gr.update()
                break
        else:
            study.tell(trial, best_val_f1)
            log_text += f"  Trial {trial_num+1} done. Best Val F1: {best_val_f1 * 100:.2f}%\n"
            yield log_text, gr.update(), None, gr.update()
        
    best_trial = study.best_trial
    log_text += f"\n=====================================\n"
    log_text += f"OPTIMIZATION COMPLETE!\n"
    log_text += f"Best Trial: {best_trial.number + 1}  |  Best Val F1: {best_trial.value * 100:.2f}%\n"
    log_text += "Best Parameters:\n"
    for k, v in best_trial.params.items():
        log_text += f"  - {k}: {v}\n"
    log_text += "=====================================\n"
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    trial_nums = [t.number + 1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    f1s = [t.value * 100 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ax.plot(trial_nums, f1s, marker='o', color='purple', linewidth=2, label="Trial F1")
    ax.set_title("Optuna Hyperparameter Optimization History")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation F1-Score (%)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    
    best_params_path = os.path.join(MODELS_DIR, "best_optuna_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "best_trial_number": best_trial.number + 1,
            "best_value": best_trial.value,
            "best_params": best_trial.params
        }, f, indent=4)
        
    results_summary  = f"Best Validation F1-Score: {best_trial.value * 100:.2f}%\n\nBest Hyperparameters:\n"
    for k, v in best_trial.params.items():
        if isinstance(v, float):
            results_summary += f"{k}: {v:.2e}\n" if v < 1e-3 else f"{k}: {v:.4f}\n"
        else:
            results_summary += f"{k}: {v}\n"
            
    yield log_text, fig, results_summary, gr.update(choices=get_optuna_studies(), value=study_name)

# ---------------------------------------------------------------------------
# INFERENCE  (consistent INFER_TRANSFORM, same as val)
# ---------------------------------------------------------------------------
def predict_image(image, model_name):
    if image is None:
        return "Please upload an image."
    if not model_name:
        return "Please select a trained model from Performance section."
        
    model_path   = os.path.join(MODELS_DIR, f"{model_name}.pth")
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(model_path):
        return "Model file not found."
    
    arch = "ResNet-18"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                meta = json.load(f)
                arch = meta.get("architecture", "ResNet-18")
        except:
            pass
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = get_model(arch)
    load_compat_state_dict(model, torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # FIX: Use the same INFER_TRANSFORM as validation — no augmentation
    image = image.convert('RGB')
    img_t = INFER_TRANSFORM(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs   = torch.sigmoid(outputs).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = [probs.item()]
        
    results = {label: float(prob) for label, prob in zip(ALL_LABELS, probs)}
    return results

# ---------------------------------------------------------------------------
# Optuna study utilities
# ---------------------------------------------------------------------------
def get_optuna_studies():
    if optuna is None:
        return []
    db_path = os.path.abspath(os.path.join(MODELS_DIR, "optuna_studies.db"))
    if not os.path.exists(db_path):
        return []
    try:
        db_path_url = db_path.replace(os.sep, '/')
        storage_url = f"sqlite:///{db_path_url}"
        summaries = optuna.get_all_study_summaries(storage=storage_url)
        return [s.study_name for s in summaries]
    except Exception as e:
        print(f"Error loading studies: {e}")
        return []

def get_optuna_study_details(study_name):
    if optuna is None or not study_name:
        return "No Optuna library or study name selected.", None, None
        
    db_path = os.path.abspath(os.path.join(MODELS_DIR, "optuna_studies.db"))
    db_path_url = db_path.replace(os.sep, '/')
    storage_url = f"sqlite:///{db_path_url}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        return f"Error loading study: {e}", None, None
        
    stats  = f"Study Name: {study_name}\n"
    stats += f"Total Trials: {len(study.trials)}\n"
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    stats += f"Completed Trials: {len(completed_trials)}\n"
    
    if len(completed_trials) > 0:
        best_trial = study.best_trial
        stats += f"Best Trial Number: {best_trial.number + 1}\n"
        stats += f"Best Validation F1-Score: {best_trial.value * 100:.2f}%\n"
        stats += "\nBest Hyperparameters:\n"
        for k, v in best_trial.params.items():
            if isinstance(v, float):
                stats += f"  - {k}: {v:.2e}\n" if v < 1e-3 else f"  - {k}: {v:.4f}\n"
            else:
                stats += f"  - {k}: {v}\n"
    else:
        stats += "\nNo completed trials yet."

    trial_data = []
    for t in study.trials:
        row = {
            "Trial":       t.number + 1,
            "State":       t.state.name,
            "F1-Score (%)": round(t.value * 100, 2) if t.value is not None else None,
        }
        for k, v in t.params.items():
            row[k] = round(v, 6) if isinstance(v, float) else v
        trial_data.append(row)
        
    df = pd.DataFrame(trial_data) if trial_data else pd.DataFrame()
    
    fig = None
    if len(completed_trials) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        trial_nums = [t.number + 1 for t in completed_trials]
        f1s        = [t.value * 100 for t in completed_trials]
        ax.plot(trial_nums, f1s, marker='o', color='purple', linewidth=2, label="Trial F1")
        
        running_max = []
        curr_max = -1
        for val in f1s:
            curr_max = max(curr_max, val)
            running_max.append(curr_max)
        ax.plot(trial_nums, running_max, linestyle='--', color='darkorange', linewidth=2, label="Best F1 (Running)")
        
        ax.set_title(f"Optimization History for {study_name}")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Validation F1-Score (%)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        
    return stats, df, fig

# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def get_active_checkpoints():
    if not os.path.exists(MODELS_DIR):
        return []
    checkpoints = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_checkpoint.pth"):
            model_name = f.replace("_checkpoint.pth", "")
            path = os.path.join(MODELS_DIR, f)
            try:
                ckpt   = torch.load(path, map_location='cpu', weights_only=False)
                epoch  = ckpt.get('epoch', 0)
                config = ckpt.get('config', {})
                arch   = config.get('architecture', 'Unknown')
                epochs = config.get('epochs', '?')
                checkpoints.append((model_name, f"{model_name} (Arch: {arch}, Epoch: {epoch+1}/{epochs})"))
            except Exception:
                checkpoints.append((model_name, f"{model_name} (Unknown state)"))
    return checkpoints

def load_checkpoint_info_to_ui(selected_checkpoint_display):
    if not selected_checkpoint_display or selected_checkpoint_display == "None (Start New)":
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    model_name      = selected_checkpoint_display.split(" ")[0]
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
    try:
        ckpt             = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg              = ckpt.get('config', {})
        arch             = cfg.get('architecture', gr.update())
        epochs           = cfg.get('epochs', gr.update())
        batch_size       = cfg.get('batch_size', gr.update())
        selected_diseases = cfg.get('selected_diseases', gr.update())
        balanced_sampling = cfg.get('balanced_sampling', gr.update())
        balanced_size    = cfg.get('balanced_size', gr.update())
        
        return (model_name, arch, epochs, batch_size, balanced_sampling, balanced_size, selected_diseases)
    except Exception as e:
        print(f"Error loading checkpoint metadata to UI: {e}")
        return model_name, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def update_checkpoint_dropdown():
    ckpt_list = ["None (Start New)"] + [disp for _, disp in get_active_checkpoints()]
    return gr.Dropdown(choices=ckpt_list, value="None (Start New)", label="Resume Training from Checkpoint")

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# NIH Chest X-ray Model Trainer & Predictor")
    gr.Markdown("Switch between default and dark mode using your browser's theme settings, or use Gradio's built-in toggle.")
    
    with gr.Tab("1. Train Model"):
        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(choices=["None (Start New)"], value="None (Start New)", label="Resume Training from Checkpoint")
            refresh_checkpoint_btn = gr.Button("🔄 Refresh Checkpoints", size="sm")
        with gr.Row():
            model_name_input    = gr.Textbox(label="Model Name (optional)", placeholder="my_resnet")
            architecture_input  = gr.Dropdown(choices=["ResNet-18", "MobileNet-V2", "DenseNet-121", "Simple CNN", "Hybrid Model"], value="ResNet-18", label="Base Architecture")
        with gr.Row():
            epochs_input     = gr.Slider(minimum=1, maximum=50, value=3, step=1, label="Epochs")
            batch_size_input = gr.Slider(minimum=4, maximum=128, value=16, step=4, label="Batch Size")
        with gr.Row():
            balanced_input      = gr.Checkbox(label="Enable Balanced Sampling", value=True)
            balanced_size_input = gr.Dropdown(choices=["50", "100", "150", "200", "500", "1000", "2000", "5000"], value="100", label="Balanced Sample Size (images per selected disease)", visible=True)
        with gr.Row():
            with gr.Column():
                diseases_input = gr.CheckboxGroup(choices=ALL_LABELS, label="Target Diseases (Select at least one for Balanced Sampling)", info="Select specific diseases to train on a targeted subset.")
                with gr.Row():
                    select_all_btn   = gr.Button("Select All")
                    deselect_all_btn = gr.Button("Clear Selection")
            
        train_btn    = gr.Button("Train Model", variant="primary")
        train_output = gr.Textbox(label="Live Terminal Output", lines=10, max_lines=20)
        
    with gr.Tab("2. Performance"):
        refresh_btn    = gr.Button("Refresh Models")
        model_dropdown = gr.Dropdown(choices=[], label="Select Model")
        with gr.Row():
            perf_plot  = gr.Plot(label="Performance Metrics (Train vs Val)")
            perf_stats = gr.Textbox(label="Final Stats")
            
    with gr.Tab("3. Inference"):
        infer_model_dropdown = gr.Dropdown(choices=[], label="Select Model for Inference")
        with gr.Row():
            image_input       = gr.Image(type="pil", label="Upload X-ray Image")
            prediction_output = gr.Label(num_top_classes=5, label="Disease Predictions")
        predict_btn = gr.Button("Predict Disease", variant="primary")
        
    with gr.Tab("4. Hyperparameter Tuning (Optuna)"):
        gr.Markdown("### Optimize hyper-parameters using Optuna. Dataset is split 80% train / 20% val at patient level.")
        with gr.Row():
            with gr.Column():
                optuna_study_name_input = gr.Textbox(label="Study Name (for resuming/saving)", value="optuna_study", placeholder="optuna_study")
                optuna_trials_input     = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Number of Trials")
                optuna_epochs_input     = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Epochs per Trial")
                optuna_archs_input      = gr.CheckboxGroup(choices=["ResNet-18", "MobileNet-V2", "DenseNet-121", "Simple CNN", "Hybrid Model"], value=["ResNet-18", "MobileNet-V2"], label="Base Architectures to Search")
                optuna_opts_input       = gr.CheckboxGroup(choices=["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"], value=["Adam", "AdamW", "SGD"], label="Optimizers to Search")
                optuna_batch_input      = gr.CheckboxGroup(choices=["8", "16", "32"], value=["16", "32"], label="Batch Sizes to Search")
            with gr.Column():
                optuna_balanced_input      = gr.Checkbox(label="Enable Balanced Sampling", value=True)
                optuna_balanced_size_input = gr.Dropdown(choices=["50", "100", "150", "200", "500", "1000", "2000", "5000"], value="100", label="Balanced Sample Size (images per selected disease)")
                optuna_diseases_input      = gr.CheckboxGroup(choices=ALL_LABELS, label="Target Diseases", info="Select specific diseases to train on a targeted subset.")
                with gr.Row():
                    optuna_select_all_btn   = gr.Button("Select All")
                    optuna_deselect_all_btn = gr.Button("Clear Selection")
                    
        optuna_tune_btn = gr.Button("Start Hyperparameter Optimization", variant="primary")
        
        with gr.Row():
            optuna_log_output = gr.Textbox(label="Optimization Terminal Output", lines=10, max_lines=20)
            with gr.Column():
                optuna_plot_output   = gr.Plot(label="Optimization History Graph")
                optuna_params_output = gr.Textbox(label="Best Hyperparameters Found", lines=8)

    with gr.Tab("5. Hyperparameter Tuning Results"):
        gr.Markdown("### View hyperparameter tuning history and details of completed studies.")
        with gr.Row():
            results_study_dropdown = gr.Dropdown(choices=[], label="Select Study")
            refresh_studies_btn    = gr.Button("🔄 Refresh Studies", size="sm")
            
        with gr.Row():
            with gr.Column(scale=1):
                results_stats_output = gr.Textbox(label="Study Statistics & Best Hyperparameters", lines=12)
            with gr.Column(scale=2):
                results_plot_output = gr.Plot(label="Study Optimization History")
                
        gr.Markdown("### All Trial History")
        results_trials_df = gr.Dataframe(label="Trials List", interactive=False)
                
    # -----------------------------------------------------------------------
    # Event wiring
    # -----------------------------------------------------------------------
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
        outputs=[
            model_name_input,
            architecture_input,
            epochs_input,
            batch_size_input,
            balanced_input,
            balanced_size_input,
            diseases_input
        ]
    )
    refresh_checkpoint_btn.click(fn=update_checkpoint_dropdown, inputs=None, outputs=[checkpoint_dropdown])
    
    # FIX: on_refresh returns 4 values — wire all 4 outputs
    def on_refresh():
        models_dd       = update_model_dropdown()
        ckpt_dd         = update_checkpoint_dropdown()
        studies_dd      = gr.Dropdown(choices=get_optuna_studies(), label="Select Study")
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
        inputs=[
            optuna_study_name_input,
            optuna_trials_input,
            optuna_epochs_input,
            optuna_archs_input,
            optuna_opts_input,
            optuna_batch_input,
            optuna_diseases_input,
            optuna_balanced_input,
            optuna_balanced_size_input
        ],
        outputs=[
            optuna_log_output,
            optuna_plot_output,
            optuna_params_output,
            results_study_dropdown
        ]
    )
    
    refresh_studies_btn.click(fn=lambda: gr.Dropdown(choices=get_optuna_studies(), label="Select Study"), inputs=None, outputs=[results_study_dropdown])
    results_study_dropdown.change(
        fn=get_optuna_study_details,
        inputs=[results_study_dropdown],
        outputs=[results_stats_output, results_trials_df, results_plot_output]
    )

if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port, share=False)
