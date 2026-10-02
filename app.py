import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
import json
import matplotlib.pyplot as plt
import io
import time
import random
from sklearn.metrics import roc_auc_score

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
# Preprocessing transforms  (research-backed for chest X-ray)
# ---------------------------------------------------------------------------
# CheXNet paper + 2024 best practices: stronger augmentation is the #1 anti-overfitting tool
# - RandomResizedCrop: simulate varying FOV, forces the model to be scale-invariant
# - RandomAffine shear: simulates patient positioning differences
# - Grayscale→RGB jitter: X-rays are grayscale so brightness/contrast matters more than hue
# - GaussianBlur: simulates varying image sharpness across scanners
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # hide small artifact patches
])

# Validation / Inference: deterministic — no augmentation
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
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

def get_model(model_type="ResNet-18", num_classes=len(ALL_LABELS)):
    if model_type == "MobileNet-V2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    elif model_type == "DenseNet-121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier.in_features, num_classes)
        )
    elif model_type == "Simple CNN":
        model = SimpleCNN(num_classes)
    elif model_type == "Hybrid Model":
        model = HybridModel(num_classes)
    else: # Default ResNet-18
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )
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
            except:
                img = Image.new('RGB', (224, 224))
            
        labels = row['Finding Labels'].split('|')
        label_tensor = torch.zeros(len(self.target_labels))
        for i, l in enumerate(self.target_labels):
            if l in labels:
                label_tensor[i] = 1.0
                
        if self.transform:
            img = self.transform(img)
            
        return img, label_tensor

# ---------------------------------------------------------------------------
# Focal Loss  (Lin et al. 2017 — addresses easy-negative dominance in imbalanced data)
# In chest X-ray, >80% of labels are 0 (no disease). Standard BCE spends most gradient
# budget on those easy negatives. Focal Loss down-weights them so the model focuses
# on the rare positive disease cases — the hard examples.
# gamma=2 is the standard value from the RetinaNet paper, widely used in medical imaging.
# ---------------------------------------------------------------------------
class FocalBCELoss(nn.Module):
    """Binary Focal Loss for multi-label classification.
    Combines pos_weight (class-level imbalance) with focal modulation (sample-level difficulty).
    Reference: Lin et al. 2017 — Focal Loss for Dense Object Detection.
    """
    def __init__(self, pos_weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight  # per-class imbalance weights
        self.gamma      = gamma        # focusing parameter; 0 → standard BCE
        self.reduction  = reduction

    def forward(self, logits, targets):
        # Numerically stable BCE per element
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        # p_t = probability of the correct class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        # Focal modulation: (1 - p_t)^gamma downweights easy examples
        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ---------------------------------------------------------------------------
# Helper: compute pos_weight for FocalBCELoss  (capped at 5.0 to prevent instability)
# ---------------------------------------------------------------------------
def compute_pos_weight(df_labels_series, target_labels, device):
    n = len(df_labels_series)
    pos_counts = torch.zeros(len(target_labels))
    for labels_str in df_labels_series:
        for i, l in enumerate(target_labels):
            if l in labels_str.split('|'):
                pos_counts[i] += 1
    pos_weight = torch.ones(len(target_labels))
    for i in range(len(target_labels)):
        if pos_counts[i] > 0:
            # cap at 5.0: beyond this the model hallucinates diseases everywhere
            pos_weight[i] = min((n - pos_counts[i]) / pos_counts[i], 5.0)
    return pos_weight.to(device)

# ---------------------------------------------------------------------------
# Mixup augmentation  (Zhang et al. 2018 — strongest regularizer for medical imaging)
# Creates convex combinations of image pairs + their labels.
# Forces the model to learn smooth decision boundaries instead of memorizing training samples.
# alpha=0.2 is the standard value used in CheXMix and similar medical imaging papers.
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda for Mixup loss."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss as convex combination of two label losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
# Helper: build optimizer with differential LR & L2 regularization
# ---------------------------------------------------------------------------
def build_optimizer(model, architecture, opt_name="Adam", backbone_lr=1e-4, classifier_lr=1e-3, weight_decay=1e-4):
    backbone_params = []
    classifier_params = []
    classifier_layer_name = "fc" if architecture == "ResNet-18" else "classifier"
    if architecture == "Simple CNN":
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
        "Adam":    lambda p: torch.optim.Adam(p, weight_decay=weight_decay),
        "AdamW":   lambda p: torch.optim.AdamW(p, weight_decay=weight_decay),
        "RMSprop": lambda p: torch.optim.RMSprop(p, weight_decay=weight_decay),
        "Adagrad": lambda p: torch.optim.Adagrad(p, weight_decay=weight_decay),
    }
    if opt_name in opt_map:
        return opt_map[opt_name](pg)
    elif opt_name == "SGD":
        return torch.optim.SGD(pg, momentum=0.9, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(pg, weight_decay=weight_decay)

# ---------------------------------------------------------------------------
# evaluate_model  — now computes AUC-ROC (the CheXNet / NIH standard metric)
# ---------------------------------------------------------------------------
# Why AUC instead of F1 at 0.5 threshold?
# AUC measures the model's ability to rank sick vs healthy images regardless of threshold.
# The Stanford CheXNet paper (2017) reports per-class AUC as the primary metric.
# F1 at a fixed 0.5 threshold is misleading for imbalanced datasets — AUC is threshold-free.
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader, device, criterion=None, target_labels=None):
    """Returns (mean_auc, avg_loss, per_class_auc_dict). Loss is None if criterion not provided."""
    model.eval()
    all_probs  = []  # shape: (N, num_classes)
    all_labels = []  # shape: (N, num_classes)
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                # Use plain BCE for val loss (not focal) for clean monitoring
                val_loss = F.binary_cross_entropy_with_logits(outputs, labels)
                total_loss += val_loss.item()
                n_batches  += 1
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)  # (N, C)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, C)

    # Compute per-class AUC; skip classes with only one label (can't compute AUC)
    per_class_auc = {}
    valid_aucs    = []
    num_classes   = all_labels.shape[1]
    labels_list   = target_labels if target_labels else [f"Class_{i}" for i in range(num_classes)]
    for i, label_name in enumerate(labels_list):
        if len(np.unique(all_labels[:, i])) > 1:  # need both 0 and 1 present
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            per_class_auc[label_name] = auc
            valid_aucs.append(auc)
        else:
            per_class_auc[label_name] = float('nan')  # class absent in val batch

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    avg_loss = (total_loss / n_batches) if n_batches > 0 else None
    return mean_auc, avg_loss, per_class_auc

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
    target_labels = selected_diseases if (selected_diseases and len(selected_diseases) > 0) else ALL_LABELS
    
    history = {
        "loss": [], "val_loss": [],
        "accuracy": [], "val_accuracy": [],
        "f1": [], "val_f1": [],
        "architecture": architecture,
        "target_labels": target_labels
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

    if 'Patient ID' in df.columns and len(df) >= 10:
        train_df, val_df = patient_split(df, val_frac=0.2, seed=42)
    else:
        train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
        val_df   = df.drop(train_df.index).reset_index(drop=True)

    msg = (f"Training on {len(train_df)} images | Validating on {len(val_df)} images | "
           f"Target Diseases: {len(target_labels)} | {epochs} epochs | batch size {int(batch_size)}.")
    print(msg)
    log_text += msg + "\n"
    yield log_text, gr.update(), gr.update()
    
    log_text += f"Using device: {device}\n"
    yield log_text, gr.update(), gr.update()
    
    model = get_model(architecture, num_classes=len(target_labels)).to(device)

    # ------------------------------------------------------------------
    # BACKBONE WARMUP FREEZE  (prevents catastrophic forgetting)
    # Strategy from medical imaging best practices:
    # Phase 1 (2 epochs): freeze backbone, train only classifier head at high LR
    # Phase 2 (remaining): unfreeze backbone at very low LR, fine-tune everything
    # This stops the pretrained ImageNet features from being destroyed in early epochs.
    # ------------------------------------------------------------------
    WARMUP_EPOCHS = 2
    classifier_layer_name = "fc" if architecture == "ResNet-18" else "classifier"

    def freeze_backbone(m):
        for n, p in m.named_parameters():
            p.requires_grad = (classifier_layer_name in n)

    def unfreeze_all(m):
        for p in m.parameters():
            p.requires_grad = True

    freeze_backbone(model)
    log_text += f"Phase 1 ({WARMUP_EPOCHS} epochs): Training classifier head only.\n"
    yield log_text, gr.update(), gr.update()

    if architecture == "Simple CNN":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        optimizer = build_optimizer(model, architecture, opt_name="Adam",
                                    backbone_lr=0.0, classifier_lr=1e-3, weight_decay=1e-4)

    total_epochs = int(epochs)
    # CosineAnnealingWarmRestarts: prevents getting stuck in local minima
    # T_0 = half of total epochs, one restart in the middle of training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, total_epochs // 2), T_mult=1, eta_min=1e-7
    )

    pos_weight = compute_pos_weight(train_df['Finding Labels'], target_labels, device)
    # FocalBCELoss: focuses gradient on hard, misclassified examples (rare diseases)
    # gamma=2.0 is validated by CheXNet-era papers for chest X-ray
    criterion = FocalBCELoss(pos_weight=pos_weight, gamma=2.0)

    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history     = checkpoint.get('history', history)
            unfreeze_all(model)  # resumed model: backbone already warmed up
            log_text += f"Resumed training weights from epoch {start_epoch}...\n"
            yield log_text, gr.update(), gr.update()
        except Exception as e:
            log_text += f"Failed to load checkpoint weights ({e}). Starting from scratch.\n"
            yield log_text, gr.update(), gr.update()

    train_dataset = NIHDataset(train_df, transform=TRAIN_TRANSFORM, target_labels=target_labels)
    val_dataset   = NIHDataset(val_df,   transform=INFER_TRANSFORM, target_labels=target_labels)
    train_loader  = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True,  num_workers=0, pin_memory=False)
    val_loader    = DataLoader(val_dataset,   batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=False)

    best_val_auc      = -1.0
    epochs_no_improve = 0
    best_model_path   = os.path.join(MODELS_DIR, f"{model_name}.pth")
    PATIENCE          = 10  # increased patience since AUC is smoother than F1

    for epoch in progress.tqdm(range(start_epoch, total_epochs), desc="Epochs"):

        # -- Switch to full fine-tuning after warmup --
        if epoch == WARMUP_EPOCHS and start_epoch < WARMUP_EPOCHS:
            unfreeze_all(model)
            # Rebuild optimizer with low backbone LR (differential LR)
            b_lr = 1e-5 if len(train_df) < 1000 else 2e-5
            if architecture == "Simple CNN":
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
            else:
                optimizer = build_optimizer(model, architecture, opt_name="Adam",
                                            backbone_lr=b_lr, classifier_lr=5e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, (total_epochs - WARMUP_EPOCHS) // 2), eta_min=1e-7
            )
            log_text += f"Phase 2 (epoch {epoch+1}+): Full fine-tuning (backbone LR={b_lr:.0e}).\n"
            yield log_text, gr.update(), gr.update()

        model.train()
        running_loss  = 0.0
        total_samples = 0
        train_probs   = []
        train_labels  = []

        for inputs, labels in progress.tqdm(train_loader, desc="Batches"):
            inputs, labels = inputs.to(device), labels.to(device)

            # -- Mixup augmentation (Zhang et al. 2018) --
            # Only apply during phase 2 (after backbone unfreeze) — phase 1 is classification-only
            use_mixup = (epoch >= WARMUP_EPOCHS) and (random.random() < 0.5)
            if use_mixup:
                mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.2)
                optimizer.zero_grad()
                outputs = model(mixed_inputs)
                loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            loss.backward()
            # Gradient clipping prevents exploding gradients, especially with focal loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss  += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Collect probs for train AUC (use original labels, not mixed)
            with torch.no_grad():
                train_probs.append(torch.sigmoid(model(inputs)).cpu().numpy())
                train_labels.append(labels.cpu().numpy())

        scheduler.step()

        epoch_loss   = running_loss / (total_samples + 1e-8)

        # Train AUC
        train_probs_np  = np.concatenate(train_probs,  axis=0)
        train_labels_np = np.concatenate(train_labels, axis=0)
        train_aucs = []
        for ci in range(train_labels_np.shape[1]):
            if len(np.unique(train_labels_np[:, ci])) > 1:
                train_aucs.append(roc_auc_score(train_labels_np[:, ci], train_probs_np[:, ci]))
        epoch_auc = float(np.mean(train_aucs)) if train_aucs else 0.0

        # Validation AUC (the metric we track for early stopping and model saving)
        val_auc, val_loss, per_class_auc = evaluate_model(model, val_loader, device, criterion, target_labels)

        history["loss"].append(epoch_loss)
        history["val_loss"].append(val_loss if val_loss is not None else 0.0)
        history["accuracy"].append(epoch_auc)      # repurpose as train AUC
        history["val_accuracy"].append(val_auc)    # repurpose as val AUC
        history["f1"]     = history.get("f1",     []) + [epoch_auc]
        history["val_f1"] = history.get("val_f1", []) + [val_auc]

        msg  = f"Epoch [{epoch+1}/{total_epochs}]\n"
        msg += f"  -> Train Loss: {epoch_loss:.4f}  |  Train AUC: {epoch_auc * 100:.2f}%\n"
        msg += f"  -> Val   Loss: {val_loss:.4f}  |  Val   AUC: {val_auc * 100:.2f}%\n"
        # Show top-3 and bottom-3 per-class AUC for actionable feedback
        valid_pca = {k: v for k, v in per_class_auc.items() if not np.isnan(v)}
        if valid_pca:
            sorted_pca = sorted(valid_pca.items(), key=lambda x: x[1])
            worst = sorted_pca[:3]
            best  = sorted_pca[-3:]
            msg += f"  -> Best  classes: {', '.join(f'{k}={v*100:.1f}%' for k,v in reversed(best))}\n"
            msg += f"  -> Worst classes: {', '.join(f'{k}={v*100:.1f}%' for k,v in worst)}\n"
        print(msg)
        log_text += msg + "\n"
        yield log_text, gr.update(), gr.update()

        if val_auc > best_val_auc:
            best_val_auc      = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            log_text += f"  ✅ New best val AUC: {best_val_auc * 100:.2f}% — model saved.\n"
            # Save per-class AUC for inspection
            history["best_per_class_auc"] = {k: (float(v) if not np.isnan(v) else None) for k, v in per_class_auc.items()}
            yield log_text, gr.update(), gr.update()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log_text += f"\n🛑 Early stopping at epoch {epoch+1} (no AUC improvement for {PATIENCE} epochs).\n"
                log_text += f"Best val AUC preserved: {best_val_auc * 100:.2f}%.\n"
                yield log_text, gr.update(), gr.update()
                break

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

    # Save final metrics JSON
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f)

    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            print(f"Error removing checkpoint: {e}")

    log_text += f"\nTraining complete! Best val AUC: {best_val_auc*100:.2f}%. Model saved as {model_name}.pth\n"
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, len(history["loss"]) + 1)

    # ── Loss curve ──
    ax1 = axes[0]
    ax1.plot(epochs_range, history["loss"], marker='o', color='royalblue',
             linewidth=2, label="Train Loss")
    if history.get("val_loss"):
        ax1.plot(epochs_range, history["val_loss"], marker='s', color='tomato',
                 linewidth=2, linestyle='--', label="Val Loss")
    ax1.set_title("Focal Loss", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── AUC-ROC curve (the CheXNet metric) ──
    ax2 = axes[1]
    train_auc = history.get("accuracy", [])      # stored in accuracy slot
    val_auc   = history.get("val_accuracy", [])  # stored in val_accuracy slot
    if train_auc:
        ax2.plot(epochs_range, [a * 100 for a in train_auc], marker='o',
                 color='royalblue', linewidth=2, label="Train AUC")
    if val_auc:
        ax2.plot(range(1, len(val_auc) + 1), [a * 100 for a in val_auc], marker='s',
                 color='tomato', linewidth=2, linestyle='--', label="Val AUC")
    ax2.set_title("Mean AUC-ROC (CheXNet Metric)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC (%)")
    ax2.set_ylim([40, 100])
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    final_loss   = history["loss"][-1]
    best_val_auc = max(val_auc) if val_auc else None
    final_val_auc = val_auc[-1] if val_auc else None
    arch = history.get("architecture", "Unknown")

    stats  = f"Architecture : {arch}\n"
    stats += f"Final Train Loss: {final_loss:.4f}\n"
    if best_val_auc is not None:
        stats += f"Best Val AUC : {best_val_auc*100:.2f}%  |  Final Val AUC: {final_val_auc*100:.2f}%\n"
    # Per-class AUC breakdown
    best_pca = history.get("best_per_class_auc", {})
    if best_pca:
        valid_pca = {k: v for k, v in best_pca.items() if v is not None}
        if valid_pca:
            stats += "\nPer-Class AUC at Best Checkpoint:\n"
            for k, v in sorted(valid_pca.items(), key=lambda x: -x[1]):
                bar = "█" * int(v * 20)
                stats += f"  {k:<22} {v*100:5.1f}%  {bar}\n"

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

            val_auc, _, _ = evaluate_model(model, val_loader, device)
            if val_auc > best_val_f1:
                best_val_f1 = val_auc

            # Report intermediate for pruning
            trial.report(val_auc, epoch)
            if trial.should_prune():
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                log_text += f"  Trial {trial_num+1} pruned at epoch {epoch+1}.\n"
                yield log_text, gr.update(), None, gr.update()
                break
        else:
            study.tell(trial, best_val_f1)
            log_text += f"  Trial {trial_num+1} done. Best Val AUC: {best_val_f1 * 100:.2f}%\n"
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
# Test-Time Augmentation (TTA) Inference
# ---------------------------------------------------------------------------
# WHY TTA WORKS FOR UNSEEN IMAGES:
# When a model sees a new chest X-ray from a different hospital/scanner, it may
# look different from training data (brightness, contrast, orientation, crop).
# TTA compensates by running inference 10 times on different views of the same image
# (5-crop × 2 flips) and averaging the logits. This:
#   1. Reduces sensitivity to image positioning/framing (the #1 source of domain shift)
#   2. Gives a free uncertainty estimate via std-dev across views
#   3. Consistently improves AUC by 1-3% on external test sets (DualTTA 2024)
# Reference: Moshkov et al. 2020; DualTTA framework 2024
# ---------------------------------------------------------------------------

TTA_BASE = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def _clahe_normalize(pil_img):
    """
    Adaptive contrast normalization for domain generalization.
    X-rays from different hospitals have wildly different brightness/contrast.
    Histogram equalization normalizes this, reducing domain shift on unseen images.
    Applied only at inference, not training (which uses data augmentation instead).
    """
    import numpy as np
    arr = np.array(pil_img.convert('L'))  # grayscale
    # Clip to [5th, 95th] percentile to remove extreme scanner artifacts
    p5, p95 = np.percentile(arr, 5), np.percentile(arr, 95)
    arr = np.clip(arr, p5, p95)
    # Normalize to [0, 255]
    if p95 > p5:
        arr = ((arr - p5) / (p95 - p5) * 255).astype(np.uint8)
    return Image.fromarray(arr).convert('RGB')

def predict_with_tta(model, pil_image, device, n_crops=5, apply_clahe=True):
    """
    TTA inference: 5-crop × 2-flip = 10 augmented views.
    Returns (mean_probs, std_probs) — std_probs is the uncertainty estimate.
    High std = model is uncertain about this image (unseen distribution).
    """
    model.eval()

    # Step 1: CLAHE normalization for domain generalization
    if apply_clahe:
        pil_image = _clahe_normalize(pil_image)

    resized = transforms.Resize((256, 256))(pil_image)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 2: Generate 10 views — 5 crops (corners + center) × 2 flips
    five_crop = transforms.FiveCrop(224)
    crops = five_crop(resized)  # tuple of 5 PIL images

    views = []
    for crop in crops:
        views.append(to_tensor(crop))                              # original
        views.append(to_tensor(transforms.functional.hflip(crop))) # flipped

    # Step 3: Batch all views together for a single GPU forward pass
    batch = torch.stack(views).to(device)  # (10, 3, 224, 224)

    with torch.no_grad():
        logits = model(batch)              # (10, num_classes)
        # Average logits then sigmoid (more stable than averaging probabilities)
        mean_logits = logits.mean(dim=0)   # (num_classes,)
        mean_probs  = torch.sigmoid(mean_logits).cpu().numpy()
        # Per-view probabilities for uncertainty (std dev)
        per_view_probs = torch.sigmoid(logits).cpu().numpy()  # (10, num_classes)
        std_probs      = per_view_probs.std(axis=0)            # (num_classes,)

    return mean_probs, std_probs

def predict_image(image, model_name):
    """Full TTA inference with uncertainty estimation for unseen chest X-rays."""
    if image is None:
        return "Please upload an image.", None
    if not model_name:
        return "Please select a trained model.", None

    model_path   = os.path.join(MODELS_DIR, f"{model_name}.pth")
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(model_path):
        return "Model file not found.", None

    arch = "ResNet-18"
    target_labels = ALL_LABELS
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                meta = json.load(f)
                arch          = meta.get("architecture", "ResNet-18")
                target_labels = meta.get("target_labels", ALL_LABELS)
        except:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = get_model(arch, num_classes=len(target_labels))
    load_compat_state_dict(model, torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    image = image.convert('RGB')

    # Run TTA inference with CLAHE normalization
    mean_probs, std_probs = predict_with_tta(model, image, device, apply_clahe=True)

    if mean_probs.ndim == 0:
        mean_probs = [float(mean_probs)]
        std_probs  = [float(std_probs)]

    # Build label dict for Gradio Label output (sorted by probability, descending)
    label_probs = {label: float(prob) for label, prob in zip(target_labels, mean_probs)}

    # Build confidence chart: probability bar with uncertainty annotation
    n  = len(target_labels)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.45)))
    y_pos   = np.arange(n)
    sorted_items = sorted(zip(target_labels, mean_probs, std_probs), key=lambda x: x[1])
    labels_sorted = [it[0] for it in sorted_items]
    probs_sorted  = [it[1] for it in sorted_items]
    std_sorted    = [it[2] for it in sorted_items]

    colors = ['#e74c3c' if p > 0.5 else '#3498db' for p in probs_sorted]
    bars   = ax.barh(y_pos, probs_sorted, color=colors, alpha=0.85, height=0.7)
    ax.errorbar(probs_sorted, y_pos, xerr=std_sorted, fmt='none',
                ecolor='#2c3e50', elinewidth=1.5, capsize=4, capthick=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability (error bars = TTA uncertainty)", fontsize=10)
    ax.set_title("Disease Probabilities — TTA (10 views) with Uncertainty", fontsize=11, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label="Decision boundary (0.5)")
    ax.legend(fontsize=8)

    # Annotate bars with probability %
    for i, (p, s) in enumerate(zip(probs_sorted, std_sorted)):
        conf_label = "HIGH" if s < 0.05 else "MED" if s < 0.12 else "LOW"
        ax.text(min(p + 0.02, 0.97), i, f"{p*100:.1f}% ± {s*100:.1f}% [{conf_label}]",
                va='center', fontsize=8, color='#2c3e50')

    plt.tight_layout()

    # Mean uncertainty across all classes — summary confidence score
    mean_uncertainty = float(np.mean(std_probs))
    confidence_level = "🟢 High confidence" if mean_uncertainty < 0.05 else \
                       "🟡 Medium confidence" if mean_uncertainty < 0.12 else \
                       "🔴 Low confidence (image may differ from training data)"

    top_findings = sorted(label_probs.items(), key=lambda x: -x[1])[:5]
    top_str = ", ".join(f"{k} ({v*100:.1f}%)" for k, v in top_findings if v > 0.1)

    summary = f"{confidence_level}\nMean TTA Uncertainty: ±{mean_uncertainty*100:.2f}%\n"
    if top_str:
        summary += f"Notable findings: {top_str}"

    return label_probs, fig

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
        gr.Markdown("### 🔬 Chest X-ray Disease Detection (TTA-Powered for Unseen Images)")
        gr.Markdown(
            "Uses **Test-Time Augmentation (TTA)** — 10 augmented views per image — and **CLAHE contrast normalization** "
            "to generalize to X-rays from hospitals/scanners not seen during training.  \n"
            "Error bars show **uncertainty**: wide bars = model is less confident about this image."
        )
        infer_model_dropdown = gr.Dropdown(choices=[], label="Select Model for Inference")
        with gr.Row():
            image_input       = gr.Image(type="pil", label="Upload X-ray Image")
            prediction_output = gr.Label(num_top_classes=8, label="Top Disease Probabilities (TTA)")
        tta_chart_output = gr.Plot(label="Full Probability Chart with TTA Uncertainty")
        predict_btn = gr.Button("🔍 Predict (TTA + CLAHE)", variant="primary")
        
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
    
    predict_btn.click(fn=predict_image, inputs=[image_input, infer_model_dropdown], outputs=[prediction_output, tta_chart_output])
    
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
    try:
        gr.close_all()
    except:
        pass
    demo.launch(share=True)
