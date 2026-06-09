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

try:
    import optuna
except ImportError:
    optuna = None

DATA_DIR = r"d:\NIH_Chest_Xray_Project"
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

def load_compat_state_dict(model, state_dict):
    """
    Loads state_dict into model, dynamically adapting between older format (nn.Linear)
    and newer format (nn.Sequential(nn.Dropout, nn.Linear)).
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
    model.load_state_dict(new_state, strict=False)

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

def train_model(model_name, architecture, epochs, batch_size, selected_diseases, balanced_sampling, balanced_size, progress=gr.Progress()):
    if not model_name:
        model_name = f"model_{int(time.time())}"
        
    log_text = f"Initializing training for model: {model_name} (Arch: {architecture})...\n"
    yield log_text, gr.update()
        
    df = pd.read_csv(CSV_PATH)
    
    # Filter dataset to only include images that actually exist on disk
    df = df[df['Image Index'].isin(all_image_paths.keys())].reset_index(drop=True)
    
    if selected_diseases:
        if balanced_sampling:
            sample_size = int(balanced_size)
            log_text += f"Using Balanced Sampling: {sample_size} images for each of the {len(selected_diseases)} selected diseases...\n"
            yield log_text, gr.update()
            dfs = []
            for d in selected_diseases:
                d_df = df[df['Finding Labels'].str.contains(d)].copy()
                sample_n = min(len(d_df), sample_size)
                if sample_n > 0:
                    dfs.append(d_df.sort_values(['Patient ID', 'Follow-up #']).head(sample_n))
            if dfs:
                df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
            else:
                log_text += "No images found for the selected diseases.\n"
                yield log_text, update_model_dropdown()
                return
        else:
            log_text += f"Using all matching images for the selected diseases...\n"
            yield log_text, gr.update()
            mask = df['Finding Labels'].apply(lambda x: any(d in x for d in selected_diseases))
            df = df[mask].reset_index(drop=True)
    else:
        log_text += f"No diseases selected. Training on the entire dataset of {len(df)} images...\n"
        yield log_text, gr.update()
        df = df.reset_index(drop=True)
        
    msg = f"Training on {len(df)} images for {epochs} epochs with batch size {batch_size}."
    print(msg)
    log_text += msg + "\n"
    yield log_text, gr.update()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = NIHDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_text += f"Using device: {device}\n"
    yield log_text, gr.update()
    
    model = get_model(architecture).to(device)
    
    # Ensure ALL parameters are fully trainable (requires_grad = True) so the backbone adapts to X-ray details
    for param in model.parameters():
        param.requires_grad = True
        
    # Configure optimizer with differential learning rates for pretrained models
    if architecture == "Simple CNN":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif architecture == "Hybrid Model":
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": 1e-5},
            {"params": classifier_params, "lr": 1e-3}
        ])
    else:
        # Pretrained models: ResNet-18, MobileNet-V2, DenseNet-121
        # Set up differential learning rates: 1e-5 for backbone (slow, stable fine-tuning), 1e-3 for classifier (fast learning)
        backbone_params = []
        classifier_params = []
        
        # Identify the classifier layer name
        classifier_name = "fc" if architecture == "ResNet-18" else "classifier"
        
        for name, param in model.named_parameters():
            if classifier_name in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
                
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": 1e-5},
            {"params": classifier_params, "lr": 1e-3}
        ])
        
    # Calculate positive weights for highly imbalanced datasets to boost F1-Score (capped at 10.0 for stability)
    pos_counts = torch.zeros(len(ALL_LABELS))
    for labels_str in df['Finding Labels']:
        labels = labels_str.split('|')
        for i, l in enumerate(ALL_LABELS):
            if l in labels:
                pos_counts[i] += 1
                
    pos_weight = torch.ones(len(ALL_LABELS))
    for i in range(len(ALL_LABELS)):
        if pos_counts[i] > 0:
            pos_weight[i] = min((len(df) - pos_counts[i]) / pos_counts[i], 10.0)
            
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Check for existing checkpoint to resume training
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint.pth")
    start_epoch = 0
    history = {"loss": [], "accuracy": [], "architecture": architecture}
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            log_text += f"Found existing checkpoint for '{model_name}'. Resuming training from epoch {start_epoch + 1}...\n"
            yield log_text, gr.update()
        except Exception as e:
            log_text += f"Failed to load checkpoint ({e}). Starting training from scratch.\n"
            yield log_text, gr.update()
            
    for epoch in progress.tqdm(range(start_epoch, int(epochs)), desc="Epochs"):
        # Put entire model in train mode to allow both backbone fine-tuning and dropout regularization
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0
        
        for inputs, labels in progress.tqdm(dataloader, desc="Batches"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            # Calculate standard average binary accuracy per label (proper multi-label metric)
            correct += (preds == labels).float().sum().item()
            total += inputs.size(0) * len(ALL_LABELS)
            
            # Accumulate TP, FP, FN (Micro F1 metrics) for positive class disease detection
            tp += ((preds == 1) & (labels == 1)).float().sum().item()
            fp += ((preds == 1) & (labels == 0)).float().sum().item()
            fn += ((preds == 0) & (labels == 1)).float().sum().item()
            
        epoch_loss = running_loss / (total / len(ALL_LABELS))
        
        # Calculate Micro F1-Score (uninflated disease detection metric)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        epoch_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_f1)  # Use F1-Score as the primary Accuracy!
        
        msg = f"Epoch [{epoch+1}/{epochs}] -\n"
        msg += f"  -> Current Training Loss: {epoch_loss:.4f}\n"
        msg += f"  -> Current Training Accuracy (F1-Score): {epoch_f1 * 100:.2f}%\n"
        print(msg)
        log_text += msg + "\n"
        yield log_text, gr.update()
        
        # Save epoch checkpoint in case environment disconnects
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
        
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(history, f)
        
    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            print(f"Error removing checkpoint: {e}")
            
    log_text += f"Training complete! Model saved as {model_name}.pth\n"
    yield log_text, update_model_dropdown()

def update_model_dropdown():
    models_list = [f.replace(".pth", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    return gr.Dropdown(choices=models_list, label="Select Model")

def get_performance(model_name):
    if not model_name:
        return None, "No model selected."
        
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(metrics_path):
        return None, "No metrics found for this model."
        
    with open(metrics_path, "r") as f:
        history = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = range(1, len(history["loss"]) + 1)
    
    ax1.plot(epochs, history["loss"], marker='o', color='blue', linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Backward compatibility: Check if this was trained under the old scheme
    if "f1_score" in history:
        # Plot F1-score as primary line, and show binary accuracy as comparison
        ax2.plot(epochs, [f * 100 for f in history["f1_score"]], marker='o', color='green', linewidth=2, label="Accuracy (F1-Score)")
        ax2.plot(epochs, [a * 100 for a in history["accuracy"]], marker='x', linestyle='--', color='gray', alpha=0.6, label="Binary Accuracy (Old)")
        final_acc = history["f1_score"][-1]
    else:
        # New scheme: history["accuracy"] is F1-score itself!
        ax2.plot(epochs, [a * 100 for a in history["accuracy"]], marker='o', color='green', linewidth=2, label="Accuracy (F1-Score)")
        final_acc = history["accuracy"][-1]
        
    ax2.set_title("Training Accuracy (F1-Score)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    final_loss = history["loss"][-1]
    arch = history.get("architecture", "Unknown")
    stats = f"Architecture: {arch} | Final Loss: {final_loss:.4f} | Final Accuracy (F1-Score): {final_acc*100:.2f}%"
    
    return fig, stats

def evaluate_model(model, dataloader, device):
    model.eval()
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            tp += ((preds == 1) & (labels == 1)).float().sum().item()
            fp += ((preds == 1) & (labels == 0)).float().sum().item()
            fn += ((preds == 0) & (labels == 1)).float().sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

def optuna_tune_model(num_trials, epochs_per_trial, selected_architectures, selected_optimizers, selected_batch_sizes, selected_diseases, balanced_sampling, balanced_size, progress=gr.Progress()):
    if optuna is None:
        yield "Error: Optuna library is not installed. Please run 'pip install optuna' to enable hyperparameter tuning.", gr.update(), None
        return
        
    if not selected_architectures:
        yield "Error: Please select at least one base architecture to search over.", gr.update(), None
        return
        
    if not selected_optimizers:
        yield "Error: Please select at least one optimizer to search over.", gr.update(), None
        return
        
    if not selected_batch_sizes:
        yield "Error: Please select at least one batch size to search over.", gr.update(), None
        return

    log_text = f"Starting Optuna Hyperparameter Optimization Study...\n"
    log_text += f"Config: Trials={num_trials}, Epochs per Trial={epochs_per_trial}\n"
    log_text += f"Architectures to evaluate: {selected_architectures}\n"
    log_text += f"Optimizers to evaluate: {selected_optimizers}\n"
    log_text += f"Batch Sizes to evaluate: {selected_batch_sizes}\n"
    yield log_text, gr.update(), None
    
    df = pd.read_csv(CSV_PATH)
    df = df[df['Image Index'].isin(all_image_paths.keys())].reset_index(drop=True)
    
    if selected_diseases:
        if balanced_sampling:
            sample_size = int(balanced_size)
            log_text += f"Using Balanced Sampling: {sample_size} images per selected disease...\n"
            yield log_text, gr.update(), None
            dfs = []
            for d in selected_diseases:
                d_df = df[df['Finding Labels'].str.contains(d)].copy()
                sample_n = min(len(d_df), sample_size)
                if sample_n > 0:
                    dfs.append(d_df.sort_values(['Patient ID', 'Follow-up #']).head(sample_n))
            if dfs:
                df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
            else:
                yield log_text + "No images found for the selected diseases.\n", gr.update(), None
                return
        else:
            log_text += "Using all matching images for target diseases...\n"
            yield log_text, gr.update(), None
            mask = df['Finding Labels'].apply(lambda x: any(d in x for d in selected_diseases))
            df = df[mask].reset_index(drop=True)
    else:
        log_text += f"No target diseases specified. Running optimization on full dataset of {len(df)} images...\n"
        yield log_text, gr.update(), None
        
    if len(df) < 5:
        yield log_text + f"Error: Dataset size too small ({len(df)} images). Please select more diseases or increase sampling size.\n", gr.update(), None
        return
        
    # Train/Validation Split (80% / 20%)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    log_text += f"Split details: {len(train_df)} training samples, {len(val_df)} validation samples.\n"
    yield log_text, gr.update(), None
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_text += f"Running on device: {device}\n"
    yield log_text, gr.update(), None
    
    # Calculate loss weights
    pos_counts = torch.zeros(len(ALL_LABELS))
    for labels_str in train_df['Finding Labels']:
        labels = labels_str.split('|')
        for i, l in enumerate(ALL_LABELS):
            if l in labels:
                pos_counts[i] += 1
                
    pos_weight = torch.ones(len(ALL_LABELS))
    for i in range(len(ALL_LABELS)):
        if pos_counts[i] > 0:
            pos_weight[i] = min((len(train_df) - pos_counts[i]) / pos_counts[i], 10.0)
            
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Optuna study Setup
    study = optuna.create_study(direction="maximize")
    
    for trial_idx in range(int(num_trials)):
        trial = study.ask()
        trial_num = trial.number
        
        arch = trial.suggest_categorical("architecture", selected_architectures)
        opt_name = trial.suggest_categorical("optimizer", selected_optimizers)
        b_size = int(trial.suggest_categorical("batch_size", selected_batch_sizes))
        dropout_val = trial.suggest_float("dropout", 0.1, 0.5)
        
        classifier_lr = trial.suggest_float("classifier_lr", 1e-4, 1e-2, log=True)
        if arch != "Simple CNN":
            backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 1e-4, log=True)
        else:
            backbone_lr = 0.0
            
        t_log = f"\n--- [Trial {trial_num+1}/{num_trials}] ---\n"
        t_log += f"Parameters:\n"
        t_log += f"  - Architecture: {arch}\n"
        t_log += f"  - Optimizer: {opt_name}\n"
        t_log += f"  - Batch Size: {b_size}\n"
        t_log += f"  - Dropout: {dropout_val:.2f}\n"
        t_log += f"  - Classifier LR: {classifier_lr:.2e}\n"
        if arch != "Simple CNN":
            t_log += f"  - Backbone LR: {backbone_lr:.2e}\n"
        
        log_text += t_log + "Training and validating model...\n"
        yield log_text, gr.update(), None
        
        train_dataset = NIHDataset(train_df, transform=transform)
        val_dataset = NIHDataset(val_df, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)
        
        model = get_model(arch)
        
        # Inject dropout
        if arch == "ResNet-18":
            model.fc = nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Linear(model.fc[1].in_features, len(ALL_LABELS))
            )
        elif arch == "MobileNet-V2":
            model.classifier[0] = nn.Dropout(dropout_val)
        elif arch == "DenseNet-121":
            model.classifier[0] = nn.Dropout(dropout_val)
        elif arch == "Simple CNN":
            model.classifier[0] = nn.Dropout(dropout_val)
        elif arch == "Hybrid Model":
            model.classifier[0] = nn.Dropout(dropout_val)
            
        model.to(device)
        
        for param in model.parameters():
            param.requires_grad = True
            
        if arch == "Simple CNN":
            params = model.parameters()
            if opt_name == "Adam":
                optimizer = torch.optim.Adam(params, lr=classifier_lr)
            elif opt_name == "AdamW":
                optimizer = torch.optim.AdamW(params, lr=classifier_lr)
            elif opt_name == "SGD":
                optimizer = torch.optim.SGD(params, lr=classifier_lr, momentum=0.9)
            elif opt_name == "RMSprop":
                optimizer = torch.optim.RMSprop(params, lr=classifier_lr)
            elif opt_name == "Adagrad":
                optimizer = torch.optim.Adagrad(params, lr=classifier_lr)
            else:
                optimizer = torch.optim.Adam(params, lr=classifier_lr)
        else:
            backbone_params = []
            classifier_params = []
            classifier_layer_name = "fc" if arch == "ResNet-18" else "classifier"
            for name, param in model.named_parameters():
                if classifier_layer_name in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
                    
            if opt_name == "Adam":
                optimizer = torch.optim.Adam([
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": classifier_params, "lr": classifier_lr}
                ])
            elif opt_name == "AdamW":
                optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": classifier_params, "lr": classifier_lr}
                ])
            elif opt_name == "SGD":
                optimizer = torch.optim.SGD([
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": classifier_params, "lr": classifier_lr}
                ], momentum=0.9)
            elif opt_name == "RMSprop":
                optimizer = torch.optim.RMSprop([
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": classifier_params, "lr": classifier_lr}
                ])
            elif opt_name == "Adagrad":
                optimizer = torch.optim.Adagrad([
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": classifier_params, "lr": classifier_lr}
                ])
            else:
                optimizer = torch.optim.Adam([
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": classifier_params, "lr": classifier_lr}
                ])
                
        best_val_f1 = 0.0
        for epoch in progress.tqdm(range(int(epochs_per_trial)), desc=f"Trial {trial_num+1} Epochs"):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            val_f1 = evaluate_model(model, val_loader, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                
        study.tell(trial, best_val_f1)
        log_text += f"Trial {trial_num+1} Completed. Best Validation F1-Score: {best_val_f1 * 100:.2f}%\n"
        yield log_text, gr.update(), None
        
    best_trial = study.best_trial
    log_text += f"\n=====================================\n"
    log_text += f"OPTIMIZATION COMPLETE!\n"
    log_text += f"Best Trial Number: {best_trial.number + 1}\n"
    log_text += f"Best Validation F1-Score: {best_trial.value * 100:.2f}%\n"
    log_text += f"Best Parameters:\n"
    for k, v in best_trial.params.items():
        log_text += f"  - {k}: {v}\n"
    log_text += f"=====================================\n"
    
    # Generate History Plot
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
        
    results_summary = f"Best Validation F1-Score: {best_trial.value * 100:.2f}%\n\nBest Hyperparameters:\n"
    for k, v in best_trial.params.items():
        if isinstance(v, float):
            results_summary += f"{k}: {v:.2e}\n" if v < 1e-3 else f"{k}: {v:.4f}\n"
        else:
            results_summary += f"{k}: {v}\n"
            
    yield log_text, fig, results_summary

def predict_image(image, model_name):
    if image is None:
        return "Please upload an image."
    if not model_name:
        return "Please select a trained model from Performance section."
        
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
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
    model = get_model(arch)
    load_compat_state_dict(model, torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        if probs.ndim == 0:
            probs = [probs.item()]
        
    results = {label: float(prob) for label, prob in zip(ALL_LABELS, probs)}
    return results

with gr.Blocks() as demo:
    gr.Markdown("# NIH Chest X-ray Model Trainer & Predictor")
    gr.Markdown("Switch between default and dark mode using your browser's theme settings, or use Gradio's built-in toggle.")
    
    with gr.Tab("1. Train Model"):
        with gr.Row():
            model_name_input = gr.Textbox(label="Model Name (optional)", placeholder="my_resnet")
            architecture_input = gr.Dropdown(choices=["ResNet-18", "MobileNet-V2", "DenseNet-121", "Simple CNN", "Hybrid Model"], value="ResNet-18", label="Base Architecture")
        with gr.Row():
            epochs_input = gr.Slider(minimum=1, maximum=50, value=3, step=1, label="Epochs")
            batch_size_input = gr.Slider(minimum=4, maximum=128, value=16, step=4, label="Batch Size")
        with gr.Row():
            balanced_input = gr.Checkbox(label="Enable Balanced Sampling", value=True)
            balanced_size_input = gr.Dropdown(choices=["50", "100", "150", "200"], value="100", label="Balanced Sample Size (images per selected disease)", visible=True)
        with gr.Row():
            with gr.Column():
                diseases_input = gr.CheckboxGroup(choices=ALL_LABELS, label="Target Diseases (Select at least one for Balanced Sampling)", info="Select specific diseases to train on a targeted subset.")
                with gr.Row():
                    select_all_btn = gr.Button("Select All")
                    deselect_all_btn = gr.Button("Clear Selection")
            
        train_btn = gr.Button("Train Model", variant="primary")
        train_output = gr.Textbox(label="Live Terminal Output", lines=10, max_lines=20)
        
    with gr.Tab("2. Performance"):
        refresh_btn = gr.Button("Refresh Models")
        model_dropdown = gr.Dropdown(choices=[], label="Select Model")
        with gr.Row():
            perf_plot = gr.Plot(label="Performance Metrics")
            perf_stats = gr.Textbox(label="Final Stats")
            
    with gr.Tab("3. Inference"):
        infer_model_dropdown = gr.Dropdown(choices=[], label="Select Model for Inference")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload X-ray Image")
            prediction_output = gr.Label(num_top_classes=5, label="Disease Predictions")
        predict_btn = gr.Button("Predict Disease", variant="primary")
        
    with gr.Tab("4. Hyperparameter Tuning (Optuna)"):
        gr.Markdown("### Optimize hyper-parameters using Optuna. The dataset is split 80% for training and 20% for validation evaluation.")
        with gr.Row():
            with gr.Column():
                optuna_trials_input = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Number of Trials")
                optuna_epochs_input = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Epochs per Trial")
                optuna_archs_input = gr.CheckboxGroup(choices=["ResNet-18", "MobileNet-V2", "DenseNet-121", "Simple CNN", "Hybrid Model"], value=["ResNet-18", "MobileNet-V2"], label="Base Architectures to Search")
                optuna_opts_input = gr.CheckboxGroup(choices=["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"], value=["Adam", "AdamW", "SGD"], label="Optimizers to Search")
                optuna_batch_input = gr.CheckboxGroup(choices=["8", "16", "32"], value=["16", "32"], label="Batch Sizes to Search")
            with gr.Column():
                optuna_balanced_input = gr.Checkbox(label="Enable Balanced Sampling", value=True)
                optuna_balanced_size_input = gr.Dropdown(choices=["50", "100", "150", "200"], value="100", label="Balanced Sample Size (images per selected disease)")
                optuna_diseases_input = gr.CheckboxGroup(choices=ALL_LABELS, label="Target Diseases", info="Select specific diseases to train on a targeted subset.")
                with gr.Row():
                    optuna_select_all_btn = gr.Button("Select All")
                    optuna_deselect_all_btn = gr.Button("Clear Selection")
                    
        optuna_tune_btn = gr.Button("Start Hyperparameter Optimization", variant="primary")
        
        with gr.Row():
            optuna_log_output = gr.Textbox(label="Optimization Terminal Output", lines=10, max_lines=20)
            with gr.Column():
                optuna_plot_output = gr.Plot(label="Optimization History Graph")
                optuna_params_output = gr.Textbox(label="Best Hyperparameters Found", lines=8)
                
    def toggle_balanced_size(balanced):
        return gr.update(visible=balanced)
        
    balanced_input.change(fn=toggle_balanced_size, inputs=[balanced_input], outputs=[balanced_size_input])
    
    select_all_btn.click(fn=lambda: gr.update(value=ALL_LABELS), outputs=diseases_input)
    deselect_all_btn.click(fn=lambda: gr.update(value=[]), outputs=diseases_input)
    
    train_btn.click(
        fn=train_model, 
        inputs=[model_name_input, architecture_input, epochs_input, batch_size_input, diseases_input, balanced_input, balanced_size_input], 
        outputs=[train_output, model_dropdown]
    )
    
    def on_refresh():
        dropdown = update_model_dropdown()
        return dropdown, dropdown
        
    refresh_btn.click(fn=on_refresh, inputs=None, outputs=[model_dropdown, infer_model_dropdown])
    
    demo.load(fn=on_refresh, inputs=None, outputs=[model_dropdown, infer_model_dropdown])
    
    model_dropdown.change(fn=get_performance, inputs=[model_dropdown], outputs=[perf_plot, perf_stats])
    infer_model_dropdown.change(fn=lambda x: x, inputs=[infer_model_dropdown], outputs=[model_dropdown])
    
    predict_btn.click(fn=predict_image, inputs=[image_input, infer_model_dropdown], outputs=[prediction_output])
    
    optuna_balanced_input.change(fn=toggle_balanced_size, inputs=[optuna_balanced_input], outputs=[optuna_balanced_size_input])
    optuna_select_all_btn.click(fn=lambda: gr.update(value=ALL_LABELS), outputs=optuna_diseases_input)
    optuna_deselect_all_btn.click(fn=lambda: gr.update(value=[]), outputs=optuna_diseases_input)
    
    optuna_tune_btn.click(
        fn=optuna_tune_model,
        inputs=[
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
            optuna_params_output
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
