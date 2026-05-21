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
    
    history = {"loss": [], "accuracy": [], "architecture": architecture}
    
    for epoch in progress.tqdm(range(int(epochs)), desc="Epochs"):
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
        
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    metrics_path = os.path.join(MODELS_DIR, f"{model_name}_metrics.json")
    
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(history, f)
        
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
        with open(metrics_path, "r") as f:
            meta = json.load(f)
            arch = meta.get("architecture", "ResNet-18")
        
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
    
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    results = {label: float(prob) for label, prob in zip(ALL_LABELS, probs)}
    return results

with gr.Blocks() as demo:
    gr.Markdown("# NIH Chest X-ray Model Trainer & Predictor")
    gr.Markdown("Switch between default and dark mode using your browser's theme settings, or use Gradio's built-in toggle.")
    
    with gr.Tab("1. Train Model"):
        with gr.Row():
            model_name_input = gr.Textbox(label="Model Name (optional)", placeholder="my_resnet")
            architecture_input = gr.Dropdown(choices=["ResNet-18", "MobileNet-V2", "DenseNet-121", "Simple CNN"], value="ResNet-18", label="Base Architecture")
        with gr.Row():
            epochs_input = gr.Slider(minimum=1, maximum=50, value=3, step=1, label="Epochs")
            batch_size_input = gr.Slider(minimum=4, maximum=128, value=16, step=4, label="Batch Size")
        with gr.Row():
            balanced_input = gr.Checkbox(label="Enable Balanced Sampling", value=True)
            balanced_size_input = gr.Dropdown(choices=["50", "100", "150", "200"], value="100", label="Balanced Sample Size (images per selected disease)", visible=True)
        with gr.Row():
            diseases_input = gr.CheckboxGroup(choices=ALL_LABELS, label="Target Diseases (Select at least one for Balanced Sampling)", info="Select specific diseases to train on a targeted subset.")
            
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
        
    def toggle_balanced_size(balanced):
        return gr.update(visible=balanced)
        
    balanced_input.change(fn=toggle_balanced_size, inputs=[balanced_input], outputs=[balanced_size_input])
    
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

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
