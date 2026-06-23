# LungSight: NIH Chest X-ray Multi-Label Classification 🫁📊

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-FF5A5F?style=for-the-badge&logo=fastapi&logoColor=white)](https://gradio.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Optuna](https://img.shields.io/badge/Optuna-283593?style=for-the-badge&logo=googlekeep&logoColor=white)](https://optuna.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

LungSight is an end-to-end deep learning application featuring a Web GUI to train, evaluate, and perform inference on chest X-ray images. It is built using **PyTorch** for model training and **Gradio** for an interactive, web-based user interface. The project utilizes the **NIH Chest X-ray 14** dataset to classify 14 distinct thoracic pathologies alongside a "No Finding" status.

---

## 📖 Table of Contents

- [🌟 Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [🚀 Installation & Local Setup](#-installation--local-setup)
- [🔬 Web App Walkthrough](#-web-app-walkthrough)
- [📊 Technical Architecture & Metrics](#-technical-architecture--metrics)
- [🚀 Cloud-Ready (Colab)](#-cloud-ready-colab)

---

## 🌟 Key Features

### 1. Multi-Label Pathological Classification
Detects 14 medical findings: *Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax*, and *No Finding*.

### 2. Diverse Model Architectures
- **ResNet-18:** Fine-tuned pretrained network with customized dropout and classification layers.
- **MobileNet-V2:** Lightweight backbone, ideal for low-latency scenarios.
- **DenseNet-121:** Highly efficient feature-reuse backbone, particularly powerful for medical imaging.
- **Simple CNN:** A custom, lightweight convolutional neural network built from scratch.
- **Hybrid Model:** Ensembled architecture combining concatenated features from ResNet-18, MobileNet-V2, and DenseNet-121 backbones.

### 3. Gradio Web Interface
- **Train Model Tab:** Select base architectures, define epochs and batch sizes, select target diseases, and enable balanced sampling.
- **Performance Tab:** Interactively refresh and select trained models to visualize loss and F1-Score training curves.
- **Inference Tab:** Upload any chest X-ray image (PNG/JPG) to get top-5 pathology predictions with confidence probabilities.
- **Hyperparameter Tuning Tab:** Run automated search studies powered by **Optuna** to optimize optimizer types, architectures, dropouts, and learning rates.

### 4. Class Imbalance & Optimization Strategies
- **Weighted BCE Loss:** Implements Binary Cross-Entropy with Logits loss weighted by positive class frequencies ($pos\_weight = \min(\frac{N - P_i}{P_i}, 10.0)$) to handle severe class imbalance.
- **Balanced Sampling:** Restricts dataset size per epoch to a fixed sample size per disease to prevent dominant pathologies from skewing learning.
- **Differential Learning Rates:** Pretrained backbones are fine-tuned slowly (learning rate `1e-5`) while new classifiers are trained rapidly (learning rate `1e-3`).
- **Checkpoint Resiliency:** Automatically saves and resumes training from epoch checkpoints.

---

## 📂 Project Structure

```text
LungSight/
├── app.py                     # Main application entry point (PyTorch model & Gradio Web UI)
├── app.ipynb                  # Google Colab Jupyter Notebook for cloud GPU training
├── Data_Entry_2017.csv        # NIH Chest X-ray dataset metadata file (images, demographics, labels)
├── BBox_List_2017.csv         # Bounding box coordinates for pathology localization
├── train_val_list.txt         # Split list containing official training/validation image indexes
├── test_list.txt              # Split list containing official testing image indexes
├── models/                    # Saved model state dicts (*.pth) and training logs (*_metrics.json)
│   └── best_optuna_params.json # Saved hyperparameter study results
├── .gitignore                 # Configured to ignore datasets, model weights, PDFs, and cache
└── images_001/ - images_012/  # Dataset folders containing chest X-ray images (untracked)
```

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
- Python 3.8 or higher is recommended.
- A CUDA-compatible GPU is highly recommended for model training, though CPU fallback is fully supported.

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install torch torchvision pandas gradio optuna matplotlib pillow
```

### 3. Run the Web Application
Launch the local Gradio server:
```bash
python app.py
```
After running, the console will output a local URL (typically `http://127.0.0.1:7860`). Open this address in your web browser to access the interface.

---

## 🔬 Web App Walkthrough

### 🏋️ Tab 1: Train Model
*   **Model Name:** Provide a custom name for your model.
*   **Base Architecture:** Select from ResNet-18, MobileNet-V2, DenseNet-121, Simple CNN, or Hybrid Model.
*   **Hyperparameters:** Set Epochs and Batch Size.
*   **Balanced Sampling:** Toggle balanced sampling and choose the target sample size (images per disease category).
*   **Target Diseases:** Select specific disease categories to train on a targeted subset, or clear selections to train on all images.
*   Click **Train Model** to run. Live terminal training updates will display in the output box.

### 📈 Tab 2: Performance
*   Select a trained model from the dropdown.
*   View training loss and validation F1-Score progress charts.
*   View final performance statistics including architecture details and final scores.

### 🔍 Tab 3: Inference
*   Select a trained model for inference.
*   Drag-and-drop or upload a chest X-ray image.
*   Click **Predict Disease** to view a bar chart of the top-5 disease predictions with classification probabilities.

### ⚙️ Tab 4: Hyperparameter Tuning (Optuna)
*   Configure the number of trials and epochs per trial.
*   Select search space criteria: architectures, optimizers (Adam, AdamW, SGD, RMSprop, Adagrad), and batch sizes.
*   Click **Start Hyperparameter Optimization** to start the search. The app will log trial details and export the best parameters to `models/best_optuna_params.json`.

---

## 📊 Technical Architecture & Metrics

*   **Primary Metric:** Micro-averaged **F1-Score** is used as the primary evaluation metric. Medical classification accuracy is easily inflated due to the high frequency of negative ("No Finding") tags, making F1-Score a far more reliable indicator of pathological detection performance.
*   **Optimization:** The BCE loss weights positive predictions by positive-to-negative class ratios to penalize false negatives more heavily on rare disease labels.
*   **Model Checkpointing:** If training is interrupted, restarting with the same model name will automatically locate `models/<name>_checkpoint.pth` and resume from the last completed epoch.

---

## 🚀 Cloud-Ready (Colab)

LungSight includes `app.ipynb` configured with Google Drive mounts, dependency installation scripts, and Gradio share configuration to easily train on high-performance cloud GPUs.
