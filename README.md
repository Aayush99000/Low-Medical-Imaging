# Low-Medical-Imaging

This project tackles one of medical AI challenges: developing reliable diagnostic models when data is scarce. Through rigorous experimentation, we empirically validate whether transfer learning can overcome the "small data problem" that plagues specialized medical imaging applications. By comparing models trained from scratch against those leveraging pre-trained knowledge, we provide concrete evidence for best practices in low-resource clinical AI deployment.

# 🎯 Problem Statement

Medical AI development faces a critical data scarcity crisis. Creating effective diagnostic tools requires large, expertly annotated datasets, which are often impossible to obtain due to:

🔒 HIPAA privacy regulations limiting data sharing
💰 High annotation costs requiring medical expert time
🏥 Data rarity for specialized conditions and rare diseases

Training complex CNNs from scratch on small datasets leads to severe overfitting and unreliable performance on new patients. Transfer learning offers a promising solution, but requires rigorous empirical validation.

# 🔬 Project Overview

This project provides empirical validation of transfer learning efficacy in low-resource medical imaging scenarios through a controlled three-way comparison:
Three Model Architecture Comparison
ModelDescriptionPurposeModel A (Scratch)Simple CNN trained only on small datasetBaseline performanceModel B (ImageNet Transfer)ResNet-50 pre-trained on ImageNet, fine-tuned on medical dataGeneral feature transferModel C (Medical Transfer)Model pre-trained on broader medical corpusDomain-specific transfer
All models are evaluated under identical conditions to ensure fair comparison.

We utilize three distinct medical imaging datasets to simulate real-world transfer scenarios:  
**1. Chest X-ray Images (Pneumonia)**
Source: Guangzhou Women and Children's Medical Center  
Task: Pneumonia detection from chest radiographs  
Link: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**2. HAM10000 Dermatoscopic Images**
Source: Harvard Dataverse  
Task: Classification of common pigmented skin lesions  
Link: [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)

**3. CheXpert Dataset**
Source: Stanford ML Group  
Task: Multi-label chest radiograph classification  
Link: [Stanford ML Group](https://arxiv.org/abs/1901.07031)

**Data Strategy**

Training Set: Deliberately constrained to 100-200 labeled images per condition  
Test Set: Large, unbiased holdout for reliable evaluation  
Augmentation: Aggressive augmentation

# Dataset Description

**1.Chest X-Ray Images (Pneumonia)**  
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.  
<img width="1084" height="551" alt="Screenshot 2025-11-12 at 15 57 13" src="images/img1.png" />

**2.HAM10000 Dermatoscopic**
The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
<img width="1049" height="706" alt="Screenshot 2025-11-12 at 16 03 42" src="images/img2.png" />

**2.CheXpert**
CheXpert is a large public dataset of 224,000+ chest X-rays from 65,000+ patients collected at Stanford Hospital. The dataset includes labels for 14 thoracic pathologies (including Pneumonia, Edema, Atelectasis, Pleural Effusion, etc.) with four label types: positive, negative, uncertain, and unmentioned. It's widely used for developing and benchmarking automated chest X-ray interpretation models.
<img width="1049" height="706" alt="Screenshot 2025-11-12 at 16 03 42" src="images/img3.png" />

## 🛠️ PyTorch Pipeline: Training & Evaluation

This project uses **PyTorch** for building and training transfer learning models. The pipeline consists of:

1. **Data Preprocessing** (`image_preprocessing.py`) — Custom PyTorch Dataset class with built-in augmentation
2. **Model Architecture** (`models.py`) — Transfer learning models (VGG19, ResNet50, InceptionV3, DenseNet121)
3. **Training** (`train.py`) — Training loop with early stopping, checkpoint saving, and fine-tuning
4. **Evaluation** (`evaluate.py`) — Model evaluation with comprehensive metrics

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key requirements:**

- PyTorch (CPU/GPU): `torch`, `torchvision`
- Computer Vision: `opencv-python`, `Pillow`
- ML Tools: `scikit-learn`, `numpy`, `pandas`, `matplotlib`
- Utilities: `tqdm`

#### 2. Dataset Structure

Place your dataset in this structure:

```
dataset/
├── train/
│   ├── Normal/
│   │   ├── image1.jpeg
│   │   └── image2.jpeg
│   └── Pneumonia/
│       ├── image1.jpeg
│       └── image2.jpeg
├── val/
│   ├── Normal/
│   └── Pneumonia/
└── test/
    ├── Normal/
    └── Pneumonia/
```

#### 3. Train a Model

```python
from train import train_model

# Train VGG19 with transfer learning
results = train_model(
    model_name="vgg19",
    dataset_root="./dataset",
    epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    freeze_base=True,
    fine_tune_epochs=10,
    fine_tune_layers=50,
    checkpoint_dir="./checkpoints"
)
```

Or run via command line:

```bash
python train.py
```

**Supported models:**

- `vgg19` — VGG19 with 16 convolutional layers
- `resnet50` — ResNet50 (152 layers deep)
- `inceptionv3` — InceptionV3 with multi-scale convolutions
- `densenet121` — DenseNet121 with dense connections

**Training Options:**

- `freeze_base`: Freeze base model weights initially (True) or train from scratch (False)
- `fine_tune_epochs`: Number of epochs for fine-tuning unfrozen layers
- `fine_tune_layers`: Number of base model layers to unfreeze for fine-tuning

#### 4. Evaluate a Model

```python
from evaluate import evaluate_model, compare_models

# Evaluate single model
results = evaluate_model(
    model_path="./checkpoints/vgg19_20251120_120000_final.pt",
    dataset_root="./dataset",
    model_name="vgg19",
    split="test"
)

# Compare multiple models
model_paths = {
    "vgg19": "./checkpoints/vgg19_final.pt",
    "resnet50": "./checkpoints/resnet50_final.pt",
    "densenet121": "./checkpoints/densenet121_final.pt"
}
comparison = compare_models(model_paths, dataset_root, split="test")
```

Or run via command line:

```bash
python evaluate.py
```

### Key Features

#### PyTorch Dataset Class (`ChestXrayDataset`)

```python
from image_preprocessing import ChestXrayDataset, get_data_loaders

# Create dataset
dataset = ChestXrayDataset(
    dataset_root="./dataset",
    split="train",
    image_size=(224, 224),
    augment=True,
    normalize=True,
    augmentation_strength=0.5
)

# Or use convenience function for all splits
train_loader, val_loader, test_loader = get_data_loaders(
    dataset_root="./dataset",
    batch_size=32,
    image_size=(224, 224),
    num_workers=4
)
```

**Preprocessing Steps:**

- Load grayscale X-ray images
- Resize to target size (default: 224×224)
- Convert grayscale to RGB (3 channels)
- Apply data augmentation (rotation, flip, brightness, shift)
- Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#### Transfer Learning Architecture

Each model includes:

- Pre-trained base model (ImageNet weights)
- Adaptive average pooling
- Classification head:
  - Dense(512) → ReLU → BatchNorm → Dropout(0.5)
  - Dense(256) → ReLU → BatchNorm → Dropout(0.3)
  - Dense(num_classes)

#### Training Features

- **Early Stopping**: Stops training if validation loss doesn't improve for 10 epochs
- **Learning Rate Scheduling**: Reduces learning rate by 0.5x if validation loss plateaus
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Fine-tuning**: Unfreezes last N layers after initial training for domain adaptation
- **Mixed Precision** (optional): Support for mixed precision training on GPUs

#### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Project Structure

```
.
├── image_preprocessing.py    # PyTorch Dataset class & DataLoaders
├── models.py                 # Transfer learning model architectures
├── train.py                  # Training loop with early stopping
├── evaluate.py               # Model evaluation & comparison
├── data.py                   # Utility functions (if needed)
├── utils.py                  # Additional utilities
├── main.py                   # Entry point (optional)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── dataset/                  # Dataset directory (structure above)
└── checkpoints/              # Saved models & results
    ├── vgg19_YYYYMMDD_HHMMSS_final.pt
    ├── vgg19_YYYYMMDD_HHMMSS_results.json
    └── ...
```

### Example Workflow

```python
# 1. Prepare data
from image_preprocessing import get_data_loaders, get_dataset_statistics

stats = get_dataset_statistics("./dataset")
print(f"Dataset stats: {stats}")

train_loader, val_loader, test_loader = get_data_loaders(
    dataset_root="./dataset",
    batch_size=32
)

# 2. Build and train model
from train import train_model

results = train_model(
    model_name="vgg19",
    dataset_root="./dataset",
    epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    freeze_base=True,
    fine_tune_epochs=10
)

# 3. Evaluate model
from evaluate import evaluate_model

eval_results = evaluate_model(
    model_path=results["model_path"],
    dataset_root="./dataset",
    model_name="vgg19",
    split="test"
)

print(f"Test Accuracy: {eval_results['metrics']['accuracy']:.4f}")
print(f"Test F1-Score: {eval_results['metrics']['f1']:.4f}")
```

### Tips for Best Results

1. **Data Augmentation**: Use stronger augmentation for small datasets to prevent overfitting
2. **Learning Rate**: Start with 1e-4 for frozen base model, reduce to 1e-5 during fine-tuning
3. **Batch Size**: Use larger batches (32-64) for stable gradient estimates with limited data
4. **Early Stopping**: Monitor validation loss; stop if no improvement for 10+ epochs
5. **Fine-tuning**: Unfreeze last 20-50 layers after initial training for better accuracy
6. **GPU**: Use CUDA-enabled GPU for 10-20x faster training

### Troubleshooting

- **Out of Memory**: Reduce batch size, freeze more layers, or use gradient accumulation
- **Poor Validation Accuracy**: Increase fine-tuning epochs, adjust learning rate, add more augmentation
- **Slow Training**: Ensure PyTorch is using GPU (check `torch.cuda.is_available()`)
- **Import Errors**: Reinstall dependencies with `pip install -r requirements.txt --upgrade`
