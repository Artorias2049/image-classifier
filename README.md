# Vehicle Classification Model

This repository contains code for training and evaluating a deep learning model for vehicle classification.

## Project Structure

```
.
├── models/
│   └── models.py             # Model architecture definitions
├── preprocessing/
│   ├── clean_data.py         # Data cleaning utilities using CleanVision
│   └── preprocessor.py       # Image preprocessing utilities
├── utils/
│   ├── dataloader.py         # DataLoader with augmentations
│   ├── metrics.py            # Evaluation metrics
│   └── training_plots.py     # Utilities for generating training plots
├── train.py                  # Main training script
├── inference.py              # Inference script for model evaluation
├── export_onnx.py            # Script to export model to ONNX format
├── analysis.py               # Dataset analysis utilities
├── README.md
└── requirements.txt          # Package dependencies
```

## Setup and Installation

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset Preparation

The dataset should be organized in the following structure:
```
vehicle_dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Dataset Analysis

You can analyze your dataset statistics using the analysis script:

```
python analysis.py
```

This will generate:
- Class distribution analysis
- Image size statistics
- Aspect ratio distribution
- Color statistics
- Outlier detection

## Training the Model

1. Before training, run the preprocessing and data cleaning step:
   ```
   python -m preprocessing.preprocessor --input ./vehicle_dataset --output ./cleaned_dataset
   ```

2. Run the training script:
   ```
   python train.py
   ```

   Optional arguments:
   - `--model-name`: Model architecture to use (default: "efficientnet_b0")
   - `--num-epochs`: Number of training epochs (default: 100)
   - `--batch-size`: Batch size (default: 32)
   - `--learning-rate`: Initial learning rate (default: 0.001)
   - `--weight-decay`: Weight decay for optimizer (default: 1e-4)
   - `--dropout-rate`: Dropout rate (default: 0.6)
   - `--label-smoothing`: Label smoothing factor (default: 0.2)
   - `--patience`: Patience for early stopping (default: 10)
   - `--warmup-epochs`: Number of warmup epochs (default: 5)
   - `--no-mixup`: Disable mixup augmentation
   - `--no-preprocess`: Skip preprocessing steps
   - `--augmentation`: Augmentation strength ("none", "light", "medium", "strong")
   - `--image-size`: Target image size (default: 244)
   - `--input-dir`: Path to input dataset (default: "./vehicle_dataset")
   - `--output-dir`: Path to output dataset (default: "./cleaned_dataset")
   - `--logs-dir`: Path to save logs (default: "./train_logs")
   - `--plots-dir`: Path to save plots (default: "./plots")
   - `--num-workers`: Number of workers for data loading (default: 0)

3. Training outputs will be saved to:
   - `train_logs/`: Model checkpoints and TensorBoard logs
   - `plots/`: Training curves, confusion matrices, and other visualizations

4. Monitor training progress with TensorBoard:
   ```
   tensorboard --logdir=train_logs/tensorboard
   ```

## Inference and Evaluation

1. Evaluate the model on validation set:
   ```
   python inference.py
   ```

   Optional arguments:
   - `--model-path`: Path to the trained model (default: "train_logs/model_best.pth")
   - `--model-name`: Model architecture name (default: "efficientnet_b0")
   - `--num-classes`: Number of classes (default: 12)
   - `--dataset-path`: Path to the validation dataset (default: "./cleaned_dataset/val")
   - `--train-path`: Path to the training dataset (default: "./cleaned_dataset/train")
   - `--samples-per-class`: Number of samples per class to evaluate (default: 2)
   - `--output-dir`: Directory to save visualization outputs (default: "plots")
   - `--image-size`: Target image size (default: 244)
   - `--use-tta`: Enable test-time augmentation

2. The script will output:
   - Per-class accuracy
   - Comparison between standard inference and test-time augmentation
   - Overall performance metrics
   - Visualizations of predictions and confusion matrices

## Exporting to ONNX Format

1. To export the trained model to ONNX format:
   ```
   python export_onnx.py
   ```

   Optional arguments:
   - `--model-path`: Path to the trained model (default: "train_logs/model_best.pth")
   - `--output-path`: Path to save the ONNX model (default: "vehicle_classifier.onnx")
   - `--model-name`: Model architecture name (default: "efficientnet_b0")
   - `--num-classes`: Number of output classes (default: 12)
   - `--dataset-path`: Path to the training dataset for class names (default: "./cleaned_dataset/train")
   - `--classes-output`: Path to save the classes.txt file (default: "classes.txt")
   - `--no-verify`: Skip ONNX model verification

## Key Features

- **Data Analysis**: Automated analysis of dataset statistics and distributions
- **Data Cleaning**: Automatic removal of problematic images using CleanVision
- **Data Preprocessing**: Consistent image resizing with padding (244×244)
- **Advanced Augmentations**: RandomAugment, RandomErasing, Mixup
- **Model Architecture Options**: ResNet, EfficientNet, MobileNet
- **Training Optimizations**:
  - Mixed precision training
  - Gradient clipping
  - Learning rate scheduling with warmup
  - Early stopping
- **Performance Evaluation**:
  - Accuracy, Precision, Recall, F1-score
  - Per-class accuracy tracking
  - Confusion matrix visualization
- **Test-Time Augmentation** for improved inference
- **Visualization Tools**:
  - Training curves
  - Prediction examples
  - TTA comparison
  - Per-class performance

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- cleanvision
- numpy
- matplotlib
- scikit-learn
- opencv-python
- tensorboard
- onnx (for model export)
- onnxruntime (for verification)
- seaborn (for plotting)
- LaTeX (for report generation)

See `requirements.txt` for specific versions.
