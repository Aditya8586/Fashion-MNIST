# Fashion MNIST CNN with Explainability Analysis

A comprehensive deep learning project demonstrating image classification on Fashion MNIST with integrated explainability techniques (SHAP and LIME) for model interpretation.

## Overview

This project builds and trains a convolutional neural network (CNN) on the Fashion MNIST dataset and compares two leading explainability frameworks to understand model predictions at the instance level.

### Key Components
- **Image Classifier**: CNN with 2 convolutional blocks, max pooling, dropout regularization
- **Explainability Methods**: SHAP (Shapley values) and LIME (local interpretable explanations)
- **Fidelity Evaluation**: Spearman correlation between SHAP and LIME, deletion fidelity, local R² scores

## Dataset

**Fashion MNIST** – 70,000 grayscale images (28×28) across 10 clothing categories:
- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Standard 60:10 train-test split

## Architecture

```
Input (28×28×1)
    ↓
Conv2D (32 filters, 3×3 kernel) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3 kernel) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Flatten → 1600 units
    ↓
Dense (128 units) + ReLU + Dropout (0.3)
    ↓
Dense (10 units) + Softmax
```

**Total Parameters**: ~225,000

## Installation

### Requirements
- Python 3.8+
- TensorFlow/Keras 2.10+
- NumPy, Matplotlib
- SHAP, LIME, SciPy, scikit-image

### Setup
```bash
pip install tensorflow numpy matplotlib shap lime scipy scikit-image
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook Fashion_MNIST.ipynb
```

### Main Workflow
1. Load and normalize Fashion MNIST data
2. Build and compile CNN model
3. Train on 60,000 training samples
4. Generate predictions on test set
5. Initialize SHAP and LIME explainers
6. Compare explanations via Spearman correlation
7. Evaluate fidelity metrics (deletion test, local R²)

## Results

### Model Performance
- Achieves high accuracy on Fashion MNIST test set
- Model summary with layer-wise parameter counts provided

### Explainability Comparison
- **SHAP vs. LIME Correlation**: Spearman rank correlation computed on aggregated pixel attributions
- **LIME Local Fidelity (R²)**: Measures how well the local surrogate model approximates the CNN
- **SHAP Deletion Fidelity**: Evaluates explanation faithfulness by measuring probability drop when top-20% attributed pixels are removed

## Example Output
```
Instance 0: spearman p = 0.459 (p=0.011)
Mean p: 0.459

LIME local fidelity (R²): 0.5301
SHAP deletion fidelity: prob drop = 0.4252
```

## Key Findings
- Demonstrates practical application of SHAP and LIME on image classification
- Quantifies agreement between two major explainability approaches
- Provides fidelity metrics to assess explanation quality
- Highlights trade-offs and complementary insights between methods

## References
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [Why Should I Trust You?: Explaining the Predictions of Any Classifier (LIME)](https://arxiv.org/abs/1602.04938)


