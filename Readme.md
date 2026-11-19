# Exploring the Neural Networks Fundamentals with SVHN Image Classification

A comprehensive empirical study of Multi-Layer Perceptrons (MLPs) on the **Street View House Numbers (SVHN)** dataset. This project spans the entire deep learning workflow: from benchmarking classical distance-based baselines and building a mathematically verified MLP from scratch in NumPy, to designing a leak-free 5-fold cross-validation hyperparameter tuning pipeline, training an optimized PyTorch MLP, and testing its robustness under label poisoning.

> **Full Technical Report:** For complete mathematical derivations, methodology, plots, and in-depth analysis of all experiments, see [`neural_networks_report.pdf`](./neural_networks_report.pdf).

---

## Dataset Overview

The **SVHN** dataset consists of $32 \times 32$ RGB street-view digit images across $10$ classes (`0`–`9`), characterized by real-world noise (varying illumination, motion blur, diverse fonts, and distracting adjacent digits) and natural class imbalance:

* **Full Training Set:** $73,257$ images (imbalanced)
* **Official Test Set:** $26,032$ images (imbalanced)
* **Feature Representations Evaluated:**
  1. **Flattened Pixels:** Standardized raw pixel vectors ($D = 3072$)
  2. **PCA:** Principal Component Analysis retaining $95\%$ variance ($D = 53$–$54$)
  3. **ResNet18 Embeddings:** Pretrained CNN feature vectors ($D = 512$, standardized)

---

## Repository Structure

```text
.
├── neural_networks_report.pdf           # Full academic report with theory, plots & analysis
├── src/
│   ├── data_loader.py                   # SVHN loading, normalization, balanced & stratified subsampling
│   ├── features.py                      # Flattening, PCA, and ResNet18 feature extraction
│   ├── baselines.py                     # 1-NN, 3-NN, and Nearest Class Centroid evaluators
│   ├── mlp.py                           # Custom NumPy MLPFromScratch + Gradient Checking
│   ├── pytorch_mlp.py                   # PyTorch MLPClassifier (Adam, L2, SGDR, Class Weights, Checkpointing)
│   └── grid_search.py                   # Leak-free 5-Fold Stratified CV Grid Search
├── notebooks/
│   ├── baselines.ipynb                  # Baseline benchmarking (1-NN, 3-NN, NCC) & dataset visualization
│   ├── gradient_checking.ipynb          # Finite-difference validation of custom backpropagation
│   ├── mlp_scratch_initialization.ipynb # Weight initialization study (Zeros, Random, Xavier, He, Optimal)
│   ├── uat_studies.ipynb                # Universal Approximation Theorem & network capacity study
│   ├── grid_search.ipynb                # 108-config 5-Fold CV hyperparameter & feature ablation study
│   ├── optimal_mlp.ipynb                # Full-dataset Optimal MLP training, curves & qualitative analysis
│   └── label_poisoning.ipynb            # Robustness evaluation under 0%–60% synthetic label noise

```

## Experiments

### 1. Distance-Based Baseline Models (`baselines.ipynb`)
Evaluated **Nearest Class Centroid (NCC)**, **1-Nearest Neighbor (1-NN)**, and **3-Nearest Neighbors (3-NN)** on the full imbalanced SVHN dataset using **Macro F1-Score** across all three feature representations.

### 2. Custom MLP Validation — Gradient Checking (`gradient_checking.ipynb`)
Implemented a Multi-Layer Perceptron from scratch in NumPy (`src/mlp.py`) and verified analytical backpropagation against numerical **Finite-Difference Approximation** ($\epsilon = 10^{-7}$) on a lightweight `[128, 64]` network using a 3-image mini-batch.

### 3. Weight Initialization Study (`mlp_scratch_initialization.ipynb`)
Benchmarked 5 initialization strategies (`zeros`, `random`, `Xavier`, `he`, `optimal`) on a balanced subset ($15,000$ train / $3,000$ test) with PCA ($53$ components) and a `[128, 128]` ReLU architecture.

### 4. Universal Approximation Theorem & Capacity Analysis (`uat_studies.ipynb`)
Evaluated a single-hidden-layer PyTorch MLP without regularization across widths $[8, 16, 32, 64, 128, 256, 512, 1024]$ on the balanced PCA subset.

### 5. 5-Fold Stratified CV Grid Search & Ablation Study (`grid_search.ipynb`)
Executed an exhaustive **108-configuration** Grid Search using **5-Fold Stratified Cross-Validation** on a $5,000$-sample imbalanced subset, testing:
* **Feature Extraction:** `['flatten', 'pca', 'resnet']`
* **Hidden Layers:** `[[64, 64], [128, 128], [256, 256], [256, 128, 64]]`
* **Learning Rate:** $[10^{-2}, 10^{-3}, 10^{-4}]$
* **Weight Decay ($L_2$):** $[10^{-3}, 10^{-4}, 10^{-5}]$

All feature scaling and PCA projections were fitted strictly **inside** each CV fold to prevent data leakage.

### 6. Optimal PyTorch MLP Training & Evaluation (`optimal_mlp.ipynb`)
Trained the winning configuration on the full imbalanced SVHN dataset ($68,257$ train / $5,000$ stratified validation / $26,032$ official test) using:
* **Class-Weighted Cross-Entropy Loss** to counter SVHN class imbalance
* **Adam Optimizer** with explicit $L_2$ Weight Decay ($0.001$)
* **Cosine Annealing with Warm Restarts (SGDR)** ($T_0 = 15$, $60$ total epochs = $4$ complete cycles)
* **Macro F1-Based Model Checkpointing** to retain the best weights across learning rate restarts


### 7. Label Poisoning & Noise Robustness Study (`label_poisoning.ipynb`)
Evaluated the degradation and noise-memorization dynamics of the Optimal MLP when injecting synthetic uniform label noise at levels **$[0\%, 10\%, 20\%, 40\%, 60\%]$** into the training labels while evaluating strictly on the **clean official test set** ($26,032$ images).

---

## Getting Started

### Prerequisites
Install the required Python packages (Python 3.10+ recommended):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision tqdm jupyter
```

### Running the Experiments
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Launch Jupyter Notebook / Lab from the project root:
   ```bash
   jupyter notebook
   ```
3. Run the notebooks inside the `notebooks/` directory in order. The SVHN dataset is automatically downloaded to `data/raw/` via `torchvision.datasets.SVHN` on the first run.

---

## Documentation

All theoretical background (Gradient Checking, Initialization theory, Universal Approximation Theorem, Adam, SGDR, Class Weighting, and Label Poisoning), learning curves, confusion/qualitative plots, and full discussions are documented in [`neural_networks_report.pdf`](./neural_networks_report.pdf).