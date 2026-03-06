# 🧠 ML from Scratch — Regression, Classification & Neural Networks in Pure NumPy

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.2-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-00C28B?style=flat)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-mewadaatharva14-181717?style=flat&logo=github)](https://github.com/mewadaatharva14)

> Three machine learning models implemented **completely from scratch** using NumPy —
> no sklearn estimators, no autograd, no black boxes.
> Every gradient is derived and computed by hand.

---

## 📌 Overview

This repository implements Polynomial Regression, Logistic Regression, and a
Multi-Layer Perceptron from the mathematical foundations up. The goal is not
just to get the models working — but to demonstrate a complete understanding
of the forward pass, loss computation, gradient derivation, and parameter update
at every single step. Each model follows a clean, consistent interface and is
accompanied by a Jupyter notebook with inline math, training curves, and
evaluation visualizations.

---

## 🗂️ Project Structure

```
ml-from-scratch/
├── src/
│   ├── __init__.py
│   ├── polynomial_regression.py   ← Degree-2 regression: GD + Normal Equation
│   ├── logistic_regression.py     ← Binary classifier: Sigmoid + BCE + L2
│   └── neural_network.py          ← MLP 784→128→10: ReLU + Softmax + Backprop
│
├── notebooks/
│   ├── 01_polynomial_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   └── 03_neural_network.ipynb
│
├── configs/
│   ├── regression_config.yaml
│   ├── logistic_config.yaml
│   └── nn_config.yaml
│
├── assets/                        ← saved plots and loss curves
├── data/                          ← datasets downloaded automatically
├── .gitignore
├── requirements.txt
├── README.md
└── train.py                       ← unified entry point
```

---

## 🏗️ Models

### 1 · Polynomial Regression
**Dataset:** California Housing (sklearn) — 20,640 samples, 8 features  
**Task:** Predict median house value from housing features

**Model:**

$$\hat{y} = X^2 \cdot W_2 + X \cdot W_1 + b$$

**Two training methods implemented:**

| Method | Description | Complexity |
|---|---|---|
| Gradient Descent | Iterative weight update via MSE gradients | O(N · F · epochs) |
| Normal Equation | Closed-form pseudoinverse solution | O(F³) — one shot |

**Gradient derivations:**

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{2}{N} X^T (\hat{y} - y) \qquad \frac{\partial \mathcal{L}}{\partial W_2} = \frac{2}{N} (X^2)^T (\hat{y} - y)$$

**Results:**

| Method | Test MSE | Test MAE | Test R² |
|---|---|---|---|
| Gradient Descent | — | — | — |
| Normal Equation  | — | — | — |

> Run `python train.py --model regression --config configs/regression_config.yaml` to fill this table.

---

### 2 · Logistic Regression
**Dataset:** Breast Cancer Wisconsin (sklearn) — 569 samples, 30 features  
**Task:** Binary classification — Malignant (0) vs Benign (1)

**Model:**

$$z = X \cdot W + b \qquad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Loss — Binary Cross-Entropy + L2 Regularization:**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i + \varepsilon) + (1 - y_i) \log(1 - \hat{y}_i + \varepsilon) \right] + \frac{\lambda}{2N} \sum W^2$$

**Gradient:**

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} X^T (\hat{y} - y) + \frac{\lambda}{N} W$$

> Bias **b** is not regularized — standard practice, as regularizing bias
> adds no benefit and can harm convergence.

**Results:**

| Metric | Value |
|---|---|
| Accuracy  | — |
| Precision | — |
| Recall    | — |
| F1 Score  | — |

> Run `python train.py --model logistic --config configs/logistic_config.yaml` to fill this table.

---

### 3 · Neural Network — MLP with Manual Backpropagation
**Dataset:** FashionMNIST (torchvision) — 60,000 images, 10 classes  
**Architecture:** 784 → 128 → 10 (fully connected, no convolutions)

**Forward Pass:**

| Step | Operation | Shape |
|---|---|---|
| Input | Flattened pixels | (N, 784) |
| Hidden pre-activation | Z₁ = X @ W₁ + b₁ | (N, 128) |
| Hidden activation | A₁ = ReLU(Z₁) | (N, 128) |
| Output pre-activation | Z₂ = A₁ @ W₂ + b₂ | (N, 10) |
| Output activation | A₂ = Softmax(Z₂) | (N, 10) |

**Loss — Categorical Cross-Entropy + L2:**

$$\mathcal{L} = -\frac{1}{N} \sum_{i} \sum_{c} y_{ic} \log(\hat{y}_{ic} + \varepsilon) + \frac{\lambda}{2N}(\|W_1\|^2 + \|W_2\|^2)$$

**Backpropagation — full chain rule:**

```
Layer 2:
  dZ2 = A2 - y_one_hot               ← combined Softmax + CCE derivative
  dW2 = (A1.T @ dZ2) / N + λ/N · W2
  db2 = sum(dZ2, axis=0) / N

Layer 1 (through ReLU gate):
  dA1 = dZ2 @ W2.T
  dZ1 = dA1 * ReLU'(Z1)              ← ReLU'(z) = 1 if z > 0 else 0
  dW1 = (X.T  @ dZ1) / N + λ/N · W1
  db1 = sum(dZ1, axis=0) / N
```

**Weight Initialization — He Init:**

$$W \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{\text{fan\_in}}}\right)$$

> He initialization compensates for ReLU zeroing ~50% of neurons per pass.
> Naive `* 0.01` initialization causes signal to die before reaching the output layer.

**Results:**

| Metric | Value |
|---|---|
| Validation Accuracy | — |
| Validation Loss     | — |

> Run `python train.py --model nn --config configs/nn_config.yaml` to fill this table.

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/mewadaatharva14/ml-from-scratch.git
cd ml-from-scratch
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train any model**
```bash
# Polynomial Regression
python train.py --model regression --config configs/regression_config.yaml

# Logistic Regression
python train.py --model logistic --config configs/logistic_config.yaml

# Neural Network MLP
python train.py --model nn --config configs/nn_config.yaml
```

All datasets download automatically on first run.  
Loss curves and plots are saved to `assets/` after each training run.

---

## 📓 Notebooks

Each notebook runs end-to-end with a single `Run All` — no setup required
beyond the steps above.

| Notebook | Dataset | What it shows |
|---|---|---|
| `01_polynomial_regression.ipynb` | California Housing | Loss curve · GD vs Normal Equation comparison · Predicted vs Actual scatter |
| `02_logistic_regression.ipynb` | Breast Cancer | Loss curve · Probability distribution by class · Top 10 feature weights |
| `03_neural_network.ipynb` | FashionMNIST | Loss + accuracy curves · Per-class accuracy bar chart · 16-image prediction grid |

To open notebooks in VS Code:
```bash
jupyter notebook notebooks/
```

---

## 🔑 Key Implementation Details

**Why `axis=0` vs `axis=1` matters:**  
`np.mean(X, axis=0)` computes the mean of each feature column across all samples — giving a vector of shape `(features,)`. Using `axis=1` would average across features per sample instead, which is wrong for standardization.

**Why `.reshape(-1, 1)` for targets:**  
NumPy broadcasting requires `y` to be `(N, 1)` not `(N,)` when computing `y_hat - y`. Shape `(N,)` minus shape `(N, 1)` would broadcast incorrectly in matrix operations.

**Why `_standardize_fit` vs `_standardize_transform` are separate methods:**  
`fit` computes mean/std from training data only and stores them. `transform` reuses those stored statistics. If we recomputed statistics on test data, test distribution information would influence the model — this is data leakage.

**Why `log_var` instead of `variance` directly in numerical computations:**  
Variance is always non-negative. Computing in log space removes that constraint, making optimization unconstrained and numerically more stable.

**Why `keepdims=True` in softmax:**  
`np.max(Z, axis=1, keepdims=True)` returns shape `(N, 1)` instead of `(N,)`. This allows broadcasting against `Z` of shape `(N, 10)`. Without `keepdims=True`, the subtraction would fail or broadcast incorrectly.

---

## 📚 References

| Resource | Link |
|---|---|
| California Housing Dataset | [sklearn docs](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) |
| Breast Cancer Wisconsin Dataset | [sklearn docs](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset) |
| FashionMNIST Dataset | [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) |
| He Initialization Paper | [He et al. 2015](https://arxiv.org/abs/1502.01852) |
| CS229 Lecture Notes (Stanford) | [Machine Learning Theory](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with 🧠 by <a href="https://github.com/mewadaatharva14">mewadaatharva14</a>
</p>
