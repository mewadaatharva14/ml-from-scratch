"""
Training Entry Point
=====================
Trains any model in this repository from the command line.

Usage:
    python train.py --model regression --config configs/regression_config.yaml
    python train.py --model logistic   --config configs/logistic_config.yaml
    python train.py --model nn         --config configs/nn_config.yaml

Arguments:
    --model  : one of {regression, logistic, nn}
    --config : path to the YAML config file for that model
"""

import argparse
import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for all OS
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ML model from scratch using NumPy."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["regression", "logistic", "nn"],
        help="Model to train: regression | logistic | nn",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file, e.g. configs/regression_config.yaml",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """
    Load and parse a YAML config file into a Python dict.

    Parameters
    ----------
    config_path : str — path to .yaml file

    Returns
    -------
    config : dict
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs: configs/regression_config.yaml, "
            f"configs/logistic_config.yaml, configs/nn_config.yaml"
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ------------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------------

def load_regression_data(config: dict) -> tuple:
    """
    Load and prepare California Housing dataset for polynomial regression.

    Steps:
        1. Fetch from sklearn
        2. Reshape y to (N, 1)
        3. Z-score standardize X and y
        4. Shuffle with fixed seed
        5. Split into train/test

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    from sklearn.datasets import fetch_california_housing

    print("Loading California Housing dataset...")
    housing    = fetch_california_housing()
    X          = housing.data                       # (20640, 8)
    y          = housing.target.reshape(-1, 1)      # (20640, 1)

    # standardize
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    y_mean, y_std = np.mean(y, axis=0), np.std(y, axis=0)
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std

    # shuffle
    seed = config["reproducibility"]["random_seed"]
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # split
    split   = int((1 - config["data"]["test_size"]) * len(X))
    return X[:split], X[split:], y[:split], y[split:]


def load_logistic_data(config: dict) -> tuple:
    """
    Load and prepare Breast Cancer dataset for logistic regression.

    Returns raw (unstandardized) splits.
    Standardization is handled inside LogisticRegression.fit()
    to prevent data leakage.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    from sklearn.datasets import load_breast_cancer

    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X    = data.data                                # (569, 30)
    y    = data.target                              # (569,)

    # shuffle
    seed = config["reproducibility"]["random_seed"]
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # split
    split   = int((1 - config["data"]["test_size"]) * len(X))
    return X[:split], X[split:], y[:split], y[split:]


def load_nn_data(config: dict) -> tuple:
    """
    Load and prepare FashionMNIST dataset for the MLP.

    Steps:
        1. Download via torchvision (cached after first run)
        2. Convert to NumPy
        3. Flatten 28x28 → 784
        4. Normalize pixels to [0, 1]
        5. Split into train/val using config val_split index

    Returns
    -------
    X_train, X_val, y_train, y_val : np.ndarray
    """
    from torchvision import datasets

    print("Loading FashionMNIST dataset...")
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True
    )

    # .data  → torch.Tensor (60000, 28, 28) uint8
    # .numpy() → np.ndarray (60000, 28, 28) uint8
    X = train_data.data.numpy()                     # (60000, 28, 28)
    y = train_data.targets.numpy()                  # (60000,)

    # flatten: 28x28 → 784
    # reshape(-1, 784): -1 infers N automatically
    X = X.reshape(-1, 784)                          # (60000, 784)

    # normalize: uint8 [0, 255] → float [0, 1]
    X = X / 255.0                                   # (60000, 784)

    # split
    split = config["data"]["val_split"]             # 55000
    return X[:split], X[split:], y[:split], y[split:]


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def save_loss_curve(
    loss_history: list[float],
    model_name:   str,
    save_dir:     str = "assets",
) -> None:
    """
    Save training loss curve as PNG to assets/.

    Parameters
    ----------
    loss_history : list of float — loss value at each epoch
    model_name   : str — used in filename and plot title
    save_dir     : str — directory to save into
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="#1A6BC4", linewidth=1.8, label="Training Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss",  fontsize=12)
    plt.title(f"{model_name} — Training Loss Curve", fontsize=14)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"loss_curve_{model_name.lower()}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved → {save_path}")


def save_accuracy_curve(
    accuracy_history: list[float],
    model_name:       str,
    save_dir:         str = "assets",
) -> None:
    """
    Save training accuracy curve as PNG to assets/.
    Only used for the neural network model.

    Parameters
    ----------
    accuracy_history : list of float — accuracy at each epoch
    model_name       : str
    save_dir         : str
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(accuracy_history, color="#00C28B", linewidth=1.8, label="Train Accuracy")
    plt.xlabel("Epoch",    fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"{model_name} — Training Accuracy Curve", fontsize=14)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"accuracy_curve_{model_name.lower()}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Accuracy curve saved → {save_path}")


# ------------------------------------------------------------------
# Model runners
# ------------------------------------------------------------------

def run_regression(config: dict) -> None:
    from src import PolynomialRegression

    X_train, X_test, y_train, y_test = load_regression_data(config)

    print(f"\nX_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    print("\nStarting training...\n")

    model = PolynomialRegression(config)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print("\n── Evaluation on Test Set ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    save_loss_curve(model.loss_history, model_name="Polynomial_Regression")


def run_logistic(config: dict) -> None:
    from src import LogisticRegression

    X_train, X_test, y_train, y_test = load_logistic_data(config)

    print(f"\nX_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    print("\nStarting training...\n")

    model = LogisticRegression(config)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print("\n── Evaluation on Test Set ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    save_loss_curve(model.loss_history, model_name="Logistic_Regression")


def run_nn(config: dict) -> None:
    from src import NeuralNetwork

    X_train, X_val, y_train, y_val = load_nn_data(config)

    print(f"\nX_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape}    y_val   : {y_val.shape}")
    print("\nStarting training...\n")

    model = NeuralNetwork(config)
    model.fit(X_train, y_train, X_val, y_val)

    metrics = model.evaluate(X_val, y_val)
    print("\n── Evaluation on Validation Set ────────────")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")

    save_loss_curve(model.loss_history,     model_name="Neural_Network")
    save_accuracy_curve(model.accuracy_history, model_name="Neural_Network")


# ------------------------------------------------------------------
# Dispatch table
# ------------------------------------------------------------------

RUNNERS = {
    "regression": run_regression,
    "logistic":   run_logistic,
    "nn":         run_nn,
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)

    print(f"\n{'='*50}")
    print(f"  Model  : {args.model}")
    print(f"  Config : {args.config}")
    print(f"{'='*50}\n")

    RUNNERS[args.model](config)

    print("\nDone.")