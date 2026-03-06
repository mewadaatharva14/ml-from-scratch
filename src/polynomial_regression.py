# ml-from-scratch/src/polynomial_regression.py
"""
Polynomial Regression from Scratch
====================================
Implements degree-2 polynomial regression using two methods:
  1. Gradient Descent  — iterative weight update via MSE gradients
  2. Normal Equation   — closed-form optimal solution via pseudoinverse

Dataset : California Housing (sklearn)
Input   : X of shape (N, 8)  — 8 housing features
Output  : y of shape (N, 1)  — median house value (standardized)
"""

import numpy as np


class PolynomialRegression:
    """
    Degree-2 Polynomial Regression trained via Gradient Descent.
    Also computes the closed-form Normal Equation solution for comparison.

    Parameters
    ----------
    config : dict
        Parsed YAML config. Expected keys:
            model.degree                 : int
            model.weight_init_scale      : float
            training.learning_rate       : float
            training.epochs              : int
            training.log_interval        : int
            reproducibility.random_seed  : int
    """

    def __init__(self, config: dict) -> None:
        self.config           = config
        self.lr               = config["training"]["learning_rate"]
        self.epochs           = config["training"]["epochs"]
        self.log_interval     = config["training"]["log_interval"]
        self.init_scale       = config["model"]["weight_init_scale"]
        self.seed             = config["reproducibility"]["random_seed"]

        # Gradient Descent weights — initialised in fit()
        self.w1:   np.ndarray | None = None   # (n_features, 1)  linear term
        self.w2:   np.ndarray | None = None   # (n_features, 1)  quadratic term
        self.bias: np.ndarray | None = None   # (1, 1)           intercept

        # Normal Equation solution — computed inside fit()
        self.w_star: np.ndarray | None = None  # (2*n_features+1, 1)

        # training history
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_poly_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Build the polynomial design matrix [1, X, X²].

        Concatenates a bias column of ones, the original features,
        and the squared features into one matrix.

        Shape trace:
            ones   : (N, 1)
            X      : (N, 8)
            X**2   : (N, 8)
            result : (N, 17)   ← 1 + 8 + 8 = 17 columns

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        X_poly : np.ndarray, shape (N, 2*n_features + 1)
        """
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X, X ** 2])

    def _compute_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Compute the closed-form Normal Equation solution and store as w_star.

        Formula: W* = pinv(X_poly.T @ X_poly) @ X_poly.T @ y

        Uses pseudoinverse (pinv) instead of inv because X_poly.T @ X_poly
        may be singular or near-singular when features are correlated.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) — standardized features
        y : np.ndarray, shape (N, 1)          — standardized targets
        """
        X_poly      = self._build_poly_matrix(X)            # (N, 17)
        self.w_star = (
            np.linalg.pinv(X_poly.T @ X_poly)               # (17, 17)
            @ X_poly.T                                       # (17, N)
            @ y                                              # (N,  1)
        )                                                    # → (17, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        """
        Initialise weights, run gradient descent, then compute Normal Equation.

        Both GD and Normal Equation are trained on the same X, y so that
        evaluate() can compare them on the same test set fairly.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features) — standardized features
        y : np.ndarray, shape (N, 1)          — standardized targets

        Returns
        -------
        self : PolynomialRegression — for method chaining
        """
        np.random.seed(self.seed)

        n, n_features = X.shape

        # weight initialisation — small random values
        self.w1   = np.random.randn(n_features, 1) * self.init_scale  # (8, 1)
        self.w2   = np.random.randn(n_features, 1) * self.init_scale  # (8, 1)
        self.bias = np.ones((1, 1))                                    # (1, 1)

        self.loss_history = []

        # ── Gradient Descent ──────────────────────────────────────────
        for epoch in range(self.epochs):

            # forward pass
            # Shape: (N,8)@(8,1) + (N,8)@(8,1) + (1,1) → (N,1)
            y_pred = (X ** 2) @ self.w2 + X @ self.w1 + self.bias

            error = y_pred - y                          # (N, 1)
            loss  = float(np.mean(error ** 2))          # scalar MSE
            self.loss_history.append(loss)

            # gradients — chain rule on MSE
            # X.T is (8,N), error is (N,1) → result (8,1) matches w1 shape
            grad_w1   = (2.0 / n) * (X.T       @ error)   # (8, 1)
            grad_w2   = (2.0 / n) * ((X**2).T  @ error)   # (8, 1)
            grad_bias = (2.0 / n) * np.sum(error)          # scalar

            self.w1   -= self.lr * grad_w1
            self.w2   -= self.lr * grad_w2
            self.bias -= self.lr * grad_bias

            if epoch % self.log_interval == 0:
                print(f"  Epoch {epoch:>5} | MSE Loss: {loss:.6f}")

        # ── Normal Equation — computed once after GD finishes ─────────
        # This is why w_star was None before: it needs X and y,
        # which only exist after fit() is called.
        self._compute_normal_equation(X, y)
        print(f"\nNormal Equation solution computed. w_star shape: {self.w_star.shape}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Gradient Descent weights.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (N, 1)
        """
        return (X ** 2) @ self.w2 + X @ self.w1 + self.bias

    def predict_normal_equation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Normal Equation weights.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (N, 1)
        """
        X_poly = self._build_poly_matrix(X)     # (N, 17)
        return X_poly @ self.w_star             # (N, 17) @ (17, 1) → (N, 1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute MSE and RMSE for both GD and Normal Equation on given data.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
        y : np.ndarray, shape (N, 1)

        Returns
        -------
        metrics : dict with keys gd_mse, gd_rmse, ne_mse, ne_rmse
        """
        y_pred_gd = self.predict(X)                  # (N, 1)
        y_pred_ne = self.predict_normal_equation(X)  # (N, 1)

        gd_mse = float(np.mean((y - y_pred_gd) ** 2))
        ne_mse = float(np.mean((y - y_pred_ne) ** 2))

        return {
            "gd_mse":  round(gd_mse,             6),
            "gd_rmse": round(float(np.sqrt(gd_mse)), 6),
            "ne_mse":  round(ne_mse,             6),
            "ne_rmse": round(float(np.sqrt(ne_mse)), 6),
        }

    def get_params(self) -> dict:
        """Return current model parameters."""
        return {
            "w1":     self.w1,
            "w2":     self.w2,
            "bias":   self.bias,
            "w_star": self.w_star,
        }