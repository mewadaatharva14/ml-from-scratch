# ml-from-scratch/src/neural_network.py
import numpy as np

class NeuralNetwork:


    def __init__(self, config: dict) -> None:
        self.input_dim         = config["data"]["input_dim"]
        self.hidden_dim        = config["model"]["hidden_dim"]
        self.num_classes       = config["data"]["num_classes"]
        self.lr                = config["training"]["learning_rate"]
        self.epochs            = config["training"]["epochs"]
        self.log_interval      = config["training"]["log_interval"]
        self.lambda_reg        = config["training"]["lambda_reg"]
        self.weight_init_scale = config["model"]["weight_init_scale"]
        self.seed              = config["reproducibility"]["random_seed"]

        # layer 1 weights — initialised in fit()
        self.W1: np.ndarray | None = None   # (input_dim, hidden_dim)  = (784, 128)
        self.b1: np.ndarray | None = None   # (1, hidden_dim)           = (1, 128)

        # layer 2 weights — initialised in fit()
        self.W2: np.ndarray | None = None   # (hidden_dim, num_classes) = (128, 10)
        self.b2: np.ndarray | None = None   # (1, num_classes)           = (1, 10)

        # cache — stored during forward, reused during backward
        self._Z1: np.ndarray | None = None  # pre-activation layer 1
        self._A1: np.ndarray | None = None  # post-activation layer 1
        self._Z2: np.ndarray | None = None  # pre-activation layer 2
        self._A2: np.ndarray | None = None  # post-activation layer 2 (predictions)

        # training history
        self.loss_history:     list[float] = []
        self.accuracy_history: list[float] = []

    def _set_seed(self) -> None:
        np.random.seed(self.seed)

    def _init_weights(self) -> None:
        self.W1 = np.random.randn(
            self.input_dim, self.hidden_dim
        ) * np.sqrt(2.0 / self.input_dim)           # He init: √(2 / fan_in)

        self.b1 = np.zeros((1, self.hidden_dim))     # (1, 128)

        self.W2 = np.random.randn(
            self.hidden_dim, self.num_classes
        ) * np.sqrt(2.0 / self.hidden_dim)           # He init: √(2 / fan_in)

        self.b2 = np.zeros((1, self.num_classes))
    
    def _relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def _relu_derivative(self, Z: np.ndarray) -> np.ndarray:

        return (Z > 0).astype(float)
    
    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def _one_hot(self, y: np.ndarray) -> np.ndarray:

        return np.eye(self.num_classes)[y]
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        self._Z1 = X @ self.W1 + self.b1       # (N, 128)
        self._A1 = self._relu(self._Z1)         # (N, 128)
        self._Z2 = self._A1 @ self.W2 + self.b2 # (N, 10)
        self._A2 = self._softmax(self._Z2)      # (N, 10)
        return self._A2
    
    def _backward(self,
        X:         np.ndarray,
        y_one_hot: np.ndarray,
        n:         int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dZ2 = self._A2 - y_one_hot                          # (N, 10)
        dW2 = (self._A1.T @ dZ2) / n                        # (128, 10)
        dW2 += (self.lambda_reg / n) * self.W2              # L2 term
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n        # (1, 10)

        # ── Layer 1 ───────────────────────────────────────────────────
        dA1 = dZ2 @ self.W2.T                               # (N, 128)
        dZ1 = dA1 * self._relu_derivative(self._Z1)         # (N, 128)
        dW1 = (X.T @ dZ1) / n                               # (784, 128)
        dW1 += (self.lambda_reg / n) * self.W1              # L2 term
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n        # (1, 128)

        return dW1, db1, dW2, db2
    
    def _compute_loss(
        self,
        y_hat:     np.ndarray,
        y_one_hot: np.ndarray,
        n:         int,
    ) -> float:
        
        eps      = 1e-10
        cce_loss = -(1.0 / n) * np.sum(y_one_hot * np.log(y_hat + eps))
        l2       = (self.lambda_reg / (2.0 * n)) * (
            np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))
        )
        return float(cce_loss + l2)
    

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
    ) -> "NeuralNetwork":
        
        self._set_seed()
        self._init_weights()

        n          = X_train.shape[0]
        y_one_hot  = self._one_hot(y_train)

        for epoch in range(self.epochs):

            # forward
            y_hat = self._forward(X_train)     # (N, 10)

            # loss
            loss  = self._compute_loss(y_hat, y_one_hot, n)
            self.loss_history.append(loss)

            # accuracy on train
            train_acc = self._accuracy(y_hat, y_train)
            self.accuracy_history.append(train_acc)

            # backward
            dW1, db1, dW2, db2 = self._backward(X_train, y_one_hot, n)

            # parameter update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            # logging
            if epoch % self.log_interval == 0:
                log_str = (
                    f"Epoch {epoch:>3d} | "
                    f"Loss: {loss:.6f} | "
                    f"Train Acc: {train_acc:.4f}"
                )
                if X_val is not None and y_val is not None:
                    val_hat = self._forward(X_val)
                    val_acc = self._accuracy(val_hat, y_val)
                    log_str += f" | Val Acc: {val_acc:.4f}"
                print(log_str)

        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (N, 784)

        Returns
        -------
        np.ndarray, shape (N, 10) — probabilities summing to 1 per row
        """
        return self._forward(X)
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class indices.

        argmax over axis=1 → picks the class with highest probability
        per sample.

        Parameters
        ----------
        X : np.ndarray, shape (N, 784)

        Returns
        -------
        np.ndarray, shape (N,) — integer class indices in [0, 9]
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) 
    

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute accuracy and loss on given data.

        Parameters
        ----------
        X : np.ndarray, shape (N, 784)
        y : np.ndarray, shape (N,) — integer labels

        Returns
        -------
        metrics : dict with keys loss, accuracy
        """
        y_hat     = self.predict_proba(X)
        y_one_hot = self._one_hot(y)
        n         = X.shape[0]

        loss     = self._compute_loss(y_hat, y_one_hot, n)
        accuracy = self._accuracy(y_hat, y)

        return {
            "loss":     round(loss,     6),
            "accuracy": round(accuracy, 4),
        }
    

    def get_params(self) -> dict:
        """
        Return all current model parameters.

        Returns
        -------
        params : dict with keys W1, b1, W2, b2
        """
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
    
    def _accuracy(
        self,
        y_hat: np.ndarray,
        y:     np.ndarray,
    ) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        y_hat : np.ndarray, shape (N, 10) — predicted probabilities
        y     : np.ndarray, shape (N,)    — true integer labels

        Returns
        -------
        accuracy : float in [0, 1]
        """
        predictions = np.argmax(y_hat, axis=1)   # (N,)
        return float(np.mean(predictions == y))