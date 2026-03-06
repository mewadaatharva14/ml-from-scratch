
import numpy as np

class LogisticRegression:
    def __init__(self, config: dict) -> None:
        self.lr                = config["training"]["learning_rate"]
        self.epochs            = config["training"]["epochs"]
        self.log_interval      = config["training"]["log_interval"]
        self.lambda_reg        = config["training"]["lambda_reg"]
        self.threshold         = config["training"]["threshold"]
        self.weight_init_scale = config["model"]["weight_init_scale"]
        self.seed              = config["reproducibility"]["random_seed"]

        # initialised in fit() once input dimension is known
        self.weights: np.ndarray | None = None   # (features, 1)
        self.bias:    float             = 0.0

        # training history
        self.loss_history: list[float] = []

    
    def _set_seed(self) -> None:
        np.random.seed(self.seed)


    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:

        self.X_mean = np.mean(X, axis=0)   # (features,) — mean of each column
        self.X_std  = np.std(X, axis=0)    # (features,) — std of each column
        return (X - self.X_mean) / self.X_std


    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:

        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    

    def _forward(self, X: np.ndarray) -> np.ndarray:

        z = X @ self.weights + self.bias   # (N, 1)
        return self._sigmoid(z)            # (N, 1)
    
    def _compute_loss(self, y_hat: np.ndarray, y: np.ndarray, n: int,
    ) -> float:
        
        eps      = 1e-7
        bce_loss = -(1.0 / n) * np.sum(
            y * np.log(y_hat + eps) + (1.0 - y) * np.log(1.0 - y_hat + eps)
        )
        l2_penalty = (self.lambda_reg / (2.0 * n)) * np.sum(np.square(self.weights))
        return float(bce_loss + l2_penalty)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegression":

        self._set_seed()

        # ensure y is column vector (N, 1)
        y_train = y_train.reshape(-1, 1)

        # standardize — fit stats on train only
        X_std = self._standardize_fit(X_train)

        n, features = X_std.shape

        # weight initialisation: small random values close to zero
        # large init → sigmoid saturates → vanishing gradients immediately
        self.weights = np.random.randn(features, 1) * self.weight_init_scale
        self.bias    = 0.0

        for epoch in range(self.epochs):

            # forward pass → (N, 1) probabilities
            y_hat = self._forward(X_std)

            # loss
            loss = self._compute_loss(y_hat, y_train, n)
            self.loss_history.append(loss)

            # gradients
            error = y_hat - y_train                                    # (N, 1)
            dW    = (1.0 / n) * (X_std.T @ error)                     # (features, 1)
            dW   += (self.lambda_reg / n) * self.weights               # L2 term
            db    = (1.0 / n) * np.sum(error)                         # scalar

            # parameter update
            self.weights -= self.lr * dW
            self.bias    -= self.lr * db

            if epoch % self.log_interval == 0:
                print(f"Epoch {epoch:>4d} | Loss: {loss:.6f}")

        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        X_std = self._standardize_transform(X)
        return self._forward(X_std)
    
    def predict(self, X: np.ndarray) -> np.ndarray:

        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
    

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:

        y      = y.reshape(-1, 1)
        y_pred = self.predict(X)

        tp = int(np.sum((y_pred == 1) & (y == 1)))
        tn = int(np.sum((y_pred == 0) & (y == 0)))
        fp = int(np.sum((y_pred == 1) & (y == 0)))
        fn = int(np.sum((y_pred == 0) & (y == 1)))

        accuracy  = (tp + tn) / len(y)
        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f1        = (2.0 * precision * recall) / (precision + recall + 1e-7)

        return {
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }
    
    def get_params(self) -> dict:
        return {
            "weights": self.weights,
            "bias":    self.bias,
        }

