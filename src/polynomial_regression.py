import numpy as np
from sklearn.datasets import fetch_california_housing

class PolynomialRegression:

    def __init__(self, config : dict):
        self.config = config
        self.lr = config["training"]["learning_rate"]
        self.epochs = config["training"]["epochs"]
        self.log_intervals = config["traninig"]["log_intervals"]
        self.init_scale   = config["model"]["weight_init_scale"]
        self.seed         = config["reproducibility"]["random_seed"]

        self.w1 = None
        self.w2 = None
        self.bais = None

        # Normal equation
        self.w_star = None

        self.loss_history: list[float] = []
    
    def load_data(self) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
        
        np.random.seed(self.seed)
        housing        = fetch_california_housing()
        X, y           = housing.data, housing.target.reshape(-1, 1)
        self.x_mean = np.mean(X, axis=0)   # shape: (8,)
        self.x_std  = np.std(X,  axis=0)   # shape: (8,)
        self.y_mean = np.mean(y, axis=0)   # shape: (1,)
        self.y_std  = np.std(y,  axis=0)   # shape: (1,)

        X_std = (X - self.x_mean) / self.x_std
        y_std = (y - self.y_mean) / self.y_std

        n          = len(X_std)
        indices    = np.random.permutation(n)
        split      = int((1 - self.config["data"]["test_size"]) * n)

        X_std, y_std = X_std[indices], y_std[indices]

        X_train, X_test = X_std[:split], X_std[split:]
        y_train, y_test = y_std[:split], y_std[split:]

        return X_train, X_test, y_train, y_test
    
    def fit(self, X: np.ndarray, y: np.ndarray):

        np.random.seed(self.seed)

        n, n_features = X.shape


        self.w1   = np.random.randn(n_features, 1) * self.init_scale
        self.w2   = np.random.randn(n_features, 1) * self.init_scale
        self.bias = np.ones((1, 1))

        self.loss_history = []

        for epoch in range(self.epochs):
            y_pred = (X ** 2) @ self.w2 + X @ self.w1 + self.bias

            error = y_pred - y                        # (N, 1)
            loss  = np.mean(error ** 2)               # scalar
            self.loss_history.append(loss)

            grad_w1   = (2 / n) * (X.T       @ error)   # (n_features, 1)
            grad_w2   = (2 / n) * ((X**2).T  @ error)   # (n_features, 1)
            grad_bias = (2 / n) * np.sum(error)

            self.w1   -= self.lr * grad_w1
            self.w2   -= self.lr * grad_w2
            self.bias -= self.lr * grad_bias

            if epoch % self.log_interval == 0:
                print(f"  Epoch {epoch:>5} | MSE Loss: {loss:.6f}")

    def normal_equation(self, X: np.ndarray, y: np.ndarray):

        n = X.shape[0]

        ones        = np.ones((n, 1))                      # (N, 1)
        X_poly      = np.hstack([ones, X, X ** 2])         # (N, 2*f+1)

        self.w_star = (np.linalg.pinv(X_poly.T @ X_poly)
                       @ X_poly.T
                       @ y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:

        return (X ** 2) @ self.w2 + X @ self.w1 + self.bias
    
    def predict_normal_equation(self, X: np.ndarray) -> np.ndarray:

        n     = X.shape[0]
        ones  = np.ones((n, 1))
        X_poly = np.hstack([ones, X, X ** 2])
        return X_poly @ self.w_star
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:

        y_pred_gd = self.predict(X)
        y_pred_ne = self.predict_normal_equation(X)

        gd_mse  = float(np.mean((y - y_pred_gd) ** 2))
        ne_mse  = float(np.mean((y - y_pred_ne) ** 2))

        return {
            "gd_mse":  gd_mse,
            "gd_rmse": float(np.sqrt(gd_mse)),
            "ne_mse":  ne_mse,
            "ne_rmse": float(np.sqrt(ne_mse)),
        }
    
    def get_params(self) -> dict:

        return {
            "w1":     self.w1,
            "w2":     self.w2,
            "bias":   self.bias,
            "w_star": self.w_star,
        }
    

