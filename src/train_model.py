import numpy as np
import pickle
import os
import pandas as pd
from data_preprocessing import preprocess_data

def mse(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return float(np.mean((y_true - y_pred)**2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot)

class LinearRegressionNormalEq:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y)
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.theta = None

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n = X_b.shape[1]
        I = np.eye(n)
        I[0,0] = 0
        self.theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ (X_b.T @ y)
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

class LassoRegressionSGD:
    def __init__(self, alpha=0.1, learning_rate=0.01, n_iter=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.theta = None

    def fit(self, X, y):
        # ensure y is a column vector
        y = y.reshape(-1, 1)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        self.theta = np.zeros((n, 1))

        for iteration in range(self.n_iter):
            batch_size = min(10000, m)
            indices = np.random.choice(m, batch_size, replace=False)
            X_batch = X_b[indices]            # (batch_size, n)
            y_batch = y[indices].reshape(-1, 1)  # (batch_size, 1)

            y_pred = X_batch @ self.theta    # (batch_size, 1)
            # (batch_size,1) - (batch_size,1) -> (batch_size,1)
            gradients = (2.0 / batch_size) * (X_batch.T @ (y_pred - y_batch))  # (n,1)

            # L1 penalty (subgradient)
            l1_penalty = self.alpha * np.sign(self.theta)
            l1_penalty[0, 0] = 0  # don't penalize intercept
            gradients += l1_penalty

            self.theta -= self.learning_rate * gradients

        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

class PolynomialRegression:
    def __init__(self, degree=2, max_poly_features=10):
        self.degree = degree
        self.max_poly_features = max_poly_features
        self.theta = None
        self.selected_features = None

    def _select_polynomial_features(self, X):
        n_features = min(self.max_poly_features, X.shape[1])
        feature_vars = np.var(X, axis=0)
        selected_indices = np.argsort(feature_vars)[-n_features:]
        return selected_indices

    def _poly_features(self, X):
        if self.selected_features is None:
            self.selected_features = self._select_polynomial_features(X)

        X_selected = X[:, self.selected_features]
        # simple polynomial expansion: original selected, squared terms
        feats = [X_selected, X_selected ** 2]
        return np.hstack(feats)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        Xp = self._poly_features(X)
        Xpi = np.c_[np.ones((Xp.shape[0], 1)), Xp]

        alpha = 10.0
        n = Xpi.shape[1]
        I = np.eye(n)
        I[0,0] = 0

        self.theta = np.linalg.pinv(Xpi.T @ Xpi + alpha * I) @ (Xpi.T @ y)
        return self

    def predict(self, X):
        Xp = self._poly_features(X)
        Xpi = np.c_[np.ones((Xp.shape[0], 1)), Xp]
        return Xpi @ self.theta

def train_models(data_path):
    X, y, mu, sigma, features, target_col = preprocess_data(data_path)

    models = [
        ("regression_model1.pkl", LinearRegressionNormalEq()),
        ("regression_model2.pkl", RidgeRegression(alpha=100.0)),
        ("regression_model3.pkl", LassoRegressionSGD(alpha=1.0, learning_rate=0.01, n_iter=1000)),
        ("regression_model_final.pkl", PolynomialRegression(degree=2, max_poly_features=8))
    ]

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for model_name, model in models:
        try:
            # Fit model - pass y as-is, each model ensures y has correct shape
            model.fit(X, y)

            y_pred = model.predict(X)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            train_mse = mse(y, y_pred)
            train_rmse = rmse(y, y_pred)
            train_r2 = r2_score(y, y_pred)

            # Save model
            model_data = {
                'model': model,
                'mu': mu,
                'sigma': sigma,
                'features': features,
                'target_col': target_col
            }

            with open(f"models/{model_name}", "wb") as f:
                pickle.dump(model_data, f)

            # Save final model results
            if model_name == "regression_model_final.pkl":
                with open("results/train_metrics.txt", "w") as f:
                    f.write("Regression Metrics:\n")
                    f.write(f"Mean Squared Error (MSE): {train_mse:.2f}\n")
                    f.write(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}\n")
                    f.write(f"R-squared (R²) Score: {train_r2:.2f}\n")

                pred_df = pd.DataFrame(y_pred.ravel())
                pred_df.to_csv("results/train_predictions.csv", index=False, header=False)

        except Exception:
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    train_models("data/train_data.csv")
