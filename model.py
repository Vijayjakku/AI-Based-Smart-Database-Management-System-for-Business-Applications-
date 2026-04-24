# model.py
"""
Step 3: Model Training
-----------------------
Two models are provided (as referenced in your literature review):

  1. XGBoost (Feng et al., 2024)  ← primary model
     - Gradient boosted trees; fast, interpretable, strong baseline
     - Easy to tune with cross-validation

  2. Neural Network (ALECE-inspired, Li et al., 2023)  ← optional
     - A simple MLP to compare against XGBoost
     - You can extend this with attention heads for your dissertation

Both models predict log(actual_rows + 1); we exponentiate at inference.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Optional – comment out if you don't want PyTorch yet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MODEL_DIR, TEST_SIZE, RANDOM_STATE, XGB_PARAMS, N_FOLDS


# ══════════════════════════════════════════════════════════════════════════════
# 1.  XGBoost Cardinality Estimator  (your primary model)
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostCardinalityEstimator:
    """
    Wraps XGBRegressor with convenience methods for training,
    evaluation, saving, and loading.

    Usage:
        model = XGBoostCardinalityEstimator()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        model.save()
    """

    def __init__(self, params: dict = None):
        self.params = params or XGB_PARAMS
        self.model = XGBRegressor(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """
        Fit the XGBoost model. Supports optional validation set for
        early stopping (recommended for your experiments).
        """
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,          # print loss every 50 rounds
        )
        print("XGBoost training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict actual row counts (un-log-transformed).
        Returns raw row count predictions (not log scale).
        """
        log_preds = self.model.predict(X)
        return np.expm1(log_preds)   # reverse the log1p transform

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Run k-fold cross-validation and return mean ± std of RMSE.
        Useful for your evaluation chapter.
        """
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            self.model, X, y,
            cv=kf, scoring="neg_root_mean_squared_error"
        )
        rmse_scores = -scores
        result = {
            "mean_rmse": float(rmse_scores.mean()),
            "std_rmse":  float(rmse_scores.std()),
        }
        print(f"Cross-validation RMSE: {result['mean_rmse']:.4f} ± {result['std_rmse']:.4f}")
        return result

    def feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Return a sorted DataFrame of feature importances.
        Useful for your dissertation's analysis section.
        """
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        return df

    def save(self, filename: str = "xgb_cardinality_model.pkl"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, filename: str = "xgb_cardinality_model.pkl"):
        path = os.path.join(MODEL_DIR, filename)
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Simple MLP (Neural Network baseline, ALECE-inspired)
# ══════════════════════════════════════════════════════════════════════════════

class CardinalityMLP(nn.Module):
    """
    A simple multi-layer perceptron for cardinality estimation.

    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × 3 → Output

    This mirrors the feed-forward layers in ALECE (Li et al., 2023).
    You can extend this with attention mechanisms for your project.
    """

    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64],
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # single output: log(actual_rows)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class NeuralCardinalityEstimator:
    """
    Wraps CardinalityMLP with a PyTorch training loop.

    Usage:
        model = NeuralCardinalityEstimator(input_dim=X.shape[1])
        model.train(X_train, y_train, X_val, y_val)
        preds = model.predict(X_test)
    """

    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64],
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 128):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CardinalityMLP(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _to_tensor(self, X, y=None):
        X_t = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        if y is not None:
            y_t = torch.tensor(np.array(y), dtype=torch.float32).to(self.device)
            return X_t, y_t
        return X_t

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_t, y_t = self._to_tensor(X_train, y_train)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0:
                val_str = ""
                if X_val is not None:
                    val_loss = self._val_loss(X_val, y_val)
                    val_str = f"  |  val_loss: {val_loss:.4f}"
                print(f"Epoch {epoch:3d}/{self.epochs}  train_loss: {epoch_loss/len(loader):.4f}{val_str}")

    def _val_loss(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            X_t, y_t = self._to_tensor(X_val, y_val)
            preds = self.model(X_t)
            return self.criterion(preds, y_t).item()

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X)
            log_preds = self.model(X_t).cpu().numpy()
        return np.expm1(log_preds)

    def save(self, filename: str = "mlp_cardinality_model.pt"):
        path = os.path.join(MODEL_DIR, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Neural model saved to {path}")

    def load(self, filename: str = "mlp_cardinality_model.pt"):
        path = os.path.join(MODEL_DIR, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Train both models
# ══════════════════════════════════════════════════════════════════════════════

def train_all(X: pd.DataFrame, y: pd.Series):
    """
    Convenience function: split data, train both models, return them.
    Call this from train.py.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )

    print("=" * 60)
    print("Training XGBoost model...")
    xgb_model = XGBoostCardinalityEstimator()
    xgb_model.train(X_train, y_train, X_val, y_val)

    print("\nTraining Neural Network model...")
    nn_model = NeuralCardinalityEstimator(input_dim=X_train.shape[1])
    nn_model.train(X_train, y_train, X_val, y_val)

    return xgb_model, nn_model, X_test, y_test
