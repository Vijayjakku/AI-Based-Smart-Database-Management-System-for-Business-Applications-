# feature_engineering.py
"""
Step 2: Feature Engineering
----------------------------
Transforms raw plan node data into numeric features
that the ML model can learn from.

Key insight: cardinality estimation errors come from the GAP
between estimated_rows and actual_rows. Our model learns to
predict actual_rows more accurately than PostgreSQL's default
cost-based estimator.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from config import MODEL_DIR


# Node types present in PostgreSQL plans (add more if you see others)
NODE_TYPES = [
    "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
    "Nested Loop", "Hash Join", "Merge Join",
    "Aggregate", "Sort", "Hash", "Limit", "Append",
]

JOIN_TYPES = ["Inner", "Left", "Right", "Full", "Semi", "Anti", "None"]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the CSV produced by data_collection.py."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(df.dtypes)
    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    One-hot encode node_type and join_type.
    When fit=True (training), saves the encoder for later use.
    When fit=False (inference), loads the saved encoder.

    Args:
        df:  Raw dataframe
        fit: True during training, False during inference

    Returns:
        DataFrame with encoded columns added, originals dropped
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    encoder_path = os.path.join(MODEL_DIR, "label_encoders.pkl")

    if fit:
        # One-hot encode the categorical columns
        node_dummies = pd.get_dummies(df["node_type"], prefix="node")
        join_dummies = pd.get_dummies(df["join_type"], prefix="join")

        # Save the column names so inference can align columns
        meta = {
            "node_columns": list(node_dummies.columns),
            "join_columns":  list(join_dummies.columns),
        }
        joblib.dump(meta, encoder_path)
    else:
        meta = joblib.load(encoder_path)
        node_dummies = pd.get_dummies(df["node_type"], prefix="node").reindex(
            columns=meta["node_columns"], fill_value=0
        )
        join_dummies = pd.get_dummies(df["join_type"], prefix="join").reindex(
            columns=meta["join_columns"], fill_value=0
        )

    df = df.drop(columns=["node_type", "join_type",
                           "relation_name", "index_name", "filter"])
    df = pd.concat([df, node_dummies, join_dummies], axis=1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that help the model learn:
      - log-scale row counts (cardinalities are log-normal)
      - estimation error ratio (useful for analysis, not a training feature)
      - filter selectivity proxy

    Args:
        df: Raw plan node DataFrame

    Returns:
        DataFrame with new feature columns
    """
    # PostgreSQL row estimates are log-normally distributed → log-transform
    df["log_estimated_rows"] = np.log1p(df["estimated_rows"])
    df["log_estimated_cost"] = np.log1p(df["estimated_cost"])
    df["log_estimated_width"] = np.log1p(df["estimated_width"])

    # Filter selectivity: what fraction of rows survived the filter?
    df["filter_selectivity"] = np.where(
        df["estimated_rows"] > 0,
        1 - (df["rows_removed_by_filter"] / (df["estimated_rows"] + 1e-6)),
        1.0,
    )

    return df


def prepare_features(df: pd.DataFrame, fit: bool = True):
    """
    Full feature preparation pipeline:
      1. Engineer derived features
      2. Encode categoricals
      3. Select final feature columns
      4. Scale (optional but good practice)

    Args:
        df:  Raw plan node DataFrame
        fit: True for training, False for inference

    Returns:
        X (features DataFrame), y (log-actual-rows Series)
    """
    df = engineer_features(df)
    df = encode_categoricals(df, fit=fit)

    # Target variable: predict log(actual_rows + 1)
    # We use log scale because cardinalities span many orders of magnitude
    y = np.log1p(df["actual_rows"])

   # Drop columns that are not features
    drop_cols = ["actual_rows", "actual_time"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ❌ REMOVE leakage / unavailable features
    X = X.drop(columns=[
        "q_error",
        "estimation_ratio",
        "shared_hit_blocks",
        "shared_read_blocks",
        "depth"
    ], errors="ignore")

    # Keep only numeric
    X = X.select_dtypes(include=["int64", "float64"])



    print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")
    return X, y

# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_raw_data("data/raw_plan_data.csv")
    X, y = prepare_features(df, fit=True)
    print("\nSample features:")
    print(X.head(3))
    print("\nTarget (log actual rows):", y.head(3).values)
