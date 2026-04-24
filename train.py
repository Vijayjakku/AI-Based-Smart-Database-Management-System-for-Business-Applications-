# train.py
"""
Step 5: Main Training Script
-----------------------------
This is your entry point. Run this to:
  1. Load collected data
  2. Prepare features
  3. Train XGBoost + Neural Network models
  4. Evaluate both vs PostgreSQL baseline
  5. Save models and results

Usage:
    python train.py
    python train.py --data data/raw_plan_data.csv
"""

import argparse
import numpy as np
import pandas as pd

from feature_engineering import load_raw_data, prepare_features
from model import train_all, XGBoostCardinalityEstimator, NeuralCardinalityEstimator
from evaluation import (
    evaluate_model, compare_with_postgres_baseline,
    plot_q_error_cdf, q_error, save_results
)


def main(data_path: str):
    print("=" * 60)
    print("Cardinality Estimation Training Pipeline")
    print("=" * 60)

    # ── 1. Load and prepare data ──────────────────────────────────
    print("\n[1/4] Loading and preparing features...")
    df = load_raw_data(data_path)
    X, y = prepare_features(df, fit=True)

    # Keep the PostgreSQL estimates as a baseline column
    # (log-transformed for fair comparison)
    postgres_log_estimates = np.log1p(df["estimated_rows"].values)

    # ── 2. Train models ───────────────────────────────────────────
    print("\n[2/4] Training models...")
    xgb_model, nn_model, X_test, y_test = train_all(X, y)

    # Get the PostgreSQL baseline estimates for the same test rows
    test_indices = y_test.index
    pg_baseline_test = postgres_log_estimates[test_indices]

    # ── 3. Evaluate ───────────────────────────────────────────────
    print("\n[3/4] Evaluating models...")

    xgb_preds_log = np.log1p(xgb_model.predict(X_test))
    nn_preds_log  = np.log1p(nn_model.predict(X_test))
    y_test_arr    = y_test.values

    xgb_metrics = evaluate_model(y_test_arr, xgb_preds_log, "XGBoost")
    nn_metrics  = evaluate_model(y_test_arr, nn_preds_log,  "Neural Network (MLP)")

    comparison = compare_with_postgres_baseline(
        y_test_arr, pg_baseline_test, xgb_preds_log
    )
    print("\n── Comparison vs PostgreSQL Baseline ──")
    print(comparison.to_string())

    # Q-error CDF (save to results/)
    plot_q_error_cdf({
        "PostgreSQL (baseline)": q_error(np.expm1(y_test_arr), np.expm1(pg_baseline_test)),
        "XGBoost":               q_error(np.expm1(y_test_arr), xgb_model.predict(X_test)),
        "Neural Network (MLP)":  q_error(np.expm1(y_test_arr), nn_model.predict(X_test)),
    }, save_path="results/q_error_cdf.png")

    # ── 4. Save models and results ────────────────────────────────
    print("\n[4/4] Saving models and results...")
    xgb_model.save()
    nn_model.save()
    save_results([xgb_metrics, nn_metrics], "model_comparison.json")

    # Feature importance (great for dissertation analysis section)
    importance_df = xgb_model.feature_importance(list(X_test.columns))
    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv("results/feature_importance.csv", index=False)

    print("\nDone! Check results/ and models/ directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/raw_plan_data.csv",
        help="Path to collected plan data CSV"
    )
    args = parser.parse_args()
    main(args.data)
