# evaluation.py
"""
Step 4: Evaluation
-------------------
Metrics used in the cardinality estimation literature (as cited in your report):

  - Q-Error  : the standard metric in query optimisation research.
               Measures how far off an estimate is, multiplicatively.
               Q-error = max(actual/estimated, estimated/actual)
               A perfect estimate → Q-error = 1.0

  - RMSE     : root mean squared error on log-scale predictions.

  - MAE      : mean absolute error on log-scale predictions.

  - R²       : how much variance our model explains.

  - Median Q-Error and percentiles (50th, 90th, 95th, 99th) are the
               standard reporting format in papers like Lero and ALECE.

Extended in Phase 2 to include JOB benchmark analysis functions.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use("Agg")   # headless — safe on Windows/servers without a display
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import RESULTS_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 1. Core cardinality estimation metrics
# ══════════════════════════════════════════════════════════════════════════════

def q_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Compute Q-error for each prediction.

    Q-error = max(actual / predicted, predicted / actual)

    Both arrays should be raw row counts (not log-transformed).
    Clamps predictions to at least 1 to avoid division by zero.

    Args:
        actual:    True row counts (from EXPLAIN ANALYZE actual_rows)
        predicted: Model-predicted row counts

    Returns:
        Array of Q-error values (≥ 1.0 for each prediction)
    """
    actual    = np.maximum(actual, 1.0)
    predicted = np.maximum(predicted, 1.0)
    return np.maximum(actual / predicted, predicted / actual)


def evaluate_model(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                   model_name: str = "Model") -> dict:
    """
    Full evaluation of a cardinality estimator.

    Args:
        y_true_log:  Ground-truth log(actual_rows + 1)
        y_pred_log:  Predicted log(actual_rows + 1)
        model_name:  Label for printing / saving

    Returns:
        dict of metrics (suitable for saving to JSON or a DataFrame)
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    qerr = q_error(y_true, y_pred)

    metrics = {
        "model":            model_name,
        "rmse_log":         float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log":          float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2":               float(r2_score(y_true_log, y_pred_log)),
        "q_error_median":   float(np.median(qerr)),
        "q_error_90th":     float(np.percentile(qerr, 90)),
        "q_error_95th":     float(np.percentile(qerr, 95)),
        "q_error_99th":     float(np.percentile(qerr, 99)),
        "q_error_max":      float(np.max(qerr)),
        "pct_within_2x":    float(np.mean(qerr <= 2.0) * 100),
        "pct_within_10x":   float(np.mean(qerr <= 10.0) * 100),
    }

    print(f"\n{'=' * 50}")
    print(f"Results for: {model_name}")
    print(f"{'=' * 50}")
    print(f"  RMSE (log scale):        {metrics['rmse_log']:.4f}")
    print(f"  MAE  (log scale):        {metrics['mae_log']:.4f}")
    print(f"  R²:                      {metrics['r2']:.4f}")
    print(f"  Q-Error  Median:         {metrics['q_error_median']:.2f}×")
    print(f"  Q-Error  90th pct:       {metrics['q_error_90th']:.2f}×")
    print(f"  Q-Error  95th pct:       {metrics['q_error_95th']:.2f}×")
    print(f"  Q-Error  99th pct:       {metrics['q_error_99th']:.2f}×")
    print(f"  Q-Error  Max:            {metrics['q_error_max']:.2f}×")
    print(f"  Estimates within  2×:    {metrics['pct_within_2x']:.1f}%")
    print(f"  Estimates within 10×:    {metrics['pct_within_10x']:.1f}%")

    return metrics


def compare_with_postgres_baseline(y_true_log: np.ndarray,
                                   y_postgres_log: np.ndarray,
                                   y_model_log: np.ndarray) -> pd.DataFrame:
    """
    Side-by-side comparison of PostgreSQL's built-in estimator
    vs your ML model.

    This is the key table for your evaluation chapter.
    """
    postgres_metrics = evaluate_model(y_true_log, y_postgres_log, "PostgreSQL (baseline)")
    model_metrics    = evaluate_model(y_true_log, y_model_log,    "ML Model")

    comparison = pd.DataFrame([postgres_metrics, model_metrics]).set_index("model")
    return comparison


def plot_q_error_cdf(q_errors_dict: dict, save_path: str = None):
    """
    Plot the CDF of Q-errors for one or more models.
    Standard visualisation format in the cardinality estimation literature.

    Args:
        q_errors_dict: {"Model Name": np.ndarray_of_q_errors, ...}
        save_path:     Optional path to save the figure
    """
    plt.figure(figsize=(8, 5))

    for name, qerr in q_errors_dict.items():
        sorted_qerr = np.sort(qerr)
        cdf = np.arange(1, len(sorted_qerr) + 1) / len(sorted_qerr)
        plt.plot(sorted_qerr, cdf, label=name)

    plt.xscale("log")
    plt.xlabel("Q-Error (log scale)")
    plt.ylabel("Cumulative Fraction of Queries")
    plt.title("Q-Error CDF — Cardinality Estimation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    plt.close()


def save_results(metrics_list: list, filename: str = "results.json"):
    """Save a list of metric dicts to JSON for your dissertation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Results saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. JOB Benchmark analysis  (Phase 2 — uses benchmark_runner output)
# ══════════════════════════════════════════════════════════════════════════════

def load_benchmark_results(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV produced by benchmark_runner.py into a DataFrame.

    Args:
        csv_path: Path to benchmark results CSV

    Returns:
        DataFrame with one row per JOB query
    """
    df = pd.read_csv(csv_path)
    # Parse JSON-encoded hints column
    if "hints_used" in df.columns:
        df["hints_used"] = df["hints_used"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
    print(f"Loaded {len(df)} benchmark results from {csv_path}")
    return df


def analyse_benchmark(df: pd.DataFrame) -> dict:
    """
    Compute aggregate statistics from benchmark runner output.

    Args:
        df: DataFrame from load_benchmark_results()

    Returns:
        dict of summary statistics
    """
    valid = df[~df["skipped"].fillna(False) & df["speedup"].notna()]

    speedups      = valid["speedup"].values
    q_improvements = valid["q_error_improvement"].dropna().values

    stats = {
        "total_queries":         len(df),
        "successful_queries":    len(valid),
        "skipped_queries":       len(df) - len(valid),
        "speedup_median":        float(np.median(speedups)),
        "speedup_mean":          float(np.mean(speedups)),
        "speedup_max":           float(np.max(speedups)),
        "speedup_min":           float(np.min(speedups)),
        "pct_queries_faster":    float(np.mean(speedups > 1.0) * 100),
        "q_error_improvement_mean": float(np.mean(q_improvements)) if len(q_improvements) else 0.0,
        "avg_hints_injected":    float(valid["hint_count"].mean()),
        "queries_with_hints":    int((valid["hint_count"] > 0).sum()),
    }

    print("\n── JOB Benchmark Summary ──────────────────────────────────────")
    print(f"  Queries run:          {stats['total_queries']}")
    print(f"  Queries succeeded:    {stats['successful_queries']}")
    print(f"  Wall-time speedup:    median {stats['speedup_median']:.2f}×  "
          f"mean {stats['speedup_mean']:.2f}×  max {stats['speedup_max']:.2f}×")
    print(f"  Queries faster:       {stats['pct_queries_faster']:.1f}%")
    print(f"  Avg hints injected:   {stats['avg_hints_injected']:.1f}")
    print(f"  Queries with hints:   {stats['queries_with_hints']}")
    print("─" * 60)

    return stats


def plot_speedup_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Bar chart of per-query wall-time speedup (ML vs PostgreSQL default).
    Queries sorted by speedup — great for your evaluation chapter figure.
    """
    valid = df[~df["skipped"].fillna(False) & df["speedup"].notna()].copy()
    valid = valid.sort_values("speedup", ascending=False)

    fig, ax = plt.subplots(figsize=(max(10, len(valid) * 0.3), 5))
    colors = ["#2ecc71" if s >= 1.0 else "#e74c3c" for s in valid["speedup"]]
    ax.bar(valid["query_id"], valid["speedup"], color=colors)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, label="No change (1×)")
    ax.set_xlabel("JOB Query ID")
    ax.set_ylabel("Speedup (PG time / ML time)")
    ax.set_title("Wall-time Speedup — ML-guided vs Default PostgreSQL Planner")
    ax.legend()
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Speedup chart saved to {save_path}")
    plt.close()


def plot_q_error_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Box plot comparing Q-error distribution:
    PostgreSQL default vs ML-guided plan per query.
    """
    valid = df[~df["skipped"].fillna(False)].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        valid["pg_q_error_median"].dropna().values,
        valid["ml_q_error_median"].dropna().values,
    ]
    ax.boxplot(data, labels=["PostgreSQL\n(default)", "ML-guided"], patch_artist=True,
               medianprops={"color": "red", "linewidth": 2})
    ax.set_ylabel("Median Q-Error per Query")
    ax.set_title("Q-Error Distribution — PostgreSQL vs ML-Guided Planner")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Q-error comparison plot saved to {save_path}")
    plt.close()


def generate_full_report(benchmark_csv: str,
                         output_dir: str = None) -> str:
    """
    One-shot: load benchmark results and produce all evaluation outputs.

    Args:
        benchmark_csv: Path to CSV from benchmark_runner.py
        output_dir:    Directory for output files (defaults to RESULTS_DIR)

    Returns:
        Path to the JSON summary file
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    df    = load_benchmark_results(benchmark_csv)
    stats = analyse_benchmark(df)

    # Save aggregate stats
    stats_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary JSON → {stats_path}")

    # Plots
    plot_speedup_distribution(df, os.path.join(output_dir, "speedup_distribution.png"))
    plot_q_error_comparison(df,   os.path.join(output_dir, "q_error_comparison.png"))

    print(f"\nAll evaluation outputs written to: {output_dir}")
    return stats_path


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    y_true     = np.random.randint(1, 10000, size=500).astype(float)
    y_postgres = y_true * np.random.uniform(0.1, 10, size=500)
    y_model    = y_true * np.random.uniform(0.5, 2.0, size=500)

    y_true_log     = np.log1p(y_true)
    y_postgres_log = np.log1p(y_postgres)
    y_model_log    = np.log1p(y_model)

    comparison = compare_with_postgres_baseline(y_true_log, y_postgres_log, y_model_log)
    print("\nComparison Table:")
    print(comparison.to_string())

    plot_q_error_cdf({
        "PostgreSQL": q_error(y_true, y_postgres),
        "ML Model":   q_error(y_true, y_model),
    }, save_path="results/q_error_cdf.png")
