"""
Join Order Benchmark Runner
-----------------------------
Orchestrates the full JOB (Join Order Benchmark) evaluation loop:

  1. Load JOB queries via job_queries.py
  2. Run each query under two conditions:
       a. Default PostgreSQL planner
       b. ML-guided planner (hints from plan_selector.py)
  3. Collect per-query metrics: wall time, Q-error per node, total nodes
  4. Save results to CSV + JSON for evaluation.py to analyse

This is the main script you run after training your model.

Usage:
    python benchmark_runner.py
    python benchmark_runner.py --sql-dir benchmark/job_queries/ --limit 20
    python benchmark_runner.py --mode score --output results/benchmark.csv
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from benchmark.job_queries import load_all_job_queries
from pg_interceptor import PlanInterceptor
from plan_selector import PlanSelector
from config import DB_CONFIG, RESULTS_DIR, MODEL_DIR


# ── Per-query result container ─────────────────────────────────────────────────

class QueryResult:
    """Stores the outcome of running a single JOB query."""

    def __init__(self, query_id: str, sql: str):
        self.query_id = query_id
        self.sql = sql
        self.timestamp = datetime.now(UTC).isoformat()

        # Default (PostgreSQL) plan metrics
        self.pg_wall_time: Optional[float] = None
        self.pg_node_count: int = 0
        self.pg_q_error_median: Optional[float] = None
        self.pg_q_error_90th: Optional[float] = None
        self.pg_q_error_max: Optional[float] = None
        self.pg_total_cost: Optional[float] = None

        # ML-guided plan metrics
        self.ml_wall_time: Optional[float] = None
        self.ml_node_count: int = 0
        self.ml_q_error_median: Optional[float] = None
        self.ml_q_error_90th: Optional[float] = None
        self.ml_q_error_max: Optional[float] = None
        self.ml_total_cost: Optional[float] = None

        # Hints injected
        self.hints_used: dict = {}
        self.hint_count: int = 0

        # Error information
        self.error: Optional[str] = None
        self.skipped: bool = False

    @property
    def speedup(self) -> Optional[float]:
        """Wall-time speedup ratio: pg_time / ml_time (> 1 = ML is faster)."""
        if self.pg_wall_time is not None and self.ml_wall_time is not None and self.ml_wall_time > 0:
            return self.pg_wall_time / self.ml_wall_time
        return None

    @property
    def q_error_improvement(self) -> Optional[float]:
        """Reduction in median Q-error (positive = ML is more accurate)."""
        if self.pg_q_error_median is not None and self.ml_q_error_median is not None:
            return self.pg_q_error_median - self.ml_q_error_median
        return None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            # PostgreSQL baseline
            "pg_wall_time": self.pg_wall_time,
            "pg_node_count": self.pg_node_count,
            "pg_q_error_median": self.pg_q_error_median,
            "pg_q_error_90th": self.pg_q_error_90th,
            "pg_q_error_max": self.pg_q_error_max,
            "pg_total_cost": self.pg_total_cost,
            # ML-guided
            "ml_wall_time": self.ml_wall_time,
            "ml_node_count": self.ml_node_count,
            "ml_q_error_median": self.ml_q_error_median,
            "ml_q_error_90th": self.ml_q_error_90th,
            "ml_q_error_max": self.ml_q_error_max,
            "ml_total_cost": self.ml_total_cost,
            # Summary
            "speedup": self.speedup,
            "q_error_improvement": self.q_error_improvement,
            "hint_count": self.hint_count,
            "hints_used": json.dumps(self.hints_used),
            "error": self.error,
            "skipped": self.skipped,
        }


def _q_errors_from_nodes(nodes) -> np.ndarray:
    """Extract Q-error values from a list of PlanNode objects."""
    errors = [n.q_error for n in nodes if getattr(n, "actual_rows", 0) > 0]
    return np.array(errors) if errors else np.array([1.0])


def _node_metrics(nodes) -> dict:
    """Compute Q-error summary stats for a plan's nodes."""
    q_errs = _q_errors_from_nodes(nodes)
    total_cost = max((getattr(n, "total_cost", 0.0) for n in nodes), default=0.0)
    return {
        "node_count": len(nodes),
        "q_error_median": float(np.median(q_errs)),
        "q_error_90th": float(np.percentile(q_errs, 90)),
        "q_error_max": float(np.max(q_errs)),
        "total_cost": total_cost,
    }


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}s"
    except Exception:
        return "N/A"


def _fmt_speedup(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}×"
    except Exception:
        return "N/A"


# ── Benchmark Runner ──────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs the Join Order Benchmark with and without ML-guided planning,
    collecting metrics for comparison.

    Args:
        db_config:     psycopg2 connection kwargs
        model_path:    Path to trained XGBoost model
        mode:          "hint" or "score" (passed to PlanSelector)
        hint_threshold: Q-error threshold to trigger hint injection (default 2.0)
        timeout_ms:    Per-query statement timeout in milliseconds
    """

    def __init__(
        self,
        db_config: dict = None,
        model_path: str = None,
        mode: str = "hint",
        hint_threshold: float = 2.0,
        timeout_ms: int = 60_000,
    ):
        self.db_config = db_config or DB_CONFIG
        self.mode = mode
        self.hint_threshold = hint_threshold
        self.timeout_ms = timeout_ms

        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "xgb_cardinality_model.pkl")
        self.model_path = model_path

        self.selector = PlanSelector(model_path=model_path, mode=mode)
        self.interceptor = PlanInterceptor(db_config=self.db_config, timeout_ms=timeout_ms)
        self.results: List[QueryResult] = []

    def run_single_query(self, query_id: str, sql: str) -> QueryResult:
        """
        Run one JOB query under both conditions and record results.

        Args:
            query_id: JOB query identifier (e.g. "1a")
            sql:      SQL query string

        Returns:
            Populated QueryResult object
        """
        qr = QueryResult(query_id, sql)

        # ── Condition A: Default PostgreSQL planner ───────────────────────────
        try:
            pg_result = self.interceptor.run_and_capture(sql)
            qr.pg_wall_time = pg_result["wall_time"]
            m = _node_metrics(pg_result["nodes"])
            qr.pg_node_count = m["node_count"]
            qr.pg_q_error_median = m["q_error_median"]
            qr.pg_q_error_90th = m["q_error_90th"]
            qr.pg_q_error_max = m["q_error_max"]
            qr.pg_total_cost = m["total_cost"]
        except Exception as e:
            qr.error = f"PG baseline error: {e}"
            qr.skipped = True
            print(f"  [SKIP] {query_id}: {e}")
            return qr

        # ── Condition B: ML-guided plan ───────────────────────────────────────
        try:
            ml_result = self.selector.select_with_hints(
                sql,
                self.interceptor,
                threshold=self.hint_threshold,
            )
            hinted = ml_result["hinted_plan"]
            qr.ml_wall_time = hinted["wall_time"]
            m2 = _node_metrics(hinted["nodes"])
            qr.ml_node_count = m2["node_count"]
            qr.ml_q_error_median = m2["q_error_median"]
            qr.ml_q_error_90th = m2["q_error_90th"]
            qr.ml_q_error_max = m2["q_error_max"]
            qr.ml_total_cost = m2["total_cost"]
            qr.hints_used = ml_result["hints_used"]
            qr.hint_count = len(qr.hints_used)

        except Exception as e:
            qr.error = f"ML plan error: {e}"
            print(f"  [ML-ERROR] {query_id}: {e}")

        return qr

    def run_benchmark(
        self,
        queries: Dict[str, str],
        limit: Optional[int] = None,
        warmup_runs: int = 1,
    ) -> List[QueryResult]:
        """
        Run the full benchmark across all supplied queries.

        Args:
            queries:     {query_id: sql} dict (from job_queries.load_all_job_queries)
            limit:       Run only the first N queries (useful for quick tests)
            warmup_runs: Number of throwaway runs to warm the PostgreSQL buffer cache

        Returns:
            List of QueryResult objects (also stored in self.results)
        """
        query_items = list(queries.items())
        if limit:
            query_items = query_items[:limit]

        total = len(query_items)
        print(f"\n{'=' * 60}")
        print(f"JOB Benchmark — {total} queries, mode={self.mode}")
        print(f"{'=' * 60}\n")

        self.interceptor.connect()

        # Optional warmup pass
        if warmup_runs > 0 and query_items:
            print(f"Warmup pass ({warmup_runs} run(s) of first query)...")
            first_sql = query_items[0][1]
            for _ in range(warmup_runs):
                try:
                    self.interceptor.run_and_capture(first_sql)
                except Exception:
                    pass
            print("Warmup done.\n")

        self.results = []
        t_start = time.perf_counter()

        for i, (qid, sql) in enumerate(query_items, 1):
            print(f"[{i:3d}/{total}] Query {qid}:")
            qr = self.run_single_query(qid, sql)
            self.results.append(qr)

            if not qr.skipped:
                print(
                    f"         PG: {_fmt_seconds(qr.pg_wall_time)}  "
                    f"ML: {_fmt_seconds(qr.ml_wall_time)}  "
                    f"Speedup: {_fmt_speedup(qr.speedup)}  "
                    f"Hints: {qr.hint_count}"
                )
            else:
                print("         Skipped.")

        elapsed = time.perf_counter() - t_start
        print(f"\nBenchmark complete in {elapsed:.1f}s")
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print a brief aggregate summary to stdout."""
        valid = [r for r in self.results if not r.skipped and r.speedup is not None]
        if not valid:
            print("No valid results to summarise.")
            return

        speedups = [r.speedup for r in valid]
        q_improvements = [r.q_error_improvement for r in valid if r.q_error_improvement is not None]

        print(f"\n{'─' * 50}")
        print(f"Summary ({len(valid)} / {len(self.results)} queries succeeded)")
        print(
            f"  Wall-time speedup — "
            f"median: {np.median(speedups):.2f}×  "
            f"mean: {np.mean(speedups):.2f}×  "
            f"max: {np.max(speedups):.2f}×"
        )
        if q_improvements:
            print(
                f"  Q-error improvement (median) — "
                f"mean: {np.mean(q_improvements):.2f}  "
                f"max: {np.max(q_improvements):.2f}"
            )
        print(f"{'─' * 50}\n")

    # ── Output helpers ────────────────────────────────────────────────────────

    def save_results_csv(self, path: str = None) -> str:
        """
        Save all QueryResult objects to a CSV file.

        Args:
            path: Output CSV path (default: results/job_benchmark_TIMESTAMP.csv)

        Returns:
            Resolved output path
        """
        if path is None:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RESULTS_DIR, f"job_benchmark_{ts}.csv")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [r.to_dict() for r in self.results]

        with open(path, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        print(f"Results saved to: {path}")
        return path

    def save_results_json(self, path: str = None) -> str:
        """Save all results as JSON (easier for downstream analysis)."""
        if path is None:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RESULTS_DIR, f"job_benchmark_{ts}.json")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

        print(f"JSON results saved to: {path}")
        return path

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a pandas DataFrame for analysis."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def __del__(self):
        try:
            self.interceptor.disconnect()
        except Exception:
            pass


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the Join Order Benchmark with ML-guided planning"
    )
    parser.add_argument(
        "--sql-dir",
        default=None,
        help="Directory containing JOB .sql files (optional; uses builtins if not set)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N queries (for quick tests)",
    )
    parser.add_argument(
        "--mode",
        choices=["hint", "score"],
        default="hint",
        help="Plan selection mode: 'hint' (pg_hint_plan) or 'score' (cost model)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Q-error threshold to trigger hint injection (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (optional; auto-named if not set)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Path to trained model (default: {os.path.join(MODEL_DIR, 'xgb_cardinality_model.pkl')})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60_000,
        help="Per-query timeout in milliseconds (default: 60000)",
    )
    args = parser.parse_args()

    queries = load_all_job_queries(sql_dir=args.sql_dir)

    runner = BenchmarkRunner(
        model_path=args.model,
        mode=args.mode,
        hint_threshold=args.threshold,
        timeout_ms=args.timeout,
    )

    runner.run_benchmark(queries, limit=args.limit)

    csv_path = runner.save_results_csv(args.output)
    json_path = runner.save_results_json(
        args.output.replace(".csv", ".json") if args.output else None
    )

    print(f"\nDone. Results at:\n  CSV : {csv_path}\n  JSON: {json_path}")


if __name__ == "__main__":
    main()