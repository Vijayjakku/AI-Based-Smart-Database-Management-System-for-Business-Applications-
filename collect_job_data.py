"""
collect_job_data.py

Collect plan-node training data from JOB queries using PostgreSQL EXPLAIN ANALYZE.

Outputs a CSV with one row per plan node, compatible with feature_engineering.py
and train.py in this project.
"""

import argparse
import csv
import os
from typing import Dict, List

import pandas as pd

from benchmark.job_queries import load_all_job_queries
from pg_interceptor import PlanInterceptor
from config import DB_CONFIG, DATA_DIR


def node_to_row(query_id: str, node) -> Dict:
    """Convert a PlanNode to a flat training-row dict."""
    return {
        "query_id": query_id,
        "node_type": node.node_type,
        "relation_name": node.relation_name or "None",
        "alias": node.alias or "None",
        "join_type": node.join_type or "None",
        "index_name": node.index_name or "None",
        "filter": node.filter or "",
        "depth": node.depth,
        "estimated_rows": node.estimated_rows,
        "estimated_width": node.estimated_width,
        "estimated_cost": node.total_cost,
        "actual_rows": node.actual_rows,
        "actual_time": node.actual_total_time,
        "rows_removed_by_filter": node.rows_removed_by_filter,
        "shared_hit_blocks": node.shared_hit_blocks,
        "shared_read_blocks": node.shared_read_blocks,
        "q_error": node.q_error,
        "estimation_ratio": node.estimation_ratio,
    }


def collect_dataset(sql_dir: str | None, limit: int | None, timeout_ms: int) -> pd.DataFrame:
    queries = load_all_job_queries(sql_dir=sql_dir)
    items = list(queries.items())
    if limit:
        items = items[:limit]

    interceptor = PlanInterceptor(db_config=DB_CONFIG, timeout_ms=timeout_ms)
    interceptor.connect()

    rows: List[Dict] = []
    total = len(items)
    print(f"Collecting training data from {total} queries...")

    for i, (query_id, sql) in enumerate(items, 1):
        try:
            result = interceptor.run_and_capture(sql)
            nodes = result["nodes"]
            for node in nodes:
                rows.append(node_to_row(query_id, node))
            print(f"[{i}/{total}] {query_id}: {len(nodes)} nodes")
        except Exception as e:
            print(f"[{i}/{total}] {query_id}: ERROR -> {e}")
            try:
                interceptor._conn.rollback()
            except Exception:
                pass

    interceptor.disconnect()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect JOB plan-node training data")
    parser.add_argument(
        "--sql-dir",
        default=None,
        help="Path to JOB .sql files (e.g. join-order-benchmark/queries). "
             "If omitted, uses only the 10 built-in sample queries."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to run."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "raw_plan_data.csv"),
        help="Output CSV path."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60000,
        help="Per-query timeout in milliseconds."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = collect_dataset(args.sql_dir, args.limit, args.timeout)

    if df.empty:
        raise SystemExit("No rows collected. Check PostgreSQL connection and query data.")

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} plan-node rows to {args.output}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
