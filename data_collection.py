# data_collection.py
"""
Step 1: Data Collection
-----------------------
Connects to PostgreSQL, runs queries with EXPLAIN ANALYZE,
and collects (estimated_rows, actual_rows, features) tuples.

This is your training data source. The more queries you run,
the better your model will be.
"""

import json
import csv
import psycopg2
import os
from config import DB_CONFIG, DATA_DIR


def get_connection():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def explain_query(conn, sql: str) -> dict:
    """
    Run EXPLAIN (ANALYZE, FORMAT JSON) on a query.
    Returns the full query plan as a Python dict.

    Args:
        conn: Active psycopg2 connection
        sql:  SQL query string to explain

    Returns:
        dict: The full JSON query plan from PostgreSQL
    """
    with conn.cursor() as cur:
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
        plan = cur.fetchone()[0][0]  # unwrap the JSON
    return plan


def extract_plan_nodes(plan: dict, nodes: list = None) -> list:
    """
    Recursively walk the query plan tree and extract every node.
    Each node contains estimated and actual row counts.

    Args:
        plan:  Query plan dict (or sub-plan)
        nodes: Accumulator list (used in recursion)

    Returns:
        list of dicts, one per plan node
    """
    if nodes is None:
        nodes = []

    node = plan.get("Plan", plan)  # top-level has a "Plan" key

    nodes.append({
        "node_type":        node.get("Node Type", "Unknown"),
        "estimated_rows":   node.get("Plan Rows", 0),
        "actual_rows":      node.get("Actual Rows", 0),
        "estimated_width":  node.get("Plan Width", 0),
        "estimated_cost":   node.get("Total Cost", 0.0),
        "actual_time":      node.get("Actual Total Time", 0.0),
        "join_type":        node.get("Join Type", "None"),
        "relation_name":    node.get("Relation Name", "None"),
        "index_name":       node.get("Index Name", "None"),
        "filter":           str(node.get("Filter", "")),
        "rows_removed_by_filter": node.get("Rows Removed by Filter", 0),
    })

    # Recurse into child plans
    for child in node.get("Plans", []):
        extract_plan_nodes(child, nodes)

    return nodes


def collect_training_data(query_list: list, output_file: str = None) -> list:
    """
    Main data collection loop.
    Runs each query, extracts plan nodes, and saves results.

    Args:
        query_list:  List of SQL query strings
        output_file: Optional CSV path to save collected data

    Returns:
        list of dicts (all plan nodes across all queries)
    """
    all_records = []
    conn = get_connection()

    print(f"Collecting data for {len(query_list)} queries...")

    for i, sql in enumerate(query_list):
        try:
            plan = explain_query(conn, sql)
            nodes = extract_plan_nodes(plan)
            all_records.extend(nodes)
            print(f"  [{i+1}/{len(query_list)}] Extracted {len(nodes)} nodes")
        except Exception as e:
            print(f"  [{i+1}] ERROR on query: {e}")
            conn.rollback()  # reset the connection after an error

    conn.close()

    # Save to CSV if requested
    if output_file and all_records:
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, output_file)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)
        print(f"Saved {len(all_records)} records to {filepath}")

    return all_records


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace these with real queries from your benchmark workload
    sample_queries = [
        "SELECT * FROM orders WHERE amount > 100",
        "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
        "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id",
    ]

    records = collect_training_data(sample_queries, output_file="raw_plan_data.csv")
    print(f"\nTotal plan nodes collected: {len(records)}")
