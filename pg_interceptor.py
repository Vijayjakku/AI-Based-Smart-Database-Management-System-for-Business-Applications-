# pg_interceptor.py
"""
PostgreSQL Plan Interceptor
-----------------------------
Intercepts query execution to capture both PostgreSQL's native query plan
(with its built-in cardinality estimates) and the actual execution results.

This module sits between your benchmark runner and PostgreSQL:
  - Forces PostgreSQL to use specific join orders / plan hints
  - Captures estimated vs actual row counts at every plan node
  - Optionally injects ML-corrected cardinality hints via pg_hint_plan

Prerequisites:
    pip install psycopg2-binary
    (optional) pg_hint_plan extension in PostgreSQL for plan injection

Usage:
    from pg_interceptor import PlanInterceptor
    interceptor = PlanInterceptor(db_config)
    result = interceptor.run_and_capture("SELECT ...")
    plan_nodes = result["nodes"]
"""

import json
import time
import psycopg2
import psycopg2.extras
from typing import Dict, List, Optional, Tuple, Any

from config import DB_CONFIG


class PlanNode:
    """
    Represents a single node in a PostgreSQL query plan tree.
    Stores both estimated and actual statistics for comparison.
    """

    def __init__(self, raw_node: dict, depth: int = 0):
        self.depth          = depth
        self.node_type      = raw_node.get("Node Type", "Unknown")
        self.relation_name  = raw_node.get("Relation Name", "")
        self.alias          = raw_node.get("Alias", "")
        self.index_name     = raw_node.get("Index Name", "")
        self.join_type      = raw_node.get("Join Type", "")
        self.filter         = raw_node.get("Filter", "")
        self.index_cond     = raw_node.get("Index Cond", "")
        self.hash_cond      = raw_node.get("Hash Cond", "")

        # Estimated (plan-time) statistics
        self.estimated_rows  = raw_node.get("Plan Rows", 0)
        self.estimated_width = raw_node.get("Plan Width", 0)
        self.startup_cost    = raw_node.get("Startup Cost", 0.0)
        self.total_cost      = raw_node.get("Total Cost", 0.0)

        # Actual (execution-time) statistics
        self.actual_rows           = raw_node.get("Actual Rows", 0)
        self.actual_loops          = raw_node.get("Actual Loops", 1)
        self.actual_startup_time   = raw_node.get("Actual Startup Time", 0.0)
        self.actual_total_time     = raw_node.get("Actual Total Time", 0.0)
        self.rows_removed_by_filter = raw_node.get("Rows Removed by Filter", 0)

        # Buffer statistics (requires BUFFERS option)
        self.shared_hit_blocks  = raw_node.get("Shared Hit Blocks", 0)
        self.shared_read_blocks = raw_node.get("Shared Read Blocks", 0)

        self.children: List["PlanNode"] = []

    @property
    def q_error(self) -> float:
        """
        Q-error for this plan node.
        Q-error = max(actual/estimated, estimated/actual)
        A perfect estimate → Q-error = 1.0
        """
        est  = max(self.estimated_rows, 1)
        act  = max(self.actual_rows, 1)
        return max(est / act, act / est)

    @property
    def estimation_ratio(self) -> float:
        """estimated / actual  (> 1 = over-estimate, < 1 = under-estimate)"""
        act = max(self.actual_rows, 1)
        return self.estimated_rows / act

    def to_dict(self) -> dict:
        """Serialise to flat dict (used for CSV / DataFrame rows)."""
        return {
            "node_type":              self.node_type,
            "relation_name":          self.relation_name,
            "alias":                  self.alias,
            "join_type":              self.join_type,
            "index_name":             self.index_name,
            "filter":                 self.filter,
            "depth":                  self.depth,
            "estimated_rows":         self.estimated_rows,
            "estimated_width":        self.estimated_width,
            "startup_cost":           self.startup_cost,
            "total_cost":             self.total_cost,
            "actual_rows":            self.actual_rows,
            "actual_loops":           self.actual_loops,
            "actual_startup_time":    self.actual_startup_time,
            "actual_total_time":      self.actual_total_time,
            "rows_removed_by_filter": self.rows_removed_by_filter,
            "shared_hit_blocks":      self.shared_hit_blocks,
            "shared_read_blocks":     self.shared_read_blocks,
            "q_error":                self.q_error,
            "estimation_ratio":       self.estimation_ratio,
        }

    def __repr__(self):
        return (f"PlanNode({self.node_type}, "
                f"est={self.estimated_rows}, act={self.actual_rows}, "
                f"q_err={self.q_error:.2f})")


def _parse_plan_tree(raw: dict, depth: int = 0) -> PlanNode:
    """
    Recursively parse a JSON query plan into a PlanNode tree.

    Args:
        raw:   Raw plan dict (from EXPLAIN JSON output)
        depth: Current tree depth (0 = root)

    Returns:
        Root PlanNode with children attached
    """
    # Top-level response has a "Plan" wrapper
    node_dict = raw.get("Plan", raw)
    node = PlanNode(node_dict, depth=depth)

    for child_raw in node_dict.get("Plans", []):
        node.children.append(_parse_plan_tree(child_raw, depth=depth + 1))

    return node


def _flatten_tree(root: PlanNode) -> List[PlanNode]:
    """Depth-first flatten of a PlanNode tree into a list."""
    result = [root]
    for child in root.children:
        result.extend(_flatten_tree(child))
    return result


class PlanInterceptor:
    """
    Connects to PostgreSQL and intercepts query plans.

    Features:
      - EXPLAIN ANALYZE with JSON output
      - Recursive plan tree parsing
      - pg_hint_plan integration for injecting cardinality hints
      - GUC (SET) parameter control (enable_hashjoin, etc.)

    Usage:
        interceptor = PlanInterceptor(DB_CONFIG)
        result = interceptor.run_and_capture("SELECT ...")
    """

    def __init__(self, db_config: dict = None, timeout_ms: int = 30_000):
        """
        Args:
            db_config:  psycopg2 connection kwargs (host, port, dbname, user, password)
            timeout_ms: Statement timeout in milliseconds (default 30s)
        """
        self.db_config  = db_config or DB_CONFIG
        self.timeout_ms = timeout_ms
        self._conn: Optional[psycopg2.extensions.connection] = None

    # ── Connection management ─────────────────────────────────────────────────

    def connect(self):
        """Open a database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.db_config)
            self._conn.set_session(autocommit=True)
            with self._conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {self.timeout_ms};")
        return self._conn

    def disconnect(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    # ── Core interception ─────────────────────────────────────────────────────

    def explain_only(self, sql: str) -> dict:
        """
        Run EXPLAIN (no ANALYZE) to get the plan without executing the query.
        Fast — use this for plan inspection without execution overhead.

        Returns:
            Raw plan dict from PostgreSQL JSON output
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
            return cur.fetchone()[0][0]

    def run_and_capture(self, sql: str,
                        buffers: bool = True) -> Dict[str, Any]:
        """
        Execute a query with EXPLAIN ANALYZE and capture the full plan.

        Args:
            sql:     SQL query string
            buffers: Include buffer hit/read statistics

        Returns:
            dict with keys:
              - "root":       PlanNode tree root
              - "nodes":      Flat list of PlanNode objects
              - "records":    List of dicts (for DataFrame / CSV)
              - "wall_time":  Total wall-clock execution time (seconds)
              - "plan_json":  Raw JSON plan from PostgreSQL
        """
        conn   = self.connect()
        opts   = "ANALYZE, BUFFERS, FORMAT JSON" if buffers else "ANALYZE, FORMAT JSON"
        t0     = time.perf_counter()

        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN ({opts}) {sql}")
            raw_plan = cur.fetchone()[0][0]

        wall_time = time.perf_counter() - t0

        root    = _parse_plan_tree(raw_plan)
        nodes   = _flatten_tree(root)
        records = [n.to_dict() for n in nodes]

        return {
            "root":      root,
            "nodes":     nodes,
            "records":   records,
            "wall_time": wall_time,
            "plan_json": raw_plan,
        }

    # ── GUC controls (for ablation / plan comparison) ─────────────────────────

    def set_guc(self, param: str, value: str):
        """
        Set a PostgreSQL GUC (Grand Unified Configuration) parameter
        for the current session.

        Common params for cardinality / plan experiments:
            enable_hashjoin    = on/off
            enable_nestloop    = on/off
            enable_mergejoin   = on/off
            enable_seqscan     = on/off
            enable_indexscan   = on/off
            join_collapse_limit = 1  (disables join reordering)
            from_collapse_limit = 1

        Args:
            param: GUC parameter name
            value: Value string
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(f"SET {param} = {value};")

    def reset_guc(self, param: str):
        """Reset a GUC parameter to its default value."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(f"RESET {param};")

    def with_forced_join_order(self, sql: str) -> Dict[str, Any]:
        """
        Capture plan with join reordering disabled.
        PostgreSQL will use the FROM-clause order as-is.

        Useful for: comparing the default join order vs ML-suggested orders.
        """
        self.set_guc("join_collapse_limit", "1")
        self.set_guc("from_collapse_limit", "1")
        try:
            return self.run_and_capture(sql)
        finally:
            self.reset_guc("join_collapse_limit")
            self.reset_guc("from_collapse_limit")

    # ── pg_hint_plan integration ──────────────────────────────────────────────

    def run_with_cardinality_hints(self, sql: str,
                                   hints: Dict[str, int]) -> Dict[str, Any]:
        """
        Inject cardinality hints via pg_hint_plan and capture the resulting plan.

        pg_hint_plan must be installed and loaded:
            CREATE EXTENSION pg_hint_plan;
            LOAD 'pg_hint_plan';

        Hints format:
            {"alias_or_table": row_count_estimate}

        Example:
            hints = {"ci": 50000, "t": 1200}
            # Generates: /*+ Rows(ci #50000) Rows(t #1200) */

        Args:
            sql:   SQL query string
            hints: Dict mapping table alias → ML-predicted row count

        Returns:
            Same structure as run_and_capture()
        """
        hint_strs = " ".join(f"Rows({alias} #{count})"
                             for alias, count in hints.items())
        hinted_sql = f"/*+ {hint_strs} */ {sql}"
        return self.run_and_capture(hinted_sql)

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def capture_batch(self, queries: Dict[str, str],
                      skip_on_error: bool = True) -> Dict[str, Dict]:
        """
        Run run_and_capture() on a batch of queries.

        Args:
            queries:       {query_id: sql_string}
            skip_on_error: If True, log errors and continue rather than raise

        Returns:
            {query_id: capture_result_dict}
        """
        results = {}
        total   = len(queries)

        for i, (qid, sql) in enumerate(queries.items(), 1):
            print(f"  [{i}/{total}] Capturing plan for query {qid}...", end=" ")
            try:
                result         = self.run_and_capture(sql)
                results[qid]   = result
                node_count     = len(result["nodes"])
                wall           = result["wall_time"]
                print(f"OK ({node_count} nodes, {wall:.2f}s)")
            except Exception as e:
                print(f"ERROR — {e}")
                if not skip_on_error:
                    raise
                # Reset connection state after error
                try:
                    self._conn.rollback()
                except Exception:
                    self._conn = None

        return results

    def get_postgres_version(self) -> str:
        """Return the PostgreSQL server version string."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            return cur.fetchone()[0]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("PlanInterceptor self-test")
    print("-" * 40)

    test_sql = "SELECT COUNT(*) FROM pg_class c JOIN pg_namespace n ON c.relnamespace = n.oid"

    with PlanInterceptor() as interceptor:
        print(f"PostgreSQL: {interceptor.get_postgres_version()}")

        result = interceptor.run_and_capture(test_sql)

        print(f"\nCaptured {len(result['nodes'])} plan nodes in {result['wall_time']:.3f}s")
        print("\nPlan nodes:")
        for node in result["nodes"]:
            indent = "  " * node.depth
            print(f"{indent}{node}")

        print("\nRecord (dict) format for first node:")
        import pprint
        pprint.pprint(result["records"][0])
