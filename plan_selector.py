# plan_selector.py
"""
ML-Guided Plan Selector
-------------------------
Uses the trained cardinality estimation model to predict more accurate
row counts, then selects the best query execution plan accordingly.

Two strategies are implemented:

  1. HintInjector  — corrects cardinality estimates via pg_hint_plan
     and lets PostgreSQL re-optimise with better stats.

  2. CostModelSelector — scores a set of candidate plans directly
     using ML-predicted cardinalities and a simple cost model.

References:
    Bao (Marcus et al., 2021): Learned cardinalities for plan selection
    Lero (Zhu et al., 2023):   Learning-based query optimiser
    ALECE (Li et al., 2023):   Attention-based cardinality estimator

Usage:
    from plan_selector import PlanSelector
    selector = PlanSelector(model_path="models/xgb_cardinality_model.pkl")
    result = selector.select_and_run(sql, interceptor)
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from typing import Dict, List, Optional, Tuple, Any

from pg_interceptor import PlanInterceptor, PlanNode
from feature_engineering import engineer_features, encode_categoricals
from config import MODEL_DIR


# ── Feature extraction from a live PlanNode ───────────────────────────────────

def plan_node_to_feature_dict(node: PlanNode) -> dict:
    """
    Convert a PlanNode (from pg_interceptor) into the feature dict
    expected by the ML model.

    This mirrors the columns produced by data_collection.extract_plan_nodes().
    """
    return {
        "node_type":              node.node_type,
        "estimated_rows":         node.estimated_rows,
        "actual_rows":            node.actual_rows,       # 0 at inference time
        "estimated_width":        node.estimated_width,
        "estimated_cost":         node.total_cost,
        "actual_time":            node.actual_total_time, # 0 at inference time
        "join_type":              node.join_type or "None",
        "relation_name":          node.relation_name or "None",
        "index_name":             node.index_name or "None",
        "filter":                 node.filter or "",
        "rows_removed_by_filter": node.rows_removed_by_filter,
    }


def extract_features_from_nodes(nodes: List[PlanNode]) -> pd.DataFrame:
    raw_dicts = [plan_node_to_feature_dict(n) for n in nodes]
    df = pd.DataFrame(raw_dicts)

    # Same feature engineering as training
    df = engineer_features(df)

    # Drop non-feature columns
    drop_cols = ["actual_rows", "actual_time"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # CRITICAL FIX: match training features exactly
    X = X.select_dtypes(include=["int64", "float64"])

    return X


# ── Core Plan Selector ────────────────────────────────────────────────────────

class PlanSelector:
    """
    Uses a trained cardinality model to improve PostgreSQL's plan selection.

    Two modes:
      - hint_mode:  Inject corrected estimates via pg_hint_plan
      - score_mode: Score candidate plans by their total estimated cost
                    using ML cardinalities

    Args:
        model_path: Path to a saved XGBoost / joblib model
        mode:       "hint" (default) or "score"
    """

    def __init__(self, model_path: str = None, mode: str = "hint"):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "xgb_cardinality_model.pkl")

        self.mode = mode
        self.model = self._load_model(model_path)
        print(f"[PlanSelector] Loaded model from {model_path} (mode={mode})")

    def _load_model(self, path: str):
        """Load a joblib-serialised model."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Run train.py first to generate the model file."
            )
        return joblib.load(path)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_cardinalities(self, nodes: List[PlanNode]) -> np.ndarray:
        """
        Predict actual row counts for a list of plan nodes.

        Args:
            nodes: PlanNode objects from PlanInterceptor

        Returns:
            np.ndarray of predicted row counts (one per node)
        """
        X = extract_features_from_nodes(nodes)
        log_preds = self.model.predict(X)
        return np.expm1(log_preds)  # reverse log1p transform

    def predict_single_node(self, node: PlanNode) -> float:
        """Predict actual row count for a single PlanNode."""
        return self.predict_cardinalities([node])[0]

    # ── Hint injection mode ───────────────────────────────────────────────────

    def build_hints(self, nodes: List[PlanNode],
                    threshold_q_error: float = 2.0) -> Dict[str, int]:
        """
        Identify plan nodes where PostgreSQL's estimates are likely wrong,
        and build a pg_hint_plan Rows() hint dict.

        Only generates hints for nodes where the model's predicted rows
        differ from the PostgreSQL estimate by more than threshold_q_error.

        Args:
            nodes:              PlanNode objects
            threshold_q_error:  Minimum Q-error to trigger a hint (default 2x)

        Returns:
            Dict: {table_alias_or_name: predicted_row_count}
        """
        ml_preds = self.predict_cardinalities(nodes)
        hints    = {}

        for node, ml_pred in zip(nodes, ml_preds):
            if not node.relation_name:
                continue  # skip non-scan nodes (joins, aggregates)

            pg_est = max(node.estimated_rows, 1)
            ml_est = max(ml_pred, 1)
            q_err  = max(pg_est / ml_est, ml_est / pg_est)

            if q_err >= threshold_q_error:
                alias = node.alias or node.relation_name
                hints[alias] = int(round(ml_pred))

        return hints

    def select_with_hints(self, sql: str,
                          interceptor: PlanInterceptor,
                          threshold: float = 2.0) -> Dict[str, Any]:
        """
        Full pipeline for hint-based plan selection:
          1. Get PostgreSQL's default plan (EXPLAIN only — no execution)
          2. Predict ML cardinalities for each node
          3. Inject hints for nodes with high Q-error
          4. Execute with corrected hints and capture results

        Args:
            sql:          SQL query string
            interceptor:  Active PlanInterceptor
            threshold:    Q-error threshold to trigger hints

        Returns:
            dict with:
              - "default_plan":   Initial plan (no hints)
              - "hinted_plan":    Plan after hint injection
              - "hints_used":     Dict of injected hints
              - "improvement":    Wall-time improvement (seconds)
        """
        # Step 1: get plan without executing (fast)
        raw_default = interceptor.explain_only(sql)
        from pg_interceptor import _parse_plan_tree, _flatten_tree
        default_root  = _parse_plan_tree(raw_default)
        default_nodes = _flatten_tree(default_root)

        # Step 2: predict cardinalities for each node
        hints = self.build_hints(default_nodes, threshold_q_error=threshold)

        if not hints:
            print("[PlanSelector] No hints needed — PostgreSQL estimates look good.")
            result = interceptor.run_and_capture(sql)
            return {
                "default_plan": result,
                "hinted_plan":  result,
                "hints_used":   {},
                "improvement":  0.0,
            }

        print(f"[PlanSelector] Injecting {len(hints)} cardinality hints: {hints}")

        # Step 3: execute default plan to get baseline wall time
        default_result = interceptor.run_and_capture(sql)
        default_time   = default_result["wall_time"]

        # Step 4: execute with hints
        hinted_result = interceptor.run_with_cardinality_hints(sql, hints)
        hinted_time   = hinted_result["wall_time"]

        improvement = default_time - hinted_time
        pct = (improvement / default_time * 100) if default_time > 0 else 0

        print(f"[PlanSelector] Wall time: {default_time:.3f}s → {hinted_time:.3f}s "
              f"({pct:+.1f}%)")

        return {
            "default_plan": default_result,
            "hinted_plan":  hinted_result,
            "hints_used":   hints,
            "improvement":  improvement,
        }

    # ── Score mode: rank candidate plans ─────────────────────────────────────

    def score_plan(self, nodes: List[PlanNode]) -> float:
        """
        Score a plan's expected cost using ML-corrected cardinalities.

        Simple cost model:
            cost = sum(node.total_cost × correction_factor)
        where correction_factor = ml_predicted_rows / pg_estimated_rows

        A lower score = a better (cheaper) plan.

        Args:
            nodes: Flat list of PlanNodes for a candidate plan

        Returns:
            Float cost score (lower is better)
        """
        if not nodes:
            return float("inf")

        ml_preds = self.predict_cardinalities(nodes)
        total_score = 0.0

        for node, ml_pred in zip(nodes, ml_preds):
            pg_est = max(node.estimated_rows, 1)
            ml_est = max(ml_pred, 1)
            correction = ml_est / pg_est
            # Re-weight PostgreSQL's cost estimate using ML correction
            corrected_cost = node.total_cost * correction
            total_score += corrected_cost

        return total_score

    def select_best_plan(self, sql: str,
                         interceptor: PlanInterceptor,
                         join_orders: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Given multiple candidate SQL variants (with different join hints or
        FROM-clause orderings), select the one with the lowest ML-corrected cost.

        Args:
            sql:          Original SQL query
            interceptor:  Active PlanInterceptor
            join_orders:  Optional list of SQL variants to evaluate.
                          If None, only the default plan is evaluated.

        Returns:
            dict with:
              - "best_sql":     The SQL variant with lowest predicted cost
              - "best_score":   The score of the winning plan
              - "all_scores":   List of (sql, score) pairs
              - "best_result":  Full capture result for the best plan
        """
        candidates = [sql]
        if join_orders:
            candidates.extend(join_orders)

        scored = []
        for candidate_sql in candidates:
            try:
                raw = interceptor.explain_only(candidate_sql)
                from pg_interceptor import _parse_plan_tree, _flatten_tree
                root  = _parse_plan_tree(raw)
                nodes = _flatten_tree(root)
                score = self.score_plan(nodes)
                scored.append((candidate_sql, score))
            except Exception as e:
                print(f"[PlanSelector] Error scoring plan: {e}")
                scored.append((candidate_sql, float("inf")))

        # Pick the best
        scored.sort(key=lambda x: x[1])
        best_sql, best_score = scored[0]

        print(f"[PlanSelector] Best plan score: {best_score:.2f} "
              f"(from {len(candidates)} candidates)")

        best_result = interceptor.run_and_capture(best_sql)

        return {
            "best_sql":    best_sql,
            "best_score":  best_score,
            "all_scores":  scored,
            "best_result": best_result,
        }

    # ── Reporting ─────────────────────────────────────────────────────────────

    def node_correction_report(self, nodes: List[PlanNode]) -> pd.DataFrame:
        """
        Generate a per-node comparison report:
        PostgreSQL estimate vs ML prediction vs actual (if available).

        Args:
            nodes: PlanNode list from a captured plan

        Returns:
            DataFrame with one row per node
        """
        ml_preds = self.predict_cardinalities(nodes)
        rows = []

        for node, ml_pred in zip(nodes, ml_preds):
            pg_est = node.estimated_rows
            actual = node.actual_rows

            rows.append({
                "node_type":    node.node_type,
                "relation":     node.relation_name or node.alias,
                "pg_estimate":  pg_est,
                "ml_estimate":  int(round(ml_pred)),
                "actual_rows":  actual,
                "pg_q_error":   node.q_error,
                "ml_q_error":   (max(actual, 1) / max(ml_pred, 1))
                                 if actual > 0
                                 else None,
            })

        df = pd.DataFrame(rows)
        return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    model_path = os.path.join(MODEL_DIR, "xgb_cardinality_model.pkl")

    if not os.path.exists(model_path):
        print(f"[ERROR] No trained model found at {model_path}")
        print("Run `python train.py` first to train and save the model.")
        sys.exit(1)

    selector = PlanSelector(model_path=model_path, mode="hint")

    test_sql = (
        "SELECT COUNT(*) FROM pg_class c "
        "JOIN pg_namespace n ON c.relnamespace = n.oid "
        "WHERE n.nspname = 'public'"
    )

    with PlanInterceptor() as interceptor:
        print("\n── Hint-based selection ──")
        result = selector.select_with_hints(test_sql, interceptor)
        print(f"Hints used: {result['hints_used']}")
        print(f"Improvement: {result['improvement']:.3f}s")

        print("\n── Node correction report ──")
        nodes  = result["hinted_plan"]["nodes"]
        report = selector.node_correction_report(nodes)
        print(report.to_string(index=False))
