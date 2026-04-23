"""
NeuroFlix — Evaluation Module
===============================
Comprehensive evaluation metrics for recommendation systems.

Metrics:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - Precision@K
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - Hit Rate@K
  - Coverage
  - Diversity

Supports comparative evaluation across all models.

Author: Mohit Raj
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


# ======================================================================
# Rating Prediction Metrics
# ======================================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ======================================================================
# Ranking Metrics
# ======================================================================
def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Precision@K: Fraction of recommended items that are relevant.
    
    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground-truth) item IDs
        k: Number of top recommendations to consider
    """
    if k == 0:
        return 0.0
    rec_at_k = recommended[:k]
    hits = sum(1 for item in rec_at_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Recall@K: Fraction of relevant items that are recommended.
    """
    if not relevant:
        return 0.0
    rec_at_k = recommended[:k]
    hits = sum(1 for item in rec_at_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    
    Measures ranking quality — higher-ranked relevant items get more credit.
    """
    if not relevant or k == 0:
        return 0.0

    rec_at_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(rec_at_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions are 1-indexed

    # Ideal DCG
    ideal_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Hit Rate@K: 1 if at least one relevant item is in top-K, else 0."""
    rec_at_k = recommended[:k]
    return 1.0 if any(item in relevant for item in rec_at_k) else 0.0


def average_precision(recommended: List[int], relevant: set) -> float:
    """Average Precision (AP) for a single user."""
    if not relevant:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)

    return sum_precisions / len(relevant) if relevant else 0.0


# ======================================================================
# System-Level Metrics
# ======================================================================
def coverage(
    all_recommendations: List[List[int]], total_items: int
) -> float:
    """
    Catalog Coverage: Fraction of all items that appear in recommendations.
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def diversity(recommendations: List[int], item_features: Dict) -> float:
    """
    Intra-list Diversity: Average dissimilarity between recommended items.
    Measures how diverse the recommendations are.
    """
    if len(recommendations) < 2:
        return 0.0

    # Simple genre-based diversity
    genre_sets = []
    for item_id in recommendations:
        if item_id in item_features:
            genre_sets.append(set(item_features[item_id]))
        else:
            genre_sets.append(set())

    total_dissimilarity = 0.0
    count = 0
    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            union = genre_sets[i] | genre_sets[j]
            intersection = genre_sets[i] & genre_sets[j]
            if union:
                jaccard = len(intersection) / len(union)
                total_dissimilarity += 1 - jaccard
                count += 1

    return total_dissimilarity / count if count > 0 else 0.0


# ======================================================================
# Comprehensive Evaluation
# ======================================================================
class RecommenderEvaluator:
    """
    Comprehensive evaluator that computes all metrics for a recommender.
    
    Supports side-by-side comparison of multiple models.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        relevance_threshold: float = 3.5,
        k_values: List[int] = None,
    ):
        self.test_df = test_df
        self.train_df = train_df
        self.movies_df = movies_df
        self.relevance_threshold = relevance_threshold
        self.k_values = k_values or [5, 10, 20]

        # Pre-compute ground truth
        self.user_relevant_items = self._compute_relevant_items()
        self.results: Dict[str, Dict] = {}

    def _compute_relevant_items(self) -> Dict[int, set]:
        """Compute relevant items per user from test set."""
        relevant = {}
        for user_id, group in self.test_df.groupby("userId"):
            liked = group[group["rating"] >= self.relevance_threshold]
            relevant[user_id] = set(liked["movieId"].values)
        return relevant

    def evaluate_model(
        self,
        model_name: str,
        predict_fn,
        recommend_fn,
        sample_users: Optional[int] = 100,
    ) -> Dict[str, float]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model_name: Name for display
            predict_fn: fn(user_id, movie_id) → predicted_rating
            recommend_fn: fn(user_id, top_n) → list of movie_ids
            sample_users: Number of users to evaluate (None = all)
        """
        logger.info("Evaluating model: %s …", model_name)

        users = list(self.user_relevant_items.keys())
        if sample_users and len(users) > sample_users:
            rng = np.random.RandomState(42)
            users = list(rng.choice(users, size=sample_users, replace=False))

        # Rating prediction metrics
        all_true = []
        all_pred = []

        # Ranking metrics
        metrics_per_k = {k: defaultdict(list) for k in self.k_values}
        all_recommendations = []

        for user_id in users:
            user_test = self.test_df[self.test_df["userId"] == user_id]
            if user_test.empty:
                continue

            relevant = self.user_relevant_items.get(user_id, set())

            # Rating predictions
            for _, row in user_test.iterrows():
                try:
                    pred = predict_fn(user_id, row["movieId"])
                    all_true.append(row["rating"])
                    all_pred.append(pred)
                except Exception:
                    pass

            # Ranking predictions
            try:
                max_k = max(self.k_values)
                recs = recommend_fn(user_id, max_k)
                if isinstance(recs, pd.DataFrame) and "movieId" in recs.columns:
                    rec_ids = recs["movieId"].tolist()
                elif isinstance(recs, list):
                    rec_ids = recs
                else:
                    rec_ids = []

                all_recommendations.append(rec_ids)

                for k in self.k_values:
                    metrics_per_k[k]["precision"].append(
                        precision_at_k(rec_ids, relevant, k)
                    )
                    metrics_per_k[k]["recall"].append(
                        recall_at_k(rec_ids, relevant, k)
                    )
                    metrics_per_k[k]["ndcg"].append(
                        ndcg_at_k(rec_ids, relevant, k)
                    )
                    metrics_per_k[k]["hit_rate"].append(
                        hit_rate_at_k(rec_ids, relevant, k)
                    )
            except Exception as e:
                logger.warning("Ranking eval failed for user %d: %s", user_id, e)

        # Compile results
        results = {"model": model_name}

        if all_true and all_pred:
            y_true = np.array(all_true)
            y_pred = np.array(all_pred)
            results["RMSE"] = round(rmse(y_true, y_pred), 4)
            results["MAE"] = round(mae(y_true, y_pred), 4)

        for k in self.k_values:
            if metrics_per_k[k]["precision"]:
                results[f"Precision@{k}"] = round(
                    np.mean(metrics_per_k[k]["precision"]), 4
                )
                results[f"Recall@{k}"] = round(
                    np.mean(metrics_per_k[k]["recall"]), 4
                )
                results[f"NDCG@{k}"] = round(
                    np.mean(metrics_per_k[k]["ndcg"]), 4
                )
                results[f"HitRate@{k}"] = round(
                    np.mean(metrics_per_k[k]["hit_rate"]), 4
                )

        # Coverage
        if all_recommendations:
            total_items = self.movies_df["movieId"].nunique()
            results["Coverage"] = round(
                coverage(all_recommendations, total_items), 4
            )

        self.results[model_name] = results
        logger.info("Evaluation complete for %s: RMSE=%.4f",
                     model_name, results.get("RMSE", float("nan")))
        return results

    def compare_models(self) -> pd.DataFrame:
        """Generate a comparison table of all evaluated models."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for name, metrics in self.results.items():
            rows.append(metrics)

        df = pd.DataFrame(rows)
        if "model" in df.columns:
            df = df.set_index("model")
        return df

    def get_summary(self) -> str:
        """Generate a text summary of evaluation results."""
        if not self.results:
            return "No models evaluated yet."

        comparison = self.compare_models()
        lines = ["=" * 60, "  NEUROFLIX — MODEL EVALUATION SUMMARY", "=" * 60, ""]

        for model_name, metrics in self.results.items():
            lines.append(f"📊 {model_name}")
            lines.append("-" * 40)
            for k, v in metrics.items():
                if k != "model":
                    lines.append(f"  {k:>20s}: {v}")
            lines.append("")

        # Best model
        if len(self.results) > 1:
            rmse_scores = {
                name: m.get("RMSE", float("inf"))
                for name, m in self.results.items()
            }
            best = min(rmse_scores, key=rmse_scores.get)
            lines.append(f"🏆 Best Model (by RMSE): {best} ({rmse_scores[best]:.4f})")

        return "\n".join(lines)
