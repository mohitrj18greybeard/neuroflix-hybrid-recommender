"""
NeuroFlix — Hybrid Recommendation Engine
==========================================
Production-grade hybrid system combining Content-Based and Collaborative
Filtering using multiple fusion strategies.

Strategies:
  1. Weighted Hybrid — linear combination of scores
  2. Switching Hybrid — context-aware model selection
  3. Cascade Hybrid — sequential refinement

Includes the unified `recommend()` API function.

Author: Mohit Raj
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pickle

import numpy as np
import pandas as pd

from .content_based import ContentBasedEngine
from .collaborative import CollaborativeFilteringEngine

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class HybridRecommenderEngine:
    """
    Hybrid Recommendation Engine that fuses Content-Based and
    Collaborative Filtering signals.
    
    Design Choices & Trade-offs:
    ─────────────────────────────
    • Weighted Hybrid: Best overall performance. Balances personalization
      (collaborative) with content relevance. Weight α controls the mix.
      
    • Switching Hybrid: Addresses cold-start. Uses content-based for new
      users (few ratings) and collaborative for established users.
      
    • Cascade: Two-stage approach — first filter with content-based,
      then re-rank with collaborative. Good precision but slower.
    """

    def __init__(
        self,
        content_engine: Optional[ContentBasedEngine] = None,
        collab_engine: Optional[CollaborativeFilteringEngine] = None,
        strategy: str = "weighted",  # "weighted", "switching", "cascade"
        alpha: float = 0.6,  # weight for collaborative in weighted hybrid
        cold_start_threshold: int = 10,  # min ratings before switching to CF
    ):
        self.content_engine = content_engine
        self.collab_engine = collab_engine
        self.strategy = strategy
        self.alpha = alpha  # α * collab + (1-α) * content
        self.cold_start_threshold = cold_start_threshold
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
    ) -> "HybridRecommenderEngine":
        """
        Fit both sub-engines and store reference data.
        
        Args:
            movies_df: Enriched movies DataFrame
            ratings_df: Full ratings DataFrame
            train_df: Training split (used for collab fitting)
        """
        self.movies_df = movies_df
        self.ratings_df = ratings_df

        fit_ratings = train_df if train_df is not None else ratings_df

        # Fit Content-Based engine
        if self.content_engine is None:
            self.content_engine = ContentBasedEngine()
        logger.info("Fitting Content-Based sub-engine …")
        self.content_engine.fit(movies_df)

        # Fit Collaborative engine
        if self.collab_engine is None:
            self.collab_engine = CollaborativeFilteringEngine(method="svd", n_factors=50)
        logger.info("Fitting Collaborative sub-engine …")
        self.collab_engine.fit(fit_ratings, movies_df)

        logger.info("Hybrid engine ready (strategy=%s, α=%.2f)", self.strategy, self.alpha)
        return self

    # ==================================================================
    # UNIFIED RECOMMENDATION API
    # ==================================================================
    def recommend(
        self,
        user_id: Optional[int] = None,
        movie_id: Optional[int] = None,
        top_n: int = 10,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        ╔══════════════════════════════════════════════════════════╗
        ║  UNIFIED RECOMMENDATION API                            ║
        ║  recommend(user_id=None, movie_id=None, top_n=10)      ║
        ╚══════════════════════════════════════════════════════════╝
        
        Behavior:
          - user_id only   → personalized recommendations for user
          - movie_id only  → similar movies (content + collaborative)
          - both           → "movies like X for user Y"
          - neither        → popular/trending movies
        
        Args:
            user_id:  User to recommend for (optional)
            movie_id: Movie to find similarities for (optional)
            top_n:    Number of recommendations to return
            strategy: Override the default strategy
            
        Returns:
            DataFrame with: movieId, title, score, genres, year, explanation, source
        """
        active_strategy = strategy or self.strategy

        if user_id is not None and movie_id is not None:
            return self._recommend_movie_for_user(user_id, movie_id, top_n, active_strategy)
        elif user_id is not None:
            return self._recommend_for_user(user_id, top_n, active_strategy)
        elif movie_id is not None:
            return self._recommend_similar_movies(movie_id, top_n)
        else:
            return self._recommend_popular(top_n)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------
    def _recommend_for_user(
        self, user_id: int, top_n: int, strategy: str
    ) -> pd.DataFrame:
        """Personalized recommendations using the selected strategy."""

        if strategy == "weighted":
            return self._weighted_hybrid_user(user_id, top_n)
        elif strategy == "switching":
            return self._switching_hybrid_user(user_id, top_n)
        elif strategy == "cascade":
            return self._cascade_hybrid_user(user_id, top_n)
        else:
            return self._weighted_hybrid_user(user_id, top_n)

    def _weighted_hybrid_user(self, user_id: int, top_n: int) -> pd.DataFrame:
        """
        Weighted Hybrid: α * collaborative_score + (1-α) * content_score
        
        Normalizes both scores to [0, 1] before combining.
        """
        # Get content-based recommendations (expanded pool)
        pool_size = top_n * 5
        content_recs = self.content_engine.recommend_for_user(
            user_id, self.ratings_df, top_n=pool_size
        )
        collab_recs = self.collab_engine.recommend_for_user(
            user_id, self.ratings_df, top_n=pool_size
        )

        # Build score maps
        content_scores = {}
        for _, row in content_recs.iterrows():
            content_scores[row["movieId"]] = row["similarity_score"]

        collab_scores = {}
        score_col = "predicted_rating" if "predicted_rating" in collab_recs.columns else "similarity_score"
        for _, row in collab_recs.iterrows():
            collab_scores[row["movieId"]] = row[score_col]

        # Combine all candidate movies
        all_movies = set(content_scores.keys()) | set(collab_scores.keys())

        # Normalize scores
        c_vals = list(content_scores.values())
        cb_vals = list(collab_scores.values())
        c_min, c_max = (min(c_vals), max(c_vals)) if c_vals else (0, 1)
        cb_min, cb_max = (min(cb_vals), max(cb_vals)) if cb_vals else (0, 1)

        def norm(val, vmin, vmax):
            if vmax == vmin:
                return 0.5
            return (val - vmin) / (vmax - vmin)

        results = []
        for mid in all_movies:
            c_score = norm(content_scores.get(mid, c_min), c_min, c_max)
            cb_score = norm(collab_scores.get(mid, cb_min), cb_min, cb_max)
            hybrid_score = self.alpha * cb_score + (1 - self.alpha) * c_score

            # Source tracking
            sources = []
            if mid in content_scores:
                sources.append("content")
            if mid in collab_scores:
                sources.append("collaborative")

            title = self._get_movie_title(mid)
            genres = self._get_movie_genres(mid)
            year = self._get_movie_year(mid)

            results.append({
                "movieId": int(mid),
                "title": title,
                "score": round(hybrid_score, 4),
                "content_score": round(c_score, 4),
                "collab_score": round(cb_score, 4),
                "genres": genres,
                "year": year,
                "explanation": f"Hybrid (α={self.alpha}): CF={cb_score:.2f} + CB={c_score:.2f}",
                "source": " + ".join(sources),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return pd.DataFrame(results[:top_n])

    def _switching_hybrid_user(self, user_id: int, top_n: int) -> pd.DataFrame:
        """
        Switching Hybrid: Use content-based for cold-start users,
        collaborative for established users.
        """
        if self.ratings_df is not None:
            user_count = len(self.ratings_df[self.ratings_df["userId"] == user_id])
        else:
            user_count = 0

        if user_count < self.cold_start_threshold:
            logger.info(
                "User %d has %d ratings (< %d) — using Content-Based.",
                user_id, user_count, self.cold_start_threshold,
            )
            recs = self.content_engine.recommend_for_user(
                user_id, self.ratings_df, top_n=top_n
            )
            recs["source"] = "content-based (cold-start)"
            recs["score"] = recs["similarity_score"]
            return recs
        else:
            logger.info(
                "User %d has %d ratings — using Collaborative Filtering.",
                user_id, user_count,
            )
            recs = self.collab_engine.recommend_for_user(
                user_id, self.ratings_df, top_n=top_n
            )
            recs["source"] = "collaborative"
            recs["score"] = recs["predicted_rating"] / 5.0
            return recs

    def _cascade_hybrid_user(self, user_id: int, top_n: int) -> pd.DataFrame:
        """
        Cascade Hybrid: Content-based generates candidates,
        collaborative re-ranks them.
        """
        # Stage 1: Content-based candidate generation
        candidates = self.content_engine.recommend_for_user(
            user_id, self.ratings_df, top_n=top_n * 3
        )

        if candidates.empty:
            return candidates

        # Stage 2: Re-rank with collaborative scores
        movie_ids = candidates["movieId"].tolist()
        collab_scores = self.collab_engine.predict_scores(user_id, movie_ids)

        candidates = candidates.copy()
        candidates["collab_score"] = collab_scores
        candidates["score"] = (
            0.4 * candidates["similarity_score"] +
            0.6 * (candidates["collab_score"] / 5.0)
        )
        candidates["source"] = "cascade (CB → CF)"
        candidates = candidates.sort_values("score", ascending=False).head(top_n)

        return candidates.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Movie similarity
    # ------------------------------------------------------------------
    def _recommend_similar_movies(self, movie_id: int, top_n: int) -> pd.DataFrame:
        """Find similar movies using both engines."""
        content_sims = self.content_engine.similar_movies(movie_id, top_n=top_n * 2)
        collab_sims = self.collab_engine.similar_movies(movie_id, top_n=top_n * 2)

        # Merge scores
        combined = {}
        for _, row in content_sims.iterrows():
            mid = row["movieId"]
            combined[mid] = {
                "content_sim": row["similarity_score"],
                "collab_sim": 0,
                "title": row["title"],
                "genres": row.get("genres", ""),
                "year": row.get("year", None),
                "explanation": row.get("explanation", ""),
            }

        for _, row in collab_sims.iterrows():
            mid = row["movieId"]
            if mid in combined:
                combined[mid]["collab_sim"] = row["similarity_score"]
            else:
                combined[mid] = {
                    "content_sim": 0,
                    "collab_sim": row["similarity_score"],
                    "title": row["title"],
                    "genres": row.get("genres", ""),
                    "year": row.get("year", None),
                    "explanation": "Collaborative similarity",
                }

        results = []
        for mid, info in combined.items():
            score = 0.5 * info["content_sim"] + 0.5 * info["collab_sim"]
            results.append({
                "movieId": int(mid),
                "title": info["title"],
                "score": round(score, 4),
                "content_score": round(info["content_sim"], 4),
                "collab_score": round(info["collab_sim"], 4),
                "genres": info["genres"],
                "year": info["year"],
                "explanation": info["explanation"],
                "source": "hybrid similarity",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return pd.DataFrame(results[:top_n])

    def _recommend_movie_for_user(
        self, user_id: int, movie_id: int, top_n: int, strategy: str
    ) -> pd.DataFrame:
        """Recommend movies similar to movie_id but personalized for user_id."""
        similar = self._recommend_similar_movies(movie_id, top_n=top_n * 3)

        if similar.empty:
            return similar

        # Re-rank based on user preferences
        movie_ids = similar["movieId"].tolist()
        collab_preds = self.collab_engine.predict_scores(user_id, movie_ids)

        similar = similar.copy()
        similar["user_affinity"] = collab_preds / 5.0
        similar["score"] = 0.5 * similar["score"] + 0.5 * similar["user_affinity"]
        similar["explanation"] = similar.apply(
            lambda r: f"Similar to selected movie, personalized for you ({r['source']})",
            axis=1,
        )
        similar = similar.sort_values("score", ascending=False).head(top_n)
        return similar.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Popular / Trending
    # ------------------------------------------------------------------
    def _recommend_popular(self, top_n: int = 10) -> pd.DataFrame:
        """Return popular/trending movies."""
        if self.movies_df is None:
            return pd.DataFrame()

        if "bayesian_avg" in self.movies_df.columns:
            popular = self.movies_df.nlargest(top_n, "bayesian_avg")
        else:
            popular = self.movies_df.head(top_n)

        results = []
        for _, row in popular.iterrows():
            results.append({
                "movieId": int(row["movieId"]),
                "title": row.get("title", ""),
                "score": round(float(row.get("bayesian_avg", 3.0)) / 5.0, 4),
                "genres": row.get("genres_str", ""),
                "year": row.get("year", None),
                "explanation": "Trending / highly rated",
                "source": "popularity",
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Batch prediction for evaluation
    # ------------------------------------------------------------------
    def predict_scores(
        self, user_id: int, movie_ids: List[int]
    ) -> np.ndarray:
        """Predict hybrid scores for evaluation."""
        content_scores = self.content_engine.predict_scores(
            user_id, movie_ids, self.ratings_df
        )
        collab_scores = self.collab_engine.predict_scores(user_id, movie_ids)

        # Normalize
        c_min, c_max = content_scores.min(), content_scores.max()
        cb_min, cb_max = collab_scores.min(), collab_scores.max()

        if c_max > c_min:
            content_norm = (content_scores - c_min) / (c_max - c_min)
        else:
            content_norm = np.full_like(content_scores, 0.5)

        if cb_max > cb_min:
            collab_norm = (collab_scores - cb_min) / (cb_max - cb_min)
        else:
            collab_norm = np.full_like(collab_scores, 0.5)

        # Scale back to rating range
        hybrid = self.alpha * collab_norm + (1 - self.alpha) * content_norm
        return 1.0 + 4.0 * hybrid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_movie_title(self, movie_id: int) -> str:
        if self.movies_df is not None:
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not match.empty:
                return str(match.iloc[0].get("title", f"Movie {movie_id}"))
        return f"Movie {movie_id}"

    def _get_movie_genres(self, movie_id: int) -> str:
        if self.movies_df is not None:
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not match.empty:
                return str(match.iloc[0].get("genres_str", ""))
        return ""

    def _get_movie_year(self, movie_id: int) -> Optional[float]:
        if self.movies_df is not None:
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not match.empty:
                return match.iloc[0].get("year", None)
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """Save the hybrid engine configuration."""
        if path is None:
            path = MODELS_DIR / "hybrid_engine_config.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "strategy": self.strategy,
            "alpha": self.alpha,
            "cold_start_threshold": self.cold_start_threshold,
        }
        with open(str(path), "wb") as f:
            pickle.dump(config, f)

        # Save sub-engines
        self.content_engine.save()
        self.collab_engine.save()

        logger.info("Hybrid engine saved.")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HybridRecommenderEngine":
        """Load a fitted hybrid engine."""
        if path is None:
            path = MODELS_DIR / "hybrid_engine_config.pkl"

        with open(str(path), "rb") as f:
            config = pickle.load(f)

        content_engine = ContentBasedEngine.load()
        collab_engine = CollaborativeFilteringEngine.load()

        engine = cls(
            content_engine=content_engine,
            collab_engine=collab_engine,
            strategy=config["strategy"],
            alpha=config["alpha"],
            cold_start_threshold=config["cold_start_threshold"],
        )
        return engine


# ======================================================================
# PUBLIC API FUNCTION
# ======================================================================
def recommend(
    user_id: Optional[int] = None,
    movie_id: Optional[int] = None,
    top_n: int = 10,
    strategy: str = "weighted",
) -> pd.DataFrame:
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║  recommend(user_id=None, movie_id=None, top_n=10)          ║
    ║                                                            ║
    ║  Clean, unified API for all recommendation scenarios:      ║
    ║    • user_id → personalized recommendations                ║
    ║    • movie_id → similar movies                             ║
    ║    • both → personalized similar movies                    ║
    ║    • neither → popular/trending                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    engine = HybridRecommenderEngine.load()
    return engine.recommend(
        user_id=user_id,
        movie_id=movie_id,
        top_n=top_n,
        strategy=strategy,
    )
