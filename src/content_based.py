"""
NeuroFlix — Content-Based Filtering Engine
============================================
Production-grade content-based recommendation using TF-IDF vectorization
and cosine similarity on movie metadata (genres, tags, titles).

Supports:
  - Movie-to-movie similarity (item-based)
  - User profile recommendations (aggregated user taste)
  - Cold-start handling for new users
  - Explainability: shows WHY a movie was recommended

Author: Mohit Raj
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ContentBasedEngine:
    """
    Content-Based Filtering using TF-IDF on movie metadata.
    
    Features used:
      - Genres (one-hot + text)
      - User-generated tags
      - Movie titles (for semantic matching)
    
    The engine builds a TF-IDF matrix and computes cosine similarity
    to find similar movies or build user taste profiles.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.movie_ids: Optional[np.ndarray] = None
        self.movie_id_to_idx: Dict[int, int] = {}
        self.movies_df: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedEngine":
        """
        Build the TF-IDF matrix from the enriched movies DataFrame.
        
        Expects columns: movieId, content_features, genres_str, tags_str, clean_title
        """
        logger.info("Fitting Content-Based engine on %d movies …", len(movies_df))
        self.movies_df = movies_df.copy()
        self.movie_ids = movies_df["movieId"].values
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.movie_ids)
        }

        # Build content features
        content = movies_df["content_features"].fillna("").values

        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content)
        self.tfidf_matrix = normalize(self.tfidf_matrix, norm="l2")

        self._fitted = True
        logger.info(
            "Content-Based engine fitted — TF-IDF matrix shape: %s",
            self.tfidf_matrix.shape,
        )
        return self

    # ------------------------------------------------------------------
    # Movie-to-movie recommendations
    # ------------------------------------------------------------------
    def similar_movies(
        self,
        movie_id: int,
        top_n: int = 10,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """
        Find the top-N most similar movies to a given movie.
        
        Returns DataFrame with: movieId, title, similarity_score, explanation
        """
        self._check_fitted()

        if movie_id not in self.movie_id_to_idx:
            logger.warning("Movie %d not in index — returning empty.", movie_id)
            return pd.DataFrame(columns=["movieId", "title", "similarity_score", "explanation"])

        idx = self.movie_id_to_idx[movie_id]
        movie_vec = self.tfidf_matrix[idx]

        # Compute cosine similarity against all movies
        sim_scores = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()

        if exclude_self:
            sim_scores[idx] = -1.0

        # Get top-N indices
        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = []
        source_title = self.movies_df.iloc[idx]["clean_title"]
        source_genres = self.movies_df.iloc[idx].get("genres_str", "")

        for i in top_indices:
            target = self.movies_df.iloc[i]
            explanation = self._explain_similarity(
                source_genres, source_title,
                target.get("genres_str", ""), target.get("clean_title", ""),
            )
            results.append({
                "movieId": int(self.movie_ids[i]),
                "title": target.get("title", target.get("clean_title", "")),
                "similarity_score": round(float(sim_scores[i]), 4),
                "genres": target.get("genres_str", ""),
                "year": target.get("year", None),
                "explanation": explanation,
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # User-profile recommendations
    # ------------------------------------------------------------------
    def recommend_for_user(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        top_n: int = 10,
        min_rating: float = 3.5,
    ) -> pd.DataFrame:
        """
        Build a user taste profile from their highly-rated movies
        and recommend similar unseen movies.
        """
        self._check_fitted()

        # Get user's ratings
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if user_ratings.empty:
            logger.info("User %d has no ratings — returning popular movies.", user_id)
            return self._cold_start_recommendations(top_n)

        # Filter for liked movies
        liked = user_ratings[user_ratings["rating"] >= min_rating]
        if liked.empty:
            liked = user_ratings.nlargest(5, "rating")

        # Build user profile vector (weighted average of liked movie vectors)
        weights = liked["rating"].values
        liked_ids = liked["movieId"].values
        liked_indices = [
            self.movie_id_to_idx[mid]
            for mid in liked_ids
            if mid in self.movie_id_to_idx
        ]

        if not liked_indices:
            return self._cold_start_recommendations(top_n)

        # Weight by rating
        weight_arr = np.array([
            weights[i] for i, mid in enumerate(liked_ids)
            if mid in self.movie_id_to_idx
        ])
        weight_arr = weight_arr / weight_arr.sum()

        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        for i, idx in enumerate(liked_indices):
            user_profile += weight_arr[i] * self.tfidf_matrix[idx].toarray().flatten()

        user_profile = user_profile.reshape(1, -1)
        user_profile = normalize(user_profile, norm="l2")

        # Compute similarity
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        # Exclude already-rated movies
        rated_ids = set(user_ratings["movieId"].values)
        for mid in rated_ids:
            if mid in self.movie_id_to_idx:
                sim_scores[self.movie_id_to_idx[mid]] = -1.0

        # Top-N
        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = []
        for i in top_indices:
            target = self.movies_df.iloc[i]
            results.append({
                "movieId": int(self.movie_ids[i]),
                "title": target.get("title", target.get("clean_title", "")),
                "similarity_score": round(float(sim_scores[i]), 4),
                "genres": target.get("genres_str", ""),
                "year": target.get("year", None),
                "explanation": f"Matches your taste profile ({target.get('genres_str', 'N/A')})",
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Cold-start fallback
    # ------------------------------------------------------------------
    def _cold_start_recommendations(self, top_n: int = 10) -> pd.DataFrame:
        """Return popular/highly-rated movies for cold-start users."""
        if self.movies_df is None:
            return pd.DataFrame()

        if "bayesian_avg" in self.movies_df.columns:
            popular = self.movies_df.nlargest(top_n, "bayesian_avg")
        elif "rating_mean" in self.movies_df.columns:
            popular = self.movies_df.nlargest(top_n, "rating_mean")
        else:
            popular = self.movies_df.head(top_n)

        results = []
        for _, row in popular.iterrows():
            results.append({
                "movieId": int(row["movieId"]),
                "title": row.get("title", row.get("clean_title", "")),
                "similarity_score": round(float(row.get("bayesian_avg", row.get("rating_mean", 0))) / 5.0, 4),
                "genres": row.get("genres_str", ""),
                "year": row.get("year", None),
                "explanation": "Popular & highly-rated (cold-start recommendation)",
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def _explain_similarity(
        self,
        source_genres: str,
        source_title: str,
        target_genres: str,
        target_title: str,
    ) -> str:
        """Generate a human-readable explanation for similarity."""
        source_set = set(source_genres.split()) if source_genres else set()
        target_set = set(target_genres.split()) if target_genres else set()
        common = source_set & target_set - {"(no", "genres", "listed)"}

        if common:
            return f"Shares genres: {', '.join(sorted(common))}"
        return "Similar content profile"

    def get_top_features(self, movie_id: int, top_n: int = 10) -> List[str]:
        """Get the top TF-IDF features for a movie (for explainability)."""
        self._check_fitted()
        if movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        movie_vec = self.tfidf_matrix[idx].toarray().flatten()
        top_indices = np.argsort(movie_vec)[::-1][:top_n]
        return [feature_names[i] for i in top_indices if movie_vec[i] > 0]

    # ------------------------------------------------------------------
    # Batch scoring for evaluation
    # ------------------------------------------------------------------
    def predict_scores(
        self, user_id: int, movie_ids: List[int], ratings_df: pd.DataFrame
    ) -> np.ndarray:
        """Predict content-based scores for a list of movies for a user."""
        self._check_fitted()

        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if user_ratings.empty:
            return np.full(len(movie_ids), 2.5)

        liked = user_ratings[user_ratings["rating"] >= 3.5]
        if liked.empty:
            liked = user_ratings.nlargest(5, "rating")

        liked_ids = liked["movieId"].values
        liked_indices = [
            self.movie_id_to_idx[mid]
            for mid in liked_ids
            if mid in self.movie_id_to_idx
        ]

        if not liked_indices:
            return np.full(len(movie_ids), 2.5)

        weights = np.array([
            liked[liked["movieId"] == self.movie_ids[idx]]["rating"].values[0]
            for idx in liked_indices
            if self.movie_ids[idx] in liked["movieId"].values
        ])
        if len(weights) == 0:
            return np.full(len(movie_ids), 2.5)

        weights = weights / weights.sum()

        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        for i, idx in enumerate(liked_indices[:len(weights)]):
            user_profile += weights[i] * self.tfidf_matrix[idx].toarray().flatten()

        user_profile = user_profile.reshape(1, -1)
        user_profile = normalize(user_profile, norm="l2")

        scores = []
        for mid in movie_ids:
            if mid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[mid]
                sim = cosine_similarity(user_profile, self.tfidf_matrix[idx]).flatten()[0]
                # Scale to rating range [1, 5]
                scores.append(1.0 + 4.0 * max(0, sim))
            else:
                scores.append(2.5)

        return np.array(scores)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """Save the fitted engine to disk."""
        if path is None:
            path = MODELS_DIR / "content_based_engine.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "movie_ids": self.movie_ids,
            "movie_id_to_idx": self.movie_id_to_idx,
            "movies_df": self.movies_df,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
        }
        with open(str(path), "wb") as f:
            pickle.dump(state, f)
        logger.info("Content-Based engine saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ContentBasedEngine":
        """Load a fitted engine from disk."""
        if path is None:
            path = MODELS_DIR / "content_based_engine.pkl"

        with open(str(path), "rb") as f:
            state = pickle.load(f)

        engine = cls(
            max_features=state["max_features"],
            ngram_range=state["ngram_range"],
            min_df=state["min_df"],
            max_df=state["max_df"],
        )
        engine.tfidf_vectorizer = state["tfidf_vectorizer"]
        engine.tfidf_matrix = state["tfidf_matrix"]
        engine.movie_ids = state["movie_ids"]
        engine.movie_id_to_idx = state["movie_id_to_idx"]
        engine.movies_df = state["movies_df"]
        engine._fitted = True
        return engine

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("ContentBasedEngine not fitted — call .fit() first.")
