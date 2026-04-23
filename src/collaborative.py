"""
NeuroFlix — Collaborative Filtering Engine
=============================================
Production-grade collaborative filtering with both memory-based
and model-based approaches.

Memory-Based:
  - User-User similarity (cosine)
  - Item-Item similarity (cosine)

Model-Based:
  - SVD (Truncated SVD / matrix factorization)
  - ALS-style via iterative optimization

Author: Mohit Raj
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class CollaborativeFilteringEngine:
    """
    Collaborative Filtering engine supporting:
    
    1. Memory-Based:
       - User-User CF (find similar users, recommend what they liked)
       - Item-Item CF (find similar items to what user liked)
    
    2. Model-Based:
       - Truncated SVD (matrix factorization)
       - Predicts latent user/item factors for rating estimation
    """

    def __init__(
        self,
        n_factors: int = 50,
        method: str = "svd",  # "svd", "user_user", "item_item"
    ):
        self.n_factors = n_factors
        self.method = method

        # Matrices
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_item_sparse: Optional[csr_matrix] = None
        self.user_ids: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None
        self.user_id_to_idx: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}

        # SVD components
        self.U: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        self.predicted_ratings: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        # Similarity matrices (memory-based)
        self.user_similarity: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None

        # Movies DataFrame for display
        self.movies_df: Optional[pd.DataFrame] = None

        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> "CollaborativeFilteringEngine":
        """
        Fit the collaborative filtering model.
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating
            movies_df: Optional movies DataFrame for display
        """
        logger.info("Fitting Collaborative Filtering engine (method=%s) …", self.method)
        self.movies_df = movies_df

        # Build user-item matrix
        self._build_user_item_matrix(ratings_df)

        if self.method == "svd":
            self._fit_svd()
        elif self.method == "user_user":
            self._fit_user_user()
        elif self.method == "item_item":
            self._fit_item_item()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True
        logger.info("Collaborative Filtering engine fitted successfully.")
        return self

    def _build_user_item_matrix(self, ratings_df: pd.DataFrame):
        """Build sparse user-item rating matrix."""
        self.user_ids = ratings_df["userId"].unique()
        self.item_ids = ratings_df["movieId"].unique()

        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}

        # Pivot to dense matrix (for MovieLens-small this is feasible)
        self.user_item_matrix = ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

        self.global_mean = ratings_df["rating"].mean()

        # Sparse representation
        row_ind = ratings_df["userId"].map(self.user_id_to_idx).values
        col_ind = ratings_df["movieId"].map(self.item_id_to_idx).values
        data = ratings_df["rating"].values
        self.user_item_sparse = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )

        logger.info(
            "User-Item matrix: %d users × %d items, %d ratings",
            len(self.user_ids), len(self.item_ids), len(ratings_df),
        )

    # ------------------------------------------------------------------
    # SVD (Model-Based)
    # ------------------------------------------------------------------
    def _fit_svd(self):
        """Truncated SVD for matrix factorization."""
        logger.info("Running Truncated SVD with %d factors …", self.n_factors)

        # Center ratings
        matrix = self.user_item_sparse.toarray().astype(float)
        user_means = np.true_divide(
            matrix.sum(axis=1),
            (matrix != 0).sum(axis=1),
            where=(matrix != 0).sum(axis=1) != 0,
        )
        user_means = np.nan_to_num(user_means, nan=self.global_mean)

        # Mean-center
        matrix_centered = matrix.copy()
        for i in range(matrix.shape[0]):
            mask = matrix[i] != 0
            matrix_centered[i, mask] -= user_means[i]

        # SVD decomposition
        n_factors = min(self.n_factors, min(matrix_centered.shape) - 1)
        self.U, self.sigma, self.Vt = svds(
            csr_matrix(matrix_centered), k=n_factors
        )

        # Sort by singular values (descending)
        idx = np.argsort(-self.sigma)
        self.U = self.U[:, idx]
        self.sigma = self.sigma[idx]
        self.Vt = self.Vt[idx, :]

        # Reconstruct predicted ratings
        sigma_diag = np.diag(self.sigma)
        self.predicted_ratings = (
            self.U @ sigma_diag @ self.Vt + user_means.reshape(-1, 1)
        )

        # Clip to valid range
        self.predicted_ratings = np.clip(self.predicted_ratings, 0.5, 5.0)

        logger.info(
            "SVD complete — explained variance captured by %d factors.",
            n_factors,
        )

    # ------------------------------------------------------------------
    # User-User CF (Memory-Based)
    # ------------------------------------------------------------------
    def _fit_user_user(self):
        """Compute user-user cosine similarity."""
        logger.info("Computing User-User similarity matrix …")
        matrix = self.user_item_sparse.toarray()
        self.user_similarity = cosine_similarity(matrix)
        np.fill_diagonal(self.user_similarity, 0)
        logger.info("User-User similarity matrix: %s", self.user_similarity.shape)

    # ------------------------------------------------------------------
    # Item-Item CF (Memory-Based)
    # ------------------------------------------------------------------
    def _fit_item_item(self):
        """Compute item-item cosine similarity."""
        logger.info("Computing Item-Item similarity matrix …")
        matrix = self.user_item_sparse.toarray().T  # items × users
        self.item_similarity = cosine_similarity(matrix)
        np.fill_diagonal(self.item_similarity, 0)
        logger.info("Item-Item similarity matrix: %s", self.item_similarity.shape)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict a single rating."""
        self._check_fitted()

        if user_id not in self.user_id_to_idx:
            return self.global_mean
        if movie_id not in self.item_id_to_idx:
            return self.global_mean

        u_idx = self.user_id_to_idx[user_id]
        i_idx = self.item_id_to_idx[movie_id]

        if self.method == "svd" and self.predicted_ratings is not None:
            return float(self.predicted_ratings[u_idx, i_idx])

        elif self.method == "user_user" and self.user_similarity is not None:
            return self._predict_user_user(u_idx, i_idx)

        elif self.method == "item_item" and self.item_similarity is not None:
            return self._predict_item_item(u_idx, i_idx)

        return self.global_mean

    def _predict_user_user(self, u_idx: int, i_idx: int, k: int = 30) -> float:
        """Predict using top-K similar users."""
        sim_users = self.user_similarity[u_idx]
        top_k_users = np.argsort(sim_users)[::-1][:k]

        matrix = self.user_item_sparse.toarray()
        numerator = 0.0
        denominator = 0.0

        for v_idx in top_k_users:
            if matrix[v_idx, i_idx] > 0:
                numerator += sim_users[v_idx] * matrix[v_idx, i_idx]
                denominator += abs(sim_users[v_idx])

        if denominator == 0:
            return self.global_mean
        return np.clip(numerator / denominator, 0.5, 5.0)

    def _predict_item_item(self, u_idx: int, i_idx: int, k: int = 30) -> float:
        """Predict using top-K similar items."""
        sim_items = self.item_similarity[i_idx]
        matrix = self.user_item_sparse.toarray()

        # Items rated by this user
        rated_mask = matrix[u_idx] > 0
        rated_indices = np.where(rated_mask)[0]

        if len(rated_indices) == 0:
            return self.global_mean

        # Get similarities to rated items
        sims = sim_items[rated_indices]
        top_k_idx = np.argsort(sims)[::-1][:k]

        numerator = 0.0
        denominator = 0.0
        for idx in top_k_idx:
            j = rated_indices[idx]
            if sims[idx] > 0:
                numerator += sims[idx] * matrix[u_idx, j]
                denominator += sims[idx]

        if denominator == 0:
            return self.global_mean
        return np.clip(numerator / denominator, 0.5, 5.0)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def recommend_for_user(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get top-N recommendations for a user."""
        self._check_fitted()

        if user_id not in self.user_id_to_idx:
            logger.info("User %d not in training data — cold start.", user_id)
            return self._cold_start_recommendations(top_n)

        u_idx = self.user_id_to_idx[user_id]

        # Get all predicted ratings for this user
        if self.method == "svd" and self.predicted_ratings is not None:
            scores = self.predicted_ratings[u_idx].copy()
        else:
            scores = np.array([
                self.predict(user_id, mid)
                for mid in self.item_ids
            ])

        # Exclude already-rated movies
        rated_movies = set(
            ratings_df[ratings_df["userId"] == user_id]["movieId"].values
        )
        for mid in rated_movies:
            if mid in self.item_id_to_idx:
                scores[self.item_id_to_idx[mid]] = -1.0

        # Top-N
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for i_idx in top_indices:
            mid = int(self.item_ids[i_idx])
            title = self._get_movie_title(mid)
            genres = self._get_movie_genres(mid)
            year = self._get_movie_year(mid)
            results.append({
                "movieId": mid,
                "title": title,
                "predicted_rating": round(float(scores[i_idx]), 4),
                "genres": genres,
                "year": year,
                "explanation": f"Predicted rating: {scores[i_idx]:.2f} ({self.method})",
            })

        return pd.DataFrame(results)

    def similar_movies(
        self,
        movie_id: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Find similar movies using collaborative signals."""
        self._check_fitted()

        if movie_id not in self.item_id_to_idx:
            return pd.DataFrame(columns=["movieId", "title", "similarity_score"])

        i_idx = self.item_id_to_idx[movie_id]

        if self.method == "item_item" and self.item_similarity is not None:
            sim_scores = self.item_similarity[i_idx].copy()
        elif self.method == "svd" and self.Vt is not None:
            # Use item latent factors for similarity
            item_factors = (np.diag(self.sigma) @ self.Vt).T
            item_factors_norm = normalize(item_factors, norm="l2")
            sim_scores = cosine_similarity(
                item_factors_norm[i_idx].reshape(1, -1),
                item_factors_norm,
            ).flatten()
            sim_scores[i_idx] = -1.0
        else:
            # Fallback: compute from user-item matrix
            matrix = self.user_item_sparse.toarray().T
            sim_scores = cosine_similarity(
                matrix[i_idx].reshape(1, -1), matrix
            ).flatten()
            sim_scores[i_idx] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            mid = int(self.item_ids[idx])
            results.append({
                "movieId": mid,
                "title": self._get_movie_title(mid),
                "similarity_score": round(float(sim_scores[idx]), 4),
                "genres": self._get_movie_genres(mid),
                "year": self._get_movie_year(mid),
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Batch scoring for evaluation
    # ------------------------------------------------------------------
    def predict_scores(
        self, user_id: int, movie_ids: List[int]
    ) -> np.ndarray:
        """Predict ratings for a batch of movies."""
        return np.array([self.predict(user_id, mid) for mid in movie_ids])

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------
    def _cold_start_recommendations(self, top_n: int = 10) -> pd.DataFrame:
        """Popularity-based fallback for unknown users."""
        if self.movies_df is not None and "bayesian_avg" in self.movies_df.columns:
            popular = self.movies_df.nlargest(top_n, "bayesian_avg")
        elif self.movies_df is not None:
            popular = self.movies_df.head(top_n)
        else:
            return pd.DataFrame()

        results = []
        for _, row in popular.iterrows():
            results.append({
                "movieId": int(row["movieId"]),
                "title": row.get("title", ""),
                "predicted_rating": float(row.get("bayesian_avg", row.get("rating_mean", 3.0))),
                "genres": row.get("genres_str", ""),
                "year": row.get("year", None),
                "explanation": "Popular movie (cold-start recommendation)",
            })
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_movie_title(self, movie_id: int) -> str:
        if self.movies_df is not None:
            match = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not match.empty:
                return str(match.iloc[0].get("title", match.iloc[0].get("clean_title", f"Movie {movie_id}")))
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
        """Save the fitted engine to disk."""
        if path is None:
            path = MODELS_DIR / f"collaborative_{self.method}_engine.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "method": self.method,
            "n_factors": self.n_factors,
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "user_id_to_idx": self.user_id_to_idx,
            "item_id_to_idx": self.item_id_to_idx,
            "global_mean": self.global_mean,
            "U": self.U,
            "sigma": self.sigma,
            "Vt": self.Vt,
            "predicted_ratings": self.predicted_ratings,
            "user_similarity": self.user_similarity,
            "item_similarity": self.item_similarity,
            "movies_df": self.movies_df,
            "user_item_sparse": self.user_item_sparse,
        }
        with open(str(path), "wb") as f:
            pickle.dump(state, f)
        logger.info("Collaborative engine (%s) saved to %s", self.method, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None, method: str = "svd") -> "CollaborativeFilteringEngine":
        """Load a fitted engine from disk."""
        if path is None:
            path = MODELS_DIR / f"collaborative_{method}_engine.pkl"

        with open(str(path), "rb") as f:
            state = pickle.load(f)

        engine = cls(
            n_factors=state["n_factors"],
            method=state["method"],
        )
        engine.user_ids = state["user_ids"]
        engine.item_ids = state["item_ids"]
        engine.user_id_to_idx = state["user_id_to_idx"]
        engine.item_id_to_idx = state["item_id_to_idx"]
        engine.global_mean = state["global_mean"]
        engine.U = state["U"]
        engine.sigma = state["sigma"]
        engine.Vt = state["Vt"]
        engine.predicted_ratings = state["predicted_ratings"]
        engine.user_similarity = state["user_similarity"]
        engine.item_similarity = state["item_similarity"]
        engine.movies_df = state["movies_df"]
        engine.user_item_sparse = state["user_item_sparse"]
        engine._fitted = True
        return engine

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("CollaborativeFilteringEngine not fitted — call .fit() first.")
