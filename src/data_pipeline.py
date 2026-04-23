"""
NeuroFlix — Data Pipeline Module
================================
Production-grade data ingestion, preprocessing, and feature engineering
for the MovieLens dataset. Handles data download, cleaning, feature
extraction, and train/test splitting with temporal awareness.

Author: Mohit Raj
"""

import os
import zipfile
import urllib.request
import logging
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download & Extract
# ---------------------------------------------------------------------------
def download_movielens(url: str = MOVIELENS_URL, dest: Path = RAW_DIR) -> Path:
    """Download and extract the MovieLens dataset if not already present."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ml-latest-small.zip"
    extract_dir = dest / "ml-latest-small"

    if extract_dir.exists() and (extract_dir / "ratings.csv").exists():
        logger.info("MovieLens data already downloaded — skipping.")
        return extract_dir

    logger.info("Downloading MovieLens dataset from %s …", url)
    try:
        urllib.request.urlretrieve(url, str(zip_path))
    except Exception as e:
        logger.error("Download failed: %s", e)
        raise

    logger.info("Extracting archive …")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest))

    zip_path.unlink(missing_ok=True)
    logger.info("Dataset ready at %s", extract_dir)
    return extract_dir


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_raw_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load all raw CSVs into a dict of DataFrames."""
    if data_dir is None:
        data_dir = download_movielens()

    frames: Dict[str, pd.DataFrame] = {}
    for name in ("ratings", "movies", "tags", "links"):
        path = data_dir / f"{name}.csv"
        if path.exists():
            frames[name] = pd.read_csv(str(path))
            logger.info("Loaded %s  →  %s rows", name, len(frames[name]))
        else:
            logger.warning("File not found: %s", path)
    return frames


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and feature-engineer the movies DataFrame.
    
    Extracts:
    - year from title
    - individual genre columns (one-hot)
    - genre string for TF-IDF
    """
    df = movies_df.copy()

    # Extract year from title  e.g. "Toy Story (1995)"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

    # Genre processing
    df["genres_list"] = df["genres"].str.split("|")
    df["genres_str"] = df["genres"].str.replace("|", " ", regex=False)

    # One-hot genre encoding
    all_genres = sorted(
        {g for genres in df["genres_list"].dropna() for g in genres if g != "(no genres listed)"}
    )
    for genre in all_genres:
        df[f"genre_{genre}"] = df["genres_list"].apply(
            lambda gs: 1 if isinstance(gs, list) and genre in gs else 0
        )

    # Fill missing years with median
    median_year = df["year"].median()
    df["year"] = df["year"].fillna(median_year)

    # Decade feature
    df["decade"] = (df["year"] // 10 * 10).astype(int)

    return df


def preprocess_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Clean ratings and add derived features."""
    df = ratings_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["rating_year"] = df["timestamp"].dt.year
    df["rating_month"] = df["timestamp"].dt.month
    return df


def preprocess_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tags per movie into a single string."""
    if tags_df is None or tags_df.empty:
        return pd.DataFrame(columns=["movieId", "tags_str"])
    
    df = tags_df.copy()
    df["tag"] = df["tag"].astype(str).str.lower().str.strip()
    agg = df.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    agg.columns = ["movieId", "tags_str"]
    return agg


def compute_movie_statistics(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-movie rating statistics for popularity & quality signals."""
    stats = ratings_df.groupby("movieId").agg(
        rating_count=("rating", "size"),
        rating_mean=("rating", "mean"),
        rating_std=("rating", "std"),
        rating_median=("rating", "median"),
        rating_min=("rating", "min"),
        rating_max=("rating", "max"),
    ).reset_index()

    stats["rating_std"] = stats["rating_std"].fillna(0)

    # Bayesian average (weighted rating) — similar to IMDB formula
    C = stats["rating_mean"].mean()            # global mean
    m = stats["rating_count"].quantile(0.25)   # minimum votes threshold
    stats["bayesian_avg"] = (
        (stats["rating_count"] * stats["rating_mean"] + m * C)
        / (stats["rating_count"] + m)
    )

    # Popularity score (log-scaled count)
    stats["popularity"] = np.log1p(stats["rating_count"])

    return stats


def compute_user_statistics(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user rating statistics for user profiling."""
    stats = ratings_df.groupby("userId").agg(
        user_rating_count=("rating", "size"),
        user_rating_mean=("rating", "mean"),
        user_rating_std=("rating", "std"),
    ).reset_index()
    stats["user_rating_std"] = stats["user_rating_std"].fillna(0)
    return stats


# ---------------------------------------------------------------------------
# Temporal train/test split
# ---------------------------------------------------------------------------
def temporal_train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings using a per-user temporal strategy:
    for each user, the most recent `test_ratio` ratings go to test.
    This prevents data leakage from future ratings.
    """
    ratings_sorted = ratings_df.sort_values(["userId", "timestamp"])

    train_frames, test_frames = [], []
    for _, user_df in ratings_sorted.groupby("userId"):
        n_test = max(1, int(len(user_df) * test_ratio))
        train_frames.append(user_df.iloc[:-n_test])
        test_frames.append(user_df.iloc[-n_test:])

    train = pd.concat(train_frames).reset_index(drop=True)
    test = pd.concat(test_frames).reset_index(drop=True)

    logger.info("Train: %d ratings | Test: %d ratings", len(train), len(test))
    return train, test


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def run_pipeline(force_download: bool = False) -> Dict[str, Any]:
    """
    Execute the full data pipeline:
    1. Download / load raw data
    2. Preprocess movies, ratings, tags
    3. Compute statistics
    4. Build enriched movie DataFrame
    5. Temporal train/test split
    6. Persist processed data
    
    Returns a dict with all processed DataFrames.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_marker = PROCESSED_DIR / ".pipeline_done"
    if cache_marker.exists() and not force_download:
        logger.info("Loading cached processed data …")
        return _load_processed()

    # Step 1 — raw data
    raw = load_raw_data()
    ratings_raw = raw["ratings"]
    movies_raw = raw["movies"]
    tags_raw = raw.get("tags", pd.DataFrame())

    # Step 2 — preprocess
    movies = preprocess_movies(movies_raw)
    ratings = preprocess_ratings(ratings_raw)
    tags_agg = preprocess_tags(tags_raw)

    # Step 3 — statistics
    movie_stats = compute_movie_statistics(ratings)
    user_stats = compute_user_statistics(ratings)

    # Step 4 — enrich movies
    movies_enriched = movies.merge(movie_stats, on="movieId", how="left")
    movies_enriched = movies_enriched.merge(tags_agg, on="movieId", how="left")
    movies_enriched["tags_str"] = movies_enriched["tags_str"].fillna("")

    # Combined text feature for content-based filtering
    movies_enriched["content_features"] = (
        movies_enriched["genres_str"].fillna("")
        + " " + movies_enriched["tags_str"]
        + " " + movies_enriched["clean_title"].fillna("")
    )

    # Step 5 — split
    train, test = temporal_train_test_split(ratings)

    # Step 6 — persist
    movies_enriched.to_csv(str(PROCESSED_DIR / "movies_enriched.csv"), index=False)
    ratings.to_csv(str(PROCESSED_DIR / "ratings.csv"), index=False)
    train.to_csv(str(PROCESSED_DIR / "train_ratings.csv"), index=False)
    test.to_csv(str(PROCESSED_DIR / "test_ratings.csv"), index=False)
    user_stats.to_csv(str(PROCESSED_DIR / "user_stats.csv"), index=False)
    cache_marker.touch()

    logger.info("Pipeline complete — processed data saved.")
    return {
        "movies": movies_enriched,
        "ratings": ratings,
        "train": train,
        "test": test,
        "user_stats": user_stats,
    }


def _load_processed() -> Dict[str, Any]:
    """Load previously-processed data from disk."""
    return {
        "movies": pd.read_csv(str(PROCESSED_DIR / "movies_enriched.csv")),
        "ratings": pd.read_csv(str(PROCESSED_DIR / "ratings.csv")),
        "train": pd.read_csv(str(PROCESSED_DIR / "train_ratings.csv")),
        "test": pd.read_csv(str(PROCESSED_DIR / "test_ratings.csv")),
        "user_stats": pd.read_csv(str(PROCESSED_DIR / "user_stats.csv")),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = run_pipeline(force_download=True)
    for k, v in result.items():
        print(f"  {k:>15s}: {v.shape}")
