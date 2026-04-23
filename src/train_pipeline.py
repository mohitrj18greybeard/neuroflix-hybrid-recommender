"""
NeuroFlix — Model Training Pipeline
=====================================
End-to-end training script that:
1. Runs the data pipeline
2. Trains Content-Based engine
3. Trains Collaborative Filtering (SVD) engine
4. Builds the Hybrid engine
5. Evaluates all models
6. Saves results and model artifacts

Author: Mohit Raj
"""

import logging
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import run_pipeline
from src.content_based import ContentBasedEngine
from src.collaborative import CollaborativeFilteringEngine
from src.hybrid import HybridRecommenderEngine
from src.evaluation import RecommenderEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def train_all():
    """Execute the complete training pipeline."""
    logger.info("=" * 60)
    logger.info("  NEUROFLIX — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Data Pipeline
    # ------------------------------------------------------------------
    logger.info("\n📦 Step 1: Running data pipeline …")
    data = run_pipeline(force_download=True)
    movies = data["movies"]
    ratings = data["ratings"]
    train = data["train"]
    test = data["test"]

    logger.info(
        "Data loaded — Movies: %d, Ratings: %d, Train: %d, Test: %d",
        len(movies), len(ratings), len(train), len(test),
    )

    # ------------------------------------------------------------------
    # Step 2: Content-Based Engine
    # ------------------------------------------------------------------
    logger.info("\n🎬 Step 2: Training Content-Based engine …")
    cb_engine = ContentBasedEngine(max_features=5000, ngram_range=(1, 2))
    cb_engine.fit(movies)
    cb_engine.save()
    logger.info("Content-Based engine trained and saved.")

    # ------------------------------------------------------------------
    # Step 3: Collaborative Filtering Engine (SVD)
    # ------------------------------------------------------------------
    logger.info("\n🤝 Step 3: Training Collaborative Filtering engine (SVD) …")
    cf_engine = CollaborativeFilteringEngine(method="svd", n_factors=50)
    cf_engine.fit(train, movies)
    cf_engine.save()
    logger.info("Collaborative Filtering engine trained and saved.")

    # ------------------------------------------------------------------
    # Step 4: Hybrid Engine
    # ------------------------------------------------------------------
    logger.info("\n🔀 Step 4: Building Hybrid engine …")
    hybrid_engine = HybridRecommenderEngine(
        content_engine=cb_engine,
        collab_engine=cf_engine,
        strategy="weighted",
        alpha=0.6,
        cold_start_threshold=10,
    )
    hybrid_engine.movies_df = movies
    hybrid_engine.ratings_df = ratings
    hybrid_engine.save()
    logger.info("Hybrid engine built and saved.")

    # ------------------------------------------------------------------
    # Step 5: Evaluation
    # ------------------------------------------------------------------
    logger.info("\n📊 Step 5: Evaluating all models …")
    evaluator = RecommenderEvaluator(
        test_df=test,
        train_df=train,
        movies_df=movies,
        relevance_threshold=3.5,
        k_values=[5, 10, 20],
    )

    # Evaluate Content-Based
    logger.info("Evaluating Content-Based model …")
    evaluator.evaluate_model(
        model_name="Content-Based (TF-IDF)",
        predict_fn=lambda uid, mid: cb_engine.predict_scores(uid, [mid], ratings)[0],
        recommend_fn=lambda uid, k: cb_engine.recommend_for_user(uid, ratings, top_n=k),
        sample_users=100,
    )

    # Evaluate Collaborative (SVD)
    logger.info("Evaluating Collaborative Filtering (SVD) …")
    evaluator.evaluate_model(
        model_name="Collaborative (SVD)",
        predict_fn=lambda uid, mid: cf_engine.predict(uid, mid),
        recommend_fn=lambda uid, k: cf_engine.recommend_for_user(uid, ratings, top_n=k),
        sample_users=100,
    )

    # Evaluate Hybrid
    logger.info("Evaluating Hybrid (Weighted) …")
    evaluator.evaluate_model(
        model_name="Hybrid (Weighted α=0.6)",
        predict_fn=lambda uid, mid: hybrid_engine.predict_scores(uid, [mid])[0],
        recommend_fn=lambda uid, k: hybrid_engine.recommend(user_id=uid, top_n=k),
        sample_users=100,
    )

    # ------------------------------------------------------------------
    # Step 6: Save Results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = evaluator.compare_models()
    comparison.to_csv(str(RESULTS_DIR / "model_comparison.csv"))

    summary = evaluator.get_summary()
    with open(str(RESULTS_DIR / "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    # Save as JSON too
    results_json = {}
    for model_name, metrics in evaluator.results.items():
        results_json[model_name] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in metrics.items()
        }
    with open(str(RESULTS_DIR / "evaluation_results.json"), "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    logger.info("\n" + summary)
    logger.info("\n✅ Training pipeline complete! All models saved to %s", MODELS_DIR)

    return {
        "evaluator": evaluator,
        "comparison": comparison,
        "engines": {
            "content_based": cb_engine,
            "collaborative": cf_engine,
            "hybrid": hybrid_engine,
        },
    }


if __name__ == "__main__":
    import numpy as np
    train_all()
