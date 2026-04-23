"""
NeuroFlix — Streamlit Cloud Entry Point
=========================================
This is the main entry point for Streamlit Cloud deployment.
It handles data download, model training, and dashboard launch
all within the cloud environment.

Author: Mohit Raj
"""

import sys
import os
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def ensure_data_and_models():
    """Download data and train models if not already present."""
    models_dir = PROJECT_ROOT / "models"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    cb_model = models_dir / "content_based_engine.pkl"
    cf_model = models_dir / "collaborative_svd_engine.pkl"
    
    if cb_model.exists() and cf_model.exists():
        return True
    
    with st.spinner("🚀 First-time setup: Downloading data & training models (this takes ~2 minutes)..."):
        try:
            from src.data_pipeline import run_pipeline
            from src.content_based import ContentBasedEngine
            from src.collaborative import CollaborativeFilteringEngine
            from src.hybrid import HybridRecommenderEngine
            
            # Run data pipeline
            st.info("📦 Downloading MovieLens dataset...")
            data = run_pipeline(force_download=True)
            movies = data["movies"]
            ratings = data["ratings"]
            train = data["train"]
            
            # Train Content-Based
            st.info("🎬 Training Content-Based engine...")
            cb = ContentBasedEngine(max_features=5000, ngram_range=(1, 2))
            cb.fit(movies)
            cb.save()
            
            # Train Collaborative
            st.info("🤝 Training Collaborative Filtering engine...")
            cf = CollaborativeFilteringEngine(method="svd", n_factors=50)
            cf.fit(train, movies)
            cf.save()
            
            # Build Hybrid
            st.info("🔀 Building Hybrid engine...")
            hybrid = HybridRecommenderEngine(
                content_engine=cb, collab_engine=cf,
                strategy="weighted", alpha=0.6,
            )
            hybrid.movies_df = movies
            hybrid.ratings_df = ratings
            hybrid.save()
            
            st.success("✅ Setup complete! Models trained and saved.")
            return True
            
        except Exception as e:
            st.error(f"❌ Setup failed: {e}")
            logger.error("Setup failed: %s", e, exc_info=True)
            return False


if __name__ == "__main__":
    # This file is run by: streamlit run streamlit_app.py
    pass

# Ensure models exist
if ensure_data_and_models():
    # Import and run the main app
    from app.streamlit_app import main
    main()
else:
    st.error("Please check the logs and try refreshing the page.")
