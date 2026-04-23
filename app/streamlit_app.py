"""
NeuroFlix — Elite Streamlit Dashboard
=======================================
Production-grade interactive movie recommendation interface.

Author: Mohit Raj
"""

import sys
import os
import logging
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.WARNING)

# ── Page Config ──
st.set_page_config(
    page_title="NeuroFlix — Hybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Hero */
.hero-section {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(229,9,20,0.08) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #e50914, #ff6b6b, #feca57);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    position: relative;
    letter-spacing: -1px;
}
.hero-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 1.1rem;
    margin-top: 0.5rem;
    position: relative;
    font-weight: 300;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(229,9,20,0.15);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e50914, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Movie Cards */
.movie-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.3rem;
    margin-bottom: 0.8rem;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}
.movie-card:hover {
    border-color: rgba(229,9,20,0.4);
    transform: translateX(5px);
    box-shadow: 0 4px 20px rgba(229,9,20,0.1);
}
.movie-card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #e50914, #ff6b6b);
    border-radius: 3px 0 0 3px;
}
.movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
}
.movie-genres {
    color: rgba(255,255,255,0.5);
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
}
.movie-score {
    display: inline-block;
    background: linear-gradient(135deg, #e50914, #ff4757);
    color: white;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
}
.movie-explanation {
    color: rgba(255,255,255,0.45);
    font-size: 0.75rem;
    margin-top: 0.4rem;
    font-style: italic;
}

/* Section Headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(229,9,20,0.3);
}

/* Comparison table */
.comparison-highlight {
    background: rgba(229,9,20,0.1);
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid rgba(229,9,20,0.2);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-right: 0.3rem;
}
.badge-cb { background: rgba(52,152,219,0.2); color: #3498db; border: 1px solid rgba(52,152,219,0.3); }
.badge-cf { background: rgba(46,204,113,0.2); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
.badge-hybrid { background: rgba(229,9,20,0.2); color: #e50914; border: 1px solid rgba(229,9,20,0.3); }

/* Rank badge */
.rank-badge {
    display: inline-block;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    border-radius: 50%;
    font-weight: 800;
    font-size: 0.8rem;
    margin-right: 0.5rem;
}
.rank-1 { background: linear-gradient(135deg, #f1c40f, #f39c12); color: #000; }
.rank-2 { background: linear-gradient(135deg, #bdc3c7, #95a5a6); color: #000; }
.rank-3 { background: linear-gradient(135deg, #e67e22, #d35400); color: #fff; }
.rank-other { background: rgba(255,255,255,0.1); color: rgba(255,255,255,0.6); }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──
@st.cache_data(ttl=3600)
def load_data():
    """Load processed data and model artifacts."""
    processed = PROJECT_ROOT / "data" / "processed"
    results = PROJECT_ROOT / "data" / "results"

    data = {}
    try:
        data["movies"] = pd.read_csv(str(processed / "movies_enriched.csv"))
        data["ratings"] = pd.read_csv(str(processed / "ratings.csv"))
        data["train"] = pd.read_csv(str(processed / "train_ratings.csv"))
        data["test"] = pd.read_csv(str(processed / "test_ratings.csv"))
        data["user_stats"] = pd.read_csv(str(processed / "user_stats.csv"))
    except FileNotFoundError:
        st.error("⚠️ Processed data not found. Run training pipeline first: `python src/train_pipeline.py`")
        st.stop()

    # Load evaluation results
    try:
        import json
        with open(str(results / "evaluation_results.json"), "r") as f:
            data["eval_results"] = json.load(f)
        data["comparison"] = pd.read_csv(str(results / "model_comparison.csv"), index_col=0)
    except Exception:
        data["eval_results"] = {}
        data["comparison"] = pd.DataFrame()

    return data


@st.cache_resource
def load_engines():
    """Load trained recommendation engines."""
    from src.content_based import ContentBasedEngine
    from src.collaborative import CollaborativeFilteringEngine
    from src.hybrid import HybridRecommenderEngine

    engines = {}
    try:
        engines["content"] = ContentBasedEngine.load()
        engines["collab"] = CollaborativeFilteringEngine.load()

        hybrid = HybridRecommenderEngine(
            content_engine=engines["content"],
            collab_engine=engines["collab"],
            strategy="weighted",
            alpha=0.6,
        )
        data = load_data()
        hybrid.movies_df = data["movies"]
        hybrid.ratings_df = data["ratings"]
        engines["hybrid"] = hybrid
    except Exception as e:
        st.error(f"⚠️ Failed to load models: {e}")
        st.stop()

    return engines


def render_movie_card(rank, title, genres, score, explanation, year=None, source=None):
    """Render a single movie recommendation card."""
    rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
    year_str = f" ({int(year)})" if year and not pd.isna(year) else ""
    source_badge = ""
    if source:
        if "content" in str(source).lower() and "collab" in str(source).lower():
            source_badge = '<span class="badge badge-hybrid">HYBRID</span>'
        elif "content" in str(source).lower():
            source_badge = '<span class="badge badge-cb">CONTENT</span>'
        elif "collab" in str(source).lower():
            source_badge = '<span class="badge badge-cf">COLLABORATIVE</span>'
        else:
            source_badge = f'<span class="badge badge-hybrid">{source.upper()}</span>'

    st.markdown(f"""
    <div class="movie-card">
        <div style="display:flex;align-items:center;">
            <span class="rank-badge {rank_class}">{rank}</span>
            <div style="flex:1;">
                <div class="movie-title">{title}{year_str}</div>
                <div class="movie-genres">{genres} {source_badge}</div>
            </div>
            <span class="movie-score">{'⭐ ' + f'{score:.1%}' if score <= 1 else '⭐ ' + f'{score:.2f}'}</span>
        </div>
        <div class="movie-explanation">💡 {explanation}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric(value, label, prefix="", suffix=""):
    """Render a metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    data = load_data()
    engines = load_engines()
    movies = data["movies"]
    ratings = data["ratings"]

    # ── Hero Section ──
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🎬 NeuroFlix</h1>
        <p class="hero-subtitle">Hybrid Movie Recommendation System — Content-Based + Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        page = st.radio(
            "Navigate",
            ["🏠 Recommendations", "📊 Model Evaluation", "🔍 Movie Explorer", "📖 System Architecture"],
            label_visibility="collapsed",
        )
        st.markdown("---")

        st.markdown("### ⚙️ Configuration")
        strategy = st.selectbox(
            "Hybrid Strategy",
            ["weighted", "switching", "cascade"],
            help="**Weighted**: Linear combination\n**Switching**: Context-aware\n**Cascade**: Two-stage refinement",
        )
        alpha = st.slider("α (CF Weight)", 0.0, 1.0, 0.6, 0.05,
                          help="Higher = more collaborative, Lower = more content-based")
        top_n = st.slider("Top-N Results", 5, 30, 10)

        st.markdown("---")
        st.markdown("### 📈 Dataset Stats")
        col1, col2 = st.columns(2)
        with col1:
            render_metric(f"{len(movies):,}", "Movies")
        with col2:
            render_metric(f"{len(ratings):,}", "Ratings")
        col3, col4 = st.columns(2)
        with col3:
            render_metric(f"{ratings['userId'].nunique():,}", "Users")
        with col4:
            n_genres = movies["genres_str"].str.split().explode().nunique() if "genres_str" in movies.columns else 0
            render_metric(f"{n_genres}", "Genres")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.7rem;'>"
            "Built by Mohit Raj<br>NeuroFlix v1.0</div>",
            unsafe_allow_html=True,
        )

    # Update engine settings
    engines["hybrid"].strategy = strategy
    engines["hybrid"].alpha = alpha

    # ── Page Router ──
    if "Recommendations" in page:
        page_recommendations(engines, data, top_n)
    elif "Model Evaluation" in page:
        page_evaluation(data)
    elif "Movie Explorer" in page:
        page_explorer(engines, data, top_n)
    elif "System Architecture" in page:
        page_architecture()


# ══════════════════════════════════════════════════════════════
#  PAGE: Recommendations
# ══════════════════════════════════════════════════════════════
def page_recommendations(engines, data, top_n):
    movies = data["movies"]
    ratings = data["ratings"]
    hybrid = engines["hybrid"]

    st.markdown('<div class="section-header">🎯 Personalized Recommendations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        user_ids = sorted(ratings["userId"].unique())
        user_id = st.selectbox("Select User ID", user_ids, index=0)
    with col2:
        movie_titles = ["(None — User-only mode)"] + sorted(movies["clean_title"].dropna().unique().tolist())
        selected_title = st.selectbox("Find movies similar to...", movie_titles)

    movie_id = None
    if selected_title != "(None — User-only mode)":
        match = movies[movies["clean_title"] == selected_title]
        if not match.empty:
            movie_id = int(match.iloc[0]["movieId"])

    # Show user profile
    user_ratings = ratings[ratings["userId"] == user_id]
    with st.expander(f"👤 User {user_id} Profile — {len(user_ratings)} ratings", expanded=False):
        if not user_ratings.empty:
            merged = user_ratings.merge(movies[["movieId", "title", "genres_str"]], on="movieId", how="left")
            top_rated = merged.nlargest(5, "rating")
            cols = st.columns(5)
            for i, (_, row) in enumerate(top_rated.iterrows()):
                with cols[i % 5]:
                    st.markdown(f"**{row['title'][:30]}...**" if len(str(row.get('title',''))) > 30 else f"**{row.get('title','N/A')}**")
                    st.markdown(f"⭐ {row['rating']}")

    # Get recommendations
    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("🧠 Computing hybrid recommendations..."):
            recs = hybrid.recommend(user_id=user_id, movie_id=movie_id, top_n=top_n)

        if recs.empty:
            st.warning("No recommendations found.")
            return

        st.markdown(f'<div class="section-header">🎬 Top {len(recs)} Recommendations</div>', unsafe_allow_html=True)

        for i, (_, row) in enumerate(recs.iterrows()):
            score_col = "score" if "score" in recs.columns else "similarity_score"
            score = row.get(score_col, row.get("predicted_rating", 0))
            render_movie_card(
                rank=i + 1,
                title=row.get("title", "Unknown"),
                genres=row.get("genres", ""),
                score=float(score),
                explanation=row.get("explanation", "Hybrid recommendation"),
                year=row.get("year", None),
                source=row.get("source", "hybrid"),
            )

        # Score distribution
        st.markdown('<div class="section-header">📊 Score Distribution</div>', unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        score_col = "score" if "score" in recs.columns else "similarity_score"
        scores = recs[score_col].values if score_col in recs.columns else recs.get("predicted_rating", pd.Series([0])).values
        colors = ['#e50914' if s >= np.percentile(scores, 75) else '#ff6b6b' if s >= np.percentile(scores, 50) else '#666' for s in scores]
        bars = ax.barh(range(len(scores)), scores, color=colors, edgecolor='none', height=0.6)
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels([str(r.get("title", ""))[:25] for _, r in recs.iterrows()], fontsize=7, color='white')
        ax.invert_yaxis()
        ax.set_xlabel("Score", color='white', fontsize=9)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='x', alpha=0.1, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════
#  PAGE: Model Evaluation
# ══════════════════════════════════════════════════════════════
def page_evaluation(data):
    st.markdown('<div class="section-header">📊 Model Evaluation & Comparison</div>', unsafe_allow_html=True)

    eval_results = data.get("eval_results", {})
    comparison = data.get("comparison", pd.DataFrame())

    if not eval_results:
        st.info("⚠️ No evaluation results found. Run training pipeline first.")
        return

    # Summary metrics
    st.markdown("### 🏆 Model Performance Comparison")
    if not comparison.empty:
        # Highlight best values
        st.dataframe(comparison.style.highlight_min(subset=[c for c in comparison.columns if "RMSE" in c or "MAE" in c], color='rgba(46,204,113,0.3)')
                     .highlight_max(subset=[c for c in comparison.columns if "Precision" in c or "Recall" in c or "NDCG" in c or "HitRate" in c or "Coverage" in c], color='rgba(46,204,113,0.3)')
                     .format("{:.4f}"))

    # Visual comparison
    st.markdown("### 📈 Visual Comparison")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    metric_groups = {
        "Rating Accuracy": ["RMSE", "MAE"],
        "Ranking Quality (K=10)": ["Precision@10", "Recall@10", "NDCG@10"],
    }

    for group_name, metrics in metric_groups.items():
        available = [m for m in metrics if m in comparison.columns]
        if not available:
            continue

        fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 3.5))
        fig.patch.set_facecolor('#0e1117')
        if len(available) == 1:
            axes = [axes]

        colors = ['#e50914', '#3498db', '#2ecc71', '#f39c12']
        for ax, metric in zip(axes, available):
            ax.set_facecolor('#0e1117')
            vals = comparison[metric].values
            model_names = [n[:15] for n in comparison.index]
            bars = ax.bar(model_names, vals, color=colors[:len(vals)], edgecolor='none', width=0.5)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

            ax.set_title(metric, fontsize=10, color='white', fontweight='bold', pad=10)
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis='y', alpha=0.1, color='white')

        fig.suptitle(group_name, fontsize=12, color='white', fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Detailed per-model breakdown
    st.markdown("### 📋 Detailed Results")
    for model_name, metrics in eval_results.items():
        with st.expander(f"📊 {model_name}", expanded=False):
            cols = st.columns(4)
            i = 0
            for k, v in metrics.items():
                if k != "model" and isinstance(v, (int, float)):
                    with cols[i % 4]:
                        render_metric(f"{v:.4f}", k)
                    i += 1


# ══════════════════════════════════════════════════════════════
#  PAGE: Movie Explorer
# ══════════════════════════════════════════════════════════════
def page_explorer(engines, data, top_n):
    st.markdown('<div class="section-header">🔍 Movie Explorer</div>', unsafe_allow_html=True)

    movies = data["movies"]
    content_engine = engines["content"]

    search = st.text_input("🔎 Search movies...", placeholder="Type a movie name...")

    if search:
        mask = movies["clean_title"].str.contains(search, case=False, na=False)
        results = movies[mask].head(20)

        if results.empty:
            st.warning("No movies found.")
            return

        for _, row in results.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                year = f" ({int(row['year'])})" if not pd.isna(row.get('year', float('nan'))) else ""
                st.markdown(f"**{row.get('title', row['clean_title'])}{year}**")
                st.caption(row.get('genres_str', ''))
            with col2:
                if st.button("Find Similar", key=f"sim_{row['movieId']}"):
                    st.session_state["explore_movie"] = int(row["movieId"])

    if "explore_movie" in st.session_state:
        mid = st.session_state["explore_movie"]
        movie_row = movies[movies["movieId"] == mid]
        if not movie_row.empty:
            info = movie_row.iloc[0]
            st.markdown(f'<div class="section-header">🎬 Similar to: {info.get("title", info["clean_title"])}</div>', unsafe_allow_html=True)

            # Show content features
            features = content_engine.get_top_features(mid, top_n=8)
            if features:
                st.markdown("**Key Features:** " + " · ".join(f"`{f}`" for f in features))

            # Get similar movies
            with st.spinner("Finding similar movies..."):
                similar = engines["hybrid"].recommend(movie_id=mid, top_n=top_n)

            if not similar.empty:
                for i, (_, row) in enumerate(similar.iterrows()):
                    score = row.get("score", row.get("similarity_score", 0))
                    render_movie_card(
                        rank=i + 1,
                        title=row.get("title", "Unknown"),
                        genres=row.get("genres", ""),
                        score=float(score),
                        explanation=row.get("explanation", ""),
                        year=row.get("year", None),
                        source=row.get("source", "hybrid"),
                    )


# ══════════════════════════════════════════════════════════════
#  PAGE: System Architecture
# ══════════════════════════════════════════════════════════════
def page_architecture():
    st.markdown('<div class="section-header">📖 System Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏗️ How NeuroFlix Works

    NeuroFlix is a **hybrid recommendation system** that intelligently combines two complementary approaches:

    ---

    #### 1️⃣ Content-Based Filtering
    - Builds **TF-IDF vectors** from movie metadata (genres, tags, titles)
    - Uses **cosine similarity** to find movies with similar content profiles
    - Builds **user taste profiles** by aggregating their highly-rated movies
    - **Strength**: Works for new users with few ratings (cold-start)
    - **Weakness**: Limited to feature-level similarity (can't capture complex preferences)

    #### 2️⃣ Collaborative Filtering (SVD)
    - Constructs a **user-item rating matrix**
    - Applies **Truncated SVD** (matrix factorization) to learn latent factors
    - Discovers hidden patterns: "Users who liked X also liked Y"
    - **Strength**: Captures complex, non-obvious taste patterns
    - **Weakness**: Requires sufficient rating history (cold-start problem)

    #### 3️⃣ Hybrid Strategies

    | Strategy | How It Works | Best For |
    |----------|-------------|----------|
    | **Weighted** | α × CF + (1-α) × CB scores | General use |
    | **Switching** | CB for cold-start, CF for active users | Mixed user base |
    | **Cascade** | CB generates candidates → CF re-ranks | High precision |

    ---

    ### 📊 Evaluation Metrics

    | Metric | What It Measures |
    |--------|-----------------|
    | **RMSE** | Rating prediction accuracy (lower = better) |
    | **MAE** | Average prediction error (lower = better) |
    | **Precision@K** | Fraction of recommended items that are relevant |
    | **Recall@K** | Fraction of relevant items that are recommended |
    | **NDCG@K** | Ranking quality (rewards correct ordering) |
    | **Coverage** | Fraction of catalog recommended |

    ---

    ### 🔧 API Design
    ```python
    # Unified API function
    recommend(user_id=42, top_n=10)          # Personalized recs
    recommend(movie_id=1, top_n=10)          # Similar movies
    recommend(user_id=42, movie_id=1, top_n=10)  # Personalized similarity
    recommend()                              # Popular/trending
    ```

    ---

    ### 🧊 Cold-Start Handling
    - **New users** (< 10 ratings): Content-based + popularity fallback
    - **New movies** (no ratings): Content features only (genres, tags)
    - **Switching hybrid** automatically detects and adapts
    """)


if __name__ == "__main__":
    main()
