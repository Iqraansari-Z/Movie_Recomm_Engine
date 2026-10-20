"""
Phase 9 — Streamlit Demo
Movie Recommendation Engine

Usage:
    streamlit run src/app.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS for a clean, minimal look ----
st.markdown("""
<style>
    .main {
        background-color: #fafafa;
    }
    .movie-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border: 1px solid #eeeeee;
        height: 100%;
    }
    .movie-title {
        font-size: 17px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .movie-genre-tag {
        display: inline-block;
        background-color: #f0f2f6;
        color: #555;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 2px 4px 2px 0;
    }
    .rank-badge {
        display: inline-block;
        background-color: #2b6cb0;
        color: white;
        font-size: 13px;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
    }
    .strategy-badge {
        display: inline-block;
        background-color: #edf2f7;
        color: #4a5568;
        font-size: 13px;
        padding: 6px 14px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    h1 {
        color: #1a1a1a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "recommendation_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    movies = pd.read_parquet(PROCESSED_DIR / "movies_final.parquet")
    return artifacts, movies


artifacts, movies = load_artifacts()

user_to_idx = artifacts["user_to_idx"]
movie_to_idx = artifacts["movie_to_idx"]
user_factors = artifacts["user_factors"]
item_factors = artifacts["item_factors"]
movie_popularity_order = artifacts["movie_popularity_order"]
popularity_score = artifacts["popularity_score"]
train_seen = artifacts["train_seen"]
train_counts_per_user = artifacts["train_counts_per_user"]
COLD_START_THRESHOLD = artifacts["cold_start_threshold"]


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------
def get_recommendations(user_id, n=10, popularity_weight=0.3):
    """
    Returns (dataframe of recommendations, strategy used).
    """
    if user_id not in user_to_idx:
        top_movie_idxs = movie_popularity_order[:n]
        strategy = "Popularity fallback — new/unknown user"
    else:
        user_idx = user_to_idx[user_id]
        n_ratings = train_counts_per_user.get(user_idx, 0)
        seen = train_seen.get(user_idx, set())

        if n_ratings <= COLD_START_THRESHOLD or user_idx not in user_factors:
            top_movie_idxs = [m for m in movie_popularity_order if m not in seen][:n]
            strategy = "Popularity fallback — cold-start user"
        else:
            u_vec = user_factors[user_idx]
            raw_scores = {
                m_idx: np.dot(u_vec, i_vec)
                for m_idx, i_vec in item_factors.items()
                if m_idx not in seen
            }
            min_s, max_s = min(raw_scores.values()), max(raw_scores.values())
            range_s = max_s - min_s if max_s != min_s else 1

            blended_scores = {
                m_idx: (popularity_weight * popularity_score.get(m_idx, 0)) +
                       ((1 - popularity_weight) * (s - min_s) / range_s)
                for m_idx, s in raw_scores.items()
            }
            top_movie_idxs = [m for m, s in sorted(blended_scores.items(), key=lambda x: -x[1])[:n]]
            strategy = "Personalized (ALS + popularity blend)"

    result = movies[movies["movie_idx"].isin(top_movie_idxs)][["movie_id", "title", "genres"]].copy()
    result["movie_idx"] = result["movie_id"].map(movie_to_idx)
    result = result.set_index("movie_idx").loc[top_movie_idxs].reset_index(drop=True)
    return result, strategy


def get_valid_user_ids():
    return sorted(user_to_idx.keys())


def get_all_genres():
    return sorted({g for gs in movies["genres"] for g in gs.split("|")})


def recommend_by_genres(selected_genres, n=10, match_all=False):
    """
    Rank movies that match the chosen genres by popularity.
    match_all=True requires every chosen genre; False requires at least one.
    """
    selected = set(selected_genres)

    def matches(genre_str):
        movie_genres = set(genre_str.split("|"))
        return selected <= movie_genres if match_all else bool(selected & movie_genres)

    pool = movies[movies["genres"].apply(matches)].copy()
    pool["pop"] = pool["movie_idx"].map(popularity_score).fillna(0)
    top = pool.sort_values("pop", ascending=False).head(n)
    return top[["movie_id", "title", "genres"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_cards(recs_df, cols_per_row=5):
    """Show a dataframe of movies as a grid of cards."""
    rows = [recs_df.iloc[i:i + cols_per_row] for i in range(0, len(recs_df), cols_per_row)]
    rank = 1
    for row_chunk in rows:
        cols = st.columns(cols_per_row)
        for col, (_, movie) in zip(cols, row_chunk.iterrows()):
            genre_tags = "".join(
                f'<span class="movie-genre-tag">{g}</span>'
                for g in movie["genres"].split("|")
            )
            with col:
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="rank-badge">#{rank}</div>
                        <div class="movie-title">{movie['title']}</div>
                        <div>{genre_tags}</div>
                    </div>
                """, unsafe_allow_html=True)
            rank += 1


# ---- Header ----
st.title("🎬 Movie Recommendation Engine")
st.markdown(
    "Personalized movie suggestions powered by **Spark ALS collaborative filtering**, "
    "with automatic fallback to popularity-based recommendations for new or cold-start users."
)
st.divider()

# ---- Sidebar (used by the personalized tab) ----
with st.sidebar:
    st.header("Personalized (user ID)")

    valid_user_ids = get_valid_user_ids()

    input_mode = st.radio(
        "Choose a user",
        ["Pick an existing user", "Enter a custom user ID"],
        label_visibility="collapsed",
    )

    if input_mode == "Pick an existing user":
        user_id = st.selectbox(
            "User ID",
            options=valid_user_ids,
            index=0,
            format_func=lambda u: f"User {u} ({train_counts_per_user.get(user_to_idx[u], 0)} ratings)",
        )
    else:
        user_id = st.number_input(
            "User ID (try an unknown number for cold-start)",
            min_value=1,
            value=1,
            step=1,
        )

    n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

    pop_weight = st.slider(
        "Popularity blend weight",
        min_value=0.0, max_value=1.0, value=0.3, step=0.1,
        help="Higher = more mainstream recommendations. Lower = more personalized/niche.",
    )

    generate = st.button("Get Recommendations", type="primary", use_container_width=True)

    st.divider()
    st.caption("Built on MovieLens 1M · Spark MLlib ALS · scikit-learn")

# Keep the last personalized result so it survives interactions in the other tab
if generate:
    with st.spinner("Generating recommendations..."):
        recs_df, strategy = get_recommendations(int(user_id), n=n_recs, popularity_weight=pop_weight)
    st.session_state["personalized"] = (recs_df, strategy)

# ---- Main content: two tabs ----
tab_personal, tab_genre = st.tabs(["Personalized (user ID)", "Pick your genres"])

with tab_personal:
    if "personalized" in st.session_state:
        recs_df, strategy = st.session_state["personalized"]
        st.markdown(f'<div class="strategy-badge">Strategy: {strategy}</div>', unsafe_allow_html=True)
        if recs_df.empty:
            st.warning("No recommendations could be generated for this user.")
        else:
            render_cards(recs_df)
    else:
        st.info("👈 Select a user in the sidebar and click **Get Recommendations** to see personalized movie suggestions.")

with tab_genre:
    st.subheader("Tell us what you like")
    st.caption("No user ID needed. Pick genres and we'll show the most popular matching movies.")

    chosen_genres = st.multiselect("Genres", options=get_all_genres())
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        match_all = st.checkbox("Must match all selected genres", value=False)
    with g_col2:
        g_n = st.slider("Number of movies", min_value=5, max_value=20, value=10, key="genre_n")

    if not chosen_genres:
        st.info("Select at least one genre to see recommendations.")
    else:
        genre_recs = recommend_by_genres(chosen_genres, n=g_n, match_all=match_all)
        mode_text = "all" if match_all else "any"
        st.markdown(
            f'<div class="strategy-badge">Strategy: Genre filter (match {mode_text} of selected) ranked by popularity</div>',
            unsafe_allow_html=True,
        )
        if genre_recs.empty:
            st.warning("No movies match that combination. Try fewer genres or untick the match-all box.")
        else:
            render_cards(genre_recs)