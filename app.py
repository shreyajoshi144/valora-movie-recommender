import streamlit as st
import pandas as pd
import os
import base64
from recommender.utils import (
    load_tmdb_movies, load_movielens_ratings, load_movielens_movies,
    map_movielens_to_tmdb, build_tmdb_ratings_matrix, get_actual_poster
)
from recommender.hybrid_engine import HybridRecommender
from recommender.evaluation import evaluate_recommender, summarize_results


def apply_velora_theme():
    # --- CSS-ONLY NETFLIX STYLE PATTERN ---
    # This creates a repeating 'V' pattern in the background without needing an image file
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: #000000;
        background-image: 
            linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Ctext x='50%25' y='50%25' font-family='Arial Black, sans-serif' font-size='60' fill='%23e50914' fill-opacity='0.07' text-anchor='middle' dominant-baseline='middle'%3EV%3C/text%3E%3C/svg%3E");
        background-size: 100px 100px;
        background-attachment: fixed;
    }}

    /* Glowing Title Effect */
    h1 {{
        color: #e50914 !important;
        text-transform: uppercase;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(229, 9, 20, 0.8), 0 0 20px rgba(229, 9, 20, 0.4) !important;
        letter-spacing: 3px;
        text-align: center;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {{
        background-color: rgba(15, 15, 15, 0.95) !important;
        backdrop-filter: blur(10px);
        border-right: 2px solid #e50914;
    }}

    /* Movie Card Styling */
    .movie-card {{
        background: rgba(30, 30, 30, 0.6);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.3s ease, border 0.3s ease;
        margin-bottom: 20px;
        min-height: 360px;
    }}

    .movie-card:hover {{
        transform: scale(1.05);
        border: 1px solid #e50914;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.5);
        background: rgba(40, 40, 40, 0.9);
    }}

    .movie-title {{
        color: white;
        font-weight: bold;
        margin-top: 10px;
        font-size: 14px;
        height: 2.5em;
        overflow: hidden;
    }}

    .score-tag {{
        color: #e50914;
        font-size: 12px;
        font-weight: bold;
        margin-top: 5px;
    }}

    /* Button Styling */
    .stButton>button {{
        background-color: #e50914 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        border: none !important;
        width: 100% !important;
        transition: 0.3s;
    }}

    .stButton>button:hover {{
        background-color: #ff0f1a !important;
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.6);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA INITIALIZATION ---
@st.cache_resource
def initialize_system():
    tmdb = load_tmdb_movies()
    ml_m = load_movielens_movies()
    ml_r = load_movielens_ratings()
    mapping = map_movielens_to_tmdb(tmdb, ml_m)
    ratings = build_tmdb_ratings_matrix(ml_r, mapping)
    engine = HybridRecommender()
    return tmdb, ratings, engine


tmdb_df, mapped_ratings, hybrid_engine = initialize_system()

# --- 3. SIDEBAR LAYOUT ---
with st.sidebar:
    st.markdown("<h2 style='color: #e50914;'>VALORA SETTINGS</h2>", unsafe_allow_html=True)
    movie_list = sorted(tmdb_df['title'].unique())
    selected_movie_name = st.selectbox("🔍 Search a Movie", [""] + movie_list)
    strategy = st.radio("Recommendation Strategy", ["Hybrid", "Content-Based", "Collaborative"])
    user_ids = sorted(mapped_ratings["userId"].unique().tolist())
    selected_user = st.selectbox("User ID (for Evaluation)", [None] + user_ids)
    top_k = st.slider("Recommendations Count", 1, 20, 6)
    predict_btn = st.button("GET RECOMMENDATIONS")

# --- 4. MAIN RESULTS AREA ---
st.title("🎬 Valora Movie Recommender")
st.write("Search a movie you love. Choose how you want recommendations. Let Valora find what you should watch next.")

if predict_btn:
    if selected_movie_name == "":
        st.error("Please search for and select a movie first!")
    else:
        seed_id = tmdb_df[tmdb_df['title'] == selected_movie_name]['tmdb_id'].values[0]

        with st.spinner('Scanning Database...'):
            recommendations = hybrid_engine.recommend(
                user_id=selected_user,
                seed_movie_id=seed_id,
                strategy=strategy,
                top_k=top_k
            )

        if recommendations:
            st.markdown("Based on your Movie Search , Valora Recommendations would be:")
            cols = st.columns(6)

            for i, rec in enumerate(recommendations):
                col_index = i % 6
                with cols[col_index]:
                    # Call the API fetcher from utils.py to get actual posters
                    img_url = get_actual_poster(rec['tmdb_id'])

                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{img_url}" style="width:100%; border-radius:5px;">
                            <div class="movie-title">{rec['title']}</div>
                            <div class="score-tag">Score: {rec.get('similarity_score', 0):.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Evaluation Metrics
            if selected_user:
                st.divider()
                st.subheader(f"📊 Evaluation for User {selected_user}")
                relevant_items = mapped_ratings[
                    (mapped_ratings["userId"] == selected_user) & (mapped_ratings["rating"] >= 4)
                    ]["tmdb_id"].tolist()


                def eval_func(u): return [r['tmdb_id'] for r in recommendations]


                results = evaluate_recommender(eval_func, {selected_user: relevant_items}, top_k=top_k)
                summary = summarize_results(results)

                c1, c2 = st.columns(2)
                c1.metric("Precision@K", f"{summary['mean_precision']:.2%}")
                c2.metric("Hit Rate", "Success" if summary['mean_hit_rate'] > 0 else "No Match")
        else:
            st.warning("No recommendations found.")
else:
    st.info("Select a movie and strategy in the sidebar to begin.")
