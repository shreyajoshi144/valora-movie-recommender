import pandas as pd
import numpy as np
import re
import requests
from difflib import get_close_matches

# -------------------------------------------------
# Title cleaning
# -------------------------------------------------
def clean_title(title: str) -> str:
    if pd.isna(title):
        return ""
    title = title.lower()
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"[^a-z0-9 ]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title

# -------------------------------------------------
# Load TMDB data
# -------------------------------------------------
def load_tmdb_movies(path="data/tmdb_5000_movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "poster_path" not in df.columns:
        df["poster_path"] = np.nan

    df = df[["id", "title", "genres", "overview", "vote_average", "popularity", "poster_path"]]
    df.rename(columns={"id": "tmdb_id"}, inplace=True)
    df["clean_title"] = df["title"].apply(clean_title)
    return df

# -------------------------------------------------
# Load MovieLens data
# -------------------------------------------------
def load_movielens_movies(path="data/movielens_movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["clean_title"] = df["title"].apply(clean_title)
    return df

def load_movielens_ratings(path="data/movielens_ratings.csv") -> pd.DataFrame:
    return pd.read_csv(path)

# -------------------------------------------------
# TMDB ↔ MovieLens mapping
# -------------------------------------------------
def map_movielens_to_tmdb(tmdb_df, ml_movies_df, cutoff=0.85):
    tmdb_titles = tmdb_df["clean_title"].tolist()
    tmdb_lookup = dict(zip(tmdb_df["clean_title"], tmdb_df["tmdb_id"]))
    rows = []
    for _, row in ml_movies_df.iterrows():
        matches = get_close_matches(row["clean_title"], tmdb_titles, n=1, cutoff=cutoff)
        if matches:
            rows.append({"movieId": row["movieId"], "tmdb_id": tmdb_lookup[matches[0]]})
    return pd.DataFrame(rows)

# -------------------------------------------------
# ⭐ CRITICAL FIX — BUILD USER RATING DATASET ⭐
# -------------------------------------------------
def build_tmdb_ratings_matrix(ratings_df, mapping_df):
    merged = ratings_df.merge(mapping_df, on="movieId", how="inner")
    return merged[["userId", "tmdb_id", "rating"]]

# -------------------------------------------------
# ⭐ FIX: REQUIRED BY COLLABORATIVE ENGINE ⭐
# -------------------------------------------------
def create_user_item_matrix(ratings_df):
    """Pivot table required for collaborative filtering."""
    return ratings_df.pivot_table(index="userId", columns="tmdb_id", values="rating")

# -------------------------------------------------
# NEW FEATURE: FETCH REAL MOVIE POSTERS
# -------------------------------------------------
def get_actual_poster(tmdb_id):
    """Fetches the actual poster URL from TMDB API using the movie ID."""
    API_KEY = "3b2b7ca05ea4688646c32685258868ea"
    if not tmdb_id or str(tmdb_id) == "nan":
        return "https://placehold.co/500x750/000000/FFFFFF?text=No+ID"
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            path = data.get('poster_path')
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception:
        pass
    return "https://placehold.co/500x750/1a1a1a/e50914?text=Poster+Unavailable"
