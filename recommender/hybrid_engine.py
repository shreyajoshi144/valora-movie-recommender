import pandas as pd
from recommender.collaborative import recommend_similar_movies_cf, prepare_collaborative_data, compute_item_similarity
from recommender.cold_start import cold_start_recommender
from recommender.content_based import content_based_recommender
from recommender.utils import load_tmdb_movies


class HybridRecommender:
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.tmdb_df = load_tmdb_movies()
        self.user_item_matrix, _ = prepare_collaborative_data()
        self.cf_similarity_df = compute_item_similarity(self.user_item_matrix)

    def recommend(self, user_id=None, seed_movie_id=None, strategy="Hybrid", top_k=None):
        k = top_k if top_k else self.top_k

        # 1. CONTENT-BASED
        if strategy == "Content-Based" and seed_movie_id:
            return content_based_recommender(seed_movie_id, top_k=k)

        # 2. COLLABORATIVE (Item-Based via Seed)
        if strategy == "Collaborative" and seed_movie_id:
            return recommend_similar_movies_cf(seed_movie_id, self.cf_similarity_df, self.tmdb_df, top_k=k)

        # 3. HYBRID (Priority Search)
        if strategy == "Hybrid" and seed_movie_id:
            content_recs = content_based_recommender(seed_movie_id, top_k=k)
            # Combine or fallback logic
            if not content_recs:
                return cold_start_recommender(user_has_history=False, top_k=k)
            return content_recs

        # DEFAULT FALLBACK (Popular Movies)
        return cold_start_recommender(user_has_history=False, top_k=k)