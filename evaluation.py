import pandas as pd
import numpy as np

# -------------------------------------------------
# Precision@K
# -------------------------------------------------
def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Computes Precision@K
    recommended_items: list of recommended item IDs
    relevant_items: list of items actually liked by the user
    """
    recommended_items = recommended_items[:k]
    hits = sum([1 for item in recommended_items if item in relevant_items])
    return hits / k


# -------------------------------------------------
# Hit Rate
# -------------------------------------------------
def hit_rate(recommended_items, relevant_items):
    """
    Computes Hit Rate
    Returns 1 if any recommended item is in relevant items, else 0
    """
    hits = any([item in relevant_items for item in recommended_items])
    return int(hits)


# -------------------------------------------------
# Evaluate recommendations for multiple users
# -------------------------------------------------
def evaluate_recommender(recommendation_func, test_data, top_k=10):
    """
    Evaluate recommender across multiple users.
    recommendation_func: function that takes user_id and returns top-K tmdb_ids
    test_data: dict mapping user_id -> list of relevant movie IDs
    Returns: DataFrame with Precision@K and Hit Rate per user
    """

    results = []

    for user_id, relevant_items in test_data.items():
        recommended_items = recommendation_func(user_id)
        precision = precision_at_k(recommended_items, relevant_items, k=top_k)
        hit = hit_rate(recommended_items, relevant_items)

        results.append({
            "user_id": user_id,
            "precision_at_{}".format(top_k): precision,
            "hit_rate": hit
        })

    return pd.DataFrame(results)


# -------------------------------------------------
# Summarize results
# -------------------------------------------------
def summarize_results(results_df):
    """
    Returns mean Precision@K and mean Hit Rate
    """
    summary = {
        "mean_precision": results_df.filter(like="precision").mean().values[0],
        "mean_hit_rate": results_df["hit_rate"].mean()
    }
    return summary
