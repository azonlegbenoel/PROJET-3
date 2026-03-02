"""
train_als.py — ALS Factorization on MovieLens 1M
Usage: python train_als.py --data_dir ./data --output_dir ./models
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

# Import de la classe ALS depuis le module partagé
from als_model import ALS

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_movielens(data_dir: str):
    """Load ratings, movies and users from MovieLens 1M .dat files or a merged CSV."""

    # Accept either raw .dat files or the merged CSV shown in the assignment screenshot
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path  = os.path.join(data_dir, "movies.dat")
    users_path   = os.path.join(data_dir, "users.dat")
    merged_csv   = os.path.join(data_dir, "merged.csv")

    if os.path.exists(merged_csv):
        print("[INFO] Loading merged CSV …")
        df = pd.read_csv(merged_csv)
        # expected columns: userId, movieId, rating, title (at minimum)
        ratings = df[["userId", "movieId", "rating"]].copy()
        movies  = df[["movieId", "title"]].drop_duplicates("movieId").copy()
    elif os.path.exists(ratings_path):
        print("[INFO] Loading raw .dat files …")
        ratings = pd.read_csv(
            ratings_path, sep="::",
            names=["userId", "movieId", "rating", "timestamp"],
            engine="python", encoding="latin-1"
        )[["userId", "movieId", "rating"]]

        movies = pd.read_csv(
            movies_path, sep="::",
            names=["movieId", "title", "genres"],
            engine="python", encoding="latin-1"
        )[["movieId", "title", "genres"]]
    else:
        raise FileNotFoundError(
            f"No recognisable data found in {data_dir}.\n"
            "Expected: ratings.dat + movies.dat  OR  merged.csv"
        )

    print(f"[INFO] Ratings: {len(ratings):,}  Movies: {len(movies):,}  "
          f"Users: {ratings['userId'].nunique():,}")
    return ratings, movies


# ─────────────────────────────────────────────
# 2. SPARSITY ANALYSIS  (anciennement section 3)
# ─────────────────────────────────────────────

def compute_sparsity(ratings: pd.DataFrame) -> float:
    n_users  = ratings["userId"].nunique()
    n_items  = ratings["movieId"].nunique()
    n_obs    = len(ratings)
    sparsity = 1.0 - n_obs / (n_users * n_items)
    print(f"\n[SPARSITY REPORT]")
    print(f"  Users      : {n_users:,}")
    print(f"  Movies     : {n_items:,}")
    print(f"  Ratings    : {n_obs:,}")
    print(f"  Matrix size: {n_users * n_items:,}")
    print(f"  Sparsity   : {sparsity*100:.2f}%")
    return sparsity


# ─────────────────────────────────────────────
# 4. COLD-START STRATEGY
# ─────────────────────────────────────────────

def build_popularity_fallback(ratings: pd.DataFrame, movies: pd.DataFrame,
                              top_n: int = 50) -> pd.DataFrame:
    """
    Wilson-score lower bound of the Bayesian average — a statistically robust
    popularity ranking that avoids promoting films with very few ratings.
    """
    stats = (ratings.groupby("movieId")["rating"]
             .agg(count="count", mean="mean")
             .reset_index())

    # Bayesian average: (v·R + m·C) / (v+m)
    m = stats["count"].quantile(0.70)   # minimum votes threshold
    C = stats["mean"].mean()            # overall mean
    stats["bayes_score"] = (
        (stats["count"] * stats["mean"] + m * C) / (stats["count"] + m)
    )
    top = (stats.sort_values("bayes_score", ascending=False)
                .head(top_n)
                .merge(movies[["movieId", "title"]], on="movieId"))
    return top


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ALS on MovieLens 1M")
    parser.add_argument("--data_dir",   default="./data",   help="Directory with MovieLens data")
    parser.add_argument("--output_dir", default="./models", help="Where to save artefacts")
    parser.add_argument("--n_factors",  type=int, default=50)
    parser.add_argument("--n_iter",     type=int, default=20)
    parser.add_argument("--reg",        type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 5a. Load
    ratings, movies = load_movielens(args.data_dir)

    # 5b. Sparsity
    sparsity = compute_sparsity(ratings)

    # 5c. Train / val split (90 / 10 random)
    np.random.seed(42)
    mask   = np.random.rand(len(ratings)) < 0.9
    train  = ratings[mask].reset_index(drop=True)
    val    = ratings[~mask].reset_index(drop=True)
    print(f"\n[SPLIT] Train: {len(train):,}  Val: {len(val):,}")

    # 5d. Fit
    print("\n[TRAINING ALS] ...")
    model = ALS(n_factors=args.n_factors, n_iter=args.n_iter, reg=args.reg)
    model.fit(train)

    # 5e. Validation RMSE
    common = val[val["userId"].isin(model.user_map) & val["movieId"].isin(model.item_map)]
    u_idx  = common["userId"].map(model.user_map).values
    i_idx  = common["movieId"].map(model.item_map).values
    true_r = common["rating"].values
    pred_r = np.clip(
        np.sum(model.user_factors[u_idx] * model.item_factors[i_idx], axis=1), 1.0, 5.0
    )
    val_rmse = float(np.sqrt(mean_squared_error(true_r, pred_r)))
    print(f"\n[VALIDATION RMSE] {val_rmse:.4f}")

    # 5f. Popularity fallback (cold-start)
    popular = build_popularity_fallback(ratings, movies, top_n=50)

    # 5g. Persist artefacts
    model.save(os.path.join(args.output_dir, "als_model.pkl"))

    movies.to_pickle(os.path.join(args.output_dir, "movies.pkl"))
    ratings.to_pickle(os.path.join(args.output_dir, "ratings.pkl"))
    popular.to_pickle(os.path.join(args.output_dir, "popular.pkl"))

    # Save metrics for Streamlit dashboard
    metrics = {
        "sparsity":     sparsity,
        "n_users":      ratings["userId"].nunique(),
        "n_items":      ratings["movieId"].nunique(),
        "n_ratings":    len(ratings),
        "val_rmse":     val_rmse,
        "train_rmse":   model.train_rmse,
        "n_factors":    args.n_factors,
        "n_iter":       args.n_iter,
        "reg":          args.reg,
        "global_mean":  float(ratings["rating"].mean()),
    }
    with open(os.path.join(args.output_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    print(f"\n[DONE] All artefacts saved to {args.output_dir}/")
    print("       Run:  streamlit run app.py")


if __name__ == "__main__":
    main()