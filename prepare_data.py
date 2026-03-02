"""
prepare_data.py — Convertit les fichiers MovieLens .dat en CSV propre
ou prépare un CSV déjà fusionné (format issu du screenshot du projet).

Usage (fichiers .dat classiques):
    python prepare_data.py --source dat --data_dir ./data

Usage (CSV déjà fusionné):
    python prepare_data.py --source csv --input merged_dataset.csv --data_dir ./data
"""

import argparse
import os
import pandas as pd
import numpy as np

def from_dat_files(data_dir: str) -> tuple:
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"), sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python", encoding="latin-1"
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"), sep="::",
        names=["movieId", "title", "genres"],
        engine="python", encoding="latin-1"
    )
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"), sep="::",
        names=["userId", "gender", "age", "occupation", "zip"],
        engine="python", encoding="latin-1"
    )
    return ratings, movies, users


def from_merged_csv(csv_path: str) -> tuple:
    """
    Handle the merged CSV visible in the screenshot:
    userId,gender,age,age_label,occupation,occupation_label,movieId,title,...
    """
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    print(f"[INFO] Colonnes trouvées: {list(df.columns)}")

    # Infer column names flexibly
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("userid", "user_id"):
            col_map["userId"] = c
        elif lc in ("movieid", "movie_id", "itemid"):
            col_map["movieId"] = c
        elif lc == "rating":
            col_map["rating"] = c
        elif lc in ("title", "movie_title"):
            col_map["title"] = c
        elif lc in ("genres", "genre"):
            col_map["genres"] = c

    # Minimum required columns
    for req in ["userId", "movieId", "rating"]:
        if req not in col_map:
            raise ValueError(f"Colonne '{req}' introuvable. Vérifiez le fichier CSV.")

    ratings = df[[col_map["userId"], col_map["movieId"], col_map["rating"]]].copy()
    ratings.columns = ["userId", "movieId", "rating"]

    movie_cols = [col_map["movieId"]]
    if "title" in col_map:
        movie_cols.append(col_map["title"])
    if "genres" in col_map:
        movie_cols.append(col_map["genres"])
    movies = df[movie_cols].drop_duplicates(col_map["movieId"]).copy()
    movies.columns = ["movieId"] + [c for c in ["title","genres"] if c in col_map.values()]

    return ratings, movies, None


def save(ratings: pd.DataFrame, movies: pd.DataFrame, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    ratings.to_csv(os.path.join(data_dir, "ratings_clean.csv"), index=False)
    movies.to_csv(os.path.join(data_dir,  "movies_clean.csv"),  index=False)
    # Also save merged for quick re-use
    merged = ratings.merge(movies, on="movieId", how="left")
    merged.to_csv(os.path.join(data_dir, "merged.csv"), index=False)
    print(f"[SAVED] ratings_clean.csv  movies_clean.csv  merged.csv  →  {data_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   choices=["dat","csv"], default="dat")
    parser.add_argument("--input",    default=None, help="Path to merged CSV (if --source csv)")
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    if args.source == "dat":
        ratings, movies, _ = from_dat_files(args.data_dir)
        print(f"[INFO] {len(ratings):,} notations  {len(movies):,} films")
        save(ratings, movies, args.data_dir)
    else:
        if not args.input:
            raise ValueError("Spécifiez --input chemin/vers/fichier.csv")
        ratings, movies, _ = from_merged_csv(args.input)
        print(f"[INFO] {len(ratings):,} notations  {len(movies):,} films")
        save(ratings, movies, args.data_dir)

    print("[DONE] Données prêtes. Lancez: python train_als.py")


if __name__ == "__main__":
    main()