"""
als_model.py — Classe ALS partagée entre train_als.py et app.py
Ce fichier DOIT être dans le même dossier que app.py et train_als.py.
"""

import time
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error


class ALS:
    """
    Alternating Least Squares Matrix Factorization.

    Minimise : Σ (r_ui - u_u · v_i)² + λ(||u_u||² + ||v_i||²)

    Chaque sous-problème est une régression ridge en forme fermée :
        u_u = (V_I^T V_I + λI)^{-1} V_I^T r_u

    Architecture distribuée (ALS > SGD) :
    - Chaque mise à jour utilisateur/item est INDÉPENDANTE → embarrassingly parallel.
    - Pas de gradient partagé, pas de learning-rate à tuner.
    - Scalabilité linéaire avec #users + #items sur Spark/MapReduce.
    """

    def __init__(self, n_factors: int = 50, n_iter: int = 20,
                 reg: float = 0.1, seed: int = 42):
        self.n_factors = n_factors
        self.n_iter    = n_iter
        self.reg       = reg
        self.seed      = seed

        self.user_factors = None
        self.item_factors = None
        self.user_ids     = None
        self.item_ids     = None
        self.user_map     = None
        self.item_map     = None
        self.global_mean  = None
        self.train_rmse   = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_index(self, ratings):
        self.user_ids = np.sort(ratings["userId"].unique())
        self.item_ids = np.sort(ratings["movieId"].unique())
        self.user_map = {u: i for i, u in enumerate(self.user_ids)}
        self.item_map = {m: i for i, m in enumerate(self.item_ids)}
        self.n_users  = len(self.user_ids)
        self.n_items  = len(self.item_ids)

    def _to_sparse(self, ratings) -> csr_matrix:
        rows = ratings["userId"].map(self.user_map).values
        cols = ratings["movieId"].map(self.item_map).values
        vals = ratings["rating"].values.astype(np.float32)
        return csr_matrix((vals, (rows, cols)),
                          shape=(self.n_users, self.n_items))

    # ── core ALS loop ─────────────────────────────────────────────────────────

    def fit(self, ratings):
        rng = np.random.RandomState(self.seed)
        self._build_index(ratings)
        self.global_mean = ratings["rating"].mean()

        R   = self._to_sparse(ratings)
        R_T = R.T.tocsr()

        self.user_factors = rng.normal(0, 0.1, (self.n_users, self.n_factors)).astype(np.float64)
        self.item_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors)).astype(np.float64)

        lI = self.reg * np.eye(self.n_factors, dtype=np.float64)

        for it in range(self.n_iter):
            t0 = time.time()
            self.user_factors = self._solve_factors(R,   self.item_factors, lI)
            self.item_factors = self._solve_factors(R_T, self.user_factors,  lI)
            rmse = self._rmse(R)
            self.train_rmse.append(rmse)
            print(f"  Iter {it+1:>2}/{self.n_iter}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")

        return self

    @staticmethod
    def _solve_factors(R: csr_matrix, fixed: np.ndarray, lI: np.ndarray) -> np.ndarray:
        n_rows    = R.shape[0]
        n_factors = fixed.shape[1]
        solved    = np.zeros((n_rows, n_factors), dtype=np.float64)

        for i in range(n_rows):
            row     = R.getrow(i)
            indices = row.indices
            if len(indices) == 0:
                continue
            values = row.data.astype(np.float64)
            F_I    = fixed[indices]
            A      = F_I.T @ F_I + lI
            b      = F_I.T @ values
            solved[i] = np.linalg.solve(A, b)

        return solved

    def _rmse(self, R: csr_matrix, max_samples: int = 50_000) -> float:
        cx  = R.tocoo()
        idx = np.random.choice(len(cx.data), min(max_samples, len(cx.data)), replace=False)
        u   = cx.row[idx]
        i   = cx.col[idx]
        r   = cx.data[idx]
        pred = np.clip(np.sum(self.user_factors[u] * self.item_factors[i], axis=1), 1.0, 5.0)
        return float(np.sqrt(mean_squared_error(r, pred)))

    # ── inference ─────────────────────────────────────────────────────────────

    def recommend(self, user_idx: int, top_k: int = 10,
                  seen_items: set = None) -> np.ndarray:
        scores = self.item_factors @ self.user_factors[user_idx]
        if seen_items:
            for si in seen_items:
                if si < len(scores):
                    scores[si] = -np.inf
        return np.argsort(scores)[::-1][:top_k]

    def predict_rating(self, user_idx: int, item_idx: int) -> float:
        return float(np.clip(
            self.user_factors[user_idx] @ self.item_factors[item_idx], 1.0, 5.0
        ))

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ALS":
        with open(path, "rb") as f:
            return pickle.load(f)