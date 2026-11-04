"""Filtrado colaborativo basado en ítems (Item-KNN) con similitud coseno."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_TOP_K = 50          # vecinos conservados por producto
DEFAULT_USER_BIAS_REG = 3.0  # regularización del sesgo de usuario (pocos ratings -> sesgo pequeño)
DEFAULT_SUPPORT_SHRINK = 0.1   # amortigua predicciones con poca evidencia (suma de similitudes baja)


# --------------------------------------------------------------------------
# Entrenamiento (se ejecuta una sola vez en scripts/prepare_data.py)
# --------------------------------------------------------------------------
def build_rating_matrix(
    user_idx: np.ndarray, item_idx: np.ndarray, ratings: np.ndarray, n_users: int, n_items: int
) -> sp.csr_matrix:
    """Matriz dispersa R de usuarios x productos con los ratings originales."""
    return sp.csr_matrix(
        (np.asarray(ratings, dtype=np.float32), (user_idx, item_idx)),
        shape=(n_users, n_items),
    )


def compute_item_similarity(
    R: sp.csr_matrix,
    top_k: int = DEFAULT_TOP_K,
    block_size: int = 1000,
    verbose: bool = True,
) -> sp.csr_matrix:
    """Similitud coseno entre productos, conservando los ``top_k`` vecinos de cada uno.

    sim(i, j) = (R_i . R_j) / (||R_i|| ||R_j||)

    Se calcula por bloques de filas para no materializar la matriz densa de
    |I| x |I|. La diagonal (un producto contra sí mismo) se excluye.
    Devuelve una CSR de |I| x |I| donde la fila i contiene los vecinos de i.
    """
    n_items = R.shape[1]
    norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    Rn = (R @ sp.diags(1.0 / norms)).tocsr().astype(np.float32)
    RnT = Rn.T.tocsr()

    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []
    row_lengths = np.zeros(n_items, dtype=np.int64)

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        block = (RnT[start:end] @ Rn).tocsr()
        indptr, indices, data = block.indptr, block.indices, block.data
        for r in range(end - start):
            lo, hi = indptr[r], indptr[r + 1]
            cols = indices[lo:hi]
            vals = data[lo:hi]
            keep = (cols != start + r) & (vals > 0)
            cols, vals = cols[keep], vals[keep]
            if len(vals) > top_k:
                top = np.argpartition(-vals, top_k - 1)[:top_k]
                cols, vals = cols[top], vals[top]
            order = np.argsort(-vals, kind="stable")
            all_cols.append(cols[order].astype(np.int32))
            all_vals.append(vals[order].astype(np.float32))
            row_lengths[start + r] = len(vals)
        if verbose and ((start // block_size) % 10 == 0 or end == n_items):
            print(f"  similitud: {end:,}/{n_items:,} productos", flush=True)

    indptr = np.concatenate([[0], np.cumsum(row_lengths)]).astype(np.int64)
    indices = np.concatenate(all_cols) if all_cols else np.array([], dtype=np.int32)
    data = np.concatenate(all_vals) if all_vals else np.array([], dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(n_items, n_items))


# --------------------------------------------------------------------------
# Inferencia (se ejecuta en cada cambio de ratings de la sesión, sin reentrenar)
# --------------------------------------------------------------------------
class ItemKNN:
    """Predice ratings para un usuario nuevo a partir de sus valoraciones.

    Para cada producto candidato i, con I_u los productos que el usuario calificó:

        b_uj   = mu + bu + b_j                          (baseline)
        Score  = b_ui + sum_j sim(j,i) * (r_uj - b_uj) / (sum_j |sim(j,i)| + shrink)

    * mu: rating medio global; b_j: sesgo (regularizado) del producto;
      bu: sesgo del usuario estimado con sus propios ratings.
    * Centrar sobre el baseline evita que las predicciones dependan solo de
      ratings absolutos (un usuario que califica todo con 5 no recomienda "todo").
    * ``shrink`` reduce el efecto de candidatos con poca evidencia.

    Solo se consideran candidatos los productos que son vecinos (top-K) de algún
    producto calificado, por lo que no hace falta entrenar nada.
    """

    def __init__(
        self,
        similarity: sp.csr_matrix,
        item_bias: np.ndarray,
        global_mean: float,
        user_bias_reg: float = DEFAULT_USER_BIAS_REG,
        support_shrink: float = DEFAULT_SUPPORT_SHRINK,
    ) -> None:
        self.similarity = similarity.tocsr()
        self.item_bias = np.asarray(item_bias, dtype=np.float64)
        self.global_mean = float(global_mean)
        self.user_bias_reg = float(user_bias_reg)
        self.support_shrink = float(support_shrink)
        self.n_items = similarity.shape[0]

    # -- utilidades -----------------------------------------------------------
    def user_bias(self, rated_idx: np.ndarray, ratings: np.ndarray) -> float:
        if len(rated_idx) == 0:
            return 0.0
        base = self.global_mean + self.item_bias[rated_idx]
        return float(np.sum(ratings - base) / (len(rated_idx) + self.user_bias_reg))

    def neighbors(self, item_idx: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Vecinos más similares (índices, similitudes) de un producto."""
        lo, hi = self.similarity.indptr[item_idx], self.similarity.indptr[item_idx + 1]
        return self.similarity.indices[lo:hi][:k], self.similarity.data[lo:hi][:k]

    # -- predicción -----------------------------------------------------------
    def predict_all(
        self, rated: Mapping[int, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rating estimado, suma de similitudes y nº de vecinos calificados por producto.

        Los productos sin ningún vecino calificado tienen ``n_neighbors == 0``
        y su predicción es solo el baseline (no son candidatos fiables).
        """
        rated_idx = np.fromiter(rated.keys(), dtype=np.int64, count=len(rated))
        ratings = np.fromiter(rated.values(), dtype=np.float64, count=len(rated))
        bu = self.user_bias(rated_idx, ratings)

        baseline_rated = self.global_mean + bu + self.item_bias[rated_idx]
        residual = ratings - baseline_rated

        sub = self.similarity[rated_idx]                    # k x |I|
        numerator = sub.T.dot(residual)                      # |I|
        support = np.asarray(abs(sub).sum(axis=0)).ravel()   # sum |sim|
        n_neighbors = np.asarray((sub > 0).sum(axis=0)).ravel()

        baseline = self.global_mean + bu + self.item_bias
        denominator = support + self.support_shrink
        correction = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )
        pred = baseline + correction
        return np.clip(pred, 1.0, 5.0), support, n_neighbors

    def recommend(
        self, rated: Mapping[int, float], n: int = 10, exclude_rated: bool = True
    ) -> pd.DataFrame:
        """Top-N productos por rating estimado (desempate por evidencia)."""
        columns = ["item_idx", "predicted_rating", "support", "n_neighbors"]
        if not rated:
            return pd.DataFrame(columns=columns)

        pred, support, n_neighbors = self.predict_all(rated)
        candidate = n_neighbors > 0
        if exclude_rated:
            candidate[np.fromiter(rated.keys(), dtype=np.int64, count=len(rated))] = False
        idx = np.flatnonzero(candidate)
        if len(idx) == 0:
            return pd.DataFrame(columns=columns)

        order = np.lexsort((-support[idx], -pred[idx]))[:n]
        chosen = idx[order]
        return pd.DataFrame(
            {
                "item_idx": chosen.astype(np.int32),
                "predicted_rating": pred[chosen],
                "support": support[chosen],
                "n_neighbors": n_neighbors[chosen].astype(np.int32),
            }
        )
