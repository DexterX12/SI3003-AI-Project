"""Recomendador basado en popularidad (media bayesiana ponderada)."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_QUANTILE = 0.90   # m = percentil 90 del número de reseñas por producto
BIAS_REGULARIZATION = 10  # lambda del sesgo de producto (se usa como baseline en Item-KNN)


def compute_popularity(
    interactions: pd.DataFrame,
    item_ids: list[str] | np.ndarray,
    quantile: float = DEFAULT_QUANTILE,
    bias_reg: float = BIAS_REGULARIZATION,
) -> tuple[pd.DataFrame, dict]:
    """Calcula el ranking de popularidad y el sesgo de cada producto.

    Score(i) = v_i / (v_i + m) * R_i + m / (v_i + m) * C

    * v_i: número de ratings del producto i
    * R_i: rating medio del producto i
    * C:   rating medio global
    * m:   percentil ``quantile`` de v_i (mínimo de reseñas para pesar como "fiable")

    Además guarda ``item_bias`` = sum(r - C) / (v_i + bias_reg), usado como baseline
    por el Item-KNN para estimar ratings.

    ``item_ids[k]`` debe ser el ``parent_asin`` con ``item_idx == k``.
    Devuelve ``(DataFrame, parámetros)``.
    """
    n_items = len(item_ids)
    idx = interactions["item_idx"].to_numpy()
    ratings = interactions["rating"].to_numpy(dtype=np.float64)

    counts = np.bincount(idx, minlength=n_items).astype(np.float64)
    sums = np.bincount(idx, weights=ratings, minlength=n_items)
    global_mean = float(ratings.mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_rating = np.where(counts > 0, sums / counts, global_mean)

    m = float(np.quantile(counts[counts > 0], quantile))
    score = counts / (counts + m) * mean_rating + m / (counts + m) * global_mean
    item_bias = (sums - counts * global_mean) / (counts + bias_reg)

    table = pd.DataFrame(
        {
            "item_idx": np.arange(n_items, dtype=np.int32),
            "parent_asin": np.asarray(item_ids),
            "n_ratings": counts.astype(np.int32),
            "mean_rating": mean_rating.astype(np.float32),
            "score": score.astype(np.float32),
            "item_bias": item_bias.astype(np.float32),
        }
    )
    table = table.sort_values(["score", "n_ratings"], ascending=False, kind="stable").reset_index(drop=True)
    table["rank"] = np.arange(1, n_items + 1, dtype=np.int32)
    params = {"global_mean": global_mean, "m": m, "quantile": quantile, "bias_reg": bias_reg}
    return table, params


def recommend_popular(
    popularity: pd.DataFrame,
    n: int = 10,
    exclude_idx: set[int] | None = None,
) -> pd.DataFrame:
    """Top-N por popularidad. ``popularity`` ya viene ordenada por ``rank``.

    La columna ``predicted_rating`` es el score ponderado, que está en la escala 1-5.
    """
    table = popularity
    if exclude_idx:
        table = table[~table["item_idx"].isin(exclude_idx)]
    top = table.head(n)
    return pd.DataFrame(
        {
            "item_idx": top["item_idx"].to_numpy(),
            "parent_asin": top["parent_asin"].to_numpy(),
            "predicted_rating": top["score"].to_numpy(dtype=np.float64),
            "support": np.nan,
            "n_neighbors": 0,
            "source": "popularity",
        }
    )
