"""Fachada del motor de recomendación: elige el algoritmo y combina resultados."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data_loader as dl
from .item_knn import ItemKNN
from .popularity import recommend_popular

ALGO_AUTO = "auto"
ALGO_POPULARITY = "popularity"
ALGO_ITEM_KNN = "item_knn"
ALGORITHM_LABELS = {
    ALGO_AUTO: "Automático (Popularity → Item-KNN)",
    ALGO_POPULARITY: "Popularity-Based",
    ALGO_ITEM_KNN: "Item-Based Collaborative Filtering",
}
SOURCE_LABELS = {
    ALGO_POPULARITY: "Popularity-Based",
    ALGO_ITEM_KNN: "Item-Based Collaborative Filtering",
}

PRODUCT_COLUMNS = [
    "parent_asin", "title", "image_url", "category", "department",
    "store", "average_rating", "rating_number", "price",
]


def resolve_strategy(requested: str, has_ratings: bool) -> tuple[str, str]:
    """Algoritmo que realmente se usa y un aviso si difiere del solicitado.

    * Sin ratings siempre se usa Popularity-Based (Item-KNN necesita al menos uno).
    * Con ratings, ``auto`` usa Item-KNN.
    """
    if requested not in ALGORITHM_LABELS:
        raise ValueError(f"Algoritmo desconocido: {requested}")
    if requested == ALGO_POPULARITY:
        return ALGO_POPULARITY, ""
    if not has_ratings:
        note = ""
        if requested == ALGO_ITEM_KNN:
            note = "Aún no calificaste productos: se muestra Popularity-Based hasta que califiques al menos uno."
        return ALGO_POPULARITY, note
    return ALGO_ITEM_KNN, ""


@dataclass
class RecommendationResult:
    """Resultado de una consulta de recomendaciones."""

    frame: pd.DataFrame      # una fila por producto, con metadata y score
    strategy: str            # algoritmo realmente usado: "popularity" o "item_knn"
    requested: str           # algoritmo solicitado: "auto", "popularity" o "item_knn"
    note: str = ""           # aviso para el usuario (p. ej. fallback a popularidad)

    @property
    def strategy_label(self) -> str:
        return SOURCE_LABELS[self.strategy]

    @property
    def asins(self) -> list[str]:
        return self.frame["parent_asin"].tolist()


class FashionRecommender:
    def __init__(
        self,
        products: pd.DataFrame,
        popularity: pd.DataFrame,
        similarity: sp.csr_matrix,
        item_map: dict[str, int],
        user_map: dict[str, int],
        info: dict,
    ) -> None:
        self.products = products.sort_values("item_idx").reset_index(drop=True)
        self.popularity = popularity
        self.item_map = item_map
        self.user_map = user_map
        self.info = info
        self.item_ids = np.array(sorted(item_map, key=item_map.get))

        bias = np.zeros(len(item_map), dtype=np.float64)
        bias[popularity["item_idx"].to_numpy()] = popularity["item_bias"].to_numpy()
        self.knn = ItemKNN(similarity, bias, global_mean=info["global_mean"])
        self._product_lookup = self.products.set_index("parent_asin")
        # Catálogo = metadata + estadísticas de popularidad (para la página Explore)
        self.catalog = self.products.merge(
            popularity[["parent_asin", "n_ratings", "mean_rating", "score", "rank"]],
            on="parent_asin",
            how="left",
        )

    # -- carga -----------------------------------------------------------------
    @classmethod
    def load(cls) -> "FashionRecommender":
        if not dl.artifacts_ready():
            raise FileNotFoundError(
                "Faltan artefactos. Ejecuta primero: python scripts/prepare_data.py"
            )
        return cls(
            products=pd.read_parquet(dl.PRODUCTS_PATH),
            popularity=pd.read_parquet(dl.POPULARITY_PATH),
            similarity=dl.load_similarity(),
            item_map=dl.load_pickle(dl.ITEM_MAPPING_PATH),
            user_map=dl.load_pickle(dl.USER_MAPPING_PATH),
            info=json.loads(dl.MODEL_INFO_PATH.read_text(encoding="utf-8")),
        )

    # -- utilidades ------------------------------------------------------------
    @property
    def n_items(self) -> int:
        return len(self.item_map)

    def to_item_idx(self, ratings: Mapping[str, float]) -> dict[int, float]:
        """Convierte {parent_asin: rating} a {item_idx: rating}, ignorando ids desconocidos."""
        return {self.item_map[a]: float(r) for a, r in ratings.items() if a in self.item_map}

    def product(self, parent_asin: str) -> pd.Series:
        return self._product_lookup.loc[parent_asin]

    def _decorate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Añade metadata del producto a un resultado con ``item_idx``."""
        if frame.empty:
            return frame.assign(**{c: pd.Series(dtype=object) for c in PRODUCT_COLUMNS})
        info = self.products.iloc[frame["item_idx"].to_numpy()][PRODUCT_COLUMNS].reset_index(drop=True)
        out = pd.concat([frame.reset_index(drop=True).drop(columns=["parent_asin"], errors="ignore"), info], axis=1)
        out.insert(0, "position", np.arange(1, len(out) + 1))
        return out

    # -- recomendación ---------------------------------------------------------
    def recommend(
        self,
        ratings: Mapping[str, float],
        n: int = 10,
        algorithm: str = ALGO_AUTO,
        exclude_rated: bool = True,
    ) -> RecommendationResult:
        """Top-N para un usuario descrito solo por sus ratings de sesión."""
        rated = self.to_item_idx(ratings)
        exclude = set(rated) if exclude_rated else set()
        strategy, note = resolve_strategy(algorithm, bool(rated))

        if strategy == ALGO_POPULARITY:
            frame = recommend_popular(self.popularity, n=n, exclude_idx=exclude)
            return RecommendationResult(self._decorate(frame), ALGO_POPULARITY, algorithm, note)

        frame = self.knn.recommend(rated, n=n, exclude_rated=exclude_rated)
        frame["parent_asin"] = self.item_ids[frame["item_idx"].to_numpy()] if len(frame) else []
        frame["source"] = ALGO_ITEM_KNN
        if len(frame) < n:
            filler = recommend_popular(self.popularity, n=n, exclude_idx=exclude | set(frame["item_idx"]))
            filler = filler.head(n - len(frame))
            note = (
                f"Item-KNN solo encontró {len(frame)} productos con evidencia suficiente; "
                "el resto se completó con Popularity-Based."
            )
            frame = pd.concat([frame, filler], ignore_index=True)
        return RecommendationResult(self._decorate(frame), ALGO_ITEM_KNN, algorithm, note)

    def compare(
        self, ratings: Mapping[str, float], n: int = 10, exclude_rated: bool = True
    ) -> dict:
        """Item-KNN frente a Popularity para el mismo usuario y el mismo N."""
        knn = self.recommend(ratings, n, ALGO_ITEM_KNN, exclude_rated)
        pop = self.recommend(ratings, n, ALGO_POPULARITY, exclude_rated)
        a, b = set(knn.asins), set(pop.asins)
        union = a | b
        return {
            "item_knn": knn,
            "popularity": pop,
            "overlap": len(a & b),
            "jaccard": len(a & b) / len(union) if union else 0.0,
        }

    # -- similitud -------------------------------------------------------------
    def similar_items(self, parent_asin: str, k: int = 10) -> pd.DataFrame:
        """Vecinos más similares a un producto (según la matriz precalculada)."""
        idx = self.item_map[parent_asin]
        neighbors, sims = self.knn.neighbors(idx, k)
        frame = pd.DataFrame({"item_idx": neighbors.astype(np.int32), "similarity": sims})
        frame["parent_asin"] = self.item_ids[neighbors] if len(frame) else []
        return self._decorate(frame)

    def similarity_submatrix(self, parent_asins: list[str]) -> np.ndarray:
        """Matriz densa de similitud entre los productos indicados (simétrica)."""
        idx = [self.item_map[a] for a in parent_asins]
        sub = self.knn.similarity[idx][:, idx].toarray()
        sym = np.maximum(sub, sub.T)
        np.fill_diagonal(sym, 1.0)
        return sym
