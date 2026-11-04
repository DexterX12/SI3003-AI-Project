"""Estado de la sesión del usuario (ratings, configuración e historial de Top-N).

Las funciones reciben ``state`` (``st.session_state`` o un ``dict``), por lo que
se pueden probar sin Streamlit. No hay cuentas ni persistencia: al cerrar la
pestaña el perfil se pierde, salvo que se descargue el CSV.
"""
from __future__ import annotations

import io
from typing import MutableMapping

import pandas as pd

from .recommender import ALGO_AUTO, FashionRecommender

MAX_HISTORY = 30
MIN_RATING, MAX_RATING = 1, 5


def init_state(state: MutableMapping) -> None:
    defaults = {
        "ratings": {},            # {parent_asin: rating 1-5}, en orden de calificación
        "algorithm": ALGO_AUTO,   # algoritmo solicitado
        "top_n": 10,
        "exclude_rated": True,
        "history": [],            # snapshots del Top-N tras cada cambio
        "step": 0,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def rate_product(state: MutableMapping, parent_asin: str, rating: int, rec: FashionRecommender) -> None:
    """Guarda (o cambia) la valoración de un producto y registra el nuevo Top-N."""
    rating = int(rating)
    if not MIN_RATING <= rating <= MAX_RATING:
        raise ValueError(f"El rating debe estar entre {MIN_RATING} y {MAX_RATING}")
    if parent_asin not in rec.item_map:
        raise KeyError(f"Producto desconocido: {parent_asin}")
    previous = state["ratings"].get(parent_asin)
    if previous == rating:
        return
    state["ratings"][parent_asin] = rating
    verb = "Calificó" if previous is None else f"Cambió {previous}★ →"
    record_snapshot(state, rec, f"{verb} {rating}★ · {_short_title(rec, parent_asin)}")


def remove_rating(state: MutableMapping, parent_asin: str, rec: FashionRecommender) -> None:
    if parent_asin in state["ratings"]:
        del state["ratings"][parent_asin]
        record_snapshot(state, rec, f"Quitó valoración · {_short_title(rec, parent_asin)}")


def reset_profile(state: MutableMapping, rec: FashionRecommender) -> None:
    """Vuelve a un usuario nuevo (sin ratings ni historial). Conserva los ajustes."""
    state["ratings"] = {}
    state["history"] = []
    state["step"] = 0
    record_snapshot(state, rec, "Perfil nuevo")


def record_snapshot(state: MutableMapping, rec: FashionRecommender, event: str = "Inicio") -> None:
    """Guarda el Top-N actual para poder ver cómo evoluciona tras cada valoración."""
    result = rec.recommend(
        state["ratings"], n=state["top_n"], algorithm=state["algorithm"],
        exclude_rated=state["exclude_rated"],
    )
    state["history"].append(
        {
            "step": state["step"],
            "event": event,
            "n_ratings": len(state["ratings"]),
            "strategy": result.strategy,
            "asins": result.asins,
            "scores": [round(float(s), 3) for s in result.frame["predicted_rating"]],
        }
    )
    state["step"] += 1
    del state["history"][:-MAX_HISTORY]


def ratings_csv(state: MutableMapping, rec: FashionRecommender) -> str:
    rows = [
        {
            "parent_asin": asin,
            "title": rec.product(asin)["title"],
            "category": rec.product(asin)["category"],
            "rating": rating,
        }
        for asin, rating in state["ratings"].items()
    ]
    frame = pd.DataFrame(rows, columns=["parent_asin", "title", "category", "rating"])
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def _short_title(rec: FashionRecommender, parent_asin: str, width: int = 45) -> str:
    title = str(rec.product(parent_asin)["title"])
    return title if len(title) <= width else title[: width - 1] + "…"
