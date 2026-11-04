"""Componentes de Streamlit compartidos por todas las páginas."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data_loader as dl
from . import session
from .recommender import ALGORITHM_LABELS, FashionRecommender, SOURCE_LABELS, resolve_strategy


# --------------------------------------------------------------------------
# Carga y configuración de página
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos precalculados…")
def get_recommender() -> FashionRecommender:
    return FashionRecommender.load()


def setup(title: str, icon: str = "👗") -> FashionRecommender:
    """Configura la página, carga artefactos, inicia la sesión y dibuja la barra lateral."""
    st.set_page_config(page_title=f"FashionRec · {title}", page_icon=icon, layout="wide")
    if not dl.artifacts_ready():
        st.error("Aún no se han generado los artefactos del modelo.")
        st.code("python scripts/prepare_data.py", language="bash")
        st.stop()
    rec = get_recommender()
    session.init_state(st.session_state)
    if not st.session_state.history:
        session.record_snapshot(st.session_state, rec, "Perfil nuevo")
    _render_sidebar(rec)
    return rec


# --------------------------------------------------------------------------
# Callbacks (se ejecutan antes de re-ejecutar el script, así las páginas ya
# dibujan con los ratings actualizados)
# --------------------------------------------------------------------------
def _on_star_change(widget_key: str, parent_asin: str) -> None:
    value = st.session_state.get(widget_key)  # índice 0-4 o None
    rec = get_recommender()
    if value is None:
        session.remove_rating(st.session_state, parent_asin, rec)
    else:
        session.rate_product(st.session_state, parent_asin, int(value) + 1, rec)


def _on_remove(parent_asin: str) -> None:
    session.remove_rating(st.session_state, parent_asin, get_recommender())


def _on_reset() -> None:
    session.reset_profile(st.session_state, get_recommender())


def _sync_setting(state_key: str, widget_key: str) -> None:
    st.session_state[state_key] = st.session_state[widget_key]


def persistent_setting(widget, label: str, state_key: str, **kwargs):
    """Widget cuyo valor sobrevive al cambiar de página.

    Streamlit descarta el estado de un widget que no se dibuja en una ejecución.
    Por eso el valor vive en ``st.session_state[state_key]`` y se vuelca al widget
    antes de crearlo.
    """
    widget_key = f"w_{state_key}"
    st.session_state[widget_key] = st.session_state[state_key]
    return widget(
        label, key=widget_key, on_change=_sync_setting, args=(state_key, widget_key), **kwargs
    )


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
def active_strategy_label() -> str:
    state = st.session_state
    strategy, _ = resolve_strategy(state.algorithm, bool(state.ratings))
    return SOURCE_LABELS[strategy]


def _render_sidebar(rec: FashionRecommender) -> None:
    ratings = st.session_state.ratings
    with st.sidebar:
        st.header("👤 Tu perfil")
        st.metric("Productos calificados", len(ratings))
        st.caption(f"Estrategia activa: **{active_strategy_label()}**")

        if ratings:
            with st.expander(f"Mis valoraciones ({len(ratings)})"):
                with st.container(height=min(60 + 52 * len(ratings), 340)):
                    for asin, rating in reversed(list(ratings.items())):
                        text, button = st.columns([5, 1], vertical_alignment="center")
                        title = str(rec.product(asin)["title"])
                        text.caption(f"{'★' * rating}{'☆' * (5 - rating)}  {title[:46]}")
                        button.button("✕", key=f"sb_rm_{asin}", on_click=_on_remove, args=(asin,),
                                      help="Quitar valoración", type="tertiary")
        else:
            st.info("Califica prendas en **Explore** para personalizar las recomendaciones.")

        st.download_button(
            "⬇ Descargar valoraciones (CSV)",
            data=session.ratings_csv(st.session_state, rec),
            file_name="mis_valoraciones.csv",
            mime="text/csv",
            disabled=not ratings,
            width="stretch",
        )
        st.button("↺ Reiniciar perfil", on_click=_on_reset, width="stretch",
                  help="Borra tus valoraciones para probar otro perfil")


# --------------------------------------------------------------------------
# Tarjetas de producto
# --------------------------------------------------------------------------
def star_rating(parent_asin: str, key_prefix: str) -> None:
    """Selector de 1 a 5 estrellas sincronizado con ``st.session_state.ratings``."""
    key = f"star_{key_prefix}_{parent_asin}"
    current = st.session_state.ratings.get(parent_asin)
    st.session_state[key] = int(current) - 1 if current else None
    st.feedback("stars", key=key, on_change=_on_star_change, args=(key, parent_asin))


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def product_card(row: pd.Series, key_prefix: str, extra: str | None = None, badge: str | None = None) -> None:
    """Imagen, nombre, categoría, rating histórico y selector de estrellas."""
    asin = row["parent_asin"]
    with st.container(border=True):
        if badge:
            st.caption(badge)
        st.image(row["image_url"], width="stretch")
        st.markdown(f"**{_truncate(str(row['title']), 70)}**", help=str(row["title"]))
        st.caption(f"{row['category']} · {row['department']}")

        avg = row.get("average_rating")
        count = row.get("rating_number")
        if pd.notna(avg):
            suffix = f" ({int(count):,} reseñas)" if pd.notna(count) else ""
            st.caption(f"Rating histórico: **{avg:.1f}**{suffix}")
        if extra:
            st.markdown(extra)

        star_rating(asin, key_prefix)
        mine = st.session_state.ratings.get(asin)
        left, right = st.columns([3, 2], vertical_alignment="center")
        left.caption(f"Tu rating: **{mine} ★**" if mine else "Sin calificar")
        if mine:
            right.button("Quitar", key=f"rm_{key_prefix}_{asin}", on_click=_on_remove, args=(asin,),
                         type="tertiary", width="stretch")


def product_grid(frame: pd.DataFrame, key_prefix: str, columns: int = 4, extra_fn=None, badge_fn=None) -> None:
    rows = [row for _, row in frame.iterrows()]
    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, row in zip(cols, rows[start:start + columns]):
            with col:
                product_card(
                    row,
                    key_prefix,
                    extra=extra_fn(row) if extra_fn else None,
                    badge=badge_fn(row) if badge_fn else None,
                )


__all__ = [
    "ALGORITHM_LABELS", "get_recommender", "setup", "persistent_setting",
    "product_card", "product_grid", "star_rating", "active_strategy_label",
]
