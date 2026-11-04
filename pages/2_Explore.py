"""Explore: catálogo de prendas y calificación de 1 a 5 estrellas."""
import math

import streamlit as st

from src import ui

rec = ui.setup("Explore", "🛍️")

PAGE_SIZE = 12
SORTS = {
    "Más populares": ("rank", True),
    "Mejor rating histórico": ("average_rating", False),
    "Más reseñas": ("rating_number", False),
    "Nombre (A-Z)": ("title", True),
}

st.title("🛍️ Explore")
st.caption("Califica las prendas que conoces o te gustan. Tus valoraciones se guardan en esta sesión.")

catalog = rec.catalog
ratings = st.session_state.ratings

with st.container(border=True):
    f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
    query = f1.text_input("Buscar", placeholder="Ej. sneakers, dress, necklace…")
    category = f2.selectbox("Categoría", ["Todas"] + sorted(catalog["category"].unique()))
    department = f3.selectbox("Departamento", ["Todos"] + sorted(catalog["department"].unique()))
    sort_name = f4.selectbox("Ordenar por", list(SORTS))
    only_rated = st.checkbox(f"Mostrar solo mis calificados ({len(ratings)})", disabled=not ratings)

view = catalog
if query.strip():
    view = view[view["title"].str.contains(query.strip(), case=False, regex=False, na=False)]
if category != "Todas":
    view = view[view["category"] == category]
if department != "Todos":
    view = view[view["department"] == department]
if only_rated:
    view = view[view["parent_asin"].isin(ratings)]
column, ascending = SORTS[sort_name]
view = view.sort_values(column, ascending=ascending, kind="stable", na_position="last")

n_pages = max(1, math.ceil(len(view) / PAGE_SIZE))
filters_signature = (query, category, department, sort_name, only_rated)
if st.session_state.get("explore_filters") != filters_signature:
    st.session_state.explore_filters = filters_signature
    st.session_state.explore_page = 1
st.session_state.explore_page = min(st.session_state.get("explore_page", 1), n_pages)


def _go(delta: int) -> None:
    st.session_state.explore_page = min(max(1, st.session_state.explore_page + delta), n_pages)


def pager(position: str) -> None:
    prev_col, text_col, next_col = st.columns([1, 3, 1], vertical_alignment="center")
    prev_col.button("← Anterior", key=f"prev_{position}", on_click=_go, args=(-1,),
                    disabled=st.session_state.explore_page <= 1, width="stretch")
    text_col.markdown(
        f"<div style='text-align:center'>{len(view):,} productos · página "
        f"<b>{st.session_state.explore_page}</b> de {n_pages}</div>",
        unsafe_allow_html=True,
    )
    next_col.button("Siguiente →", key=f"next_{position}", on_click=_go, args=(1,),
                    disabled=st.session_state.explore_page >= n_pages, width="stretch")


if view.empty:
    st.warning("Ningún producto coincide con los filtros.")
    st.stop()

pager("top")
start = (st.session_state.explore_page - 1) * PAGE_SIZE
ui.product_grid(view.iloc[start:start + PAGE_SIZE], key_prefix="explore")
pager("bottom")
