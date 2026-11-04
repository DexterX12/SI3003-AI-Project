"""Recommendations: recomendaciones personalizadas que se recalculan con cada rating."""
import streamlit as st

from src import ui
from src.recommender import ALGO_ITEM_KNN, ALGORITHM_LABELS

rec = ui.setup("Recommendations", "✨")
state = st.session_state

st.title("✨ Recommendations")

with st.container(border=True):
    c1, c2, c3 = st.columns([3, 3, 2], vertical_alignment="bottom")
    with c1:
        ui.persistent_setting(
            st.selectbox, "Algoritmo", "algorithm",
            options=list(ALGORITHM_LABELS), format_func=ALGORITHM_LABELS.get,
        )
    with c2:
        ui.persistent_setting(st.slider, "Cantidad de recomendaciones", "top_n", min_value=5, max_value=40, step=5)
    with c3:
        ui.persistent_setting(st.checkbox, "Excluir ya calificados", "exclude_rated")

result = rec.recommend(
    state.ratings, n=state.top_n, algorithm=state.algorithm, exclude_rated=state.exclude_rated
)

n_rated = len(state.ratings)
m1, m2, m3 = st.columns(3)
m1.metric("Productos que calificaste", n_rated)
m2.metric("Estrategia en uso", result.strategy_label)
m3.metric("Recomendaciones mostradas", len(result.frame))

if result.note:
    st.warning(result.note)
if n_rated == 0:
    st.info("Estás viendo los productos más populares. Califica prendas en **Explore** para personalizar el resultado.")
    st.page_link("pages/2_Explore.py", label="Ir a Explore", icon="🛍️")
elif result.strategy == ALGO_ITEM_KNN:
    st.caption(
        "Rating estimado = rating típico del producto + ajuste según tus valoraciones de productos "
        "similares. Los productos marcados como Popularity completan la lista cuando no hay "
        "suficientes vecinos con evidencia."
    )


def _extra(row) -> str:
    if row["source"] == ALGO_ITEM_KNN:
        return (
            f"**Rating estimado: {row['predicted_rating']:.1f} / 5**  \n"
            f"Similar a {int(row['n_neighbors'])} producto(s) que calificaste"
        )
    return f"**Score de popularidad: {row['predicted_rating']:.1f} / 5**"


ui.product_grid(
    result.frame,
    key_prefix="rec",
    extra_fn=_extra,
    badge_fn=lambda r: f"#{int(r['position'])} · " + ("Item-KNN" if r["source"] == ALGO_ITEM_KNN else "Popularity"),
)
