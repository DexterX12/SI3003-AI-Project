"""Dashboard: estadísticas generales del dataset."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import data_loader as dl
from src import ui
from src.preprocessing import sparsity

rec = ui.setup("Dashboard", "📊")


@st.cache_data(show_spinner="Calculando estadísticas…")
def dataset_stats() -> dict:
    inter = pd.read_parquet(dl.INTERACTIONS_PATH, columns=["user_idx", "item_idx", "rating"])
    per_user = inter.groupby("user_idx").size()
    return {
        "rating_dist": inter["rating"].round().astype(int).value_counts().sort_index(),
        "per_user": per_user.clip(upper=15).value_counts().sort_index(),
        "n_ratings": len(inter),
        "mean_rating": float(inter["rating"].mean()),
    }


stats = dataset_stats()
info = rec.info

st.title("📊 Dashboard")
st.caption(
    f"Subconjunto de Amazon Fashion: usuarios con ≥ {info['min_user_interactions']} y productos con "
    f"≥ {info['min_item_interactions']} calificaciones (filtro iterativo), de {info['raw_reviews']:,} reseñas originales."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Productos", f"{info['n_items']:,}")
c2.metric("Usuarios", f"{info['n_users']:,}")
c3.metric("Ratings", f"{stats['n_ratings']:,}")
c4.metric("Rating medio", f"{stats['mean_rating']:.2f}")
st.caption(
    f"Dispersión de la matriz usuario-producto: **{sparsity(info['n_ratings'], info['n_users'], info['n_items']):.3%}**"
)

left, right = st.columns(2)
with left:
    st.subheader("Distribución de calificaciones")
    dist = stats["rating_dist"].reindex(range(1, 6), fill_value=0)
    fig = px.bar(
        x=dist.index, y=dist.values, labels={"x": "Estrellas", "y": "Ratings"},
        text=[f"{v / dist.sum():.0%}" for v in dist.values],
    )
    fig.update_layout(xaxis={"tickmode": "linear"}, margin={"t": 10})
    st.plotly_chart(fig)
with right:
    st.subheader("Ratings por usuario")
    per_user = stats["per_user"]
    fig = px.bar(
        x=[f"{i}+" if i == 15 else str(i) for i in per_user.index], y=per_user.values,
        labels={"x": "Productos calificados por usuario", "y": "Usuarios"},
    )
    fig.update_layout(margin={"t": 10})
    st.plotly_chart(fig)

st.subheader("Productos por categoría")
cats = rec.catalog["category"].value_counts().sort_values()
fig = px.bar(x=cats.values, y=cats.index, orientation="h", labels={"x": "Productos", "y": ""})
fig.update_layout(margin={"t": 10}, height=380)
st.plotly_chart(fig)

st.subheader("Productos más populares")
st.caption(
    "Ordenados por el score de Popularity-Based: media ponderada entre el rating del producto y el rating "
    f"global, con peso según su número de reseñas (m = {info['m']:.0f})."
)
top = rec.catalog.sort_values("rank").head(10)
st.dataframe(
    top[["image_url", "title", "category", "n_ratings", "mean_rating", "score"]],
    hide_index=True,
    column_config={
        "image_url": st.column_config.ImageColumn("Imagen", width="small"),
        "title": st.column_config.TextColumn("Producto", width="large"),
        "category": "Categoría",
        "n_ratings": st.column_config.NumberColumn("Reseñas", format="%d"),
        "mean_rating": st.column_config.NumberColumn("Rating medio", format="%.2f"),
        "score": st.column_config.NumberColumn("Score", format="%.3f"),
    },
    row_height=64,
)
