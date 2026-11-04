"""Model Insights: comportamiento y comparación de los modelos."""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data_loader as dl
from src import ui
from src.recommender import ALGORITHM_LABELS, SOURCE_LABELS

rec = ui.setup("Model Insights", "🔬")
state = st.session_state
ratings = state.ratings

st.title("🔬 Model Insights")

# --- Resumen ---------------------------------------------------------------
strategy_label = ui.active_strategy_label()
n_candidates = 0
if ratings:
    _, _, n_nb = rec.knn.predict_all(rec.to_item_idx(ratings))
    n_candidates = int((n_nb > 0).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Interacciones del usuario", len(ratings))
m2.metric("Algoritmo activo", strategy_label)
m3.metric("Algoritmo solicitado", ALGORITHM_LABELS[state.algorithm].split(" (")[0])
m4.metric("Productos con evidencia CF", f"{n_candidates:,}" if ratings else "—",
          help="Productos que son vecinos (top-50) de al menos una prenda que calificaste.")

tab_topn, tab_sim, tab_cmp, tab_eval = st.tabs(
    ["Evolución del Top-N", "Similitud entre productos", "Item-KNN vs Popularity", "Evaluación offline"]
)


def short(asin: str, width: int = 40) -> str:
    title = str(rec.product(asin)["title"])
    return title if len(title) <= width else title[: width - 1] + "…"


# --- 1. Cómo cambia el Top-N -----------------------------------------------
with tab_topn:
    history = state.history
    st.caption("Cada vez que calificas, quitas o reinicias, se guarda el Top-N resultante para ver cómo se mueve.")
    if len(history) < 2:
        st.info("Califica un producto en **Explore** para ver cómo cambia el Top-N.")
    else:
        steps = pd.DataFrame(
            {
                "Paso": [h["step"] for h in history],
                "Evento": [h["event"] for h in history],
                "Ratings": [h["n_ratings"] for h in history],
                "Estrategia": [SOURCE_LABELS[h["strategy"]] for h in history],
            }
        )
        st.dataframe(steps.iloc[::-1], hide_index=True, height=min(250, 38 + 35 * len(steps)))

        labels = [f"Paso {h['step']}: {h['event']}" for h in history]
        to_pos = len(history) - 1
        chosen = st.selectbox("Comparar con el paso anterior a…", range(1, len(history)),
                              index=to_pos - 1, format_func=lambda i: labels[i])
        before, after = history[chosen - 1], history[chosen]
        pos_before = {a: i + 1 for i, a in enumerate(before["asins"])}
        pos_after = {a: i + 1 for i, a in enumerate(after["asins"])}
        score_after = dict(zip(after["asins"], after["scores"]))

        def movement(asin: str) -> str:
            if asin not in pos_before:
                return "🆕 Nuevo"
            if asin not in pos_after:
                return "⬇ Sale del Top-N"
            delta = pos_before[asin] - pos_after[asin]
            return "= igual" if delta == 0 else (f"▲ +{delta}" if delta > 0 else f"▼ {delta}")

        union = list(dict.fromkeys(after["asins"] + before["asins"]))
        moves = pd.DataFrame(
            {
                "Producto": [short(a, 60) for a in union],
                "Antes": [pos_before.get(a) for a in union],
                "Después": [pos_after.get(a) for a in union],
                "Cambio": [movement(a) for a in union],
                "Rating estimado": [score_after.get(a) for a in union],
            }
        )
        kept = len(set(before["asins"]) & set(after["asins"]))
        st.markdown(
            f"**{before['event']} → {after['event']}** · "
            f"{kept} de {len(after['asins'])} productos se mantienen, "
            f"{len(set(after['asins']) - set(before['asins']))} son nuevos."
        )
        st.dataframe(moves, hide_index=True, column_config={
            "Rating estimado": st.column_config.NumberColumn(format="%.2f"),
        })

        # Gráfico de posiciones por paso para los productos del Top-N actual
        current = history[-1]["asins"][:10]
        fig = go.Figure()
        for asin in current:
            xs, ys = [], []
            for h in history:
                if asin in h["asins"]:
                    xs.append(h["step"])
                    ys.append(h["asins"].index(asin) + 1)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=short(asin, 28),
                                     hovertext=short(asin, 80), hoverinfo="text+x+y"))
        fig.update_yaxes(autorange="reversed", title="Posición en el Top-N", dtick=1)
        fig.update_xaxes(title="Paso", dtick=1)
        fig.update_layout(title="Posición de los productos del Top-10 actual a lo largo de la sesión",
                          height=420, legend={"font": {"size": 10}})
        st.plotly_chart(fig)

# --- 2. Similitud entre productos ------------------------------------------
with tab_sim:
    st.caption(
        "Similitud coseno entre productos, calculada una sola vez con los ratings de todos los usuarios "
        "(solo se guardan los 50 vecinos más cercanos de cada producto)."
    )
    rated_asins = list(ratings)
    popular_asins = rec.catalog.sort_values("rank")["parent_asin"].head(200).tolist()
    options = rated_asins + [a for a in popular_asins if a not in ratings]
    asin = st.selectbox("Producto", options, format_func=lambda a: ("★ " if a in ratings else "") + short(a, 80),
                        help="Primero aparecen los que calificaste; escribe para buscar entre los 200 más populares.")
    neighbors = rec.similar_items(asin, k=10)
    left, right = st.columns([1, 3])
    with left:
        st.image(rec.product(asin)["image_url"], width="stretch")
    with right:
        if neighbors.empty:
            st.info("Este producto no tiene vecinos: ningún otro producto comparte usuarios con él.")
        else:
            st.dataframe(
                neighbors[["image_url", "title", "category", "similarity"]],
                hide_index=True, row_height=64,
                column_config={
                    "image_url": st.column_config.ImageColumn("Imagen", width="small"),
                    "title": st.column_config.TextColumn("Producto similar", width="large"),
                    "category": "Categoría",
                    "similarity": st.column_config.ProgressColumn(
                        "Similitud", min_value=0.0, max_value=float(max(neighbors["similarity"].max(), 0.01)),
                        format="%.3f"),
                },
            )

    if ratings:
        st.subheader("Mapa de similitud: tus productos y tus recomendaciones")
        result = rec.recommend(ratings, n=min(state.top_n, 10), algorithm=state.algorithm,
                               exclude_rated=True)
        mine = rated_asins[-8:]
        recs = [a for a in result.asins if a not in mine][:8]
        axis = mine + recs
        matrix = rec.similarity_submatrix(axis)
        names = [("★ " if a in ratings else "") + short(a, 26) for a in axis]
        fig = px.imshow(matrix, x=names, y=names, color_continuous_scale="Blues", zmin=0, zmax=max(0.3, float(np.nanmax(matrix - np.eye(len(axis))))),
                        aspect="auto", labels={"color": "Similitud"})
        fig.update_layout(height=560, margin={"t": 10})
        st.plotly_chart(fig)
        st.caption("★ = productos que calificaste. Un valor 0 indica que no están entre los 50 vecinos mutuos.")

# --- 3. Comparación con Popularity ------------------------------------------
with tab_cmp:
    n = state.top_n
    comparison = rec.compare(ratings, n=n, exclude_rated=state.exclude_rated)
    knn, pop = comparison["item_knn"], comparison["popularity"]
    if not ratings:
        st.info("Sin valoraciones, ambos algoritmos devuelven la misma lista (Item-KNN cae en Popularity).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Productos en común", f"{comparison['overlap']} de {n}")
    c2.metric("Jaccard", f"{comparison['jaccard']:.2f}",
              help="Intersección / unión de ambas listas. 1 = idénticas, 0 = totalmente distintas.")
    pop_rank = rec.catalog.set_index("parent_asin")["rank"]
    if ratings:
        avg_rank = float(pop_rank.reindex(knn.frame["parent_asin"]).mean())
        c3.metric("Ranking medio de Item-KNN en Popularity", f"#{avg_rank:,.0f}",
                  help="Cuanto más alto el número, más se aleja Item-KNN de lo que ya es popular.")

    col_knn, col_pop = st.columns(2)
    for column, title, result in ((col_knn, "Item-Based CF", knn), (col_pop, "Popularity-Based", pop)):
        with column:
            st.subheader(title)
            table = result.frame.assign(rank_pop=result.frame["parent_asin"].map(pop_rank))
            st.dataframe(
                table[["position", "image_url", "title", "predicted_rating", "rank_pop"]],
                hide_index=True, row_height=64,
                column_config={
                    "position": st.column_config.NumberColumn("#", width="small"),
                    "image_url": st.column_config.ImageColumn("Imagen", width="small"),
                    "title": st.column_config.TextColumn("Producto", width="large"),
                    "predicted_rating": st.column_config.NumberColumn("Rating est.", format="%.2f"),
                    "rank_pop": st.column_config.NumberColumn("Rank pop.", format="%d"),
                },
            )

# --- 4. Evaluación offline --------------------------------------------------
with tab_eval:
    results_path = dl.PROJECT_ROOT / "docs" / "experiment_results.json"
    st.markdown(
        "Experimento *leave-last-out*: se oculta la última interacción de cada usuario, los modelos se "
        "recalculan sin ella y se mide si el producto oculto aparece en el Top-N. "
        "Se reproduce con `python scripts/evaluate.py --users 0 --save`."
    )
    if not results_path.exists():
        st.info("Aún no hay resultados guardados. Ejecuta `python scripts/evaluate.py --users 0 --save`.")
    else:
        res = json.loads(results_path.read_text(encoding="utf-8"))[-1]
        n_eval = res["n"]
        names = {"popularity": "Popularity-Based", "item_knn": "Item-KNN", "item_knn+pop": "Item-KNN + relleno Popularity"}
        table = pd.DataFrame(
            [
                {"Algoritmo": names[k], f"HR@{n_eval}": v[f"HR@{n_eval}"],
                 f"NDCG@{n_eval}": v[f"NDCG@{n_eval}"], "Cobertura del catálogo": v["coverage"]}
                for k, v in res["ranking"].items()
            ]
        )
        st.caption(f"{res['users_evaluated']:,} usuarios evaluados · un recomendador aleatorio lograría HR@{n_eval} ≈ {res['random_hr']:.4f}")
        st.dataframe(table, hide_index=True, column_config={
            f"HR@{n_eval}": st.column_config.NumberColumn(format="%.4f"),
            f"NDCG@{n_eval}": st.column_config.NumberColumn(format="%.4f"),
            "Cobertura del catálogo": st.column_config.NumberColumn(format="percent"),
        })
        fig = px.bar(table, x="Algoritmo", y=f"HR@{n_eval}", text=table[f"HR@{n_eval}"].map("{:.3f}".format))
        fig.update_layout(height=320, margin={"t": 10})
        st.plotly_chart(fig)
        st.markdown(
            f"**RMSE del rating oculto** (menor es mejor): Item-KNN **{res['rmse']['knn']:.3f}**, "
            f"baseline (media + sesgos) {res['rmse']['baseline']:.3f}, media global {res['rmse']['global']:.3f}. "
            f"Donde Item-KNN tiene vecinos ({res['covered_share']:.1%} de los casos): "
            f"Item-KNN {res['rmse_covered']['knn']:.3f} vs baseline {res['rmse_covered']['baseline']:.3f}."
        )
