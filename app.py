"""FashionRec · página de inicio.

Ejecutar con:  streamlit run app.py
"""
import streamlit as st

from src import ui
from src.recommender import ALGORITHM_LABELS

rec = ui.setup("Inicio")

st.title("👗 FashionRec — Demo interactiva")
st.markdown(
    "Recomendador de moda con **filtrado colaborativo basado en ítems** sobre reseñas reales "
    "de Amazon Fashion. Califica prendas y observa cómo cambian tus recomendaciones al instante."
)

info = rec.info
c1, c2, c3 = st.columns(3)
c1.metric("Productos", f"{info['n_items']:,}")
c2.metric("Usuarios", f"{info['n_users']:,}")
c3.metric("Ratings", f"{info['n_ratings']:,}")

st.subheader("Cómo usarla")
st.markdown(
    """
1. **Explore**: busca prendas y califícalas de 1 a 5 estrellas (necesitas unas 5 para ver el efecto).
2. **Recommendations**: consulta tu Top-N personalizado y cambia el algoritmo o la cantidad.
3. **Model Insights**: compara Item-KNN con Popularity, mira cómo cambió tu Top-N y qué productos se parecen.
4. Usa **Reiniciar perfil** en la barra lateral para probar otro gusto desde cero.
"""
)

nav = st.columns(4)
nav[0].page_link("pages/1_Dashboard.py", label="Dashboard", icon="📊")
nav[1].page_link("pages/2_Explore.py", label="Explore", icon="🛍️")
nav[2].page_link("pages/3_Recommendations.py", label="Recommendations", icon="✨")
nav[3].page_link("pages/4_Model_Insights.py", label="Model Insights", icon="🔬")

st.subheader("Tu perfil ahora mismo")
n_rated = len(st.session_state.ratings)
if n_rated == 0:
    st.info(
        "Aún no has calificado nada, por lo que se usa **Popularity-Based**. "
        "Ve a **Explore** y califica algunas prendas."
    )
else:
    result = rec.recommend(
        st.session_state.ratings,
        n=4,
        algorithm=st.session_state.algorithm,
        exclude_rated=st.session_state.exclude_rated,
    )
    st.success(
        f"Has calificado {n_rated} producto(s). Estrategia: **{result.strategy_label}**. "
        "Primeras recomendaciones:"
    )
    ui.product_grid(
        result.frame,
        key_prefix="home",
        extra_fn=lambda r: f"**Rating estimado: {r['predicted_rating']:.1f} / 5**",
    )

with st.expander("Sobre los algoritmos"):
    st.markdown(
        f"""
* **{ALGORITHM_LABELS['popularity']}**: ordena por una media ponderada que combina la calificación
  del producto con su número de reseñas.
* **{ALGORITHM_LABELS['item_knn']}**: recomienda productos que suelen recibir valoraciones
  similares de los mismos usuarios (similitud coseno entre productos, 50 vecinos por producto).
* En modo *Automático* se usa Popularity hasta que calificas tu primer producto y luego Item-KNN.
"""
    )
