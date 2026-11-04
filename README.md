# FashionRec

Recomendador de moda con **filtrado colaborativo basado en ítems** sobre reseñas reales de
[Amazon Reviews 2023 – Amazon Fashion](https://amazon-reviews-2023.github.io/).

Algoritmos:

| Algoritmo | Propósito |
| --- | --- |
| **Popularity-Based** | Media ponderada entre el rating del producto y el global, según su número de reseñas. Se usa mientras el usuario no ha calificado nada. |
| **Item-Based CF (Item-KNN)** | Recomienda productos que suelen recibir valoraciones similares de los mismos usuarios (similitud coseno, 50 vecinos por producto). |

Más detalle en [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md), el funcionamiento del script de preparación en
[`docs/PREPARE_DATA.md`](docs/PREPARE_DATA.md) y los resultados de la comparación en
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Requisitos

* Python 3.10 o superior (probado con 3.11)
* ~3 GB libres solo si necesitas descargar los datos crudos; la app en sí usa ~3 MB de artefactos

## Instalación y ejecución

```powershell
pip install -r requirements.txt

python scripts/prepare_data.py     # una sola vez (~30 s con los datos crudos ya descargados)
streamlit run app.py
```

El proyecto ya incluye los artefactos generados (`artifacts/` y `data/processed/`), por lo que puedes
ejecutar `streamlit run app.py` directamente. Solo necesitas `prepare_data.py` si quieres regenerarlos
o cambiar los umbrales del filtro.

### Datos crudos

`prepare_data.py` espera estos archivos en `data/raw/` (o en la raíz del proyecto):

* `Amazon_Fashion.jsonl` (reseñas, ~1 GB)
* `meta_Amazon_Fashion.jsonl` (metadata, ~1.4 GB)

Para descargarlos desde Hugging Face:

```powershell
python scripts/download_data.py
```

### Opciones de `prepare_data.py`

```powershell
python scripts/prepare_data.py --min-user 2 --min-item 5 --top-k 50
```

| Opción | Por defecto | Significado |
| --- | --- | --- |
| `--min-user` | 2 | Mínimo de ratings por usuario |
| `--min-item` | 5 | Mínimo de ratings por producto |
| `--top-k` | 50 | Vecinos conservados por producto |

> **Por qué no 5-core.** En Amazon Fashion el 86 % de los usuarios tiene una sola reseña. Un 5-core estricto
> (usuarios ≥ 5 **y** productos ≥ 5, aplicado iterativamente) deja el dataset **vacío**. Se usa
> usuarios ≥ 2 y productos ≥ 5, también iterativo: un usuario con una sola reseña no aporta nada a la
> similitud entre productos. Resultado: 21 034 usuarios, 4 823 productos y 50 457 ratings.

## Cómo usar la app

1. **Explore**: busca, filtra y califica prendas con las estrellas. Con unas 5 valoraciones ya se nota el efecto.
2. **Recommendations**: elige algoritmo, cantidad y si se excluyen los ya calificados. Verás el rating estimado.
3. **Model Insights**: cómo cambió tu Top-N tras cada valoración, productos similares, comparación con Popularity y la evaluación offline.
4. **Dashboard**: estadísticas del dataset.
5. En la barra lateral: contador de calificados, lista para quitar valoraciones, **descargar CSV** y **reiniciar perfil**.

Las valoraciones viven de forma temporal en `st.session_state`. La información se pierde al cerrar la pestaña. (Descarga el CSV si quieres conservarlas).

## Estructura

```
app.py                     Inicio
pages/                     Dashboard · Explore · Recommendations · Model Insights
src/
  data_loader.py           Lectura de JSONL y carga/guardado de artefactos
  preprocessing.py         Limpieza, filtro iterativo, categorías, índices
  popularity.py            Popularity-Based
  item_knn.py              Similitud coseno top-K + predicción Item-KNN
  recommender.py           Fachada: elige algoritmo, fallback, comparación
  session.py               Estado de sesión (ratings, historial, CSV) sin dependencia de Streamlit
  ui.py                    Componentes compartidos de Streamlit
scripts/
  download_data.py         Descarga los JSONL
  prepare_data.py          Pipeline completo → artefactos
  evaluate.py              Experimento offline Popularity vs Item-KNN
tests/                     pytest (unitarias + flujo de la app)
artifacts/                 item_similarity.npz, popularity.parquet, mappings/, model_info.json
data/processed/            interactions.parquet, products.parquet
docs/                      Algoritmos, experimentos y resultados
```

### Artefactos

| Archivo | Contenido |
| --- | --- |
| `data/processed/interactions.parquet` | Ratings históricos filtrados |
| `data/processed/products.parquet` | Metadata, categoría inferida y URL de imagen |
| `artifacts/mappings/user_mapping.pkl` | `user_id → índice` |
| `artifacts/mappings/item_mapping.pkl` | `parent_asin → índice` |
| `artifacts/item_similarity.npz` | Matriz dispersa de similitud (50 vecinos por producto) |
| `artifacts/popularity.parquet` | Ranking de popularidad y sesgo de cada producto |
| `artifacts/model_info.json` | Parámetros y estadísticas de la última preparación |

El identificador de producto es siempre `parent_asin`, tanto en reseñas como en metadata.

## Limitaciones conocidas

* **Categorías inferidas.** La metadata de Amazon Fashion 2023 no trae categorías útiles (`categories` vacío
  y `main_category` constante). La categoría se infiere del título con palabras clave en inglés; es una
  aproximación y algunos productos caen en «Otros». El departamento sale de `details["Department"]` o del título.
* **Dataset muy disperso** (99.95 %). Cada producto tiene en promedio ~18 vecinos con evidencia (de un máximo de 50),
  por lo que con pocas valoraciones Item-KNN solo cubre una parte del catálogo; el resto de la lista se
  completa con Popularity.
* **Catálogo sesgado a joyería y accesorios**, reflejo de los datos (el filtro favorece productos con muchas reseñas).