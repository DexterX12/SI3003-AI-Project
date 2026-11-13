# `scripts/prepare_data.py`

Script único de preparación. Lee los JSONL crudos de Amazon Fashion, deja un subconjunto limpio y calcula
**una sola vez** todo lo que necesita la app: popularidad y similitud entre productos. Después,
`streamlit run app.py` solo carga los archivos generados; no entrena nada.

## Uso

```powershell
python scripts/prepare_data.py
python scripts/prepare_data.py --min-user 3 --min-item 3 --top-k 50
```

| Opción | Por defecto | Efecto |
| --- | --- | --- |
| `--min-user` | `2` | Mínimo de ratings por usuario que sobreviven al filtro |
| `--min-item` | `5` | Mínimo de ratings por producto que sobreviven al filtro |
| `--top-k` | `50` | Vecinos que se guardan por producto en la matriz de similitud |
| `--block-size` | `1000` | Productos procesados por bloque al calcular similitud (más alto = más rápido y más memoria) |

En la ejecución de referencia (2.5 M reseñas, datos ya en disco) tardó unos **28 s** en total.

## Entradas

| Archivo | Se busca en | Campos que se usan |
| --- | --- | --- |
| `Amazon_Fashion.jsonl` (reseñas) | `data/raw/`, luego la raíz del proyecto | `user_id`, `parent_asin`, `rating`, `timestamp` |
| `meta_Amazon_Fashion.jsonl` (metadata) | `data/raw/`, luego la raíz del proyecto | `parent_asin`, `title`, `store`, `main_category`, `details.Department`, `average_rating`, `rating_number`, `price`, `images` |

Si no están, el script falla con un mensaje que indica ejecutar `python scripts/download_data.py`.
El identificador de producto es siempre `parent_asin`, en reseñas y en metadata.
Las líneas JSON inválidas se omiten y se avisa por stderr.

## Pipeline

```
Reseñas JSONL ─► [1] leer ─► [2] limpiar ─► [3] prefiltro ─┐
                                                            ├─► [4] unir metadata ─► [5] filtro iterativo ─► índices
Metadata JSONL ─────────────────────────────────────────────┘                                                  │
                                                                                                               ▼
                                              [7] guardar artefactos ◄─ [6] popularidad + matriz R + similitud
```

Las cifras entre paréntesis son las de la ejecución de referencia con los parámetros por defecto.

### [1] Leer reseñas (`data_loader.read_reviews`)

Recorre el JSONL línea a línea y conserva solo `user_id`, `parent_asin`, `rating` y `timestamp`
(se descarta el texto de la reseña, que es lo más pesado). (2 500 939 reseñas)

### [2] Limpiar (`preprocessing.clean_interactions`)

* Elimina filas con `user_id`, `parent_asin` o `rating` nulos.
* Conserva solo ratings numéricos entre 1 y 5.
* Si un usuario reseñó el mismo producto varias veces, deja **la más reciente** (por `timestamp`).
* Deja el rating como `float32`.

(2 474 375 interacciones únicas usuario-producto)

### [3] Prefiltro

Se quedan solo los productos con al menos `--min-item` interacciones. No cambia el resultado final
(el filtro del paso 5 lo exige igualmente); existe para **leer menos metadata**, que pesa 1.4 GB y tiene 826 108 productos.
(93 728 productos candidatos)

### [4] Unir metadata (`data_loader.read_metadata` + `preprocessing.clean_products`)

* Lee la metadata solo de los productos candidatos.
* **Imagen**: elige la URL de la variante `MAIN`; si no hay, la primera disponible. Orden de calidad: `large`, `hi_res`, `thumb`.
* **Exige título e imagen**. Los productos sin alguno de los dos se eliminan, junto con sus interacciones. (4 productos eliminados)
* **Categoría**: la metadata de Amazon Fashion 2023 trae `categories` vacío y `main_category` constante, así que se infiere del
  título con palabras clave en inglés (`preprocessing._CATEGORY_PATTERNS`). Gana la primera regla que coincide, por eso el orden
  importa («dress shoes» es Calzado, no Vestidos). Lo que no coincide queda en «Otros».
* **Departamento**: se toma de `details.Department` y, si falta, del título (Mujer, Hombre, Niñas, Niños, Unisex, Sin especificar).
* `price`, `average_rating` y `rating_number` se convierten a número; los valores no válidos quedan como nulos.

### [5] Filtro iterativo (`preprocessing.kcore_filter`)

Deja usuarios con al menos `--min-user` interacciones y productos con al menos `--min-item`. Se **repite hasta que no cae
ninguna fila**, porque quitar productos baja el conteo de los usuarios y viceversa. (33 iteraciones)

> **Por qué no 5-core.** Con usuarios ≥ 5 y productos ≥ 5 el resultado es vacío: el 86 % de los usuarios tiene una sola reseña.
> Por eso el valor por defecto es usuarios ≥ 2. Si el filtro deja el dataset vacío, el script termina con
> `El filtro dejó el dataset vacío. Reduce --min-user / --min-item.`

Alternativas medidas sobre este dataset (usuarios ≥ / productos ≥):

| Usuarios ≥ / Productos ≥ | Usuarios | Productos | Ratings |
| --- | --- | --- | --- |
| 5 / 5 | 0 | 0 | 0 |
| 3 / 5 | 967 | 572 | 4 849 |
| 3 / 3 | 2 223 | 2 245 | 12 490 |
| **2 / 5 (por defecto)** | **21 034** | **4 823** | **50 457** |
| 2 / 3 | 62 209 | 23 831 | 149 270 |

Luego se crean los **índices enteros** (`build_mappings`, `add_indices`): `user_idx` y `item_idx` contiguos desde 0, asignados
en orden alfabético del identificador, por lo que son deterministas. El script comprueba con un `assert` que `products` queda
alineado con `item_idx` (fila `k` = producto `k`); el recomendador depende de esa alineación.

### [6] Popularidad y similitud

* **Popularidad** (`popularity.compute_popularity`): score bayesiano por producto, sesgo de producto (`item_bias`) y ranking.
  Guarda también `global_mean` y `m` (percentil 90 del número de ratings por producto).
* **Matriz R** (`item_knn.build_rating_matrix`): CSR de usuarios × productos con los ratings originales. Solo vive en memoria.
* **Similitud** (`item_knn.compute_item_similarity`): coseno entre productos, por bloques de `--block-size`, sin diagonal,
  conservando los `--top-k` vecinos más similares de cada producto. Fórmulas completas en [`ALGORITHMS.md`](ALGORITHMS.md).

### [7] Guardar artefactos

Se sobrescriben los archivos existentes. Los directorios se crean si no existen.

## Salidas

| Archivo | Contenido |
| --- | --- |
| `data/processed/interactions.parquet` | Ratings filtrados |
| `data/processed/products.parquet` | Metadata de los productos que quedaron |
| `artifacts/mappings/user_mapping.pkl` | `dict` `user_id → user_idx` |
| `artifacts/mappings/item_mapping.pkl` | `dict` `parent_asin → item_idx` |
| `artifacts/item_similarity.npz` | `scipy.sparse.csr_matrix` float32 de `n_items × n_items`; la fila `i` contiene los vecinos de `i`, ordenados de mayor a menor similitud |
| `artifacts/popularity.parquet` | Popularidad y sesgo por producto, ordenado por `rank` |
| `artifacts/model_info.json` | Parámetros y estadísticas de la corrida |

### Esquemas

**`interactions.parquet`** (50 457 filas)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `user_id` | str | Identificador original de Amazon |
| `parent_asin` | str | Identificador del producto |
| `user_idx` | int32 | Índice del usuario |
| `item_idx` | int32 | Índice del producto |
| `rating` | float32 | 1.0 a 5.0 |
| `timestamp` | int64 | Milisegundos desde epoch |

**`products.parquet`** (4 823 filas, ordenado por `item_idx`)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `parent_asin`, `item_idx` | str, int32 | Identificador e índice |
| `title`, `store` | str | Nombre y tienda (`store` puede ser vacío) |
| `category`, `department` | str | Inferidos, ver paso [4] |
| `average_rating`, `rating_number` | float64, int64 | Rating promedio y número de reseñas **históricos de Amazon** (todo el periodo, no solo el subconjunto) |
| `price` | float64 | Nulo en el 80 % de los productos; la app no lo usa |
| `image_url` | str | URL remota de la imagen principal |

**`popularity.parquet`** (4 823 filas)

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `item_idx`, `parent_asin` | int32, str | Identificadores |
| `n_ratings` | int32 | Ratings en el subconjunto (`v_i`) |
| `mean_rating` | float32 | Rating medio en el subconjunto (`R_i`) |
| `score` | float32 | Score bayesiano de popularidad |
| `item_bias` | float32 | Sesgo regularizado, baseline de Item-KNN |
| `rank` | int32 | Posición por `score` (1 = mejor) |

**`model_info.json`**: `created_at`, `min_user_interactions`, `min_item_interactions`, `top_k_neighbors`, `raw_reviews`,
`n_users`, `n_items`, `n_ratings`, `sparsity`, `similarity_nnz`, `global_mean`, `m`, `quantile`, `bias_reg`.
La app lo lee para mostrar cifras y para fijar `global_mean` en el recomendador.