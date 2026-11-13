# Algoritmos

FashionRec usa dos recomendadores. Todo el entrenamiento ocurre una vez en `scripts/prepare_data.py`;
cuando el usuario califica una prenda no se entrena nada, solo se usa la matriz de similitud
precalculada y sus valoraciones de la sesión.

## Flujo de recomendación

```
Usuario sin ratings ──► Popularity-Based
        │ califica una prenda
        ▼
Usuario con ratings ──► Item-Based CF ──► Top-N ordenado por rating estimado
        │ cambia / quita / agrega un rating
        └─────────────► se recalcula el Top-N (milisegundos)
```

En modo *Automático* (por defecto) esta es la lógica. El usuario también puede forzar Popularity-Based;
si fuerza Item-KNN sin ratings se muestra Popularity con un aviso (`resolve_strategy` en `src/recommender.py`).

## 1. Popularity-Based (`src/popularity.py`)

Media bayesiana ponderada (la fórmula de IMDb):

```
Score(i) = v_i / (v_i + m) · R_i  +  m / (v_i + m) · C
```

| Símbolo | Significado | Valor en este dataset |
| --- | --- | --- |
| `v_i` | número de ratings del producto en el subconjunto | por producto |
| `R_i` | rating medio del producto | por producto |
| `C` | rating medio global | 4.140 |
| `m` | percentil 90 de `v_i` (confianza mínima) | 17 |

Un producto con pocas reseñas se "encoge" hacia `C`, así que un único 5★ no lo lleva al primer puesto.
El score está en la escala 1-5 y es lo que se muestra como «Score de popularidad».

## 2. Item-Based Collaborative Filtering (`src/item_knn.py`)

### 2.1 Entrenamiento (una sola vez)

1. Matriz dispersa `R ∈ ℝ^{|U|×|I|}` con los ratings originales (usuarios × productos).
2. Similitud coseno entre columnas (productos):

   ```
   sim(i, j) = (R_i · R_j) / (‖R_i‖ ‖R_j‖)
   ```

3. Se calcula por bloques de 1000 productos para no materializar la matriz densa |I|×|I|.
4. Se descarta la diagonal y se conservan los **50 vecinos más similares** de cada producto
   (`artifacts/item_similarity.npz`, CSR de 4 823 × 4 823 con 86 245 valores distintos de cero).

Como los ratings son positivos, la similitud está en [0, 1].

### 2.2 Predicción (en cada cambio de valoraciones)

Sea `I_u` el conjunto de productos que el usuario calificó. La versión simple sería:

```
Score(u,i) = Σ_j sim(i,j) · r_uj  /  Σ_j |sim(i,j)|         j ∈ I_u
```

Esa versión depende de los ratings absolutos: un usuario que califica todo con 5★ obtendría 5.0 para
cualquier candidato. Por eso la implementación **centra sobre un baseline**:

```
b_uj    = μ + bu + b_j                                    (baseline de un par usuario-producto)
Score   = b_ui + Σ_j sim(j,i) · (r_uj − b_uj) / (Σ_j |sim(j,i)| + λ)
```

| Término | Definición |
| --- | --- |
| `μ` | rating medio global |
| `b_j` | sesgo del producto: `Σ(r − μ) / (n_j + 10)` (precalculado, en `popularity.parquet`) |
| `bu` | sesgo del usuario: `Σ_j (r_uj − μ − b_j) / (|I_u| + 3)`, calculado con sus ratings de sesión |
| `λ` | `0.1`. Amortigua candidatos con poca evidencia (suma de similitudes baja) |

* Los vecinos usados son los de las filas de los productos calificados (`sim(j, ·)`), es decir, «productos
  parecidos a lo que calificaste».
* Solo son candidatos los productos que son vecinos de al menos un producto calificado.
* El resultado se limita a [1, 5] y se ordena por rating estimado, desempatando por suma de similitudes.
* Si hay menos candidatos que el N pedido, la lista se completa con Popularity-Based y cada tarjeta indica su origen.
* Por defecto se excluyen los productos ya calificados.

### 2.3 Hiperparámetros

| Parámetro | Valor | Dónde se fija | Cómo se eligió |
| --- | --- | --- | --- |
| vecinos por producto | 50 | `--top-k` | Mínimo arbitrario |
| `λ` (shrink) | 0.1 | `DEFAULT_SUPPORT_SHRINK` | barrido en [`EXPERIMENTS.md`](EXPERIMENTS.md): mejor RMSE sin perder HR@10 |
| regularización de `bu` | 3 | `DEFAULT_USER_BIAS_REG` | con pocos ratings el sesgo de usuario debe ser pequeño |
| regularización de `b_j` | 10 | `BIAS_REGULARIZATION` | evita sesgos extremos en productos con 5 reseñas |

## 3. Preparación del dataset (`scripts/prepare_data.py`)

```
Load JSONL → Clean → Prefiltro → Join metadata → Filtro iterativo → Matriz dispersa → Similitud → Artefactos
```

1. **Clean**: elimina nulos y ratings fuera de 1-5; una sola reseña por (usuario, `parent_asin`), la más reciente.
2. **Prefiltro**: productos con ≥ 5 interacciones, para leer solo su metadata (la metadata pesa 1.4 GB).
3. **Join metadata** por `parent_asin`: se exige título e imagen.
4. **Filtro iterativo** hasta estabilizar: usuarios ≥ 2 y productos ≥ 5 interacciones (ver README: el 5-core estricto deja el dataset vacío).
5. Índices enteros contiguos y deterministas (orden alfabético) para usuarios y productos.

| Etapa | Cifra |
| --- | --- |
| Reseñas originales | 2 500 939 |
| Interacciones únicas usuario-producto | 2 474 375 |
| Productos con ≥ 5 interacciones | 93 728 |
| Resultado final | 21 034 usuarios · 4 823 productos · 50 457 ratings |
| Dispersión de la matriz | 99.95 % |

La categoría de cada producto se infiere del título por palabras clave (la metadata no trae categorías útiles).
