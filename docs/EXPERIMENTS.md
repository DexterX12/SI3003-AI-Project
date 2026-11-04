# Experimentos

Comparación básica Popularity-Based vs Item-KNN con `scripts/evaluate.py`.

## Protocolo (leave-last-out)

* Para cada uno de los **21 034 usuarios** se oculta su interacción más reciente (por `timestamp`).
* Popularidad, sesgos y matriz de similitud se **recalculan solo con el entrenamiento** (29 423 ratings), sin fuga de información.
* Se rankean todos los productos que el usuario no ha calificado y se mide si el producto oculto aparece en el Top-10.
* Métricas: **HR@10** (acierto), **NDCG@10** (acierto ponderado por posición), **cobertura** (% del catálogo que el algoritmo llega a recomendar) y **RMSE** del rating oculto.
* «Item-KNN + Popularity» es lo que ve el usuario en la app: Item-KNN y, si faltan candidatos, relleno con Popularity.

Reproducir: `python scripts/evaluate.py --users 0 --save` (≈ 30 s). Datos en `experiment_results.json`.

## Resultados (shrink λ = 0.1, 50 vecinos)

| Algoritmo | HR@10 | NDCG@10 | Cobertura |
| --- | --- | --- | --- |
| Aleatorio (referencia) | 0.0021 | – | – |
| Popularity-Based | 0.0047 | 0.0022 | 0.2 % |
| Item-KNN | 0.0379 | 0.0216 | 68.8 % |
| **Item-KNN + Popularity (app)** | **0.0402** | **0.0225** | 68.9 % |

* Item-KNN acierta el producto oculto unas **8 veces más** que Popularity (0.0379 vs 0.0047) y ≈ 18 veces más que un recomendador aleatorio.
* Popularity recomienda a todos casi la misma decena de productos (cobertura del 0.2 % del catálogo), mientras que Item-KNN personaliza y llega al 69 %.

### Predicción del rating (RMSE, menor es mejor)

| Subconjunto | Item-KNN | Baseline (μ + sesgos) | Media global |
| --- | --- | --- | --- |
| Todos los casos (21 034) | 1.196 | 1.197 | 1.270 |
| Solo donde el producto oculto es vecino de lo que calificó el usuario (7.9 %) | **1.041** | 1.062 | – |

La mejora del rating estimado es **pequeña**: el valor de Item-KNN en este dataset está en el **ranking**, no en
acertar el número de estrellas. Solo en el 7.9 % de los casos el producto oculto tenía evidencia colaborativa;
en el resto la predicción es el baseline.

## Sensibilidad al hiperparámetro de shrink (λ)

| λ | HR@10 Item-KNN | HR@10 Item-KNN + Pop | RMSE (con vecinos) |
| --- | --- | --- | --- |
| 0 | 0.0398 | 0.0421 | 1.172 |
| 0.02 | 0.0386 | 0.0409 | 1.102 |
| 0.05 | 0.0377 | 0.0400 | 1.063 |
| **0.1** | 0.0379 | 0.0402 | 1.041 |
| 0.3 | 0.0371 | 0.0394 | **1.031** |
| 1.0 | 0.0371 | 0.0394 | 1.044 |

El ranking casi no depende de λ (HR@10 entre 0.037 y 0.040). Sí cambia el RMSE: sin shrink se
sobreajustan predicciones con poca evidencia. Se eligió **λ = 0.1** como compromiso; λ = 0.3 da un RMSE un
poco mejor a costa de ~2 % relativo de HR@10. El barrido se evaluó sobre el mismo conjunto de prueba
(sin conjunto de validación aparte), así que las diferencias pequeñas entre valores de λ no son concluyentes.

## Comportamiento en la app (ejemplo)

Con las cinco primeras prendas populares calificadas (3 con 5★ y 2 con 3★):

* La estrategia pasa de Popularity-Based a Item-KNN.
* Item-KNN y Popularity comparten **1 de 10** productos (Jaccard 0.05).
* El ranking medio de popularidad de los productos recomendados por Item-KNN es **#211** (de 4 823): se alejan de lo que ya es popular.
* Cambiar el primer rating de 5★ a 1★ modifica el Top-5 (solo 2 de 5 productos se mantienen) y el ranking medio de popularidad pasa a #405.

## Limitaciones

* HR@10 absoluto es bajo (≈ 4 %) porque el catálogo tiene 4 823 productos y cada usuario aporta pocas interacciones.
* Solo se evalúa un hold-out por usuario (el último), no validación cruzada.
* El hold-out mide similitud de compra, no si al usuario le *gustó* el producto; HR no distingue ratings altos de bajos.
