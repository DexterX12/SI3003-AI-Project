"""Comparación offline Popularity vs Item-KNN (leave-last-out).

Para cada usuario evaluado se oculta su interacción más reciente. La popularidad y
la similitud se recalculan SOLO con el resto de los datos (sin fuga de información).
Luego se rankean todos los productos que el usuario no ha calificado y se mide si el
producto oculto aparece en el Top-N.

Métricas:
    HR@N, NDCG@N     acierto del producto oculto en el Top-N
    Cobertura        % del catálogo que el algoritmo llega a recomendar
    RMSE             error del rating estimado del producto oculto
                     (Item-KNN vs baseline vs media global)

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --users 5000 --n 10 --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402
from src import item_knn, popularity  # noqa: E402

RESULTS_PATH = dl.PROJECT_ROOT / "docs" / "experiment_results.json"


def split_leave_last(inter: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inter = inter.sort_values(["user_idx", "timestamp"], kind="stable")
    last = inter.groupby("user_idx").tail(1)
    return inter.drop(last.index), last


def _ndcg(rank: int) -> float:
    return 1.0 / np.log2(rank + 1)


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    item_ids: list[str],
    n_users: int,
    n: int,
    top_k: int,
    shrink: float,
    user_bias_reg: float,
    seed: int,
    similarity=None,
) -> tuple[dict, object]:
    n_items = len(item_ids)
    pop_table, pop_params = popularity.compute_popularity(train, item_ids)
    if similarity is None:
        R = item_knn.build_rating_matrix(
            train["user_idx"].to_numpy(), train["item_idx"].to_numpy(),
            train["rating"].to_numpy(), int(train["user_idx"].max()) + 1, n_items,
        )
        similarity = item_knn.compute_item_similarity(R, top_k=top_k, verbose=False)

    bias = np.zeros(n_items)
    bias[pop_table["item_idx"].to_numpy()] = pop_table["item_bias"].to_numpy()
    knn = item_knn.ItemKNN(
        similarity, bias, pop_params["global_mean"],
        user_bias_reg=user_bias_reg, support_shrink=shrink,
    )
    pop_order = pop_table["item_idx"].to_numpy()

    train_by_user = {u: g for u, g in train.groupby("user_idx")}
    rng = np.random.default_rng(seed)
    users = test["user_idx"].to_numpy()
    if n_users and n_users < len(users):
        users = rng.choice(users, size=n_users, replace=False)
    test_by_user = test.set_index("user_idx")

    hits = {"popularity": [], "item_knn": [], "item_knn+pop": []}
    ranks = {k: [] for k in hits}
    recommended = {k: set() for k in hits}
    err = {"knn": [], "baseline": [], "global": []}
    err_covered = {"knn": [], "baseline": []}

    for u in users:
        g = train_by_user.get(u)
        if g is None:
            continue
        rated = dict(zip(g["item_idx"].to_numpy().tolist(), g["rating"].to_numpy().tolist()))
        target = int(test_by_user.at[u, "item_idx"])
        target_rating = float(test_by_user.at[u, "rating"])

        # Popularity: primeros N no calificados
        seen = set(rated)
        top_pop = [i for i in pop_order[: n + len(seen)] if i not in seen][:n]

        # Item-KNN puro y con relleno de popularidad (como la app)
        knn_frame = knn.recommend(rated, n=n, exclude_rated=True)
        top_knn = knn_frame["item_idx"].tolist()
        filler = [i for i in pop_order[: 2 * n + len(seen)] if i not in seen and i not in top_knn]
        top_mix = (top_knn + filler)[:n]

        for name, lst in (("popularity", top_pop), ("item_knn", top_knn), ("item_knn+pop", top_mix)):
            recommended[name].update(lst)
            if target in lst:
                pos = lst.index(target) + 1
                hits[name].append(1.0)
                ranks[name].append(pos)
            else:
                hits[name].append(0.0)

        pred, _, nnb = knn.predict_all(rated)
        bu = knn.user_bias(np.array(list(rated)), np.array(list(rated.values())))
        base = float(np.clip(knn.global_mean + bu + knn.item_bias[target], 1, 5))
        err["knn"].append((pred[target] - target_rating) ** 2)
        err["baseline"].append((base - target_rating) ** 2)
        err["global"].append((knn.global_mean - target_rating) ** 2)
        if nnb[target] > 0:
            err_covered["knn"].append((pred[target] - target_rating) ** 2)
            err_covered["baseline"].append((base - target_rating) ** 2)

    def summary(name: str) -> dict:
        h = np.array(hits[name])
        return {
            f"HR@{n}": float(h.mean()),
            f"NDCG@{n}": float(sum(_ndcg(r) for r in ranks[name]) / len(h)),
            "coverage": len(recommended[name]) / n_items,
        }

    result = {
        "users_evaluated": len(hits["popularity"]),
        "n": n,
        "params": {"top_k": top_k, "support_shrink": shrink, "user_bias_reg": user_bias_reg},
        "ranking": {name: summary(name) for name in hits},
        "rmse": {k: float(np.sqrt(np.mean(v))) for k, v in err.items()},
        "rmse_covered": {k: float(np.sqrt(np.mean(v))) for k, v in err_covered.items()},
        "covered_share": len(err_covered["knn"]) / max(len(err["knn"]), 1),
        "random_hr": n / n_items,
    }
    return result, similarity


def print_result(res: dict) -> None:
    n = res["n"]
    print(f"\nUsuarios evaluados: {res['users_evaluated']:,}   N={n}   "
          f"parámetros={res['params']}   HR aleatorio≈{res['random_hr']:.4f}")
    rows = [
        {"Algoritmo": k, **{m: f"{v:.4f}" if m != "coverage" else f"{v:.1%}" for m, v in vals.items()}}
        for k, vals in res["ranking"].items()
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nRMSE del rating oculto (todos):        {res['rmse']}")
    print(f"RMSE donde Item-KNN tiene vecinos ({res['covered_share']:.1%} de los casos): {res['rmse_covered']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users", type=int, default=5000, help="usuarios a evaluar (0 = todos)")
    parser.add_argument("--n", type=int, default=10, help="tamaño del Top-N")
    parser.add_argument("--top-k", type=int, default=item_knn.DEFAULT_TOP_K)
    parser.add_argument("--shrink", type=float, default=item_knn.DEFAULT_SUPPORT_SHRINK)
    parser.add_argument("--user-bias-reg", type=float, default=item_knn.DEFAULT_USER_BIAS_REG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="probar varios valores de shrink")
    parser.add_argument("--save", action="store_true", help="guardar resultados en docs/experiment_results.json")
    args = parser.parse_args()

    inter = pd.read_parquet(dl.INTERACTIONS_PATH)
    item_map = dl.load_pickle(dl.ITEM_MAPPING_PATH)
    item_ids = sorted(item_map, key=item_map.get)
    train, test = split_leave_last(inter)
    print(f"Train: {len(train):,} ratings | Test: {len(test):,} usuarios (último rating de cada uno)")

    t0 = time.time()
    results = []
    similarity = None
    shrinks = [0.0, 0.02, 0.05, 0.1, 0.3, 1.0] if args.sweep else [args.shrink]
    for shrink in shrinks:
        res, similarity = evaluate(
            train, test, item_ids, args.users, args.n, args.top_k,
            shrink, args.user_bias_reg, args.seed, similarity,
        )
        print_result(res)
        results.append(res)
    print(f"\nTiempo: {time.time() - t0:.0f}s")

    if args.save:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Resultados guardados en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
