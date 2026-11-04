"""Prepara el dataset y entrena (una sola vez) los artefactos del recomendador.

Pipeline:
    Load JSONL -> Clean -> Filter (pre) -> Join metadata -> k-core -> Sparse matrix
    -> Item similarity (top-K) -> Save artifacts

Nota sobre el filtro: en Amazon Fashion el 5-core estricto (usuarios >= 5 y
productos >= 5, iterativo) deja el dataset vacío, porque el 86 % de los usuarios
tiene una sola reseña. Por defecto se usa usuarios >= 2 y productos >= 5, también
iterativo: un usuario con una sola reseña no aporta a la similitud entre productos.

Uso:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --min-user 3 --min-item 3 --top-k 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402
from src import item_knn, popularity, preprocessing as pp  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-user", type=int, default=2, help="mínimo de ratings por usuario (por defecto 2)")
    parser.add_argument("--min-item", type=int, default=5, help="mínimo de ratings por producto (por defecto 5)")
    parser.add_argument("--top-k", type=int, default=item_knn.DEFAULT_TOP_K, help="vecinos por producto")
    parser.add_argument("--block-size", type=int, default=1000, help="filas por bloque al calcular similitud")
    args = parser.parse_args()

    t0 = time.time()
    reviews_path = dl.find_raw_file(dl.REVIEWS_FILE)
    meta_path = dl.find_raw_file(dl.META_FILE)

    print(f"[1/7] Leyendo reseñas: {reviews_path}")
    raw = dl.read_reviews(reviews_path)
    n_raw = len(raw)
    print(f"      {n_raw:,} reseñas leídas")

    print("[2/7] Limpiando interacciones")
    inter = pp.clean_interactions(raw)
    del raw
    print(f"      {len(inter):,} interacciones únicas usuario-producto")

    print(f"[3/7] Prefiltro (productos con >= {args.min_item} interacciones) para acotar la lectura de metadata")
    counts = inter["parent_asin"].value_counts()
    candidates = set(counts[counts >= args.min_item].index)
    inter = inter[inter["parent_asin"].isin(candidates)]
    print(f"      {len(candidates):,} productos candidatos")

    print(f"[4/7] Uniendo metadata: {meta_path}")
    meta = dl.read_metadata(meta_path, candidates)
    products = pp.clean_products(meta)
    del meta
    print(f"      {len(products):,} productos con título e imagen")
    inter = inter[inter["parent_asin"].isin(set(products["parent_asin"]))]

    print(f"[5/7] Filtro iterativo (usuarios >= {args.min_user}, productos >= {args.min_item})")
    inter = pp.kcore_filter(inter, min_user=args.min_user, min_item=args.min_item)
    if inter.empty:
        sys.exit("El filtro dejó el dataset vacío. Reduce --min-user / --min-item.")
    user_map, item_map = pp.build_mappings(inter)
    inter = pp.add_indices(inter, user_map, item_map)
    n_users, n_items = len(user_map), len(item_map)
    print(f"      {n_users:,} usuarios, {n_items:,} productos, {len(inter):,} ratings")

    item_ids = sorted(item_map, key=item_map.get)
    products = products[products["parent_asin"].isin(item_map)].copy()
    products["item_idx"] = products["parent_asin"].map(item_map).astype("int32")
    products = products.sort_values("item_idx").reset_index(drop=True)
    assert len(products) == n_items and (products["item_idx"].to_numpy() == range(n_items)).all()

    print("[6/7] Popularidad y matriz de similitud")
    pop_table, pop_params = popularity.compute_popularity(inter, item_ids)
    R = item_knn.build_rating_matrix(
        inter["user_idx"].to_numpy(), inter["item_idx"].to_numpy(),
        inter["rating"].to_numpy(), n_users, n_items,
    )
    similarity = item_knn.compute_item_similarity(R, top_k=args.top_k, block_size=args.block_size)

    print("[7/7] Guardando artefactos")
    dl.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dl.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    inter[["user_id", "parent_asin", "user_idx", "item_idx", "rating", "timestamp"]].to_parquet(
        dl.INTERACTIONS_PATH, index=False
    )
    products.to_parquet(dl.PRODUCTS_PATH, index=False)
    dl.save_pickle(user_map, dl.USER_MAPPING_PATH)
    dl.save_pickle(item_map, dl.ITEM_MAPPING_PATH)
    dl.save_similarity(similarity)
    pop_table.to_parquet(dl.POPULARITY_PATH, index=False)

    info = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_user_interactions": args.min_user,
        "min_item_interactions": args.min_item,
        "top_k_neighbors": args.top_k,
        "raw_reviews": n_raw,
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": int(len(inter)),
        "sparsity": pp.sparsity(len(inter), n_users, n_items),
        "similarity_nnz": int(similarity.nnz),
        **pop_params,
    }
    dl.MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nListo en {time.time() - t0:.0f}s. Artefactos en: {dl.ARTIFACTS_DIR} y {dl.PROCESSED_DIR}")
    print("Siguiente paso: streamlit run app.py")


if __name__ == "__main__":
    main()
