"""Lectura de datos crudos (JSONL) y carga/guardado de artefactos procesados."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MAPPINGS_DIR = ARTIFACTS_DIR / "mappings"

REVIEWS_FILE = "Amazon_Fashion.jsonl"
META_FILE = "meta_Amazon_Fashion.jsonl"

INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"
PRODUCTS_PATH = PROCESSED_DIR / "products.parquet"
USER_MAPPING_PATH = MAPPINGS_DIR / "user_mapping.pkl"
ITEM_MAPPING_PATH = MAPPINGS_DIR / "item_mapping.pkl"
SIMILARITY_PATH = ARTIFACTS_DIR / "item_similarity.npz"
POPULARITY_PATH = ARTIFACTS_DIR / "popularity.parquet"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model_info.json"


# --------------------------------------------------------------------------
# Datos crudos
# --------------------------------------------------------------------------
def find_raw_file(name: str) -> Path:
    """Busca un archivo crudo en ``data/raw`` y, si no está, en la raíz del proyecto."""
    for folder in (RAW_DIR, PROJECT_ROOT):
        candidate = folder / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró '{name}' en {RAW_DIR} ni en {PROJECT_ROOT}. "
        "Ejecuta: python scripts/download_data.py"
    )


def iter_jsonl(path: Path, log_every: int = 500_000, label: str = "") -> Iterator[dict]:
    """Itera un archivo JSONL línea a línea, omitiendo líneas corruptas."""
    start = time.time()
    bad = 0
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if log_every and i % log_every == 0:
                print(f"  {label or path.name}: {i:,} líneas ({time.time() - start:.0f}s)", flush=True)
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  Aviso: {bad} líneas inválidas omitidas en {path.name}", file=sys.stderr)


def read_reviews(path: Path) -> pd.DataFrame:
    """Lee las reseñas conservando solo los campos necesarios."""
    users, items, ratings, stamps = [], [], [], []
    for row in iter_jsonl(path, label="reviews"):
        users.append(row.get("user_id"))
        items.append(row.get("parent_asin"))
        ratings.append(row.get("rating"))
        stamps.append(row.get("timestamp"))
    return pd.DataFrame(
        {"user_id": users, "parent_asin": items, "rating": ratings, "timestamp": stamps}
    )


def _pick_image(images: Iterable[dict] | None) -> str | None:
    """Elige la URL de la imagen principal (variante MAIN si existe)."""
    if not images:
        return None
    images = [im for im in images if isinstance(im, dict)]
    ordered = sorted(images, key=lambda im: 0 if im.get("variant") == "MAIN" else 1)
    for im in ordered:
        for key in ("large", "hi_res", "thumb"):
            if im.get(key):
                return im[key]
    return None


def read_metadata(path: Path, keep_asins: set[str]) -> pd.DataFrame:
    """Lee la metadata de los ``parent_asin`` indicados (campos mínimos para la app)."""
    records = []
    for row in iter_jsonl(path, label="metadata"):
        asin = row.get("parent_asin")
        if asin not in keep_asins:
            continue
        details = row.get("details") or {}
        records.append(
            {
                "parent_asin": asin,
                "title": row.get("title"),
                "store": row.get("store"),
                "main_category": row.get("main_category"),
                "department_raw": details.get("Department") if isinstance(details, dict) else None,
                "average_rating": row.get("average_rating"),
                "rating_number": row.get("rating_number"),
                "price": row.get("price"),
                "image_url": _pick_image(row.get("images")),
            }
        )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Artefactos procesados
# --------------------------------------------------------------------------
def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_similarity(matrix: sp.csr_matrix, path: Path = SIMILARITY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix, compressed=True)


def load_similarity(path: Path = SIMILARITY_PATH) -> sp.csr_matrix:
    return sp.load_npz(path).tocsr()


def artifacts_ready() -> bool:
    """Indica si ya se ejecutó ``scripts/prepare_data.py``."""
    required = (
        INTERACTIONS_PATH,
        PRODUCTS_PATH,
        USER_MAPPING_PATH,
        ITEM_MAPPING_PATH,
        SIMILARITY_PATH,
        POPULARITY_PATH,
    )
    return all(p.exists() for p in required)
