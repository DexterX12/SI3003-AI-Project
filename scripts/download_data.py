"""Descarga los JSONL de Amazon Reviews 2023 (categoría Amazon Fashion) a data/raw/.

Si los archivos ya existen en data/raw/ o en la raíz del proyecto no se descargan.

Uso:
    python scripts/download_data.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
FILES = {
    dl.REVIEWS_FILE: f"{BASE}/review_categories/{dl.REVIEWS_FILE}",
    dl.META_FILE: f"{BASE}/meta_categories/{dl.META_FILE}",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB ({done / total:.0%})", end="", flush=True)

    urllib.request.urlretrieve(url, partial, reporthook=progress)
    partial.replace(destination)
    print()


def main() -> None:
    for name, url in FILES.items():
        try:
            existing = dl.find_raw_file(name)
            print(f"OK  {name} ya existe en {existing}")
            continue
        except FileNotFoundError:
            pass
        print(f"Descargando {name} ...")
        _download(url, dl.RAW_DIR / name)
    print("Datos crudos listos. Siguiente paso: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
