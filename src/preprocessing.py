"""Limpieza de reseñas y metadata, filtro k-core y construcción de índices."""
from __future__ import annotations

import re

import pandas as pd


# --------------------------------------------------------------------------
# Interacciones
# --------------------------------------------------------------------------
def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina nulos y ratings inválidos y deja una sola reseña por (usuario, producto).

    Si un usuario reseñó el mismo producto varias veces se conserva la más reciente.
    """
    out = df.dropna(subset=["user_id", "parent_asin", "rating"]).copy()
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out = out[out["rating"].between(1, 5)]
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype("int64")
    out = out.sort_values("timestamp", kind="stable")
    out = out.drop_duplicates(subset=["user_id", "parent_asin"], keep="last")
    out["rating"] = out["rating"].astype("float32")
    return out.reset_index(drop=True)


def kcore_filter(
    df: pd.DataFrame,
    min_user: int = 5,
    min_item: int = 5,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    verbose: bool = True,
) -> pd.DataFrame:
    """Filtro iterativo: usuarios con >= ``min_user`` y productos con >= ``min_item`` interacciones.

    Con ``min_user == min_item == k`` es el k-core clásico. Se repite hasta que
    ya no se elimina ninguna fila, porque quitar productos reduce los conteos de
    los usuarios y viceversa.
    """
    out = df
    iteration = 0
    while True:
        iteration += 1
        user_counts = out[user_col].value_counts()
        item_counts = out[item_col].value_counts()
        keep = out[user_col].map(user_counts).ge(min_user) & out[item_col].map(item_counts).ge(min_item)
        if keep.all():
            break
        out = out[keep]
        if verbose:
            print(f"  k-core iteración {iteration}: {len(out):,} interacciones", flush=True)
    return out.reset_index(drop=True)


def build_mappings(interactions: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    """Crea índices enteros contiguos y deterministas para usuarios y productos."""
    users = sorted(interactions["user_id"].unique())
    items = sorted(interactions["parent_asin"].unique())
    return {u: i for i, u in enumerate(users)}, {a: i for i, a in enumerate(items)}


def add_indices(
    interactions: pd.DataFrame, user_map: dict[str, int], item_map: dict[str, int]
) -> pd.DataFrame:
    out = interactions.copy()
    out["user_idx"] = out["user_id"].map(user_map).astype("int32")
    out["item_idx"] = out["parent_asin"].map(item_map).astype("int32")
    return out


# --------------------------------------------------------------------------
# Productos
# --------------------------------------------------------------------------
# La metadata de Amazon Fashion (2023) trae ``categories`` vacío y ``main_category``
# constante, por lo que el tipo de prenda se infiere del título. El primer patrón
# que coincide gana, así que el orden importa (p. ej. "dress shoes" -> Calzado).
_CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("Calzado", r"\b(shoes?|sneakers?|boots?|sandals?|slippers?|loafers?|heels?|pumps?|clogs?|flats|oxfords?|moccasins?|espadrilles?|flip[- ]flops?)\b"),
    ("Ropa interior y calcetines", r"\b(socks?|underwear|bras?|panty|panties|briefs?|boxers?|lingerie|thongs?|bralettes?|shapewear|camisoles?)\b"),
    ("Joyería", r"\b(necklaces?|earrings?|bracelets?|rings?|pendants?|jewelry|anklets?|charms?|brooch(es)?|studs?)\b"),
    ("Relojes", r"\b(watch(es)?|smartwatch)\b"),
    ("Bolsos y carteras", r"\b(bags?|backpacks?|wallets?|purses?|handbags?|totes?|clutch(es)?|luggage|satchels?|crossbody)\b"),
    ("Trajes de baño", r"\b(swim\w*|bikinis?|bathing|tankinis?|one[- ]piece|trunks)\b"),
    ("Ropa de dormir", r"\b(pajamas?|pyjamas?|nightgowns?|nightshirts?|robes?|sleepwear|loungewear)\b"),
    ("Abrigos y chaquetas", r"\b(jackets?|coats?|vests?|parkas?|blazers?|windbreakers?|cardigans?|outerwear|puffer|anorak)\b"),
    ("Vestidos y faldas", r"\b(dress(es)?|gowns?|skirts?|jumpsuits?|rompers?|overalls?)\b"),
    ("Pantalones y shorts", r"\b(pants?|jeans?|shorts?|trousers?|leggings?|joggers?|sweatpants|chinos?|capris?)\b"),
    ("Camisas y tops", r"\b(shirts?|t-shirts?|tees?|blouses?|tops?|tanks?|sweaters?|hoodies?|sweatshirts?|polos?|pullovers?|tunics?|henley|jerseys?)\b"),
    ("Accesorios", r"\b(hats?|caps?|beanies?|scarf|scarves|gloves?|mittens?|belts?|sunglasses|ties?|bow ?ties?|bandanas?|masks?|keychains?|suspenders|headbands?|glasses)\b"),
    ("Disfraces", r"\b(costumes?|cosplay|halloween)\b"),
]
_COMPILED_CATEGORIES = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _CATEGORY_PATTERNS]

_DEPARTMENT_RULES: list[tuple[str, str]] = [
    ("Niñas", r"\bgirls?\b"),
    ("Niños", r"\bboys?\b"),
    ("Mujer", r"\b(women'?s?|womens|ladies|lady|female)\b"),
    ("Hombre", r"\b(men'?s?|mens|male|gentlemen)\b"),
    ("Unisex", r"\bunisex\b"),
]
_COMPILED_DEPARTMENTS = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _DEPARTMENT_RULES]


def infer_category(title: str) -> str:
    for name, pattern in _COMPILED_CATEGORIES:
        if pattern.search(title):
            return name
    return "Otros"


def normalize_department(raw: str | None, title: str) -> str:
    """Departamento (Mujer/Hombre/...) desde ``details['Department']`` o el título."""
    text = raw.strip().lower() if isinstance(raw, str) else ""
    if text:
        if "girl" in text:
            return "Niñas"
        if "boy" in text:
            return "Niños"
        if "unisex" in text:
            return "Unisex"
        if "women" in text or "lady" in text or "ladies" in text:
            return "Mujer"
        if "men" in text:
            return "Hombre"
    for name, pattern in _COMPILED_DEPARTMENTS:
        if pattern.search(title):
            return name
    return "Sin especificar"


def clean_products(meta: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la metadata: exige título e imagen, deriva categoría y departamento."""
    out = meta.copy()
    out["title"] = out["title"].fillna("").astype(str).str.strip()
    out = out[(out["title"] != "") & out["image_url"].notna()]
    out = out.drop_duplicates(subset="parent_asin", keep="first")
    out["store"] = out["store"].fillna("").astype(str).str.strip()
    out["category"] = out["title"].map(infer_category)
    out["department"] = [
        normalize_department(raw, title) for raw, title in zip(out["department_raw"], out["title"])
    ]
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["average_rating"] = pd.to_numeric(out["average_rating"], errors="coerce")
    out["rating_number"] = pd.to_numeric(out["rating_number"], errors="coerce")
    cols = [
        "parent_asin", "title", "store", "category", "department",
        "average_rating", "rating_number", "price", "image_url",
    ]
    return out[cols].reset_index(drop=True)


def sparsity(n_ratings: int, n_users: int, n_items: int) -> float:
    return float(1.0 - n_ratings / (n_users * n_items)) if n_users and n_items else float("nan")
