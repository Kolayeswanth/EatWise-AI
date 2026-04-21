from __future__ import annotations

import re
from io import BytesIO
from typing import List, Tuple

import easyocr
from PIL import Image

_reader = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def clean_ingredients(raw_text: str) -> List[str]:
    text = raw_text.lower()
    text = re.sub(r"ingredients?[:\\n]", "", text)
    text = text.replace(";", ",")
    text = re.sub(r"[^a-z0-9,()\- ]+", " ", text)

    chunks = [x.strip(" .") for x in text.split(",")]
    cleaned = []
    for chunk in chunks:
        if not chunk:
            continue
        if len(chunk) < 2:
            continue
        cleaned.append(chunk)

    seen = set()
    deduped = []
    for item in cleaned:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def extract_ingredients_from_image(image_bytes: bytes) -> Tuple[str, List[str]]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = __import__("numpy").array(image)

    reader = get_reader()
    result = reader.readtext(image_np, detail=0, paragraph=True)
    raw_text = " ".join(result)

    ingredients = clean_ingredients(raw_text)
    return raw_text, ingredients
