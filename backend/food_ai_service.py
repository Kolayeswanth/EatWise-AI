from __future__ import annotations

import base64
import json
import os
import re
from typing import List

import requests
from dotenv import load_dotenv

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

HTTP_TIMEOUT_SECONDS = 30
MAX_ITEMS = 30


def _chat_url() -> str:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
        return ""
    return (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )


def _ensure_configured() -> None:
    if not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is missing")
    if not AZURE_OPENAI_KEY:
        raise RuntimeError("AZURE_OPENAI_KEY is missing")
    if not AZURE_OPENAI_DEPLOYMENT:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is missing")


def _extract_json_array(text: str) -> List[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\\s*", "", cleaned)
    cleaned = re.sub(r"^```\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        return []

    names: List[str] = []
    seen = set()
    for item in parsed:
        value = str(item).strip()
        lowered = value.lower()
        if not value or lowered in seen:
            continue
        seen.add(lowered)
        names.append(value)
    return names


def _chat(messages: list, temperature: float = 0.2) -> str:
    _ensure_configured()

    headers = {
        "api-key": AZURE_OPENAI_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 300,
    }

    response = requests.post(_chat_url(), headers=headers, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    # Some API responses can return content blocks instead of a plain string.
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "\n".join(part for part in text_parts if part).strip()

    return str(content)


def detect_food_names(image_bytes: bytes) -> List[str]:
    if not image_bytes:
        return []

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        "Look at this food image.\n"
        "Guess the top 3 possible food names.\n"
        "Return ONLY a JSON array of names."
    )

    messages = [
        {
            "role": "system",
            "content": "You identify foods from images and return strict JSON when requested.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                },
            ],
        },
    ]

    text = _chat(messages=messages, temperature=0.1)
    names = _extract_json_array(text)
    return names[:3]


def generate_food_ingredients(food_name: str, language: str) -> List[str]:
    food = str(food_name).strip()
    target_language = str(language).strip() or "English"
    if not food:
        return []

    prompt = (
        f"List common ingredients used in {food}.\n"
        f"Return a clean list of ingredients in {target_language}.\n"
        "No explanation.\n"
        "Return ONLY a JSON array of strings."
    )

    messages = [
        {
            "role": "system",
            "content": "You generate common ingredient lists and return strict JSON arrays when requested.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = _chat(messages=messages, temperature=0.2)
    ingredients = _extract_json_array(text)

    cleaned: List[str] = []
    seen = set()
    for item in ingredients:
        token = str(item).strip()
        lowered = token.lower()
        if not token or lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(token)
        if len(cleaned) >= MAX_ITEMS:
            break

    return cleaned
