from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from supabase import Client, create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "users")

_client: Optional[Client] = None


def _normalize_allergies(allergies: Any) -> List[str]:
    if not allergies:
        return []

    if isinstance(allergies, str):
        values = [part.strip() for part in allergies.split(",")]
    else:
        values = [str(item).strip() for item in allergies if str(item).strip()]

    normalized: List[str] = []
    seen = set()
    for item in values:
        token = item.strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _normalize_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(data or {})
    profile["name"] = str(profile.get("name", "")).strip()
    profile["age"] = int(profile.get("age", 0) or 0)
    profile["preferred_language"] = str(profile.get("preferred_language", "English")).strip() or "English"
    profile["health_conditions"] = [
        str(item).strip()
        for item in profile.get("health_conditions", [])
        if str(item).strip()
    ]
    profile["allergies"] = _normalize_allergies(profile.get("allergies", []))
    return profile


def get_client() -> Client:
    global _client
    if _client is not None:
        return _client

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase environment variables are missing")

    _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


def get_user(user_id: str) -> Dict[str, Any]:
    client = get_client()
    response = client.table(SUPABASE_TABLE).select("*").eq("id", user_id).limit(1).execute()
    rows = response.data or []
    return rows[0] if rows else {}


def create_user(data: Dict[str, Any]) -> Dict[str, Any]:
    client = get_client()
    profile = _normalize_profile(data)
    profile.setdefault("id", str(uuid4()))

    response = client.table(SUPABASE_TABLE).insert(profile).execute()
    rows = response.data or []
    return rows[0] if rows else profile


def update_user(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    client = get_client()
    profile = _normalize_profile(data)

    response = client.table(SUPABASE_TABLE).update(profile).eq("id", user_id).execute()
    rows = response.data or []
    if rows:
        return rows[0]

    profile["id"] = user_id
    return profile