"""
Utilidades compartidas para clientes LLM locales.
"""
from __future__ import annotations

import json
from json import JSONDecoder
from typing import Any


JSON_DECODER = JSONDecoder()


def clip_text(text: str, max_chars: int) -> str:
    """Recorta texto largo para no saturar el prompt."""
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


_STRING_BOOL_MAP = {"true": True, "false": False, "null": None, "none": None}


def normalize_coherent_field(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Normaliza el campo "coherent" cuando el modelo lo devuelve como STRING
    en vez de como literal JSON (ej. {"coherent": "false"} en vez de
    {"coherent": false}) -- ambas formas son JSON valido, pero json.loads
    las deja como tipos de Python distintos (str vs bool/None). El codigo
    que combina el veredicto de reglas con el del LLM compara con `is False`
    / `is None`, que solo es cierto para el bool/None real: una respuesta
    correcta del modelo pero con el booleano entre comillas se ignoraba en
    silencio, como si el modelo nunca hubiera detectado nada.
    """
    if "coherent" in parsed and isinstance(parsed["coherent"], str):
        key = parsed["coherent"].strip().lower()
        # Un valor de "coherent" que no sea true/false/null (ej. "medium",
        # visto en produccion: el modelo confunde el campo de confianza con
        # el de coherencia) no es un true/false valido -- tratarlo como
        # indeterminado (None) en vez de dejar pasar la cadena original,
        # que nunca es `is False`/`is True` y quedaria silenciosamente
        # invisible para el codigo que combina el veredicto.
        parsed = {**parsed, "coherent": _STRING_BOOL_MAP.get(key)}
    return parsed


def parse_json_response(raw_text: str) -> Any:
    """
    Intenta extraer JSON incluso si el modelo anade prose o fences markdown.
    """
    text = (raw_text or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)

    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    for start_char in ("{", "["):
        start = text.find(start_char)
        if start == -1:
            continue
        try:
            parsed, _ = JSON_DECODER.raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("No JSON object found", text, 0)


def _json_candidates(text: str) -> list[str]:
    candidates = [text]

    if text.startswith("```") and text.endswith("```"):
        stripped = text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidates.append(text[start_obj : end_obj + 1])

    start_list = text.find("[")
    end_list = text.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        candidates.append(text[start_list : end_list + 1])

    unique_candidates: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip()
        if cleaned and cleaned not in unique_candidates:
            unique_candidates.append(cleaned)
    return unique_candidates
