"""
Extraccion tolerante para campos estructurados afectados por errores OCR.

No modifica el texto OCR original. Devuelve candidatos normalizados con
evidencia para que las reglas puedan dejar trazabilidad en el reporte.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import unicodedata


DATE_WITH_SEPARATORS = re.compile(r"\b(\d{1,2})[/:-](\d{1,2})[/:-](\d{2,4})\b")
MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
DATE_LONG_ES = re.compile(
    r"\b(\d{1,2})\s+de\s+(" + "|".join(MESES_ES) + r")\s+de\s+(\d{4})\b",
    flags=re.IGNORECASE,
)
COMPACT_DATE_CONTEXT = re.compile(
    r"(?:fecha|f[e3]cha|f[ée]c[ha]{1,2})\D{0,20}(\d{8,10})",
    flags=re.IGNORECASE,
)
DNI_CONTEXT = re.compile(
    r"(?:dni|nie|dmi|dn[il1]|n[il1]e|on[il1]n[il1]e|onun[il1]e)\W{0,8}"
    r"([A-Z0-9][A-Z0-9]{7}[A-Z])",
    flags=re.IGNORECASE,
)
DNI_LOOSE = re.compile(r"\b([A-Z0-9][A-Z0-9]{7}[A-Z])\b", flags=re.IGNORECASE)

OCR_DIGIT_MAP = str.maketrans(
    {
        "B": "8",
        "D": "0",
        "G": "6",
        "I": "1",
        "L": "1",
        "O": "0",
        "Q": "0",
        "S": "5",
        "Z": "2",
    }
)


@dataclass(frozen=True)
class NormalizedCandidate:
    """Candidato extraido sin perder el texto OCR original."""

    value: str
    original: str
    method: str
    confidence: str = "media"

    @property
    def details(self) -> str:
        return (
            f"Normalizado={self.value}; original={self.original}; "
            f"metodo={self.method}; confianza={self.confidence}."
        )


def find_dni_nie(text: str) -> NormalizedCandidate | None:
    """Busca DNI/NIE con tolerancia a confusiones OCR comunes."""
    cleaned = _compact_identifier_text(text)

    for pattern, method, confidence in (
        (DNI_CONTEXT, "dni_contexto_ocr", "media"),
        (DNI_LOOSE, "dni_patron_laxo", "baja"),
    ):
        for match in pattern.finditer(cleaned):
            original = match.group(1).upper()
            normalized = _normalize_dni_candidate(original)
            if _is_dni_or_nie_shape(normalized):
                return NormalizedCandidate(
                    value=normalized,
                    original=original,
                    method=method,
                    confidence=confidence,
                )

    return None


def find_date(text: str) -> NormalizedCandidate | None:
    """Busca fechas con separadores o en formato compacto cerca de 'Fecha'."""
    source = text or ""

    # Segunda pasada con digitos OCR confundibles (O/S/I/L/...) traducidos a
    # su digito real: solo se usa si la busqueda literal no encuentra nada,
    # y el patron sigue exigiendo separadores reales, asi que el riesgo de
    # inventar una fecha en texto normal es bajo.
    for candidate_source, confidence in ((source, "alta"), (source.translate(OCR_DIGIT_MAP), "media")):
        for match in DATE_WITH_SEPARATORS.finditer(candidate_source):
            candidate = _build_date(match.group(1), match.group(2), match.group(3))
            if candidate:
                return NormalizedCandidate(
                    value=candidate,
                    original=match.group(0),
                    method="fecha_con_separadores",
                    confidence=confidence,
                )

    for match in DATE_LONG_ES.finditer(source):
        month_num = MESES_ES[match.group(2).lower()]
        candidate = _build_date(match.group(1), str(month_num), match.group(3))
        if candidate:
            return NormalizedCandidate(
                value=candidate,
                original=match.group(0),
                method="fecha_larga_es",
                confidence="alta",
            )

    normalized_source = _normalize_for_search(source)
    for match in COMPACT_DATE_CONTEXT.finditer(normalized_source):
        raw = match.group(1)
        candidate = _normalize_compact_date(raw)
        if candidate is not None:
            return NormalizedCandidate(
                value=candidate,
                original=raw,
                method="fecha_compacta_con_contexto",
                confidence="media",
            )

    return None


def has_tolerant_label(patterns: list[str] | str, text: str) -> NormalizedCandidate | None:
    """
    Busca etiquetas cortas aplicando normalizacion basica y variantes OCR.

    Esta funcion es deliberadamente conservadora: no inventa etiquetas si no hay
    evidencia textual; solo tolera acentos, signos y caracteres confundidos.
    """
    normalized_text = _normalize_for_search(text)
    if isinstance(patterns, str):
        patterns = [patterns]

    for pattern in patterns:
        normalized_pattern = _normalize_for_search(pattern)
        if normalized_pattern in normalized_text:
            return NormalizedCandidate(
                value=pattern,
                original=pattern,
                method="etiqueta_normalizada",
                confidence="alta",
            )

        loose_pattern = _label_to_loose_regex(normalized_pattern)
        match = re.search(loose_pattern, normalized_text)
        if match:
            return NormalizedCandidate(
                value=pattern,
                original=match.group(0),
                method="etiqueta_ocr_laxa",
                confidence="media",
            )

    return None


def text_tail(text: str, max_chars: int = 1200) -> str:
    """Devuelve el tramo final del texto para buscar firmas/fechas."""
    source = text or ""
    if len(source) <= max_chars:
        return source
    return source[-max_chars:]


def _compact_identifier_text(text: str) -> str:
    normalized = _normalize_for_search(text).upper()
    return re.sub(r"[^A-Z0-9/:.\-\s]", " ", normalized)


def _normalize_dni_candidate(candidate: str) -> str:
    raw = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    if len(raw) != 9:
        return raw

    if raw[0] in {"X", "Y", "Z"}:
        body = raw[1:8].translate(OCR_DIGIT_MAP)
        return f"{raw[0]}{body}{raw[8]}"

    number = raw[:8].translate(OCR_DIGIT_MAP)
    return f"{number}{raw[8]}"


def _is_dni_or_nie_shape(value: str) -> bool:
    return bool(
        re.fullmatch(r"\d{8}[A-Z]", value)
        or re.fullmatch(r"[XYZ]\d{7}[A-Z]", value)
    )


def _build_date(day: str, month: str, year: str) -> str | None:
    if len(year) == 2:
        year = f"20{year}"

    try:
        parsed = datetime(int(year), int(month), int(day))
    except ValueError:
        return None

    if parsed.year < 1900 or parsed.year > 2100:
        return None

    return parsed.strftime("%d/%m/%Y")


def _normalize_compact_date(raw: str) -> str | None:
    candidates = [raw]
    if len(raw) > 8:
        candidates.extend(raw[:index] + raw[index + 1 :] for index in range(len(raw)))

    for candidate in candidates:
        if len(candidate) != 8:
            continue
        parsed = _build_date(candidate[:2], candidate[2:4], candidate[4:])
        if parsed is not None:
            return parsed

    return None


def _normalize_for_search(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", without_accents)


def _label_to_loose_regex(pattern: str) -> str:
    escaped_words = []
    for word in pattern.split():
        escaped_words.append("".join(_loose_label_char(char) for char in word))
    return r"(?<!\w)" + r"\W+".join(escaped_words) + r"(?=$|\W)"


def _loose_label_char(char: str) -> str:
    if char == "i":
        return "[i1l]"
    if char == "l":
        return "[l1!|]"
    if char == "o":
        return "[o0]"
    return re.escape(char)
