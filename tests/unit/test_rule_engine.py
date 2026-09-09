from src.rules.engine import RuleEngine
from src.rules.normalization import find_date, find_dni_nie, has_tolerant_label


class FakeDocumentOCR:
    def __init__(self, zones: dict[str, str]) -> None:
        self.zones = zones
        self.full_text = " ".join(zones.values())

    def get_text_by_zone(self, zone: str) -> str:
        return self.zones.get(zone, "")


def build_protocol() -> dict:
    return {
        "id_protocolo": "CI-TEST",
        "version": "1.0",
        "proceso_sanitario": "general",
        "tipo_documento": "consentimiento_informado",
        "fecha_vigencia": "2024-01-01",
        "campos_obligatorios": [
            {
                "nombre_zona": "identificacion_paciente",
                "tipo_contenido": "texto",
                "zona_documento": "header",
                "descripcion": "Paciente identificado",
                "patrones_esperados": ["dni"],
            },
            {
                "nombre_zona": "firma_paciente",
                "tipo_contenido": "firma",
                "zona_documento": "signature",
                "descripcion": "Firma del paciente",
                "min_count": 1,
            },
            {
                "nombre_zona": "fecha_firma",
                "tipo_contenido": "fecha",
                "zona_documento": "signature",
                "descripcion": "Fecha de firma",
                "patron_regex": r"\d{1,2}/\d{1,2}/\d{4}",
            },
        ],
        "reglas_contenido": [
            {
                "id_regla": "RC-001",
                "zona_documento": "body",
                "tipo": "keyword",
                "patron": ["riesgo"],
                "obligatoria": True,
                "mensaje_error": "Falta la mencion de riesgos",
            }
        ],
        "reglas_coherencia": [
            {
                "id_regla": "RCH-001",
                "tipo": "campo_presente_si",
                "campo_origen": "firma_paciente",
                "campo_destino": "fecha_firma",
                "condicion": "Si hay firma debe haber fecha",
                "mensaje_error": "Hay firma pero no fecha",
            }
        ],
    }


def test_rule_engine_returns_valid_when_all_checks_pass() -> None:
    protocol = build_protocol()
    document = FakeDocumentOCR(
        {
            "header": "Nombre: Maria Garcia DNI: 12345678A",
            "body": "Se explican los riesgos y beneficios del procedimiento.",
            "signature": "Fecha: 15/03/2025",
        }
    )

    result = RuleEngine(protocol).validate(
        document,
        signatures=[{"x": 10, "y": 10}],
        checkboxes=[],
    )

    assert result.verdict == "valid"
    assert result.checks_passed == result.checks_total
    assert result.issues == []


def test_rule_engine_flags_missing_date_when_signature_exists() -> None:
    protocol = build_protocol()
    document = FakeDocumentOCR(
        {
            "header": "Nombre: Maria Garcia DNI: 12345678A",
            "body": "Se explican los riesgos del procedimiento.",
            "signature": "Firma del paciente",
        }
    )

    result = RuleEngine(protocol).validate(
        document,
        signatures=[{"x": 10, "y": 10}],
        checkboxes=[],
    )

    # Falta la fecha de firma (campo obligatorio): un campo obligatorio ausente
    # nunca permite "valid" (ver hallazgo 10 en apuntes_hallazgos.md), pero
    # tampoco degrada por si solo a "inconsistent" -- eso requeriria una
    # incoherencia semantica real, no solo puntaje bajo por acumulacion de
    # campos ausentes (ver hallazgo 14, que corrigio exactamente este patron).
    assert result.verdict == "incomplete"
    assert any(issue.rule_id == "CAMPO-fecha_firma" for issue in result.issues)
    assert any(issue.rule_id == "RCH-001" for issue in result.issues)


def test_find_dni_nie_normalizes_common_ocr_errors() -> None:
    candidate = find_dni_nie("ONUNIE. B2069897X")

    assert candidate is not None
    assert candidate.value == "82069897X"
    assert candidate.original == "B2069897X"


def test_find_date_accepts_compact_date_with_context() -> None:
    candidate = find_date("Fecha: 240772024 Lugar: Madrid")

    assert candidate is not None
    assert candidate.value == "24/07/2024"
    assert candidate.original == "240772024"


def test_tolerant_label_accepts_common_final_l_ocr_error() -> None:
    candidate = has_tolerant_label(
        "firma del profesional",
        "Firma del paciente: Firma del profesiona!",
    )

    assert candidate is not None
    assert candidate.value == "firma del profesional"


def test_rule_engine_uses_tolerant_extractors_without_changing_ocr_text() -> None:
    protocol = build_protocol()
    protocol["campos_obligatorios"].append(
        {
            "nombre_zona": "firma_profesional",
            "tipo_contenido": "firma",
            "zona_documento": "signature",
            "descripcion": "Firma del profesional",
            "min_count": 1,
            "patrones_esperados": ["firma del profesional"],
        }
    )
    protocol["campos_obligatorios"][1]["patrones_esperados"] = ["firma del paciente"]
    # NOTA: no se anaden aqui reglas_contenido tipo "regex" para el DNI con
    # error de OCR ("B" por "8") ni para la fecha compacta sin separadores
    # -- esa tolerancia solo existe en el camino de deteccion de
    # campo_obligatorio (find_dni_nie/find_date, ver hallazgos de
    # normalizacion), no en reglas_contenido tipo "regex", que hace un
    # match literal sin ninguna correccion. Anadir esos regex aqui (como
    # tenia esta prueba antes) los hacia fallar siempre por diseno,
    # independientemente de si la extraccion tolerante funcionaba bien o
    # no -- esa cobertura ya la dan test_find_dni_nie_normalizes_common_ocr_errors
    # y test_find_date_accepts_compact_date_with_context por separado.
    document = FakeDocumentOCR(
        {
            "header": "Nombre: Sanchez Pro, Alberto ONUNIE. B2069897X",
            "body": "Se explican los riesgos del procedimiento.",
            # La fecha y las etiquetas de firma van en su propia zona
            # ("signature", la misma que declaran fecha_firma/firma_*) en
            # vez de en "legal": esa zona se excluye a proposito de la
            # busqueda de fechas de respaldo (ver comentario extenso en
            # engine.py, hallazgo 19) porque ahi vive de forma consistente
            # la fecha de emision del aviso legal, no la fecha de firma
            # real -- ponerla en "legal" aqui solo probaria esa exclusion
            # deliberada, no la extraccion tolerante que este test quiere
            # verificar.
            "legal": "Documento sujeto a la normativa de proteccion de datos vigente.",
            "signature": (
                "Fecha: 240772024 Lugar: Madrid "
                "Firma del paciente: Firma del profesional: E ES]"
            ),
        }
    )

    result = RuleEngine(protocol).validate(
        document,
        signatures=[{"x": 10, "y": 10}, {"x": 100, "y": 10}],
        checkboxes=[],
    )

    assert result.verdict == "valid"
    assert "ONUNIE. B2069897X" in document.full_text
