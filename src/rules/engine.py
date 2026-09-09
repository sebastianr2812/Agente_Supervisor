"""
Motor de validacion basado en reglas YAML.
Carga protocolos y evalua documentos contra sus reglas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
import re
import unicodedata

import yaml

from .normalization import OCR_DIGIT_MAP, find_date, find_dni_nie, has_tolerant_label, text_tail

if TYPE_CHECKING:
    from src.ocr.extractor import DocumentOCR


DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
REQUIRED_PROTOCOL_KEYS = {
    "id_protocolo",
    "version",
    "proceso_sanitario",
    "tipo_documento",
    "fecha_vigencia",
    "campos_obligatorios",
    "reglas_contenido",
    "reglas_coherencia",
}

# Umbral del puntaje ponderado (0-100) por debajo del cual el veredicto ya
# no puede ser "valid". Por debajo de este umbral el veredicto es
# "incomplete", salvo que una regla de incoherencia fuerte (ver
# TIPOS_INCOHERENCIA_FUERTE) lo escale explicitamente a "inconsistent" --
# un puntaje bajo por si solo (ej. varios campos obligatorios ausentes) no
# implica incoherencia semantica, solo ausencia de datos.
UMBRAL_VALIDO = 85.0

# Un criterio con este peso o mayor se considera "critico": si falla, el
# documento nunca puede quedar como "valid" sin importar el puntaje total.
PESO_CRITICO = 3

# Tipos de regla que son una senal determinista y ya verificada (0 falsos
# positivos por diseno, ver hallazgo 12) de que el documento tiene una
# incoherencia real, no solo un dato faltante. A diferencia del resto de
# reglas (que solo restan puntaje), si UNA de estas falla el veredicto debe
# ser "inconsistent" directamente -- sin este escalado, el puntaje ponderado
# seguia cayendo dentro del rango de "incomplete" (60-85) aunque la regla
# hubiera detectado correctamente la incoherencia, porque el resto de
# comprobaciones del documento (campos presentes, etc.) seguian aportando
# suficiente puntaje para no bajar de 60.
TIPOS_INCOHERENCIA_FUERTE = {
    "rango_fisiologico",
    "ausencia_regex",
    "formato_numerico_requerido",
    "checkboxes_completos",
    "fecha_posterior",
    "identidad_duplicada",
    "ausencia_contradicha",
}

def _windowed_text_near_pattern(
    blocks: list[Any],
    patterns: list[str],
    before: int = 0,
    after: int = 12,
) -> Optional[str]:
    """
    Copia local de la misma utilidad de src/agent/supervisor.py (no se
    importa desde alli para evitar un import circular: supervisor.py ya
    importa de este modulo). Devuelve una ventana de texto alrededor del
    primer bloque que coincide con alguno de los patrones, en vez de toda
    la zona documental.
    """
    lowered_patterns = [p.lower() for p in patterns]
    for i, block in enumerate(blocks):
        text = getattr(block, "text", "") or ""
        if any(p in text.lower() for p in lowered_patterns):
            start = max(0, i - before)
            end = min(len(blocks), i + after)
            return " ".join(getattr(b, "text", "") for b in blocks[start:end])
    return None


def _windowed_value_until_next_label(
    blocks: list[Any],
    patterns: list[str],
    max_after: int = 10,
    before: int = 2,
) -> Optional[str]:
    """
    Variante de _windowed_text_near_pattern para pares de fechas que
    comparten zona: en vez de una ventana de ancho fijo, se extiende el
    valor bloque a bloque desde la etiqueta encontrada HASTA el bloque
    anterior a la proxima etiqueta (heuristica: un bloque que termina en
    ":" es la etiqueta de OTRO campo, ya que el generador dibuja siempre
    "Etiqueta:" pegado). Esto evita dos fallos opuestos de una ventana de
    ancho fijo: cortar una fecha larga en texto ("7 de febrero de 2026",
    5 bloques) o, si el propio valor esta degradado por OCR mas alla de
    reconocimiento, seguir leyendo hasta la fecha LIMPIA del campo
    siguiente y confundirla con la propia.

    `before` antepone unos pocos bloques ANTERIORES al ancla (ej. "Fecha",
    "de" antes de "nacimiento:"): sin ellos, la busqueda de fecha compacta
    tolerante (find_date -> COMPACT_DATE_CONTEXT) nunca encuentra la
    palabra "fecha" que exige como contexto, porque el ancla especifica
    por campo (ej. "nacimiento") no la incluye. El retroceso se corta en
    el primer bloque con un digito: las etiquetas nunca llevan digitos,
    solo los VALORES -- sin este corte, una etiqueta de una sola palabra
    antes del ancla (ej. "Fecha programada:") deja que `before` alcance el
    VALOR del campo anterior (ej. la fecha de nacimiento ya impresa) y lo
    confunda con el propio.
    """
    lowered_patterns = [p.lower() for p in patterns]
    for i, block in enumerate(blocks):
        text = getattr(block, "text", "") or ""
        if any(p in text.lower() for p in lowered_patterns):
            lead: list[str] = []
            for b in reversed(blocks[max(0, i - before) : i]):
                b_text = getattr(b, "text", "") or ""
                if any(ch.isdigit() for ch in b_text):
                    break
                lead.append(b_text)
            lead.reverse()
            collected = lead + [text]
            for b in blocks[i + 1 : i + 1 + max_after]:
                b_text = getattr(b, "text", "") or ""
                if b_text.rstrip().endswith(":"):
                    break
                collected.append(b_text)
            return " ".join(collected)
    return None


DEFAULT_WEIGHT_BY_CONTENT_TYPE = {
    "firma": 3,
    "fecha": 2,
    "texto": 2,
    "casilla": 1,
}
DEFAULT_WEIGHT_COHERENCIA = 3


@dataclass
class ValidationIssue:
    """Una incidencia detectada en la validacion."""

    rule_id: str
    severity: str
    message: str
    zone: Optional[str] = None
    details: Optional[str] = None


@dataclass
class ScoreBreakdownItem:
    """Contribucion de un criterio individual al puntaje ponderado."""

    criterio: str
    categoria: str
    peso: float
    aprobado: bool
    detalle: str = ""
    tipo: str = ""


@dataclass
class ValidationResult:
    """Resultado de la validacion de un documento."""

    verdict: str
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0
    weighted_score: float = 0.0
    score_breakdown: list[ScoreBreakdownItem] = field(default_factory=list)

    @property
    def score(self) -> float:
        if self.checks_total == 0:
            return 0.0
        return self.checks_passed / self.checks_total


@dataclass
class FieldValidation:
    """Estado de presencia de un campo del protocolo."""

    present: bool
    zone: str
    details: Optional[str] = None


class ProtocolLoader:
    """Carga y valida protocolos YAML."""

    def __init__(
        self,
        protocols_dir: str = "configs/protocolos",
        schema_path: str = "configs/schema_protocolo.yaml",
    ) -> None:
        self.protocols_dir = Path(protocols_dir)
        self.schema_path = Path(schema_path)
        self.protocols: dict[str, dict[str, Any]] = {}

    def load_schema(self) -> dict[str, Any]:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"No existe el esquema: {self.schema_path}")
        with self.schema_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def load_all(self) -> dict[str, dict[str, Any]]:
        if not self.protocols_dir.exists():
            raise FileNotFoundError(
                f"No existe el directorio de protocolos: {self.protocols_dir}"
            )

        loaded: dict[str, dict[str, Any]] = {}
        for yaml_file in sorted(self.protocols_dir.glob("*.yaml")):
            protocol = self.load_file(yaml_file)
            loaded[protocol["id_protocolo"]] = protocol

        self.protocols = loaded
        return self.protocols

    def load_file(self, path: str | Path) -> dict[str, Any]:
        yaml_path = Path(path)
        with yaml_path.open("r", encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle) or {}

        self._validate_protocol_shape(protocol, yaml_path)
        return protocol

    def get_protocol(self, protocol_id: str) -> Optional[dict[str, Any]]:
        if not self.protocols:
            self.load_all()
        return self.protocols.get(protocol_id)

    def _validate_protocol_shape(
        self,
        protocol: dict[str, Any],
        yaml_path: Path,
    ) -> None:
        missing = sorted(REQUIRED_PROTOCOL_KEYS - set(protocol))
        if missing:
            raise ValueError(
                f"El protocolo {yaml_path} no contiene las claves requeridas: {missing}"
            )

        if not isinstance(protocol.get("campos_obligatorios"), list):
            raise ValueError(
                f"El protocolo {yaml_path} debe definir 'campos_obligatorios' como lista"
            )
        if not isinstance(protocol.get("reglas_contenido"), list):
            raise ValueError(
                f"El protocolo {yaml_path} debe definir 'reglas_contenido' como lista"
            )
        if not isinstance(protocol.get("reglas_coherencia"), list):
            raise ValueError(
                f"El protocolo {yaml_path} debe definir 'reglas_coherencia' como lista"
            )


class RuleEngine:
    """Evalua un documento OCR contra un protocolo de validacion."""

    def __init__(self, protocol: dict[str, Any]) -> None:
        self.protocol = protocol

    def validate(
        self,
        doc: "DocumentOCR | Any",
        signatures: Optional[list[dict[str, Any]]] = None,
        checkboxes: Optional[list[dict[str, Any]]] = None,
        vl_rescue: Optional[Any] = None,
        vl_date_rescue: Optional[Any] = None,
        vl_identity_rescue: Optional[Any] = None,
    ) -> ValidationResult:
        """
        `vl_rescue`: funcion opcional `(nombre_campo: str) -> bool | None`
        para complementar el OCR en campos donde se ha verificado que un
        analisis visual directo (VL) ayuda -- ver hallazgo 18: el OCR de
        "fecha_servicio" en DLR-001-ES falla con recall 0,42, mientras que
        pedirle a VL que enumere las fechas visibles y filtrar por
        etiqueta en codigo acerto 30/30 en el mismo campo. Solo se llama
        cuando el OCR marca el campo como ausente (para no arriesgar la
        fiabilidad ya verificada del OCR en los campos donde funciona
        bien, ver limitacion de especificidad "encontrado: false" de VL en
        el hallazgo 8/16); si devuelve True, el campo se marca presente.
        Nunca se llama si `vl_rescue` es None (comportamiento identico al
        anterior, sin este parametro).

        `vl_date_rescue`: funcion opcional `() -> list[dict]` (sin
        argumentos, mismo principio que `vl_rescue` extendido a la regla
        `fecha_posterior`) que devuelve TODAS las fechas que VL lee
        directamente de la imagen junto con su etiqueta/contexto (ver
        QwenVLClient.check_labeled_date_present, ya validado 30/30 para
        "fecha_servicio"). Se usa como respaldo cuando `_date_for` no logra
        extraer una fecha del texto OCR para `fecha_nacimiento` o
        `fecha_documento` -- el caso mas comun de fallo en la regla
        `fecha_posterior` es que el propio valor numerico este degradado
        mas alla de lo que un regex tolerante puede reparar (dígitos
        fusionados, separador perdido), y VL leyendo los pixeles
        directamente no depende de esa tokenizacion. Como con `vl_rescue`,
        solo actua cuando el OCR ya fallo; nunca sustituye una fecha que el
        motor de reglas ya extrajo con exito. Se espera que el propio
        callable memorice el resultado de la llamada a VL (una sola
        invocacion por documento, reutilizada para ambos campos de fecha).

        `vl_identity_rescue`: mismo principio que `vl_date_rescue`, para la
        regla `identidad_duplicada`. Funcion opcional `() -> list[dict]`
        que devuelve TODOS los identificadores de paciente que VL lee
        directamente de la imagen (ver
        QwenVLClient.list_patient_identifiers). Se usa solo cuando el
        regex sobre el texto OCR no encuentra mas de un valor distinto
        (inconclusivo): si VL, leyendo los pixeles, SI encuentra 2+
        valores distintos, se escala la incoherencia. Nunca al reves --
        una deteccion ya confirmada por el regex de OCR nunca se descarta
        por este rescate.
        """
        signatures = signatures or []
        checkboxes = checkboxes or []

        issues: list[ValidationIssue] = []
        checks_passed = 0
        checks_total = 0
        weighted_passed = 0.0
        weighted_total = 0.0
        breakdown: list[ScoreBreakdownItem] = []
        field_status: dict[str, FieldValidation] = {}

        for field_definition in self.protocol.get("campos_obligatorios", []):
            # `informational: true` (ver hallazgo sobre campos auxiliares de
            # fecha_posterior/identidad_duplicada): el campo SE COMPRUEBA y
            # queda disponible en field_status para que otras reglas lo usen
            # (ej. comparar fecha_nacimiento vs fecha_documento), pero su
            # ausencia NO cuenta para la puntuacion ni bloquea "valid" por
            # si sola (a diferencia de un campo obligatorio real). Sin esto,
            # cada campo auxiliar nuevo anadido solo para alimentar una
            # regla de coherencia se convertia en un punto de fallo
            # independiente para el veredicto "valid" (via el fix del
            # hallazgo de "cualquier campo obligatorio ausente bloquea
            # valid"), aunque su unico proposito fuera servir de insumo a
            # otra regla, no representar informacion central del documento.
            is_informational = bool(field_definition.get("informational"))

            if not is_informational:
                checks_total += 1
            present, details = self._check_required_field(
                field_definition,
                doc,
                signatures,
                checkboxes,
            )
            field_name = field_definition["nombre_zona"]
            if not present and vl_rescue is not None:
                vl_present = vl_rescue(field_name)
                if vl_present:
                    present = True
                    details = f"Ausente por OCR, rescatado por VL. ({details})"
            field_status[field_name] = FieldValidation(
                present=present,
                zone=field_definition["zona_documento"],
                details=details,
            )

            peso = self._field_weight(field_definition)
            categoria = "campo_informativo" if is_informational else "campo_obligatorio"
            if not is_informational:
                weighted_total += peso
            breakdown.append(
                ScoreBreakdownItem(
                    criterio=f"CAMPO-{field_name}",
                    categoria=categoria,
                    peso=peso,
                    aprobado=present,
                    detalle=details,
                )
            )

            if present:
                if not is_informational:
                    checks_passed += 1
                    weighted_passed += peso
                continue

            if not is_informational:
                issues.append(
                    ValidationIssue(
                        rule_id=f"CAMPO-{field_name}",
                        severity="error",
                        message=(
                            f"Campo obligatorio ausente o incompleto: "
                            f"{field_definition['descripcion']}"
                        ),
                        zone=field_definition["zona_documento"],
                        details=details,
                    )
                )

        for rule in self.protocol.get("reglas_contenido", []):
            checks_total += 1
            found, details = self._check_content_rule(rule, doc)

            peso = self._content_rule_weight(rule)
            weighted_total += peso
            breakdown.append(
                ScoreBreakdownItem(
                    criterio=rule["id_regla"],
                    categoria="regla_contenido",
                    peso=peso,
                    aprobado=found,
                    detalle=details,
                    tipo=rule.get("tipo", ""),
                )
            )

            if found:
                checks_passed += 1
                weighted_passed += peso
                continue

            if rule.get("obligatoria", True):
                issues.append(
                    ValidationIssue(
                        rule_id=rule["id_regla"],
                        severity="error",
                        message=rule["mensaje_error"],
                        zone=rule["zona_documento"],
                        details=details,
                    )
                )

        for rule in self.protocol.get("reglas_coherencia", []):
            should_count, passed, issue = self._check_coherence_rule(
                rule,
                field_status,
                doc,
                checkboxes,
                vl_date_rescue=vl_date_rescue,
                vl_identity_rescue=vl_identity_rescue,
            )
            if not should_count:
                continue

            checks_total += 1
            peso = self._coherence_rule_weight(rule)
            weighted_total += peso

            if passed:
                detalle = f"Condicion cumplida: {rule.get('condicion', '')}"
            elif issue is not None:
                detalle = issue.message + (f" {issue.details}" if issue.details else "")
            else:
                detalle = "Condicion no cumplida."

            breakdown.append(
                ScoreBreakdownItem(
                    criterio=rule["id_regla"],
                    categoria="regla_coherencia",
                    peso=peso,
                    aprobado=passed,
                    detalle=detalle,
                    tipo=rule.get("tipo", ""),
                )
            )

            if passed:
                checks_passed += 1
                weighted_passed += peso
            elif issue is not None:
                issues.append(issue)

        weighted_score = (weighted_passed / weighted_total * 100) if weighted_total > 0 else 0.0

        # NOTA: el umbral bajo (< UMBRAL_INCOMPLETO) ya NO decide por si solo
        # el veredicto "inconsistent" -- ver mas abajo `incoherencia_fuerte`.
        # Antes, un puntaje muy bajo caia directo en "inconsistent" sin mas
        # comprobacion, lo que mezclaba dos cosas distintas: un documento con
        # VARIOS campos obligatorios ausentes (que acumulan penalizacion por
        # el campo en si mas las reglas de contenido/coherencia que dependen
        # de ese mismo campo) puede caer bajo el umbral solo por ausencia de
        # datos, sin ninguna incoherencia semantica real -- verificado en
        # casos con 2 campos obligatorios ausentes que quedaban mal
        # etiquetados como "inconsistent" en vez de "incomplete". Un puntaje
        # bajo por muchos datos ausentes sigue siendo, como mucho,
        # "incomplete"; solo una senal explicita de incoherencia (ver abajo)
        # puede llevar el veredicto a "inconsistent".
        if weighted_score >= UMBRAL_VALIDO:
            verdict = "valid"
        else:
            verdict = "incomplete"

        critical_failure = any(
            not item.aprobado and item.peso >= PESO_CRITICO for item in breakdown
        )
        if critical_failure and verdict == "valid":
            verdict = "incomplete"

        # Un campo_obligatorio ausente es, por definicion, un documento
        # incompleto, sin importar su peso: el peso solo debia graduar
        # CUAN incompleto (afectando weighted_score), no si se le puede seguir
        # llamando "valido" porque el resto del documento compensa la
        # puntuacion. Antes de esta correccion, un campo obligatorio de
        # peso bajo (ej. 1-2) podia fallar, quedar correctamente listado
        # en `issues`, y aun asi el veredicto salir "valid" si el
        # weighted_score superaba UMBRAL_VALIDO (85) por el resto de
        # comprobaciones aprobadas -- verificado que esto explicaba el
        # 83% de los documentos "incomplete" mal clasificados como
        # "valid" en la evaluacion integral.
        missing_mandatory_field = any(
            not item.aprobado and item.categoria == "campo_obligatorio"
            for item in breakdown
        )
        if missing_mandatory_field and verdict == "valid":
            verdict = "incomplete"

        # Mismo razonamiento que el fix anterior, un nivel mas arriba: si
        # una regla de un tipo pensado especificamente para detectar una
        # incoherencia real (no un dato ausente) falla, el veredicto debe
        # ser "inconsistent" directamente, sin importar el weighted_score.
        # Antes de esto, estas reglas SI se evaluaban y SI quedaban
        # registradas como fallidas en `issues`, pero como el veredicto se
        # decidia solo por el umbral del puntaje ponderado, el resto de
        # comprobaciones del documento (campos presentes, otras reglas)
        # aportaba puntaje suficiente para quedarse en el rango "incomplete"
        # (60-85) sin llegar nunca a "inconsistent" -- la deteccion
        # funcionaba pero el veredicto final la ignoraba.
        incoherencia_fuerte = any(
            not item.aprobado and item.tipo in TIPOS_INCOHERENCIA_FUERTE
            for item in breakdown
        )
        if incoherencia_fuerte:
            verdict = "inconsistent"

        return ValidationResult(
            verdict=verdict,
            issues=issues,
            checks_passed=checks_passed,
            checks_total=checks_total,
            weighted_score=round(weighted_score, 2),
            score_breakdown=breakdown,
        )

    def _field_weight(self, field_definition: dict[str, Any]) -> float:
        if "peso" in field_definition:
            return float(field_definition["peso"])
        content_type = field_definition.get("tipo_contenido", "texto")
        return float(DEFAULT_WEIGHT_BY_CONTENT_TYPE.get(content_type, 2))

    def _content_rule_weight(self, rule: dict[str, Any]) -> float:
        if "peso" in rule:
            return float(rule["peso"])
        return 2.0 if rule.get("obligatoria", True) else 1.0

    def _coherence_rule_weight(self, rule: dict[str, Any]) -> float:
        if "peso" in rule:
            return float(rule["peso"])
        return float(DEFAULT_WEIGHT_COHERENCIA)

    def _check_required_field(
        self,
        field_definition: dict[str, Any],
        doc: "DocumentOCR | Any",
        signatures: list[dict[str, Any]],
        checkboxes: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        content_type = field_definition["tipo_contenido"]
        zone = field_definition["zona_documento"]
        zone_text = self._get_zone_text(doc, zone)
        full_text = getattr(doc, "full_text", "") or ""

        if content_type == "texto":
            min_chars = int(field_definition.get("min_chars", 1))
            expected_patterns = field_definition.get("patrones_esperados", [])
            field_name = field_definition.get("nombre_zona", "")
            candidate_texts = [
                ("zona", zone_text),
                ("documento_completo", full_text),
            ]
            best_reason = "La zona no contiene texto OCR."

            for scope, text in candidate_texts:
                if not text.strip():
                    continue
                if len(text.strip()) < min_chars:
                    best_reason = (
                        f"Texto demasiado corto en {scope}: "
                        f"{len(text.strip())} chars."
                    )
                    continue
                if expected_patterns and not self._keyword_match(expected_patterns, text):
                    dni_candidate = None
                    if self._expects_identifier(field_name, expected_patterns):
                        dni_candidate = find_dni_nie(text)
                    if dni_candidate is not None:
                        return (
                            True,
                            f"Identificador tolerante encontrado en {scope}. "
                            f"{dni_candidate.details}",
                        )

                    best_reason = (
                        f"No se encontraron los patrones esperados en {scope}."
                    )
                    continue
                return True, f"Texto encontrado usando {scope}."

            return False, best_reason

        if content_type == "firma":
            min_count = int(field_definition.get("min_count", 1))
            label_patterns = field_definition.get("patrones_esperados", [])
            label_found = True
            label_details = "sin etiqueta requerida"
            if label_patterns:
                label_candidate = has_tolerant_label(
                    label_patterns,
                    " ".join([zone_text, text_tail(full_text)]),
                )
                label_found = label_candidate is not None
                label_details = (
                    label_candidate.details
                    if label_candidate
                    else "No se encontro etiqueta OCR en zona/final del documento."
                )

            present = len(signatures) >= min_count and label_found
            details = (
                f"Firmas detectadas: {len(signatures)}. "
                f"Etiqueta encontrada: {label_found}. {label_details}"
            )
            return present, details

        if content_type == "casilla":
            expected_value = field_definition.get("valor_esperado")
            if expected_value is None:
                return bool(checkboxes), f"Casillas detectadas: {len(checkboxes)}."

            normalized_expected = self._normalize_text(str(expected_value))
            if normalized_expected in {"true", "si", "marcada", "checked"}:
                present = any(item.get("checked") for item in checkboxes)
            else:
                present = any(not item.get("checked") for item in checkboxes)
            return present, f"Casillas detectadas: {len(checkboxes)}."

        if content_type == "fecha":
            pattern = field_definition.get("patron_regex", DATE_PATTERN.pattern)
            found_in_zone = bool(re.search(pattern, zone_text))
            if found_in_zone:
                return True, "Fecha encontrada en la zona de firma."

            date_candidate = find_date(zone_text)
            if date_candidate is not None:
                return True, f"Fecha tolerante encontrada en zona. {date_candidate.details}"

            # El rescate final buscaba en el texto completo del documento
            # (incluida su cola, cerca del pie de pagina) sin excluir
            # ninguna zona. Se detecto que eso encuentra sistematicamente
            # la fecha de emision/informe del aviso legal -un dato real,
            # pero de un campo distinto al buscado- y la confunde con la
            # fecha de servicio, causando falsos positivos en documentos
            # donde esta ultima genuinamente no aparece. Simplemente
            # restringir esta busqueda a la zona propia del campo
            # ("header") resulto demasiado estricto: perdia casos
            # legitimos donde el bloque cae en "body" o "signature" por
            # una clasificacion de zona imperfecta. La solucion intermedia:
            # buscar en todo el documento EXCEPTO la zona "legal" (donde
            # vive de forma consistente el aviso de confidencialidad y sus
            # fechas de emision/informe), en vez de limitarse a una sola
            # zona o de no excluir ninguna.
            non_legal_text = " ".join(
                getattr(b, "text", "") for b in getattr(doc, "blocks", [])
                if getattr(b, "zone", None) != "legal"
            )
            found_in_document = bool(re.search(pattern, non_legal_text))
            if found_in_document:
                return True, "Fecha encontrada en el documento (excluyendo aviso legal)."

            date_candidate = find_date(text_tail(non_legal_text))
            if date_candidate is not None:
                return (
                    True,
                    f"Fecha tolerante encontrada en el documento (excluyendo aviso legal). "
                    f"{date_candidate.details}",
                )

            return False, "No se detecto una fecha valida."

        return False, f"Tipo de contenido no soportado: {content_type}."

    def _check_content_rule(
        self,
        rule: dict[str, Any],
        doc: "DocumentOCR | Any",
    ) -> tuple[bool, str]:
        zone_text = self._get_zone_text(doc, rule["zona_documento"])
        full_text = getattr(doc, "full_text", "") or ""
        rule_type = rule["tipo"]

        if rule_type == "keyword":
            if self._keyword_match(rule["patron"], zone_text):
                return True, "Coincidencia por palabra clave en la zona esperada."

            if self._keyword_match(rule["patron"], full_text):
                return True, "Coincidencia por palabra clave usando el documento completo."

            return False, "No hubo coincidencia por palabra clave."

        if rule_type == "regex":
            found_in_zone = bool(
                re.search(str(rule["patron"]), zone_text, flags=re.IGNORECASE)
            )
            if found_in_zone:
                return True, "Coincidencia por expresion regular en la zona esperada."

            tolerant_candidate = self._check_tolerant_regex(rule, zone_text)
            if tolerant_candidate is not None:
                return (
                    True,
                    "Coincidencia tolerante en la zona esperada. "
                    f"{tolerant_candidate.details}",
                )

            found_in_document = bool(
                re.search(str(rule["patron"]), full_text, flags=re.IGNORECASE)
            )
            if found_in_document:
                return True, "Coincidencia por expresion regular usando el documento completo."

            full_scope = text_tail(full_text) if rule.get("zona_documento") == "signature" else full_text
            tolerant_candidate = self._check_tolerant_regex(rule, full_scope)
            if tolerant_candidate is not None:
                return (
                    True,
                    "Coincidencia tolerante usando texto auxiliar. "
                    f"{tolerant_candidate.details}",
                )

            return False, "No hubo coincidencia por expresion regular."

        if rule_type == "semantic":
            return False, "La validacion semantica se delega a la fase LLM."

        if rule_type == "ausencia_regex":
            # Regla determinista inversa: falla si el patron SI aparece, en
            # vez de fallar si no aparece. Se detecto que una comprobacion
            # por palabra clave positiva (ej. "dosis") puede autoengañarse
            # cuando el propio texto que senala la ausencia de un dato usa
            # esa misma palabra (ej. el generador del corpus escribe
            # literalmente "(sin informacion de dosis)" cuando falta la
            # dosis, lo que satisfacia por error la regla "keyword" que
            # busca la palabra "dosis"). Este tipo de regla cubre ese caso:
            # detecta la frase de ausencia explicita en vez de la palabra
            # suelta.
            found_in_zone = bool(
                re.search(str(rule["patron"]), zone_text, flags=re.IGNORECASE)
            )
            found_in_document = found_in_zone or bool(
                re.search(str(rule["patron"]), full_text, flags=re.IGNORECASE)
            )
            if found_in_document:
                return False, "Se encontro el patron de ausencia explicita (el dato senalado no esta presente)."
            return True, "No se encontro ningun indicio explicito de dato ausente."

        return False, f"Tipo de regla no soportado: {rule_type}."

    def _check_coherence_rule(
        self,
        rule: dict[str, Any],
        field_status: dict[str, FieldValidation],
        doc: "DocumentOCR | Any",
        checkboxes: Optional[list[dict[str, Any]]] = None,
        vl_date_rescue: Optional[Any] = None,
        vl_identity_rescue: Optional[Any] = None,
    ) -> tuple[bool, bool, Optional[ValidationIssue]]:
        rule_type = rule["tipo"]
        checkboxes = checkboxes or []

        if rule_type == "checkboxes_completos":
            # Regla determinista: en un checklist de seguridad (ej.
            # preoperatorio segun la OMS), un item marcado "No" no es un
            # dato ausente sino una alerta de seguridad activa -- el
            # checklist en si mismo pierde su proposito si se permite que
            # items criticos queden sin verificar. Se aproxima con la
            # proporcion de casillas marcadas del documento: por debajo del
            # umbral, se considera una incoherencia real (el documento dice
            # "verificado" en su titulo pero varios items no lo estan), no
            # solo una casilla vacia aislada.
            total = len(checkboxes)
            if total == 0:
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="warning",
                    message="No se detectaron casillas para verificar la regla.",
                )
            checked = sum(1 for c in checkboxes if c.get("checked"))
            ratio = checked / total
            umbral = rule.get("proporcion_minima", 1.0)
            if ratio >= umbral:
                return True, True, None
            return True, False, ValidationIssue(
                rule_id=rule["id_regla"],
                severity="error",
                message=rule["mensaje_error"],
                details=f"{checked}/{total} casillas marcadas ({ratio:.0%}).",
            )

        if rule_type == "campo_presente_si":
            origin = field_status.get(rule["campo_origen"])
            destination = field_status.get(rule["campo_destino"])
            if origin is None or destination is None:
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="warning",
                    message="La regla hace referencia a campos no definidos en el protocolo.",
                    details=f"{rule['campo_origen']} -> {rule['campo_destino']}",
                )

            if not origin.present:
                return True, True, None
            if destination.present:
                return True, True, None

            return True, False, ValidationIssue(
                rule_id=rule["id_regla"],
                severity="error",
                message=rule["mensaje_error"],
                zone=destination.zone,
                details=destination.details,
            )

        if rule_type == "fecha_posterior":
            # Busqueda por VENTANA alrededor del patron propio de cada campo
            # (igual que formato_numerico_requerido), no por toda la zona:
            # dos campos de fecha distintos (ej. fecha de nacimiento y fecha
            # del documento) suelen compartir la misma zona "header", y
            # extraer "la primera fecha de la zona" es ambiguo -- puede
            # devolver la fecha equivocada segun el orden de dibujo. La
            # ventana ata cada fecha a la etiqueta de su propio campo.
            origin = field_status.get(rule["campo_origen"])
            destination = field_status.get(rule["campo_destino"])
            if origin is None or destination is None:
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="warning",
                    message="La regla de fechas hace referencia a campos no definidos.",
                )

            def _date_for(field_name: str, field_val: "FieldValidation") -> Optional[datetime]:
                # Los patrones de anclaje son palabras especificas por familia
                # (ej. "expedicion", "revision:", "ingreso:"), no un termino
                # generico compartido -- evita que el ancla de una fecha
                # encuentre la etiqueta de la OTRA fecha del mismo documento.
                # La ventana en si se acota con _windowed_value_until_next_label
                # (ver docstring): se extiende hasta la etiqueta siguiente en
                # vez de un ancho fijo, para no cortar fechas largas en texto
                # ni leer accidentalmente la fecha LIMPIA del campo siguiente
                # cuando la propia esta degradada por OCR.
                field_def = next(
                    (c for c in self.protocol.get("campos_obligatorios", [])
                     if c.get("nombre_zona") == field_name),
                    {},
                )
                patterns = field_def.get("patrones_esperados") or []
                if not patterns:
                    return None
                zone_blocks = [
                    b for b in getattr(doc, "blocks", [])
                    if getattr(b, "zone", None) == field_val.zone
                ]
                window_text = _windowed_value_until_next_label(zone_blocks, patterns)
                if not window_text:
                    # La zona declarada en el protocolo puede no coincidir
                    # con la zona real donde el generador dibuja la etiqueta
                    # (ej. "Fecha de ingreso" en ADM-001 cae en zona "body",
                    # no "header"). Se cae a buscar en todo el documento antes
                    # de rendirse -- el ancla especifica por familia sigue
                    # evitando la contaminacion cruzada entre campos.
                    all_blocks = getattr(doc, "blocks", [])
                    window_text = _windowed_value_until_next_label(all_blocks, patterns)
                if not window_text:
                    return None
                return self._extract_first_date(window_text)

            origin_date = _date_for(rule["campo_origen"], origin)
            destination_date = _date_for(rule["campo_destino"], destination)

            if (origin_date is None or destination_date is None) and vl_date_rescue is not None:
                def _vl_date_for(field_name: str) -> Optional[datetime]:
                    field_def = next(
                        (c for c in self.protocol.get("campos_obligatorios", [])
                         if c.get("nombre_zona") == field_name),
                        {},
                    )
                    # Se recorta el ":" final de patrones como "revision:"
                    # (necesario para anclar en bloques OCR palabra-por-
                    # palabra, ver _windowed_value_until_next_label) --
                    # VL describe la fecha en lenguaje natural ("Fecha de
                    # revision"), sin ese caracter, asi que igualarlo
                    # literalmente nunca encontraria coincidencia.
                    patterns = [p.lower().rstrip(":") for p in (field_def.get("patrones_esperados") or [])]
                    if not patterns:
                        return None
                    try:
                        entries = vl_date_rescue() or []
                    except Exception:
                        return None
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        label = str(entry.get("label_or_context", "")).lower()
                        if any(p in label for p in patterns):
                            return self._extract_first_date(str(entry.get("date", "")))
                    return None

                if origin_date is None:
                    origin_date = _vl_date_for(rule["campo_origen"])
                if destination_date is None:
                    destination_date = _vl_date_for(rule["campo_destino"])

            if origin_date is None or destination_date is None:
                # Sin ventana confiable para AMBAS fechas no hay evidencia
                # de un defecto real (puede ser simplemente OCR degradado
                # en un documento por lo demas correcto) -- se aprueba en
                # vez de escalar una incoherencia fuerte sin base. (Un
                # `issue` devuelto junto con passed=True nunca se muestra
                # -- ver el llamador -- asi que se omite directamente.)
                return True, True, None

            if destination_date >= origin_date:
                return True, True, None

            return True, False, ValidationIssue(
                rule_id=rule["id_regla"],
                severity="error",
                message=rule["mensaje_error"],
                details=(
                    f"Fecha origen: {origin_date.isoformat()}, "
                    f"fecha destino: {destination_date.isoformat()}"
                ),
            )

        if rule_type == "identidad_duplicada":
            # Regla determinista, sin campo_origen/campo_destino: busca un
            # patron de identificador (por defecto NHC) en TODO el
            # documento y compara las coincidencias entre si, en vez de
            # comparar dos campos concretos. La mayoria de documentos solo
            # mencionan el identificador una vez (0 o 1 coincidencia unica
            # -> nada que comparar, se aprueba). Solo se marca incoherencia
            # cuando aparecen DOS O MAS valores DISTINTOS del mismo patron
            # de identificador -- ej. el NHC de cabecera no coincide con el
            # NHC re-verificado en una caja de auditoria/consentimiento.
            # "NHC" en si mismo es muy inestable en OCR (confusiones de
            # forma C<->O, H<->U observadas en el corpus: "NHC-123456" leido
            # como "NHO-123456" o "NUC-123456"), asi que el patron por
            # defecto ya no fija el prefijo literal -- exige solo que
            # empiece por "N" y tenga la forma "prefijo-digitos", y compara
            # identificadores por su parte NUMERICA (normalizada con el
            # mismo mapa de confusion letra/digito que usa el resto del
            # motor), que es el contenido real que distingue un NHC de
            # otro. Sin esto la regla practicamente nunca encontraba mas de
            # una coincidencia (0 o 1), aprobando por defecto casi siempre.
            # La racha de digitos tolera UN espacio interno (ej. "NHC-0
            # 14125", donde Tesseract corto el numero en dos bloques) -- se
            # limpia al normalizar. (Se probo tambien volver opcional el
            # separador entre prefijo y digitos, para capturar casos como
            # "NAOL19827" sin separador -- descartado tras verificar que
            # producia un falso positivo real: "Fecha emision: ...2026" en
            # un documento valido se leyo como "NEOI2026" y calzaba el
            # patron sin separador. Se prefiere perder ese rescate a cambio
            # de mantener 0 falsos positivos, ya verificado como la
            # propiedad mas valiosa de esta regla.)
            patron_id = rule.get(
                "patron_id",
                r"\bN\w{1,3}[-\s]\$?([0-9BDGILOQSZ](?:\s?[0-9BDGILOQSZ]){4,6})\b",
            )
            full_text = getattr(doc, "full_text", "") or ""
            raw_matches = re.findall(patron_id, full_text, flags=re.IGNORECASE)
            matches = {m.replace(" ", "").upper().translate(OCR_DIGIT_MAP) for m in raw_matches}
            if len(matches) <= 1 and vl_identity_rescue is not None:
                try:
                    vl_entries = vl_identity_rescue() or []
                except Exception:
                    vl_entries = []
                # Filtro determinista en Python (no una exclusion pedida al
                # modelo -- ver principio ya establecido en esta sesion: los
                # LLM/VL no siguen bien instrucciones de "ignora/excluye X"
                # aunque perciban X correctamente, aunque SI perciben X
                # correctamente). Verificado con qwen2.5vl:7b en 2 rondas
                # de prueba: primero confundio el numero de colegiado del
                # medico responsable con un identificador de paciente
                # (etiqueta "MEDICO RESPONSABLE"); tras excluir esa
                # categoria, confundio ademas una fecha con un
                # identificador (etiqueta "Fecha de expedicion"). En vez de
                # perseguir cada categoria de ruido nueva con una lista
                # negativa (dosis, nombres, CIF, fechas, colegiados..., cada
                # una descubierta por su propio falso positivo), se exige
                # en positivo que la etiqueta mencione "NHC" -- el nombre
                # real del identificador en TODOS los casos genuinos vistos
                # en el corpus (generate_nhc() en el generador). Mucho mas
                # seguro que enumerar exclusiones sin fin.
                vl_values = {
                    "".join(ch for ch in str(entry.get("value", "")).upper() if ch.isalnum())
                    for entry in vl_entries
                    if isinstance(entry, dict) and str(entry.get("value", "")).strip()
                    and "nhc" in str(entry.get("label_or_context", "")).lower()
                }
                vl_values.discard("")
                if len(vl_values) > 1:
                    return True, False, ValidationIssue(
                        rule_id=rule["id_regla"],
                        severity="error",
                        message=rule["mensaje_error"],
                        details=f"Identificadores distintos leidos por VL: {', '.join(sorted(vl_values))}",
                    )
            if len(matches) <= 1:
                return True, True, None
            return True, False, ValidationIssue(
                rule_id=rule["id_regla"],
                severity="error",
                message=rule["mensaje_error"],
                details=f"Identificadores distintos encontrados en el mismo documento: {', '.join(sorted(matches))}",
            )

        if rule_type == "ausencia_contradicha":
            # Regla determinista: coherencia entre un campo DECLARADO (ej.
            # "Medicacion habitual: Ninguna") y texto libre de otro campo
            # (ej. antecedentes). Solo se activa si el campo de origen
            # realmente declara la ausencia (patron_ausencia coincide en su
            # zona) -- si el campo no declara "Ninguna"/similar, no hay nada
            # que contradecir y se aprueba sin mirar el destino. Si SI la
            # declara, se busca en el campo destino alguna palabra clave que
            # contradiga esa ausencia (ej. el nombre de un medicamento).
            origin = field_status.get(rule["campo_origen"])
            destination = field_status.get(rule["campo_destino"])
            if origin is None or destination is None:
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="warning",
                    message="La regla hace referencia a campos no definidos.",
                )
            origin_text = self._get_zone_text(doc, origin.zone)
            if not re.search(str(rule["patron_ausencia"]), origin_text, flags=re.IGNORECASE):
                return True, True, None

            destination_text = self._get_zone_text(doc, destination.zone)
            if self._keyword_match(rule.get("patrones_contradiccion", []), destination_text):
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="error",
                    message=rule["mensaje_error"],
                    zone=destination.zone,
                )
            return True, True, None

        if rule_type == "formato_numerico_requerido":
            # Regla determinista: exige que una VENTANA de texto alrededor
            # del campo (no toda la zona documental, que puede abarcar
            # otros campos y casi siempre contiene algun digito en otra
            # parte -fechas, NHC-, lo que haria pasar la comprobacion
            # siempre sin aportar nada) tenga al menos una coincidencia del
            # patron indicado. Pensada para detectar resultados de
            # laboratorio "sin formato estandar" (ej. solo
            # "positivo"/"negativo" sin ningun valor numerico, unidad o
            # rango de referencia), donde el campo tecnicamente esta
            # "presente" (tiene texto) pero le falta el contenido
            # cuantitativo que un resultado de laboratorio real siempre
            # tiene.
            origin = field_status.get(rule["campo_origen"])
            if origin is None:
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="warning",
                    message="La regla de formato numerico hace referencia a un campo no definido.",
                )

            field_def = next(
                (c for c in self.protocol.get("campos_obligatorios", [])
                 if c.get("nombre_zona") == rule["campo_origen"]),
                {},
            )
            patterns = field_def.get("patrones_esperados") or []
            zone_blocks = [
                b for b in getattr(doc, "blocks", [])
                if getattr(b, "zone", None) == origin.zone
            ]
            window_text = _windowed_text_near_pattern(zone_blocks, patterns, before=0, after=15) \
                if zone_blocks and patterns else None
            search_text = window_text if window_text else self._get_zone_text(doc, origin.zone)

            if re.search(str(rule["patron_regex"]), search_text, flags=re.IGNORECASE):
                return True, True, None
            return True, False, ValidationIssue(
                rule_id=rule["id_regla"],
                severity="error",
                message=rule["mensaje_error"],
                zone=origin.zone,
            )

        if rule_type == "rango_fisiologico":
            # Regla determinista (sin LLM/VL): extrae valores numericos del
            # texto de un campo mediante expresiones regulares y comprueba
            # que caigan dentro de un rango fisiologicamente posible. No
            # depende de juicio clinico sutil (no evalua si un tratamiento
            # "tiene sentido"), solo limites objetivos e incuestionables
            # (ej. una saturacion de oxigeno no puede superar 100%). Sirve
            # para detectar de forma estable el tipo de incoherencia
            # "valores fisiologicos imposibles" que las reglas de contenido
            # y coherencia existentes no cubrian, sin la variabilidad de
            # calibracion que mostraron el LLM y el analisis visual para
            # esta misma categoria de defecto.
            origin = field_status.get(rule["campo_origen"])
            if origin is None:
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="warning",
                    message="La regla de rango fisiologico hace referencia a un campo no definido.",
                )

            origin_text = self._get_zone_text(doc, origin.zone)
            full_text = getattr(doc, "full_text", "") or ""
            search_text = origin_text if origin_text.strip() else full_text

            fuera_de_rango = []
            for parametro in rule.get("parametros", []):
                match = re.search(parametro["patron_regex"], search_text, flags=re.IGNORECASE)
                if not match:
                    continue
                try:
                    valor = float(match.group(1))
                except (ValueError, IndexError):
                    continue
                minimo = parametro.get("minimo")
                maximo = parametro.get("maximo")
                if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
                    fuera_de_rango.append(
                        f"{parametro.get('nombre', parametro['patron_regex'])}={match.group(1)} "
                        f"(rango esperado {minimo}-{maximo})"
                    )

            if fuera_de_rango:
                return True, False, ValidationIssue(
                    rule_id=rule["id_regla"],
                    severity="error",
                    message=rule["mensaje_error"],
                    zone=origin.zone,
                    details="; ".join(fuera_de_rango),
                )

            return True, True, None

        if rule_type == "texto_coherente":
            return False, False, None

        return True, False, ValidationIssue(
            rule_id=rule["id_regla"],
            severity="warning",
            message=f"Tipo de regla de coherencia no soportado: {rule_type}",
        )

    def _get_zone_text(self, doc: "DocumentOCR | Any", zone: str) -> str:
        zone_text = ""
        if hasattr(doc, "get_text_by_zone"):
            zone_text = doc.get_text_by_zone(zone) or ""
        if zone_text:
            return zone_text
        return getattr(doc, "full_text", "") or ""

    def _keyword_match(self, patterns: list[str] | str, text: str) -> bool:
        normalized_text = self._normalize_text(text)

        if isinstance(patterns, str):
            patterns = [patterns]

        return any(self._normalize_text(pattern) in normalized_text for pattern in patterns)

    def _extract_first_date(self, text: str) -> Optional[datetime]:
        match = DATE_PATTERN.search(text or "")
        if not match:
            candidate = find_date(text or "")
            if candidate is None:
                return None
            raw = candidate.value
        else:
            raw = match.group(0).replace("-", "/")

        day, month, year = raw.split("/")
        if len(year) == 2:
            year = f"20{year}"

        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            return None

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return "".join(char for char in normalized if not unicodedata.combining(char))

    def _expects_identifier(
        self,
        field_name: str,
        expected_patterns: list[str] | str,
    ) -> bool:
        normalized_name = self._normalize_text(field_name)
        if "identificacion" in normalized_name:
            return True

        if isinstance(expected_patterns, str):
            expected_patterns = [expected_patterns]
        normalized_patterns = [self._normalize_text(item) for item in expected_patterns]
        return any(item in {"dni", "nie", "dni/nie"} for item in normalized_patterns)

    def _check_tolerant_regex(self, rule: dict[str, Any], text: str):
        pattern = str(rule.get("patron", ""))
        message = self._normalize_text(str(rule.get("mensaje_error", "")))
        zone = rule.get("zona_documento", "")

        if "dni" in message or "nie" in message or pattern == r"\d{8}[A-Z]":
            return find_dni_nie(text)

        if "fecha" in message or zone == "signature":
            return find_date(text)

        return None
