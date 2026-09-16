"""
Agente supervisor que orquesta el pipeline OCR, reglas y LLM opcional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - se valida al construir el grafo.
    END = None
    StateGraph = None

from src.llm.mistral_client import MistralClient
from src.ocr.extractor import detect_checkboxes, detect_signatures, extract_text_blocks
from src.rules.engine import ProtocolLoader, RuleEngine, ValidationResult


@dataclass
class SupervisorState:
    """Estado documentado del agente durante el procesamiento."""

    document_path: str = ""
    protocol_id: str = ""
    protocols_dir: str = "configs/protocolos"
    use_llm: bool = False

    ocr_result: Any = None
    signatures: list[dict[str, Any]] = field(default_factory=list)
    checkboxes: list[dict[str, Any]] = field(default_factory=list)
    protocol: Optional[dict[str, Any]] = None
    rule_validation: Optional[ValidationResult] = None
    llm_validations: list[dict[str, Any]] = field(default_factory=list)

    verdict: str = ""
    report: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def ocr_step(state: dict[str, Any]) -> dict[str, Any]:
    """Extrae texto, firmas y casillas del documento."""
    if state.get("errors"):
        return state

    try:
        protocols_dir = state.get("protocols_dir", "configs/protocolos")
        protocol = ProtocolLoader(protocols_dir=protocols_dir).get_protocol(state["protocol_id"])
        lang = (protocol or {}).get("idioma", "spa")
        ocr_result = extract_text_blocks(state["document_path"], lang=lang)
        signatures = detect_signatures(state["document_path"])
        checkboxes = detect_checkboxes(state["document_path"])
    except Exception as exc:
        return _append_error(state, f"OCR error: {exc}")

    return {
        **state,
        "ocr_result": ocr_result,
        "signatures": signatures,
        "checkboxes": checkboxes,
    }


def rule_validation_step(state: dict[str, Any]) -> dict[str, Any]:
    """Valida el documento contra el protocolo YAML."""
    if state.get("errors"):
        return state

    try:
        protocols_dir = state.get("protocols_dir", "configs/protocolos")
        loader = ProtocolLoader(protocols_dir=protocols_dir)
        protocol = loader.get_protocol(state["protocol_id"])
        if protocol is None:
            return _append_error(
                state,
                f"Protocolo no encontrado: {state['protocol_id']}",
            )

        result = RuleEngine(protocol).validate(
            state["ocr_result"],
            state.get("signatures", []),
            state.get("checkboxes", []),
        )
    except Exception as exc:
        return _append_error(state, f"Rule error: {exc}")

    return {
        **state,
        "protocol": protocol,
        "rule_validation": result,
    }


def llm_validation_step(state: dict[str, Any]) -> dict[str, Any]:
    """
    Ejecuta solo las reglas semanticas delegadas al LLM.

    Si use_llm=False, deja la validacion registrada como omitida para que el
    reporte sea explicito sin cargar el modelo local.
    """
    if state.get("errors"):
        return state

    protocol = state.get("protocol")
    if protocol is None:
        try:
            protocols_dir = state.get("protocols_dir", "configs/protocolos")
            protocol = ProtocolLoader(protocols_dir=protocols_dir).get_protocol(state["protocol_id"])
        except Exception as exc:
            return _append_error(state, f"LLM setup error: {exc}")

    validations: list[dict[str, Any]] = []
    coherence_rules = _coherence_llm_rules(protocol or {})

    if not state.get("use_llm", False):
        for rule in coherence_rules:
            validations.append(
                {
                    "rule_id": rule["id_regla"],
                    "condition": rule.get("condicion", ""),
                    "skipped": True,
                    "result": {
                        "coherent": None,
                        "justification": "Validacion LLM desactivada.",
                    },
                }
            )
        return {**state, "protocol": protocol, "llm_validations": validations}

    try:
        client = state.get("_llm_client") or MistralClient()
        doc = state["ocr_result"]
        rule_result = state.get("rule_validation")

        for rule in coherence_rules:
            if _field_is_absent(rule_result, rule.get("campo_origen")) or _field_is_absent(
                rule_result, rule.get("campo_destino")
            ):
                validations.append(
                    {
                        "rule_id": rule["id_regla"],
                        "condition": rule.get("condicion", ""),
                        "skipped": True,
                        "error": False,
                        "result": {
                            "coherent": None,
                            "justification": "Campo de entrada ausente; ya registrado por separado como campo obligatorio faltante.",
                        },
                    }
                )
                continue

            text_a = _text_for_field(doc, protocol or {}, rule.get("campo_origen"))
            text_b = _text_for_field(doc, protocol or {}, rule.get("campo_destino"))
            result = client.check_coherence(
                text_a,
                text_b,
                rule.get("condicion", ""),
            )
            validations.append(
                {
                    "rule_id": rule["id_regla"],
                    "condition": rule.get("condicion", ""),
                    "skipped": False,
                    "error": False,
                    "result": result,
                }
            )
    except Exception as exc:
        error_message = f"LLM no disponible: {exc}"
        pending_rules = coherence_rules[len(validations):]
        for rule in pending_rules:
            validations.append(
                {
                    "rule_id": rule["id_regla"],
                    "condition": rule.get("condicion", ""),
                    "skipped": True,
                    "error": True,
                    "result": {
                        "coherent": None,
                        "justification": error_message,
                    },
                }
            )
        return {
            **state,
            "protocol": protocol,
            "llm_validations": validations,
            "llm_error": error_message,
        }

    return {
        **state,
        "protocol": protocol,
        "llm_validations": validations,
    }


def generate_report_step(state: dict[str, Any]) -> dict[str, Any]:
    """Genera el reporte final serializable a JSON."""
    rule_result = state.get("rule_validation")
    llm_results = state.get("llm_validations", [])
    ocr_result = state.get("ocr_result")

    verdict = _combined_verdict(state.get("errors", []), rule_result, llm_results)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "document": state.get("document_path", ""),
        "protocol": state.get("protocol_id", ""),
        "protocol_version": (state.get("protocol") or {}).get("version"),
        "verdict": verdict,
        "ocr": {
            "confidence": round(float(getattr(ocr_result, "avg_confidence", 0.0)), 2),
            "blocks": len(getattr(ocr_result, "blocks", []) or []),
            "text_chars": len(getattr(ocr_result, "full_text", "") or ""),
        },
        "signatures": {
            "found": len(state.get("signatures", []) or []),
            "items": [_signature_to_dict(item) for item in state.get("signatures", [])],
        },
        "checkboxes": [_checkbox_to_dict(item) for item in state.get("checkboxes", [])],
        "rule_checks": _rule_checks_to_dict(rule_result),
        "semantic_checks": llm_results,
        "llm_enabled": bool(state.get("use_llm", False)),
        "llm_error": state.get("llm_error"),
        "errors": list(state.get("errors", [])),
    }

    return {**state, "verdict": verdict, "report": report}


def build_supervisor_graph():
    """Construye el grafo LangGraph del agente supervisor."""
    if StateGraph is None or END is None:
        raise ImportError(
            "langgraph no esta instalado. Instala las dependencias de requirements.txt."
        )

    graph = StateGraph(dict)
    graph.add_node("ocr", ocr_step)
    graph.add_node("rules", rule_validation_step)
    graph.add_node("llm", llm_validation_step)
    graph.add_node("report", generate_report_step)

    graph.set_entry_point("ocr")
    graph.add_edge("ocr", "rules")
    graph.add_edge("rules", "llm")
    graph.add_edge("llm", "report")
    graph.add_edge("report", END)

    return graph.compile()


def supervise_document(
    document_path: str,
    protocol_id: str,
    *,
    use_llm: bool = False,
    llm_client: Any = None,
    protocols_dir: str = "configs/protocolos",
) -> dict[str, Any]:
    """Ejecuta el pipeline completo sobre un documento.

    `protocols_dir` por defecto apunta al conjunto de protocolos en ingles
    (usado por la prueba de integracion sobre ClinOCR-Bench). Para validar
    contra los protocolos en espanol evaluados en la memoria (CN-001-ES,
    DLR-001-ES, MED-001-ES, ADM-001-ES, PREOP-001-ES), pasa
    protocols_dir="configs/protocolos_es" -- que es lo que hace run_supervisor.py
    por defecto.
    """
    graph = build_supervisor_graph()
    initial_state: dict[str, Any] = {
        "document_path": document_path,
        "protocol_id": protocol_id,
        "use_llm": use_llm,
        "protocols_dir": protocols_dir,
        "errors": [],
    }
    if llm_client is not None:
        initial_state["_llm_client"] = llm_client

    final_state = graph.invoke(initial_state)
    return final_state["report"]


def _append_error(state: dict[str, Any], message: str) -> dict[str, Any]:
    return {**state, "errors": list(state.get("errors", [])) + [message]}


def _coherence_llm_rules(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        rule
        for rule in protocol.get("reglas_coherencia", [])
        if rule.get("tipo") == "texto_coherente"
    ]


def _field_is_absent(rule_result: Optional[ValidationResult], field_name: Optional[str]) -> bool:
    """True si `field_name` ya quedo marcado ausente por el motor de reglas
    (score_breakdown, criterio "CAMPO-<field_name>", aprobado=False).

    _text_for_field() no distingue "campo ausente" de "campo presente": si
    el patron esperado no aparece en su zona, cae al texto completo de esa
    zona (membrete, texto legal, ruido de OCR) en vez de devolver vacio. Un
    texto_coherente que reciba ese texto residual como si fuera el campo
    real induce al LLM a comparar de que habla cada seccion en vez de
    evaluar la condicion pedida, generando falsos positivos (ver hallazgo
    22). Si el campo ya esta registrado como ausente, esa ausencia ya se
    reporta por separado (regla CAMPO-<field_name>) y no aporta nada nuevo
    pedirle al LLM que ademas juzgue "coherencia" contra ese texto residual.
    """
    if not field_name or rule_result is None:
        return False
    target = f"CAMPO-{field_name}"
    for item in rule_result.score_breakdown:
        if item.criterio == target:
            return not item.aprobado
    return False


def _windowed_text_near_pattern(
    blocks: list[Any],
    patterns: list[str],
    before: int = 0,
    after: int = 12,
) -> Optional[str]:
    """
    Devuelve una ventana de bloques de texto alrededor del primer bloque
    que coincide con alguno de los patrones esperados del campo, en vez
    de concatenar toda la zona. Evita que un campo concreto (p. ej.
    "medico_solicitante") arrastre texto no relacionado de otros campos
    que comparten la misma zona_documento (p. ej. el membrete o una caja
    de cumplimiento normativo dentro de "header"), lo cual diluye la
    senal cuando esa zona es extensa y puede hacer que una comprobacion
    de coherencia LLM compare texto irrelevante en vez del campo real.

    `before=0` por defecto: se detecto que `before=2` podia arrastrar la
    cola de la oracion del campo ANTERIOR dentro de la misma zona (ej. el
    final de "motivo_consulta" colandose en la ventana de
    "historia_medica" cuando el patron buscado -p. ej. "antecedentes"- es
    la primera palabra del bloque encontrado), lo que llevaba al LLM a
    comparar dos fragmentos de un mismo campo como si fueran campos
    distintos y marcar una incoherencia inexistente. El patron esperado
    ya suele ser la propia etiqueta del campo (aparece al inicio de su
    bloque de texto), por lo que no hace falta contexto previo.
    """
    lowered_patterns = [p.lower() for p in patterns]
    for i, block in enumerate(blocks):
        text = getattr(block, "text", "") or ""
        if any(p in text.lower() for p in lowered_patterns):
            start = max(0, i - before)
            end = min(len(blocks), i + after)
            return " ".join(getattr(b, "text", "") for b in blocks[start:end])
    return None


def _text_for_field(
    doc: Any,
    protocol: dict[str, Any],
    field_name: Optional[str],
) -> str:
    if not field_name:
        return getattr(doc, "full_text", "") or ""

    for field_def in protocol.get("campos_obligatorios", []):
        if field_def.get("nombre_zona") != field_name:
            continue

        zone = field_def.get("zona_documento")
        patterns = field_def.get("patrones_esperados") or []
        blocks = [
            b for b in getattr(doc, "blocks", [])
            if getattr(b, "zone", None) == zone
        ] if zone else []

        if blocks and patterns:
            windowed = _windowed_text_near_pattern(blocks, patterns)
            if windowed:
                return windowed

        if zone and hasattr(doc, "get_text_by_zone"):
            text = doc.get_text_by_zone(zone)
            if text:
                return text
        return getattr(doc, "full_text", "") or ""

    return getattr(doc, "full_text", "") or ""


def _combined_verdict(
    errors: list[str],
    rule_result: Optional[ValidationResult],
    llm_results: list[dict[str, Any]],
) -> str:
    if errors:
        return "error"

    # Un "coherent: false" con confianza baja se trata como indeterminado,
    # no como fallo duro: el modelo puede reportar su mejor estimacion aun
    # sin evidencia solida (ver prompt de check_coherence), y forzar
    # "inconsistent" sobre una senal de baja confianza generaba demasiados
    # falsos positivos en la evaluacion integral. Los llm_results que no
    # incluyen "confidence" (backends antiguos) se tratan como si fuera
    # "high", para no cambiar el comportamiento de integraciones previas.
    has_semantic_failure = any(
        item.get("result", {}).get("coherent") is False
        and item.get("result", {}).get("confidence", "high") != "low"
        for item in llm_results
    )
    if has_semantic_failure:
        return "inconsistent"

    if rule_result is None:
        return "incomplete"

    has_semantic_error = any(item.get("error") for item in llm_results)
    if has_semantic_error and rule_result.verdict == "valid":
        return "incomplete"

    return rule_result.verdict


def _rule_checks_to_dict(rule_result: Optional[ValidationResult]) -> dict[str, Any]:
    if rule_result is None:
        return {
            "verdict": None,
            "passed": 0,
            "total": 0,
            "score": 0.0,
            "weighted_score": 0.0,
            "score_breakdown": [],
            "issues": [],
        }

    return {
        "verdict": rule_result.verdict,
        "passed": int(rule_result.checks_passed),
        "total": int(rule_result.checks_total),
        "score": round(float(rule_result.score), 4),
        "weighted_score": round(float(rule_result.weighted_score), 2),
        "score_breakdown": [
            {
                "criterio": item.criterio,
                "categoria": item.categoria,
                "peso": item.peso,
                "aprobado": item.aprobado,
                "detalle": item.detalle,
            }
            for item in rule_result.score_breakdown
        ],
        "issues": [
            {
                "id": issue.rule_id,
                "severity": issue.severity,
                "zone": issue.zone,
                "message": issue.message,
                "details": issue.details,
            }
            for issue in rule_result.issues
        ],
    }


def _signature_to_dict(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": int(item.get("x", 0)),
        "y": int(item.get("y", 0)),
        "width": int(item.get("width", 0)),
        "height": int(item.get("height", 0)),
        "area": int(item.get("area", 0)),
    }


def _checkbox_to_dict(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": int(item.get("x", 0)),
        "y": int(item.get("y", 0)),
        "width": int(item.get("width", 0)),
        "height": int(item.get("height", 0)),
        "checked": bool(item.get("checked", False)),
        "fill_ratio": float(item.get("fill_ratio", 0.0)),
    }
