#!/usr/bin/env python3
"""
evaluate_synthetic_corpus.py - Evaluacion integral del agente PathwayGuard
sobre el corpus sintetico con ground truth controlado.

Ejecuta el pipeline completo (OCR -> Reglas -> [LLM]) sobre cada documento
del corpus sintetico y calcula todas las metricas requeridas:

  - Matriz de confusion 3x3 del veredicto (valid/incomplete/inconsistent)
  - Precision, recall, F1 por clase y macro/micro/weighted
  - Sensibilidad y especificidad (valid vs no-valid)
  - Kappa de Cohen (agente vs ground truth)
  - Metricas por familia documental
  - Metricas de deteccion de firmas y casillas (P/R/F1)
  - Evaluacion de zonificacion
  - Intervalos de confianza bootstrap
  - Ablacion: solo reglas vs reglas+LLM vs reglas+LLM+VL

Uso:
  python evaluate_synthetic_corpus.py --corpus-dir data/synthetic_corpus_v2
                                      --protocols-dir configs/protocolos_es
                                      [--use-llm] [--use-vl]
                                      [--output-dir data/results/synthetic]

Requisitos: Tesseract OCR, OpenCV, PyYAML, Pillow (+ Ollama si --use-llm)
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Agregar raiz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

VERDICT_CLASSES = ["valid", "incomplete", "inconsistent"]


def confusion_matrix(y_true: list[str], y_pred: list[str],
                     classes: list[str] = None) -> dict:
    """Calcula matriz de confusion y metricas derivadas."""
    classes = classes or VERDICT_CLASSES
    n = len(classes)
    matrix = [[0] * n for _ in range(n)]
    class_idx = {c: i for i, c in enumerate(classes)}

    for true, pred in zip(y_true, y_pred):
        ti = class_idx.get(true)
        pi = class_idx.get(pred)
        if ti is not None and pi is not None:
            matrix[ti][pi] += 1

    # Per-class metrics
    per_class = {}
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        tn = sum(sum(row) for row in matrix) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    # Macro averages
    macro_p = sum(m["precision"] for m in per_class.values()) / n
    macro_r = sum(m["recall"] for m in per_class.values()) / n
    macro_f1 = sum(m["f1"] for m in per_class.values()) / n

    # Weighted averages
    total = sum(m["support"] for m in per_class.values())
    weighted_p = sum(m["precision"] * m["support"] for m in per_class.values()) / max(total, 1)
    weighted_r = sum(m["recall"] * m["support"] for m in per_class.values()) / max(total, 1)
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / max(total, 1)

    # Accuracy
    correct = sum(matrix[i][i] for i in range(n))
    accuracy = correct / max(total, 1)

    # Binary: valid vs non-valid (sensitivity/specificity)
    valid_idx = class_idx.get("valid", 0)
    tp_bin = matrix[valid_idx][valid_idx]
    fn_bin = sum(matrix[valid_idx]) - tp_bin
    fp_bin = sum(matrix[j][valid_idx] for j in range(n)) - tp_bin
    tn_bin = total - tp_bin - fn_bin - fp_bin
    sensitivity = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else 0.0
    specificity = tn_bin / (tn_bin + fp_bin) if (tn_bin + fp_bin) > 0 else 0.0

    return {
        "matrix": matrix,
        "classes": classes,
        "per_class": per_class,
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "weighted": {"precision": round(weighted_p, 4), "recall": round(weighted_r, 4), "f1": round(weighted_f1, 4)},
        "accuracy": round(accuracy, 4),
        "sensitivity_valid": round(sensitivity, 4),
        "specificity_valid": round(specificity, 4),
        "total": total,
        "correct": correct,
    }


def cohens_kappa(y_true: list[str], y_pred: list[str]) -> float:
    """Calcula kappa de Cohen."""
    classes = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    if n == 0:
        return 0.0

    class_idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    matrix = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        matrix[class_idx[t]][class_idx[p]] += 1

    po = sum(matrix[i][i] for i in range(k)) / n
    pe = sum(
        sum(matrix[i][j] for j in range(k)) * sum(matrix[j][i] for j in range(k))
        for i in range(k)
    ) / (n * n)

    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def bootstrap_ci(y_true: list[str], y_pred: list[str],
                 metric_fn, n_bootstrap: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> dict:
    """Calcula intervalo de confianza bootstrap para una metrica."""
    import random
    rng = random.Random(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        y_t = [y_true[i] for i in indices]
        y_p = [y_pred[i] for i in indices]
        try:
            score = metric_fn(y_t, y_p)
            scores.append(score)
        except (ZeroDivisionError, ValueError):
            continue

    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    scores.sort()
    alpha = 1 - ci
    lo = int(alpha / 2 * len(scores))
    hi = int((1 - alpha / 2) * len(scores))

    return {
        "mean": round(sum(scores) / len(scores), 4),
        "ci_lower": round(scores[lo], 4),
        "ci_upper": round(scores[min(hi, len(scores) - 1)], 4),
        "n_bootstrap": len(scores),
    }


def macro_f1(y_true, y_pred):
    """Calcula macro F1 como scalar para bootstrap."""
    cm = confusion_matrix(y_true, y_pred)
    return cm["macro"]["f1"]


def accuracy_score(y_true, y_pred):
    """Calcula accuracy como scalar para bootstrap."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)


# ---------------------------------------------------------------------------
# Pipeline de evaluacion
# ---------------------------------------------------------------------------

def load_ground_truth(corpus_dir: str) -> dict:
    """Carga el ground truth del corpus sintetico."""
    gt_path = Path(corpus_dir) / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"No se encontro ground truth en {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_document(image_path: str, protocol: dict,
                             use_llm: bool = False,
                             use_vl: bool = False,
                             llm_model: str = None) -> dict:
    """
    Ejecuta el pipeline PathwayGuard sobre un documento.
    Retorna el resultado completo incluyendo OCR, reglas y opcionalmente LLM/VL.

    IMPORTANTE: el veredicto final se calcula reutilizando la logica oficial
    de combinacion de src/agent/supervisor.py (_combined_verdict), no una
    logica ad-hoc. Esto asegura que la ablacion "reglas" vs "reglas+LLM" vs
    "reglas+LLM+VL" mida realmente el efecto de anadir cada capa semantica sobre
    el veredicto, tal como se combinan en el agente real, en vez de calcular
    resultados de LLM/VL que luego se descartan sin influir en el veredicto.
    """
    from src.ocr.extractor import extract_text_blocks, detect_signatures, detect_checkboxes
    from src.rules.engine import RuleEngine
    from src.agent.supervisor import _combined_verdict, _coherence_llm_rules, _text_for_field, _field_is_absent

    t0 = time.time()

    # 1. OCR
    try:
        doc_ocr = extract_text_blocks(image_path, lang="spa")
    except Exception as e:
        return {
            "error": f"OCR failed: {str(e)}",
            "verdict": "error",
            "elapsed_s": time.time() - t0,
        }

    # 2. Deteccion de firmas y casillas
    signatures = detect_signatures(image_path)
    checkboxes = detect_checkboxes(image_path)

    # 3. Validacion por reglas
    # vl_rescue (ver hallazgo 18): para campos donde se verifico que el OCR
    # tiene recall bajo (ej. fecha_servicio en DLR-001-ES, 0.42) y que un
    # chequeo visual dedicado de VL SI acierta de forma fiable (30/30
    # verificado), se ofrece como respaldo -- solo se llama si el OCR ya
    # marco el campo como ausente, nunca para cuestionar un "presente"
    # que el OCR ya confirmo. Este diccionario cubre el UNICO campo donde
    # se justifica un metodo dedicado (distinguir "fecha de servicio" de
    # "fecha de emision", ambigüedad que un chequeo generico de presencia
    # no resuelve); el resto de campos usa el respaldo generico de abajo.
    VL_FIELD_RESCUE_KEYWORDS = {
        "fecha_servicio": (
            ["recogida", "servicio", "muestra", "extraccion", "extracción"],
            ["emision", "emisión", "informe"],
        ),
    }
    # vl_rescue GENERICO (2026-09-07): ademas del respaldo dedicado por
    # palabra clave de arriba (fecha_servicio), se anade un respaldo GENERAL
    # para cualquier otro campo obligatorio no cubierto, usando
    # QwenVLClient.verify_document_zone -- un metodo ya escrito en el
    # cliente pero nunca conectado a ningun pipeline hasta ahora ("codigo
    # muerto"). Se le pasa la propia `descripcion` del campo tal como esta
    # en el YAML del protocolo (ya pensada para humanos, sirve igual de
    # bien como pregunta para VL). Igual que el respaldo dedicado, SOLO se
    # llama cuando el OCR ya marco el campo ausente, y solo se acepta un
    # "encontrado" con confianza alta o media (nunca "baja") para no
    # arriesgar la fiabilidad ya verificada del OCR en los campos donde
    # funciona bien.
    vl_rescue = None
    if use_vl:
        campo_descriptions = {
            c["nombre_zona"]: c.get("descripcion", "")
            for c in protocol.get("campos_obligatorios", [])
            if not c.get("informational") and c.get("descripcion")
        }
        keyword_fields = {
            c["nombre_zona"] for c in protocol.get("campos_obligatorios", [])
        } & VL_FIELD_RESCUE_KEYWORDS.keys()
        if campo_descriptions or keyword_fields:
            from src.llm.qwen_vl_client import QwenVLClient
            rescue_client = QwenVLClient()

            def vl_rescue(field_name: str, _kw_fields=keyword_fields,
                          _descriptions=campo_descriptions, _client=rescue_client) -> Optional[bool]:
                if field_name in _kw_fields:
                    accept_kw, reject_kw = VL_FIELD_RESCUE_KEYWORDS[field_name]
                    try:
                        result = _client.check_labeled_date_present(image_path, accept_kw, reject_kw)
                        return result.get("found")
                    except Exception:
                        return None
                description = _descriptions.get(field_name)
                if not description:
                    return None
                try:
                    result = _client.verify_document_zone(image_path, description, check_type="presence")
                except Exception:
                    return None
                if result.get("found") is True and result.get("confidence") in ("high", "medium"):
                    return True
                return None

    # vl_date_rescue: mismo principio que vl_rescue, aplicado a la regla de
    # coherencia "fecha_posterior" en vez de a la presencia de un campo. El
    # fallo mas comun ahi es que el propio VALOR de la fecha este degradado
    # por OCR mas alla de lo que un regex tolerante puede reparar (digitos
    # fusionados, separador perdido) -- un problema que VL, al leer los
    # pixeles directamente, no hereda. Se memoriza la llamada a VL (unica
    # por documento, cacheada) para que ambos campos de fecha (nacimiento y
    # documento) la reutilicen sin duplicar la invocacion, ya de por si
    # lenta en CPU.
    vl_date_rescue = None
    if use_vl and any(r.get("tipo") == "fecha_posterior" for r in protocol.get("reglas_coherencia", [])):
        from src.llm.qwen_vl_client import QwenVLClient
        date_rescue_client = QwenVLClient()
        _vl_dates_cache: dict[str, list] = {}

        def vl_date_rescue(_client=date_rescue_client, _cache=_vl_dates_cache) -> list:
            if "dates" not in _cache:
                try:
                    result = _client.check_labeled_date_present(image_path, [], [])
                    _cache["dates"] = result.get("all_dates_found") or []
                except Exception:
                    _cache["dates"] = []
            return _cache["dates"]

    # vl_identity_rescue: mismo principio que vl_date_rescue, para
    # "identidad_duplicada". REACTIVADO (2026-09-07): se habia descartado
    # con qwen2.5vl:3b (ver historial en apuntes_hallazgos.md) tras un falso
    # positivo real -- el modelo de 3B no se restringia de forma fiable a
    # numeros de identificacion, devolvia dosis de medicacion, nombres,
    # numeros de colegiado y fechas como si fueran "identificadores"
    # distintos. Comparado directamente contra qwen2.5vl:7b (mismo prompt,
    # mismos documentos): el modelo de 7B devuelve UNICAMENTE los 2 NHC
    # reales en los casos donde 3B alucinaba 9 entradas o fallaba con lista
    # vacia -- y ademas responde ~5x mas rapido. Se reactiva para
    # revalidar con el modelo correcto (deteccion + prueba de falsos
    # positivos) antes de confiar en el para la evaluacion final.
    vl_identity_rescue = None
    if use_vl and any(r.get("tipo") == "identidad_duplicada" for r in protocol.get("reglas_coherencia", [])):
        from src.llm.qwen_vl_client import QwenVLClient
        identity_rescue_client = QwenVLClient()
        _vl_identities_cache: dict[str, list] = {}

        def vl_identity_rescue(_client=identity_rescue_client, _cache=_vl_identities_cache) -> list:
            if "ids" not in _cache:
                try:
                    result = _client.list_patient_identifiers(image_path)
                    _cache["ids"] = result.get("identifiers_found") or []
                except Exception:
                    _cache["ids"] = []
            return _cache["ids"]

    engine = RuleEngine(protocol)
    result = engine.validate(
        doc_ocr, signatures, checkboxes,
        vl_rescue=vl_rescue, vl_date_rescue=vl_date_rescue,
        vl_identity_rescue=vl_identity_rescue,
    )

    # Veredicto solo-reglas (base para la ablacion)
    rule_only_verdict = result.verdict
    current_verdict = result.verdict

    # 4. LLM (opcional) - misma logica de combinacion que supervisor.py
    llm_checks = []
    if use_llm:
        try:
            from src.llm.ollama_client import OllamaClient
            client = OllamaClient(model=llm_model) if llm_model else OllamaClient()
            coherence_rules = _coherence_llm_rules(protocol)
            for rule in coherence_rules:
                if _field_is_absent(result, rule.get("campo_origen")) or _field_is_absent(
                    result, rule.get("campo_destino")
                ):
                    llm_checks.append({
                        "rule_id": rule["id_regla"],
                        "condition": rule.get("condicion", ""),
                        "skipped": True,
                        "error": False,
                        "result": {
                            "coherent": None,
                            "justification": "Campo de entrada ausente; ya registrado por separado como campo obligatorio faltante.",
                        },
                    })
                    continue

                text_a = _text_for_field(doc_ocr, protocol, rule.get("campo_origen"))
                text_b = _text_for_field(doc_ocr, protocol, rule.get("campo_destino"))
                try:
                    llm_result = client.check_coherence(
                        text_a[:1000], text_b[:1000],
                        rule.get("condicion", "")
                    )
                    llm_checks.append({
                        "rule_id": rule["id_regla"],
                        "condition": rule.get("condicion", ""),
                        "skipped": False,
                        "error": False,
                        "result": llm_result,
                    })
                except Exception as e:
                    llm_checks.append({
                        "rule_id": rule["id_regla"],
                        "condition": rule.get("condicion", ""),
                        "skipped": True,
                        "error": True,
                        "result": {"coherent": None, "justification": str(e)},
                    })

            # Contraindicacion alergia/medicamento (solo MED-001-ES, unica
            # familia con ambos campos). Usa la misma receta de enumeracion
            # cerrada + decision en Python que valido el hallazgo 18 (en vez
            # de pedirle al modelo un veredicto abierto "si/no hay
            # contraindicacion"), MAS un filtro determinista de familias
            # farmacologicas conocidas (ver KNOWN_ALLERGY_DRUG_FAMILIES en
            # ollama_client.py, hallazgo 2026-09-08) que ajusta la confianza
            # de la respuesta del LLM en vez de fijarla en "high" siempre --
            # ver comentario extenso alli sobre por que se prefirio esto a
            # un gate rigido de aceptar/rechazar.
            campo_names = {c["nombre_zona"] for c in protocol.get("campos_obligatorios", [])}
            if {"alergias", "medicamentos"} <= campo_names:
                alergia_text = _text_for_field(doc_ocr, protocol, "alergias")
                # NO se usa _text_for_field (ventana de ~12 bloques) para
                # medicamentos: Tesseract tokeniza palabra por palabra, y una
                # tabla de hasta 6 medicamentos x 4 columnas ocupa muchos mas
                # de 12 bloques -- el medicamento contraindicado, mezclado al
                # azar en la lista, con frecuencia queda fuera de la ventana,
                # lo que le ocultaria al LLM la evidencia que se le pide
                # evaluar. Se usa el texto completo de la zona en su lugar.
                medicamentos_field = next(
                    (c for c in protocol.get("campos_obligatorios", []) if c["nombre_zona"] == "medicamentos"),
                    None,
                )
                if medicamentos_field and hasattr(doc_ocr, "get_text_by_zone"):
                    medicamentos_text = doc_ocr.get_text_by_zone(medicamentos_field["zona_documento"]) or ""
                else:
                    medicamentos_text = _text_for_field(doc_ocr, protocol, "medicamentos")
                try:
                    alergia_result = client.check_allergy_contraindication(
                        alergia_text[:500], medicamentos_text[:1500]
                    )
                    relacionados = alergia_result.get("medicamentos_relacionados", [])
                    contraindicado = any(
                        med.lower() in medicamentos_text.lower() for med in relacionados if med
                    )
                    llm_checks.append({
                        "rule_id": "RCH-MED-ES-ALERGIA",
                        "condition": "Ningun medicamento prescrito deberia pertenecer a la misma familia farmacologica que una alergia declarada",
                        "skipped": False,
                        "error": False,
                        "result": {
                            "coherent": not contraindicado,
                            "confidence": alergia_result.get("confidence", "high"),
                            "justification": alergia_result.get("justification", ""),
                        },
                    })
                except Exception as e:
                    llm_checks.append({
                        "rule_id": "RCH-MED-ES-ALERGIA",
                        "condition": "Ningun medicamento prescrito deberia pertenecer a la misma familia farmacologica que una alergia declarada",
                        "skipped": True,
                        "error": True,
                        "result": {"coherent": None, "justification": str(e)},
                    })
        except ImportError:
            llm_checks = [{"error": True, "result": {"coherent": None, "justification": "OllamaClient no disponible"}}]

        # Aplicar la MISMA logica de combinacion que usa el agente real
        current_verdict = _combined_verdict([], result, llm_checks)

    # 5. VL (opcional) - analisis visual directo de la imagen completa.
    # El proyecto no incluye una funcion oficial de combinacion para VL (solo
    # existe para LLM textual en supervisor.py); esta logica ad-hoc replica el
    # mismo principio de diseno que _combined_verdict, con dos correcciones
    # sobre la version anterior (ver hallazgo 6 en las notas de evaluacion):
    #
    # 1. Confidence-gated: antes, CUALQUIER "coherent: false" de VL forzaba
    #    "inconsistent" sin mirar su confianza -- a diferencia del LLM
    #    textual, que ya exige confidence != "low" desde antes. Un modelo
    #    visual de 3B con conocimiento clinico limitado (ver hallazgo 8)
    #    puede reportar baja confianza y aun asi acertar por casualidad; sin
    #    el filtro, esos aciertos casuales de baja confianza pesaban igual
    #    que una deteccion segura, lo que es estadisticamente arriesgado.
    #
    # 2. Bidireccional pero con un limite deliberado: si VL confirma
    #    coherencia con confianza alta o media, puede REVERTIR una
    #    degradacion introducida por el LLM textual (o por un error previo
    #    de VL/LLM) devolviendo el veredicto al que darian las reglas solas
    #    (`rule_only_verdict`). Pero VL NUNCA puede anular un veredicto que
    #    YA coincide con `rule_only_verdict`: el motor de reglas es, con
    #    diferencia, la fuente mas fiable de esta evaluacion (0 falsos
    #    positivos verificados en sus reglas deterministas, ver hallazgos
    #    10/12/13/14), y dejar que un modelo visual con calibracion no
    #    verificada la sobrescriba pondria en riesgo esa fiabilidad a cambio
    #    de una ganancia no demostrada.
    vl_check = None
    if use_vl:
        try:
            from src.llm.qwen_vl_client import QwenVLClient
            vl_client = QwenVLClient()
            # check_urgency_coherence (ver hallazgo 8, continuacion final en
            # las notas de evaluacion) descompone la comprobacion en 2
            # llamadas simples con la logica condicional resuelta en Python
            # -- validado con 28/30 aciertos sobre los 30 documentos de
            # CN-001-ES, incluyendo 5/5 en el defecto que debe detectar.
            #
            # check_document_coherence (chequeo binario general de
            # fechas/firmas/identidad) se DESACTIVA para el resto de
            # protocolos -- ver hallazgo 17: verificado con una evaluacion
            # completa de 5 familias que este chequeo generico entra en
            # bucles de repeticion o alucina un problema falso cuando el
            # documento esta bien (el kappa agregado bajo de 0,64 a 0,60 al
            # activarlo en las 4 familias sin motivo_consulta). Se prefiere
            # no usarlo antes que arriesgar ese perjuicio ya medido.
            tiene_motivo_consulta = any(
                c.get("nombre_zona") == "motivo_consulta"
                for c in protocol.get("campos_obligatorios", [])
            )
            if tiene_motivo_consulta:
                vl_check = vl_client.check_urgency_coherence(image_path)
                vl_coherent = vl_check.get("coherent")
                vl_confidence = vl_check.get("confidence", "high")
                if vl_coherent is False and vl_confidence != "low":
                    current_verdict = "inconsistent"
                elif (
                    vl_coherent is True
                    and vl_confidence in ("high", "medium")
                    and current_verdict != rule_only_verdict
                ):
                    current_verdict = rule_only_verdict
                elif vl_coherent is None and current_verdict == "valid":
                    current_verdict = "incomplete"
        except Exception as e:
            vl_check = {"coherent": None, "justification": f"Error VL: {e}"}
            if current_verdict == "valid":
                current_verdict = "incomplete"

    elapsed = time.time() - t0

    return {
        "verdict": current_verdict,
        "rule_only_verdict": rule_only_verdict,
        "weighted_score": result.weighted_score,
        "checks_passed": result.checks_passed,
        "checks_total": result.checks_total,
        "issues": [
            {
                "rule_id": iss.rule_id,
                "severity": iss.severity,
                "zone": iss.zone,
                "message": iss.message,
                "details": iss.details,
            }
            for iss in result.issues
        ],
        "score_breakdown": [
            {
                "criterio": item.criterio,
                "categoria": item.categoria,
                "peso": item.peso,
                "aprobado": item.aprobado,
                "detalle": item.detalle,
            }
            for item in result.score_breakdown
        ],
        "ocr": {
            "confidence": round(doc_ocr.avg_confidence, 2),
            "blocks": len(doc_ocr.blocks),
            "text_chars": len(doc_ocr.full_text),
        },
        "signatures_detected": len(signatures),
        "checkboxes_detected": len(checkboxes),
        "checkboxes_checked": sum(1 for cb in checkboxes if cb.get("checked")),
        "llm_checks": llm_checks,
        "vl_check": vl_check,
        "elapsed_s": round(elapsed, 3),
    }


def evaluate_corpus(corpus_dir: str, protocols_dir: str,
                    use_llm: bool = False, use_vl: bool = False,
                    output_dir: str = None, limit: int = None,
                    seed: int = 42, partition: str = "all",
                    llm_model: str = None) -> dict:
    """
    Ejecuta la evaluacion integral del agente sobre el corpus sintetico.

    `partition`: "all" (por defecto), "dev" o "eval". El corpus v4 marca
    cada documento con gt["partition"] segun la particion definida en
    generate_synthetic_corpus.py (SUBSTATE_PLAN): "dev" (layout A/B,
    usado para fijar umbrales/regex/pesos) o "eval" (layout C, cabecera
    ampliada y firma lateral, congelado para la metrica final). Las
    metricas que se citen en la memoria como resultado final deben
    ejecutarse con partition="eval" unicamente, nunca sobre "dev" ni
    sobre el corpus mezclado, para no reutilizar documentos usados en el
    ajuste. Documentos de un corpus antiguo sin campo "partition" se
    incluyen siempre (se asume "all").
    """
    gt_data = load_ground_truth(corpus_dir)
    docs = gt_data["documents"]
    corpus_path = Path(corpus_dir)

    if partition != "all":
        # d.get("partition", partition) == partition haria que un
        # documento SIN campo "partition" (corpus antiguo y ya en desuso,
        # data/synthetic_corpus anterior al diseno v4 -- NO el corpus
        # sintetico actual de 230 documentos, data/synthetic_corpus_v2,
        # donde todo documento tiene ese campo) pasara el filtro SIEMPRE,
        # sin importar que particion se pida -- se detecto que esto hacia
        # que "--partition eval" se ejecutara en silencio sobre aquel
        # corpus antiguo completo (75/75 documentos, deprecado) en vez de
        # fallar o avisar, invalidando la metrica sin ningun error
        # visible. Ahora un documento sin ese campo se EXCLUYE
        # explicitamente en vez de incluirse por defecto.
        docs = [d for d in docs if d.get("partition") == partition]
        print(f"Particion seleccionada: '{partition}' -> {len(docs)} documentos")

    if limit is not None and limit < len(docs):
        import random as _random
        rng = _random.Random(seed)

        families = sorted(set(d["family"] for d in docs))
        states = sorted(set(d["state"] for d in docs))
        groups: dict[tuple, list] = {(f, s): [] for f in families for s in states}
        for d in docs:
            groups[(d["family"], d["state"])].append(d)

        # Orden de visita: state-mayor, family-menor. Esto garantiza que las
        # primeras N*len(families) muestras cubran TODAS las familias antes
        # de repetir un estado, en vez de agotar alfabeticamente las
        # primeras familias/estados y dejar el resto sin representacion
        # (bug detectado: con 25 grupos family x state y limit=15, un
        # round-robin simple sobre `keys` ordenadas solo recorria los
        # primeros 15 grupos alfabeticos, excluyendo MED-001-ES y
        # PREOP-001-ES por completo).
        visit_order = [(f, s) for s in states for f in families]

        sampled = []
        idx = 0
        max_iters = len(visit_order) * max(1, (len(docs) // max(len(visit_order), 1)) + 2)
        iters = 0
        while len(sampled) < limit and iters < max_iters:
            key = visit_order[idx % len(visit_order)]
            group = groups[key]
            if group:
                pick = rng.randrange(len(group))
                sampled.append(group.pop(pick))
            idx += 1
            iters += 1
            if idx % len(visit_order) == 0 and all(not g for g in groups.values()):
                break
        docs = sampled[:limit]

        covered_families = sorted(set(d["family"] for d in docs))
        print(f"Muestra estratificada: {len(docs)}/{len(gt_data['documents'])} documentos "
              f"(limit={limit}, seed={seed}) - familias cubiertas: {covered_families}")

    # Cargar protocolos directamente desde YAML (sin depender de ProtocolLoader)
    import yaml as _yaml
    protocols_path = Path(protocols_dir)
    if not protocols_path.exists():
        raise FileNotFoundError(f"No existe el directorio de protocolos: {protocols_path}")
    protocols = {}
    for yaml_file in sorted(protocols_path.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as fh:
            proto = _yaml.safe_load(fh) or {}
        pid = proto.get("id_protocolo", yaml_file.stem)
        protocols[pid] = proto
        print(f"  Protocolo cargado: {pid} <- {yaml_file.name}")
    if not protocols:
        raise RuntimeError(f"No se encontraron protocolos YAML en {protocols_path}")
    print(f"Protocolos cargados: {list(protocols.keys())}")

    # Resultados
    results = []
    y_true = []
    y_pred = []
    errors = []

    total = len(docs)
    print(f"\nEvaluando {total} documentos...")
    config_label = "reglas"
    if use_llm:
        config_label += "+LLM"
    if use_vl:
        config_label += "+VL"
    print(f"Configuracion: {config_label}\n")

    for i, doc_gt in enumerate(docs, 1):
        family = doc_gt["family"]
        state = doc_gt["state"]
        doc_id = doc_gt["doc_id"]
        image_path = str(corpus_path / doc_gt["filepath"])

        # Buscar protocolo
        protocol = protocols.get(family)
        if protocol is None:
            err_msg = f"Protocol '{family}' not found. Available: {list(protocols.keys())}"
            errors.append({"doc_id": doc_id, "error": err_msg})
            if i <= 3:  # Solo imprimir los primeros para diagnostico
                print(f"  WARN: {err_msg}")
            continue

        # Ejecutar pipeline
        result = run_pipeline_on_document(
            image_path, protocol,
            use_llm=use_llm, use_vl=use_vl, llm_model=llm_model
        )

        if result.get("verdict") == "error":
            err_msg = result.get("error", "Unknown")
            errors.append({"doc_id": doc_id, "error": err_msg})
            print(f"  [{i:3d}/{total}] ERROR en {doc_id}: {err_msg}")
            continue

        # Registrar resultado
        expected = doc_gt["expected_verdict"]
        predicted = result["verdict"]
        y_true.append(expected)
        y_pred.append(predicted)

        doc_result = {
            "doc_id": doc_id,
            "family": family,
            "state": state,
            "expected_verdict": expected,
            "predicted_verdict": predicted,
            "rule_only_verdict": result.get("rule_only_verdict"),
            "correct": expected == predicted,
            "weighted_score": result["weighted_score"],
            "ocr_confidence": result["ocr"]["confidence"],
            "ocr_blocks": result["ocr"]["blocks"],
            "checks_passed": result["checks_passed"],
            "checks_total": result["checks_total"],
            "signatures_detected": result["signatures_detected"],
            "checkboxes_detected": result["checkboxes_detected"],
            "checkboxes_checked": result["checkboxes_checked"],
            "issues": result["issues"],
            "score_breakdown": result["score_breakdown"],
            "elapsed_s": result["elapsed_s"],
            # Ground truth comparisons
            "gt_signatures_expected": doc_gt["signatures"]["expected"],
            "gt_signatures_present": doc_gt["signatures"]["present"],
            "gt_checkboxes_expected": doc_gt["checkboxes"]["expected"],
            "gt_checkboxes_checked": doc_gt["checkboxes"]["checked"],
            "gt_fields": doc_gt["fields"],
            "gt_semantic_issues": doc_gt["semantic_issues"],
        }
        results.append(doc_result)

        status = "OK" if expected == predicted else "MISS"
        pct = i / total * 100
        print(f"  [{i:3d}/{total}] ({pct:5.1f}%) {doc_id:40s} "
              f"exp={expected:12s} pred={predicted:12s} [{status}] "
              f"({result['elapsed_s']:.1f}s)")

    # --------------- Calcular metricas ---------------
    print(f"\n{'='*60}")
    print(f"Calculando metricas...")

    # Matriz de confusion
    cm = confusion_matrix(y_true, y_pred)

    # Kappa de Cohen
    kappa = cohens_kappa(y_true, y_pred)

    # Bootstrap CIs
    print("  Calculando intervalos de confianza bootstrap...")
    ci_f1 = bootstrap_ci(y_true, y_pred, macro_f1, n_bootstrap=1000)
    ci_acc = bootstrap_ci(y_true, y_pred, accuracy_score, n_bootstrap=1000)
    ci_kappa = bootstrap_ci(y_true, y_pred, cohens_kappa, n_bootstrap=1000)

    # Metricas por familia
    per_family = {}
    for family in sorted(set(r["family"] for r in results)):
        fam_results = [r for r in results if r["family"] == family]
        fam_true = [r["expected_verdict"] for r in fam_results]
        fam_pred = [r["predicted_verdict"] for r in fam_results]
        fam_cm = confusion_matrix(fam_true, fam_pred)
        per_family[family] = {
            "n_docs": len(fam_results),
            "accuracy": fam_cm["accuracy"],
            "macro_f1": fam_cm["macro"]["f1"],
            "per_class": fam_cm["per_class"],
            "correct": fam_cm["correct"],
        }

    # Metricas de firmas (solo PREOP que requiere firma)
    sig_tp = sig_fp = sig_fn = sig_tn = 0
    for r in results:
        expected_sigs = r["gt_signatures_expected"]
        detected_sigs = r["signatures_detected"]
        if expected_sigs > 0:
            if detected_sigs >= expected_sigs:
                sig_tp += 1
            else:
                sig_fn += 1
        else:
            if detected_sigs > 0:
                sig_fp += 1
            else:
                sig_tn += 1

    sig_precision = sig_tp / (sig_tp + sig_fp) if (sig_tp + sig_fp) > 0 else 0.0
    sig_recall = sig_tp / (sig_tp + sig_fn) if (sig_tp + sig_fn) > 0 else 0.0
    sig_f1 = 2 * sig_precision * sig_recall / (sig_precision + sig_recall) if (sig_precision + sig_recall) > 0 else 0.0

    # Metricas de casillas (PREOP)
    cb_tp = cb_fp = cb_fn = cb_tn = 0
    for r in results:
        expected_cb = r["gt_checkboxes_expected"]
        detected_cb = r["checkboxes_detected"]
        if expected_cb > 0:
            if detected_cb > 0:
                cb_tp += 1
            else:
                cb_fn += 1
        else:
            if detected_cb > 0:
                cb_fp += 1
            else:
                cb_tn += 1

    cb_precision = cb_tp / (cb_tp + cb_fp) if (cb_tp + cb_fp) > 0 else 0.0
    cb_recall = cb_tp / (cb_tp + cb_fn) if (cb_tp + cb_fn) > 0 else 0.0
    cb_f1 = 2 * cb_precision * cb_recall / (cb_precision + cb_recall) if (cb_precision + cb_recall) > 0 else 0.0

    # Metricas de deteccion de campos por protocolo
    field_detection = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    for r in results:
        for field_name, gt_present in r["gt_fields"].items():
            # Buscar en score_breakdown si el campo fue detectado
            campo_key = f"CAMPO-{field_name}"
            detected = any(
                item["criterio"] == campo_key and item["aprobado"]
                for item in r["score_breakdown"]
            )
            if gt_present and detected:
                field_detection[field_name]["tp"] += 1
            elif gt_present and not detected:
                field_detection[field_name]["fn"] += 1
            elif not gt_present and detected:
                field_detection[field_name]["fp"] += 1

    field_metrics = {}
    for field_name, counts in sorted(field_detection.items()):
        tp = counts["tp"]
        fn = counts["fn"]
        fp = counts["fp"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0
        field_metrics[field_name] = {
            "precision": round(p, 4),
            "recall": round(r_val, 4),
            "f1": round(f, 4),
            "tp": tp, "fn": fn, "fp": fp,
        }

    # Tiempos
    times = [r["elapsed_s"] for r in results]
    avg_time = sum(times) / len(times) if times else 0
    total_time = sum(times)

    # --------------- Ensamblar resultado final ---------------
    evaluation = {
        "metadata": {
            "evaluator": "evaluate_synthetic_corpus.py",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "configuration": config_label,
            "use_llm": use_llm,
            "use_vl": use_vl,
            "corpus_dir": corpus_dir,
            "protocols_dir": protocols_dir,
            "total_documents": total,
            "evaluated": len(results),
            "errors": len(errors),
        },
        "global_metrics": {
            "confusion_matrix": cm["matrix"],
            "classes": cm["classes"],
            "accuracy": cm["accuracy"],
            "macro_f1": cm["macro"]["f1"],
            "weighted_f1": cm["weighted"]["f1"],
            "kappa": kappa,
            "sensitivity_valid": cm["sensitivity_valid"],
            "specificity_valid": cm["specificity_valid"],
            "per_class": cm["per_class"],
            "macro": cm["macro"],
            "weighted": cm["weighted"],
        },
        "confidence_intervals": {
            "macro_f1": ci_f1,
            "accuracy": ci_acc,
            "kappa": ci_kappa,
        },
        "per_family": per_family,
        "detection_metrics": {
            "signatures": {
                "precision": round(sig_precision, 4),
                "recall": round(sig_recall, 4),
                "f1": round(sig_f1, 4),
                "tp": sig_tp, "fp": sig_fp, "fn": sig_fn, "tn": sig_tn,
            },
            "checkboxes": {
                "precision": round(cb_precision, 4),
                "recall": round(cb_recall, 4),
                "f1": round(cb_f1, 4),
                "tp": cb_tp, "fp": cb_fp, "fn": cb_fn, "tn": cb_tn,
            },
            "fields": field_metrics,
        },
        "timing": {
            "avg_per_document_s": round(avg_time, 3),
            "total_s": round(total_time, 1),
        },
        "documents": results,
        "errors": errors,
    }

    # --------------- Imprimir resumen ---------------
    print(f"\n{'='*60}")
    print(f"RESULTADOS DE EVALUACION INTEGRAL")
    print(f"{'='*60}")
    print(f"Configuracion: {config_label}")
    print(f"Documentos evaluados: {len(results)}/{total} (errores: {len(errors)})")
    print(f"\n--- Metricas globales ---")
    print(f"  Accuracy:       {cm['accuracy']:.4f}")
    print(f"  Macro F1:       {cm['macro']['f1']:.4f}  (CI 95%: [{ci_f1['ci_lower']:.4f}, {ci_f1['ci_upper']:.4f}])")
    print(f"  Weighted F1:    {cm['weighted']['f1']:.4f}")
    print(f"  Kappa Cohen:    {kappa:.4f}  (CI 95%: [{ci_kappa['ci_lower']:.4f}, {ci_kappa['ci_upper']:.4f}])")
    print(f"  Sensibilidad:   {cm['sensitivity_valid']:.4f}")
    print(f"  Especificidad:  {cm['specificity_valid']:.4f}")

    print(f"\n--- Matriz de confusion ---")
    print(f"  {'':15s} {'valid':>10s} {'incomplete':>10s} {'inconsistent':>12s}")
    for i, cls in enumerate(cm["classes"]):
        row = "  ".join(f"{cm['matrix'][i][j]:>10d}" for j in range(3))
        print(f"  {cls:15s} {row}")

    print(f"\n--- F1 por clase ---")
    for cls, metrics in cm["per_class"].items():
        print(f"  {cls:15s}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  (n={metrics['support']})")

    print(f"\n--- Por familia ---")
    for fam, fm in per_family.items():
        print(f"  {fam:15s}  acc={fm['accuracy']:.3f}  F1={fm['macro_f1']:.3f}  ({fm['correct']}/{fm['n_docs']})")

    print(f"\n--- Deteccion de firmas ---")
    print(f"  P={sig_precision:.3f}  R={sig_recall:.3f}  F1={sig_f1:.3f}  (TP={sig_tp} FP={sig_fp} FN={sig_fn} TN={sig_tn})")

    print(f"\n--- Deteccion de casillas ---")
    print(f"  P={cb_precision:.3f}  R={cb_recall:.3f}  F1={cb_f1:.3f}  (TP={cb_tp} FP={cb_fp} FN={cb_fn} TN={cb_tn})")

    print(f"\n--- Deteccion de campos ---")
    for field_name, fm in sorted(field_metrics.items()):
        print(f"  {field_name:30s}  P={fm['precision']:.3f}  R={fm['recall']:.3f}  F1={fm['f1']:.3f}")

    print(f"\n--- Tiempo ---")
    print(f"  Promedio: {avg_time:.2f}s/doc  |  Total: {total_time:.1f}s")

    # Guardar resultados
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = out_path / f"eval_{config_label}_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"\nResultados guardados en: {result_file}")

    return evaluation


def _requirements_summary(protocol: dict) -> str:
    """
    Construye una descripcion breve en lenguaje natural de los campos que
    exige un protocolo, a partir de las descripciones de
    campos_obligatorios del YAML. Se usa como contexto para
    classify_document_verdict, de modo que Qwen2.5-VL tenga un criterio de
    "completo" equivalente al que usan las reglas, sin exponerle la
    sintaxis tecnica del protocolo (regex, pesos, zonas).
    """
    descripciones = [
        c.get("descripcion", c.get("nombre_zona", ""))
        for c in protocol.get("campos_obligatorios", [])
    ]
    return "; ".join(d for d in descripciones if d)


def evaluate_vl_standalone(corpus_dir: str, protocols_dir: str,
                           output_dir: str = None, limit: int = None,
                           partition: str = "all",
                           vl_model: str = None) -> dict:
    """
    Evalua a Qwen2.5-VL de forma AISLADA: cada documento se clasifica
    directamente en las 3 clases del veredicto (valid/incomplete/
    inconsistent) mirando solo la imagen, sin pasar por OCR ni por el
    motor de reglas.

    Esto mide la capacidad del componente visual por si mismo, sin la
    limitacion estructural de la logica de combinacion usada en
    "reglas+LLM+VL" (que solo puede degradar un veredicto de las reglas, nunca
    mejorarlo) — permite comparar su kappa de 3 clases directamente contra
    "reglas" y "reglas+LLM" en igualdad de condiciones.

    Los documentos donde Qwen no da un veredicto valido (verdict=None,
    fallo de parseo o abstencion) se excluyen del calculo de metricas y se
    reportan aparte como tasa de abstencion, para no mezclar "incierto"
    con "seguro pero equivocado".
    """
    from src.llm.qwen_vl_client import QwenVLClient

    gt_data = load_ground_truth(corpus_dir)
    docs = gt_data["documents"]

    if partition != "all":
        # d.get("partition", partition) == partition haria que un
        # documento SIN campo "partition" (corpus antiguo y ya en desuso,
        # data/synthetic_corpus anterior al diseno v4 -- NO el corpus
        # sintetico actual de 230 documentos, data/synthetic_corpus_v2,
        # donde todo documento tiene ese campo) pasara el filtro SIEMPRE,
        # sin importar que particion se pida -- se detecto que esto hacia
        # que "--partition eval" se ejecutara en silencio sobre aquel
        # corpus antiguo completo (75/75 documentos, deprecado) en vez de
        # fallar o avisar, invalidando la metrica sin ningun error
        # visible. Ahora un documento sin ese campo se EXCLUYE
        # explicitamente en vez de incluirse por defecto.
        docs = [d for d in docs if d.get("partition") == partition]
        print(f"Particion seleccionada: '{partition}' -> {len(docs)} documentos")

    if limit is not None and limit < len(docs):
        families = sorted(set(d["family"] for d in docs))
        states = sorted(set(d["state"] for d in docs))
        groups: dict[tuple, list] = {(f, s): [] for f in families for s in states}
        for d in docs:
            groups[(d["family"], d["state"])].append(d)
        visit_order = [(f, s) for s in states for f in families]

        import random as _random
        rng = _random.Random(42)
        sampled = []
        idx = 0
        max_iters = len(visit_order) * max(1, (len(docs) // max(len(visit_order), 1)) + 2)
        iters = 0
        while len(sampled) < limit and iters < max_iters:
            key = visit_order[idx % len(visit_order)]
            group = groups[key]
            if group:
                sampled.append(group.pop(rng.randrange(len(group))))
            idx += 1
            iters += 1
            if idx % len(visit_order) == 0 and all(not g for g in groups.values()):
                break
        docs = sampled[:limit]
        covered = sorted(set((d["family"], d["state"]) for d in docs))
        print(f"Muestra estratificada: {len(docs)} documentos, "
              f"{len(covered)} combinaciones familia x estado cubiertas")

    import yaml as _yaml
    protocols_path = Path(protocols_dir)
    protocols = {}
    for yaml_file in sorted(protocols_path.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as fh:
            proto = _yaml.safe_load(fh) or {}
        pid = proto.get("id_protocolo", yaml_file.stem)
        protocols[pid] = proto

    client = QwenVLClient(model=vl_model) if vl_model else QwenVLClient()
    corpus_path = Path(corpus_dir)

    total = len(docs)
    print(f"\nEvaluando {total} documentos (Qwen2.5-VL AISLADO, veredicto de 3 clases)...\n")

    results = []
    abstentions = []
    t_start = time.time()

    for i, doc_gt in enumerate(docs, 1):
        family = doc_gt["family"]
        doc_id = doc_gt["doc_id"]
        image_path = str(corpus_path / doc_gt["filepath"])
        protocol = protocols.get(family, {})
        document_type = protocol.get("tipo_documento", family)
        requirements = _requirements_summary(protocol)

        t0 = time.time()
        try:
            vl_result = client.classify_document_verdict(image_path, document_type, requirements)
        except Exception as e:
            vl_result = {"verdict": None, "confidence": "low", "reason": f"Error: {e}"}
        elapsed = time.time() - t0

        verdict = vl_result.get("verdict")
        if verdict not in VERDICT_CLASSES:
            abstentions.append({"doc_id": doc_id, "reason": vl_result.get("reason", "")})
            status = "ABSTENCION"
        else:
            status = "OK" if verdict == doc_gt["expected_verdict"] else "MISS"

        results.append({
            "doc_id": doc_id,
            "family": family,
            "state": doc_gt["state"],
            "expected_verdict": doc_gt["expected_verdict"],
            "vl_verdict": verdict,
            "vl_confidence": vl_result.get("confidence"),
            "vl_reason": vl_result.get("reason"),
            "elapsed_s": round(elapsed, 3),
        })

        print(f"  [{i:3d}/{total}] ({i/total*100:5.1f}%) {doc_id:40s} "
              f"exp={doc_gt['expected_verdict']:12s} vl={str(verdict):12s} [{status}] ({elapsed:.1f}s)")

    total_time = time.time() - t_start

    decisive = [r for r in results if r["vl_verdict"] in VERDICT_CLASSES]
    y_true = [r["expected_verdict"] for r in decisive]
    y_pred = [r["vl_verdict"] for r in decisive]

    print(f"\n{'='*60}")
    print(f"RESULTADOS: Qwen2.5-VL AISLADO (veredicto de 3 clases)")
    print(f"{'='*60}")
    print(f"Documentos evaluados: {total} | Decisivos: {len(decisive)} | "
          f"Abstenciones: {len(abstentions)} ({len(abstentions)/max(total,1)*100:.1f}%)")

    metrics = None
    kappa = None
    if decisive:
        cm = confusion_matrix(y_true, y_pred)
        kappa = cohens_kappa(y_true, y_pred)
        kappa_ci = bootstrap_ci(y_true, y_pred, cohens_kappa)

        print(f"\n--- Metricas globales (solo documentos decisivos) ---")
        print(f"  Accuracy:  {cm['accuracy']:.4f}")
        print(f"  Macro F1:  {cm['macro']['f1']:.4f}")
        print(f"  Kappa Cohen: {kappa:.4f}  (CI 95%: [{kappa_ci['ci_lower']:.4f}, {kappa_ci['ci_upper']:.4f}])")

        print(f"\n--- Matriz de confusion ---")
        classes = cm["classes"]
        print(f"  {'':20s} " + " ".join(f"{c:>12s}" for c in classes))
        for i, cls in enumerate(classes):
            row = cm["matrix"][i]
            print(f"  {cls:20s} " + " ".join(f"{v:>12d}" for v in row))

        print(f"\n--- F1 por clase ---")
        for cls, m in cm["per_class"].items():
            print(f"  {cls:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['support']})")

        metrics = {
            "accuracy": cm["accuracy"], "macro_f1": cm["macro"]["f1"],
            "kappa": kappa, "kappa_ci": kappa_ci, "confusion_matrix": cm,
        }

    print(f"\n--- Tiempo ---")
    print(f"  Promedio: {total_time/max(total,1):.2f}s/doc  |  Total: {total_time:.1f}s")

    output = {
        "metadata": {
            "mode": "vl_standalone_3class",
            "vl_model": vl_model or "qwen2.5vl:7b (default)",
            "total": total, "decisive": len(decisive), "abstentions": len(abstentions),
        },
        "metrics": metrics,
        "documents": results,
        "abstentions": abstentions,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = out_path / f"eval_vl_standalone_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResultados guardados en: {result_file}")

    return output


def run_ablation(corpus_dir: str, protocols_dir: str,
                 output_dir: str = None, limit: int = None,
                 vl_limit: int = 25, partition: str = "all") -> dict:
    """
    Ejecuta ablacion completa: reglas solas, reglas+LLM, reglas+LLM+VL --
    las tres configuraciones citadas en la memoria (tablas de metricas
    globales, precision/exhaustividad/F1 y matrices de confusion). Compara
    las tres SOBRE LA MISMA MUESTRA de documentos, para que la tabla
    comparativa sea valida (misma base en las tres filas).

    IMPORTANTE -- para reproducir los numeros exactos citados en la memoria
    (kappa = 0,84 / 0,93 / 0,93 sobre la particion de evaluacion, n=115),
    esta funcion debe llamarse con partition="eval" y vl_limit=115 (o el
    tamano real de esa particion), es decir SIN submuestreo. El resultado
    final del TFM se obtuvo asi, no con el `vl_limit` por defecto de esta
    funcion. Con los valores por defecto (`vl_limit=25`), esta ablacion es
    una vista previa rapida, no el experimento reportado.

    Por defecto, "reglas" y "reglas+LLM" tambien se evaluan sobre la muestra
    de `vl_limit` documentos (no sobre el corpus completo), precisamente
    para poder comparar las tres filas de forma justa.

    `vl_limit` por defecto es 25 = 5 familias x 5 estados, lo que garantiza
    UNA muestra con cobertura completa de todas las combinaciones familia x
    estado (incluyendo la clase "valid", que con muestras mas pequenas puede
    quedar fuera por completo y distorsionar la sensibilidad/especificidad).
    Qwen2.5-VL puede tardar ~2 minutos por imagen en CPU (ver docstring de
    QwenVLClient); usa --vl-limit 115 (con --partition eval) solo si
    dispones de varias horas o GPU, para reproducir el resultado citado.

    `limit`: si se especifica explicitamente, fuerza un tamano de muestra
    distinto para "reglas"/"reglas+LLM" (por ejemplo --limit 115 para usar
    la particion completa en esas dos filas y solo aplicar el submuestreo a
    reglas+LLM+VL) a costa de que la comparacion entre filas deje de ser
    estrictamente sobre los mismos documentos.
    """
    baseline_limit = limit if limit is not None else vl_limit
    configs = [
        ("reglas", False, False, baseline_limit),
        ("reglas+LLM", True, False, baseline_limit),
        ("reglas+LLM+VL", True, True, vl_limit),
    ]

    all_results = {}
    for label, use_llm, use_vl, config_limit in configs:
        print(f"\n{'#'*60}")
        print(f"# ABLACION: {label}" + (f" (muestra de {config_limit} docs)" if config_limit else ""))
        print(f"{'#'*60}")
        try:
            result = evaluate_corpus(
                corpus_dir, protocols_dir,
                use_llm=use_llm, use_vl=use_vl,
                output_dir=output_dir, limit=config_limit,
                partition=partition
            )
            all_results[label] = {
                "accuracy": result["global_metrics"]["accuracy"],
                "macro_f1": result["global_metrics"]["macro_f1"],
                "kappa": result["global_metrics"]["kappa"],
                "sensitivity": result["global_metrics"]["sensitivity_valid"],
                "specificity": result["global_metrics"]["specificity_valid"],
                "avg_time_s": result["timing"]["avg_per_document_s"],
            }
        except Exception as e:
            print(f"  ERROR en {label}: {e}")
            all_results[label] = {"error": str(e)}

    # Tabla comparativa
    print(f"\n{'='*60}")
    print(f"TABLA COMPARATIVA DE ABLACION")
    print(f"{'='*60}")
    print(f"  {'Config':15s} {'Acc':>8s} {'F1':>8s} {'Kappa':>8s} {'Sens':>8s} {'Spec':>8s} {'t/doc':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, metrics in all_results.items():
        if "error" in metrics:
            print(f"  {label:15s} {'ERROR':>8s}")
        else:
            print(f"  {label:15s} {metrics['accuracy']:>8.3f} {metrics['macro_f1']:>8.3f} "
                  f"{metrics['kappa']:>8.3f} {metrics['sensitivity']:>8.3f} "
                  f"{metrics['specificity']:>8.3f} {metrics['avg_time_s']:>7.2f}s")

    if output_dir:
        ablation_file = Path(output_dir) / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(ablation_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResultados de ablacion guardados en: {ablation_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluacion integral del agente PathwayGuard"
    )
    parser.add_argument(
        "--corpus-dir", default="data/synthetic_corpus_v2",
        help="Directorio del corpus sintetico"
    )
    parser.add_argument(
        "--protocols-dir", default="configs/protocolos_es",
        help="Directorio de protocolos en espanol"
    )
    parser.add_argument(
        "--output-dir", default="data/results/synthetic",
        help="Directorio de salida para resultados"
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Activar validacion semantica con Mistral-7B"
    )
    parser.add_argument(
        "--use-vl", action="store_true",
        help="Activar analisis visual con Qwen2.5-VL"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Ejecutar ablacion completa (reglas, reglas+LLM, reglas+LLM+VL)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluar solo una muestra estratificada de N documentos (todas las configs)"
    )
    parser.add_argument(
        "--vl-limit", type=int, default=25,
        help="Tamano de muestra para la ablacion (25 = 5 familias x 5 estados, cobertura completa). "
             "Se usa para las 3 configs (reglas/reglas+LLM/reglas+LLM+VL) salvo que --limit indique otro valor "
             "para las dos primeras. Qwen2.5-VL es lento en CPU (~2min/doc en el peor caso)."
    )
    parser.add_argument(
        "--partition", choices=["all", "dev", "eval"], default="all",
        help="Subconjunto del corpus a evaluar segun gt['partition'] (corpus v4+). "
             "'eval' = particion congelada (layout C) que debe usarse para la metrica "
             "final citada en la memoria; 'dev' = particion de ajuste (layout A/B), "
             "solo para diagnostico durante el desarrollo; 'all' = corpus completo."
    )

    parser.add_argument(
        "--llm-model", default=None,
        help="Tag exacto del modelo Ollama a usar con --use-llm, p. ej. 'qwen3:8b'. "
             "Sin este flag se usa el backend oficial documentado en la memoria "
             "(mistral:7b via OllamaClient). Pensado para exploracion de "
             "modelos alternativos, no para el resultado final citado en el TFM."
    )
    parser.add_argument(
        "--vl-standalone", action="store_true",
        help="Evalua Qwen2.5-VL AISLADO: veredicto directo de 3 clases "
             "(valid/incomplete/inconsistent) mirando solo la imagen, sin OCR "
             "ni motor de reglas. Mide la capacidad del componente visual sin "
             "la limitacion de la logica de combinacion usada en --use-vl "
             "(que solo puede degradar un veredicto de las reglas, nunca "
             "mejorarlo). Ignora --use-llm/--use-vl si se pasa junto a ellos."
    )

    args = parser.parse_args()

    if args.vl_standalone:
        evaluate_vl_standalone(
            args.corpus_dir, args.protocols_dir,
            output_dir=args.output_dir, limit=args.limit,
            partition=args.partition, vl_model=args.llm_model
        )
    elif args.ablation:
        run_ablation(args.corpus_dir, args.protocols_dir, args.output_dir,
                     limit=args.limit, vl_limit=args.vl_limit, partition=args.partition)
    else:
        evaluate_corpus(
            args.corpus_dir, args.protocols_dir,
            use_llm=args.use_llm, use_vl=args.use_vl,
            output_dir=args.output_dir, limit=args.limit,
            partition=args.partition, llm_model=args.llm_model
        )


if __name__ == "__main__":
    main()
