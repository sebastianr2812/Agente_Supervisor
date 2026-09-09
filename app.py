"""
Supervisor de documentos clínicos - interfaz web de validacion de documentos sanitarios.

Uso:
    venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("TESSDATA_PREFIX", r"C:\Users\tcana\AppData\Local\tessdata")

import streamlit as st

# Se importa run_pipeline_on_document (no supervise_document) para reutilizar
# EXACTAMENTE el pipeline validado en la evaluacion integral (reglas + LLM +
# analisis visual dirigido, incluido el rescate de campos/fechas/identidad y
# la comprobacion de coherencia de urgencia con umbral de confianza), en vez
# de reimplementar esa logica por separado en la app y arriesgar que diverja
# de lo que realmente se midio (kappa = 0.93; ver memoria, seccion 5.11.3).
from evaluate_synthetic_corpus import run_pipeline_on_document
from src.ocr.extractor import extract_text_blocks
from src.rules.engine import ProtocolLoader

# PROTOCOLOS_DIR: la app usaba antes configs/protocolos (ingles, OCR en
# "eng") -- un conjunto de protocolos paralelo y nunca evaluado. Migrado
# (2026-09-09) a configs/protocolos_es, el unico validado en la memoria.
PROTOCOLOS_DIR = "configs/protocolos_es"

PROTOCOLS = {
    "Clinical Notes": "CN-001-ES",
    "Diagnostic & Lab Reports": "DLR-001-ES",
    "Medication Lists": "MED-001-ES",
    "Patient Admission Forms": "ADM-001-ES",
    "Preop Forms & Clinical Checklists": "PREOP-001-ES",
}
LABEL_BY_PROTOCOL = {v: k for k, v in PROTOCOLS.items()}


@st.cache_resource
def load_protocol_loader() -> ProtocolLoader:
    loader = ProtocolLoader(protocols_dir=PROTOCOLOS_DIR)
    loader.load_all()
    return loader


def get_protocol(protocol_id: str) -> dict:
    """Devuelve el diccionario del protocolo ya cargado (para pasarlo a
    run_pipeline_on_document, que opera sobre el dict, no sobre el id)."""
    return load_protocol_loader().get_protocol(protocol_id)


@st.cache_resource
def load_protocol_keywords() -> dict[str, set[str]]:
    """Palabras clave caracteristicas de cada protocolo (para clasificar el tipo de documento)."""
    loader = load_protocol_loader()
    keywords_by_protocol = {}
    for protocol_id in PROTOCOLS.values():
        protocol = loader.get_protocol(protocol_id)
        keywords: set[str] = set()
        for field in protocol.get("campos_obligatorios", []):
            for pattern in field.get("patrones_esperados", []):
                keywords.add(pattern.lower())
        keywords_by_protocol[protocol_id] = keywords
    return keywords_by_protocol


def guess_document_type(full_text: str) -> tuple[str, dict[str, int]]:
    """Compara el texto extraido por OCR contra las palabras clave de cada protocolo."""
    text_lower = full_text.lower()
    keywords_by_protocol = load_protocol_keywords()
    scores = {
        protocol_id: sum(1 for kw in keywords if kw in text_lower)
        for protocol_id, keywords in keywords_by_protocol.items()
    }
    best_protocol = max(scores, key=scores.get)
    return best_protocol, scores


@st.cache_resource
def load_criteria_text() -> dict[str, dict[str, str]]:
    """Texto en espanol (ya definido en cada protocolo _es) para cada criterio, por protocolo."""
    loader = load_protocol_loader()
    text_by_protocol: dict[str, dict[str, str]] = {}
    for protocol_id in PROTOCOLS.values():
        protocol = loader.get_protocol(protocol_id)
        texts: dict[str, str] = {}
        for f in protocol.get("campos_obligatorios", []):
            texts[f"CAMPO-{f['nombre_zona']}"] = f["descripcion"]
        for r in protocol.get("reglas_contenido", []):
            texts[r["id_regla"]] = r["mensaje_error"]
        for r in protocol.get("reglas_coherencia", []):
            texts[r["id_regla"]] = r["condicion"]
        text_by_protocol[protocol_id] = texts
    return text_by_protocol


def criterio_reason(protocol_id: str, criterio: str, aprobado: bool) -> str:
    """Construye una frase en espanol a partir de la definicion del criterio en el protocolo."""
    texts = load_criteria_text().get(protocol_id, {})
    base = texts.get(criterio, criterio)
    if criterio.startswith("CAMPO-"):
        return f"Encontrado: {base}" if aprobado else f"Ausente: {base}"
    if criterio.startswith("RCH-"):
        return f"Coherente: {base}" if aprobado else f"No coherente: {base}"
    # reglas_contenido: 'base' es el mensaje de FALLO definido en el protocolo
    # (p.ej. "No se encontro fecha de nacimiento"), no debe mostrarse tal
    # cual cuando la regla SI se cumplio.
    return f"Requisito cumplido (regla: {base})" if aprobado else f"No cumplido: {base}"


VERDICT_STYLE = {
    "valid": ("VALID", "#1a7f37", "#e6f4ea"),
    "incomplete": ("INCOMPLETE", "#9a6700", "#fff8e6"),
    "inconsistent": ("INCONSISTENT", "#b91c1c", "#fde8e8"),
    "error": ("ERROR", "#b91c1c", "#fde8e8"),
}

st.set_page_config(page_title="Supervisor de documentos clínicos", page_icon="🩺", layout="centered")

st.title("🩺 Supervisor de documentos clínicos")
st.caption("Automated healthcare document supervision — OCR + rules + local LLM")

st.divider()

doc_type_label = st.selectbox("Document type", list(PROTOCOLS.keys()))
protocol_id = PROTOCOLS[doc_type_label]

col1, col2 = st.columns(2)
with col1:
    use_llm = st.checkbox(
        "Enable LLM semantic check (Mistral)",
        value=False,
        help="Requires Ollama running locally with mistral pulled. Fast (~seconds). "
        "Only affects protocols with a texto_coherente rule — currently only Clinical Notes.",
    )
with col2:
    use_qwen = st.checkbox(
        "Enable deep visual analysis (Qwen2.5-VL)",
        value=False,
        help="Requires Ollama with qwen2.5vl:7b pulled. SLOW on CPU (~1-3 minutes per "
        "document). Looks at the raw image directly instead of relying on OCR text. "
        "NOTE: this demo button runs a general coherence check that the project's own "
        "evaluation found to hurt results when applied broadly (see memoria, hallazgo "
        "17) -- it is kept here for exploration only, not as the validated configuration "
        "(which applies VL in a targeted way per rule, see evaluate_synthetic_corpus.py).",
    )

uploaded_file = st.file_uploader(
    "Upload a scanned document",
    type=["png", "jpg", "jpeg", "pdf"],
)

if uploaded_file is not None and uploaded_file.type != "application/pdf":
    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

validate_clicked = st.button("Validate document", type="primary", disabled=uploaded_file is None)

if validate_clicked and uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.divider()

    with st.spinner("Checking document type..."):
        try:
            ocr_preview = extract_text_blocks(tmp_path, lang="spa")
            guessed_protocol, type_scores = guess_document_type(ocr_preview.full_text)
        except Exception as exc:  # noqa: BLE001
            os.unlink(tmp_path)
            st.error(f"OCR error while checking document type: {exc}")
            st.stop()

    if type_scores[guessed_protocol] == 0:
        os.unlink(tmp_path)
        st.warning(
            "Could not confidently recognize this document as any of the 5 supported "
            "types. Validation was not run — please check the image quality or upload "
            "a supported document type."
        )
        st.stop()

    if guessed_protocol != protocol_id:
        os.unlink(tmp_path)
        st.error(
            f"This document does not correspond to the selected type "
            f"(**{doc_type_label}**). It looks more like a "
            f"**{LABEL_BY_PROTOCOL[guessed_protocol]}** document. Validation was not run."
        )
        with st.expander("Why did we guess this?"):
            st.write(
                {LABEL_BY_PROTOCOL[pid]: score for pid, score in type_scores.items()}
            )
        st.stop()

    protocol = get_protocol(protocol_id)
    spinner_msg = (
        "Running the validated pipeline (rules + LLM + targeted visual analysis, "
        "this can take 1-3 minutes on CPU)..."
        if use_qwen
        else "Document type confirmed. Running rules and agent pipeline..."
    )
    with st.spinner(spinner_msg):
        try:
            report = run_pipeline_on_document(
                tmp_path, protocol, use_llm=use_llm, use_vl=use_qwen
            )
            error_message = report.get("error") if report.get("verdict") == "error" else None
        except Exception as exc:  # noqa: BLE001
            report = None
            error_message = str(exc)

    os.unlink(tmp_path)

    if error_message:
        st.error(f"Pipeline error: {error_message}")
    else:
        verdict = report["verdict"]
        label, color, bg = VERDICT_STYLE.get(verdict, ("UNKNOWN", "#374151", "#f3f4f6"))
        st.markdown(
            f"""
            <div style="background-color:{bg};border:2px solid {color};
                        border-radius:10px;padding:18px;text-align:center;margin-bottom:16px;">
                <span style="color:{color};font-size:28px;font-weight:800;letter-spacing:1px;">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Weighted score", f"{report['weighted_score']:.1f}/100")
        metric_cols[1].metric("Checks passed", f"{report['checks_passed']}/{report['checks_total']}")
        metric_cols[2].metric("OCR confidence", f"{report['ocr']['confidence']:.1f}%")
        metric_cols[3].metric("Signatures found", report["signatures_detected"])

        if use_qwen and report.get("rule_only_verdict") != verdict:
            st.caption(
                f"ℹ️ Visual analysis changed the verdict from the rules-only result "
                f"(**{report.get('rule_only_verdict')}**) to **{verdict}**."
            )

        st.subheader("Attribute breakdown")
        breakdown = report.get("score_breakdown", [])
        if breakdown:
            rows = [
                {
                    "Criterion": item["criterio"],
                    "Category": item["categoria"],
                    "Weight": item["peso"],
                    "Passed": "Yes" if item["aprobado"] else "No",
                    "Reason": criterio_reason(protocol_id, item["criterio"], item["aprobado"]),
                }
                for item in breakdown
            ]
            st.table(rows)

        issues = report.get("issues", [])
        if issues:
            st.subheader("Issues found")
            for issue in issues:
                st.warning(f"**{issue['rule_id']}** ({issue['zone'] or 'n/a'}): {issue['message']}")
        else:
            st.success("No issues found.")

        llm_checks = report.get("llm_checks", [])
        if llm_checks:
            st.subheader("Semantic validation (LLM)")
            for item in llm_checks:
                result = item.get("result", {})
                if item.get("skipped"):
                    st.info(f"{item['rule_id']}: skipped ({result.get('justification', '')})")
                else:
                    coherent = result.get("coherent")
                    icon = "✅" if coherent else "❌" if coherent is False else "❓"
                    st.write(f"{icon} **{item['rule_id']}**: {result.get('justification', '')}")

        if use_qwen:
            st.subheader("Deep visual analysis (Qwen2.5-VL)")
            vl_check = report.get("vl_check")
            if vl_check is None:
                st.info(
                    "This document type has no open-ended clinical-coherence check assigned "
                    "to visual analysis (only Clinical Notes does, see memoria hallazgo 17). "
                    "Visual analysis was still applied in a targeted way as a fallback for any "
                    "field, date, or patient identifier the rules engine could not confirm from "
                    "OCR alone — its effect, if any, is already reflected in the checks above."
                )
            elif "error" in vl_check:
                st.error(f"Visual analysis error: {vl_check['error']}")
            elif "justification" in vl_check and vl_check.get("coherent") is None and "Error VL" in str(vl_check.get("justification", "")):
                st.error(vl_check["justification"])
            else:
                coherent = vl_check.get("coherent")
                icon = "✅" if coherent else "❌" if coherent is False else "❓"
                st.write(
                    f"{icon} **Coherent:** {coherent} "
                    f"(confidence: {vl_check.get('confidence', 'n/a')})"
                )
                st.write(vl_check.get("justification", ""))

        with st.expander("Full JSON report"):
            st.json(report)
else:
    st.info("Select a document type and upload a scanned document to begin.")
