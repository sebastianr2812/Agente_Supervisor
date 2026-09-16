"""
Selecciona una muestra de documentos de ClinOCR-Bench, corre el agente sobre
cada uno, y genera dos archivos Excel con una hoja por protocolo: uno "ciego"
para revision humana a nivel de atributo (sin el veredicto ni las respuestas
del agente) y otro "clave de respuestas" con el detalle completo para
comparar despues.

Uso:
    venv\\Scripts\\python.exe scripts\\build_human_review_sample.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# TESSDATA_PREFIX: solo se fija si no viene ya definida por el entorno. No se
# hardcodea una ruta de una maquina de desarrollo concreta; si tu instalacion
# de Tesseract no encuentra sus datos de idioma por defecto, exporta la
# variable de entorno antes de ejecutar este script.

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

from src.agent.supervisor import supervise_document
from src.rules.engine import ProtocolLoader

CLINOCR_DIR = ROOT_DIR / "data" / "clinocr_bench" / "raw" / "ClinOCR-Bench"
OUTPUT_DIR = ROOT_DIR / "data" / "human_review"

# protocol_id -> list of (template_number, subset) to sample from
PROTOCOL_SOURCES: dict[str, list[tuple[int, str]]] = {
    "CN-001": [(4, "normal"), (4, "poor"), (4, "rotated"), (5, "normal"), (5, "poor")],
    "DLR-001": [
        (1, "normal"), (2, "normal"), (6, "normal"),
        (11, "tables"), (12, "tables"), (15, "tables"),
    ],
    "MED-001": [(9, "tables"), (9, "mixed")],
    "ADM-001": [(13, "tables"), (13, "mixed")],
    "PREOP-001": [(10, "tables"), (10, "mixed"), (14, "tables"), (14, "mixed")],
}

SAMPLES_PER_PROTOCOL = 9


def find_samples(template: int, subset: str, count: int, used: set[str]) -> list[Path]:
    pattern = f"template_{template}_sample_*_{subset}.jpg"
    candidates = sorted((CLINOCR_DIR / "scans" / subset).glob(pattern))
    selected = []
    for path in candidates:
        if path.name in used:
            continue
        selected.append(path)
        used.add(path.name)
        if len(selected) >= count:
            break
    return selected


def build_sample_for_protocol(protocol_id: str) -> list[Path]:
    sources = PROTOCOL_SOURCES[protocol_id]
    per_source = max(1, SAMPLES_PER_PROTOCOL // len(sources) + 1)
    used: set[str] = set()
    documents: list[Path] = []

    for template, subset in sources:
        documents.extend(find_samples(template, subset, per_source, used))
        if len(documents) >= SAMPLES_PER_PROTOCOL:
            break

    return documents[:SAMPLES_PER_PROTOCOL]


def field_columns(protocol: dict) -> list[dict]:
    """Devuelve los campos_obligatorios del protocolo con su peso y descripcion."""
    return [
        {
            "nombre_zona": f["nombre_zona"],
            "peso": f.get("peso", 2),
            "descripcion": f["descripcion"],
        }
        for f in protocol.get("campos_obligatorios", [])
    ]


def field_result(breakdown: list[dict], nombre_zona: str) -> str:
    target = f"CAMPO-{nombre_zona}"
    for item in breakdown:
        if item["criterio"] == target:
            return "Yes" if item["aprobado"] else "No"
    return "?"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loader = ProtocolLoader()
    loader.load_all()

    blind_wb = Workbook()
    blind_wb.remove(blind_wb.active)
    answer_wb = Workbook()
    answer_wb.remove(answer_wb.active)

    grand_total = 0

    for protocol_id in PROTOCOL_SOURCES:
        protocol = loader.get_protocol(protocol_id)
        fields = field_columns(protocol)
        documents = build_sample_for_protocol(protocol_id)
        print(f"{protocol_id}: {len(documents)} documentos, {len(fields)} atributos")

        blind_ws = blind_wb.create_sheet(title=protocol_id)
        answer_ws = answer_wb.create_sheet(title=protocol_id)

        blind_header = ["ID", "Ruta de la imagen"]
        answer_header = ["ID", "Ruta de la imagen"]
        for field in fields:
            col_label = f"{field['descripcion']} (peso {field['peso']})"
            blind_header.append(col_label)
            answer_header.append(col_label)
        blind_header += ["Tu veredicto (valid/incomplete/inconsistent)", "Notas"]
        answer_header += ["Veredicto del agente", "Puntaje ponderado", "Verificaciones"]

        blind_ws.append(blind_header)
        answer_ws.append(answer_header)
        for ws in (blind_ws, answer_ws):
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        for i, doc_path in enumerate(documents, start=1):
            rel_path = doc_path.relative_to(ROOT_DIR)
            doc_id = f"{protocol_id}-{i:02d}"

            try:
                report = supervise_document(str(doc_path), protocol_id, use_llm=False)
                verdict = report["verdict"]
                score = report["rule_checks"]["weighted_score"]
                checks = f"{report['rule_checks']['passed']}/{report['rule_checks']['total']}"
                breakdown = report["rule_checks"]["score_breakdown"]
                field_values = [field_result(breakdown, f["nombre_zona"]) for f in fields]
            except Exception as exc:  # noqa: BLE001
                verdict = f"ERROR: {exc}"
                score = 0.0
                checks = "-"
                field_values = ["?" for _ in fields]

            blind_row = [doc_id, str(rel_path)] + ["" for _ in fields] + ["", ""]
            answer_row = [doc_id, str(rel_path)] + field_values + [verdict, score, checks]
            blind_ws.append(blind_row)
            answer_ws.append(answer_row)
            grand_total += 1

        blind_ws.column_dimensions["A"].width = 14
        blind_ws.column_dimensions["B"].width = 55
        answer_ws.column_dimensions["A"].width = 14
        answer_ws.column_dimensions["B"].width = 55
        for col_idx in range(3, 3 + len(fields)):
            letter = chr(ord("A") + col_idx - 1) if col_idx <= 26 else "A" + chr(ord("A") + col_idx - 27)
            blind_ws.column_dimensions[letter].width = 30
            answer_ws.column_dimensions[letter].width = 30

    blind_path = OUTPUT_DIR / "revision_humana_CIEGA.xlsx"
    answer_path = OUTPUT_DIR / "revision_humana_CLAVE_RESPUESTAS.xlsx"
    blind_wb.save(blind_path)
    answer_wb.save(answer_path)

    print(f"\nPlanilla ciega (para revisores): {blind_path}")
    print(f"Clave de respuestas (NO compartir con revisores): {answer_path}")
    print(f"Total documentos: {grand_total}")


if __name__ == "__main__":
    main()
