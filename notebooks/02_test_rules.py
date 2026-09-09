"""
Script de prueba para la Fase 2.
Ejecuta OCR + validacion por reglas sobre un documento de ejemplo.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table

from src.ocr.extractor import (
    detect_checkboxes,
    detect_signatures,
    extract_text_blocks,
)
from src.rules import ProtocolLoader, RuleEngine

console = Console()

image_path = "data/synthetic/ejemplo_consentimiento_01.png"
protocol_id = "CI-001"

console.print("[bold]1. Carga del protocolo[/bold]")
loader = ProtocolLoader()
protocol = loader.get_protocol(protocol_id)
if protocol is None:
    raise ValueError(f"No se encontro el protocolo: {protocol_id}")
console.print(f"  Protocolo cargado: {protocol['id_protocolo']}")

console.print("\n[bold]2. OCR del documento[/bold]")
ocr_result = extract_text_blocks(image_path)
signatures = detect_signatures(image_path)
checkboxes = detect_checkboxes(image_path)
console.print(f"  Bloques OCR: {len(ocr_result.blocks)}")
console.print(f"  Confianza media: {ocr_result.avg_confidence:.1f}%")
console.print(f"  Firmas detectadas: {len(signatures)}")
console.print(f"  Casillas detectadas: {len(checkboxes)}")

console.print("\n[bold]2.1 Texto por zona[/bold]")
for zone in ["header", "body", "legal", "signature"]:
    zone_text = ocr_result.get_text_by_zone(zone).strip()
    preview = zone_text[:250].replace("\n", " ")
    console.print(f"  [{zone}] {len(zone_text)} chars", markup=False)
    console.print(f"    {preview}..." if preview else "    <vacio>")

if signatures:
    console.print("\n[bold]2.2 Firmas detectadas[/bold]")
    for index, signature in enumerate(signatures, start=1):
        console.print(
            f"  {index}. x={signature['x']} y={signature['y']} "
            f"w={signature['width']} h={signature['height']} "
            f"area={signature['area']}"
        )

console.print("\n[bold]3. Validacion por reglas[/bold]")
result = RuleEngine(protocol).validate(
    ocr_result,
    signatures=signatures,
    checkboxes=checkboxes,
)
console.print(f"  Veredicto: {result.verdict}")
console.print(f"  Checks: {result.checks_passed}/{result.checks_total}")
console.print(f"  Score: {result.score:.2%}")

if result.issues:
    table = Table(title="Incidencias detectadas")
    table.add_column("Rule ID")
    table.add_column("Severidad")
    table.add_column("Zona")
    table.add_column("Mensaje")
    for issue in result.issues:
        table.add_row(
            issue.rule_id,
            issue.severity,
            issue.zone or "-",
            issue.message,
        )
    console.print(table)
else:
    console.print("  No se detectaron incidencias.")
