"""
Script de prueba para el pipeline OCR.
Usa un documento de ejemplo para verificar que todo funciona.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table

from src.ocr.extractor import detect_checkboxes, detect_signatures, extract_text_blocks

console = Console()

image_path = "data/synthetic/ejemplo_consentimiento_01.png"

console.print("[bold]1. Extraccion de texto[/bold]")
result = extract_text_blocks(image_path)
console.print(f"  Bloques extraidos: {len(result.blocks)}")
console.print(f"  Confianza media: {result.avg_confidence:.1f}%")
console.print(f"  Texto completo ({len(result.full_text)} chars):")
console.print(result.full_text[:500])

console.print("\n[bold]2. Texto por zonas[/bold]")
for zone in ["header", "body", "legal", "signature"]:
    text = result.get_text_by_zone(zone)
    if text:
        console.print(f"  [{zone}]: {text[:100]}...", markup=False)

console.print("\n[bold]3. Deteccion de firmas[/bold]")
signatures = detect_signatures(image_path)
console.print(f"  Firmas detectadas: {len(signatures)}")

console.print("\n[bold]4. Deteccion de casillas[/bold]")
checkboxes = detect_checkboxes(image_path)
table = Table(title="Casillas detectadas")
table.add_column("Posicion")
table.add_column("Marcada")
table.add_column("Fill ratio")
for checkbox in checkboxes:
    table.add_row(
        f"({checkbox['x']}, {checkbox['y']})",
        "X" if checkbox["checked"] else "-",
        str(checkbox["fill_ratio"]),
    )
console.print(table)
