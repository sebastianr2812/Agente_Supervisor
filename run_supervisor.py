"""
CLI para ejecutar el agente supervisor sobre un documento.

Uso:
    python run_supervisor.py <imagen> <protocolo_id>
    python run_supervisor.py <imagen> <protocolo_id> --use-llm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.supervisor import supervise_document
from src.llm.mistral_client import MistralClient
from src.llm.ollama_client import OllamaClient


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el agente supervisor de documentacion sanitaria.",
    )
    parser.add_argument("image_path", help="Ruta de la imagen o PDF a validar.")
    parser.add_argument("protocol_id", help="ID del protocolo YAML. Ejemplo: CI-001.")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Activa las validaciones semanticas con un LLM local.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "llama-cpp"],
        default="ollama",
        help="Backend LLM local. Por defecto usa Ollama.",
    )
    parser.add_argument(
        "--ollama-model",
        default="mistral:7b",
        help="Modelo de Ollama si se usa --llm-backend ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="URL del servicio Ollama.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=300,
        help="Timeout en segundos para cada llamada a Ollama.",
    )
    parser.add_argument(
        "--model-path",
        default="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        help="Ruta del modelo GGUF si se usa --llm-backend llama-cpp.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directorio donde se guardara el reporte JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path)

    console.print("\n[bold]Agente Supervisor de Documentacion Sanitaria[/bold]")
    console.print(f"Documento: {image_path}")
    console.print(f"Protocolo: {args.protocol_id}")
    if args.use_llm:
        llm_label = _llm_label(args)
    else:
        llm_label = "desactivado"
    console.print(f"LLM: {llm_label}\n")

    llm_client = None
    if args.use_llm:
        llm_client = _build_llm_client(args)

    with console.status("Procesando documento..."):
        report = supervise_document(
            str(image_path),
            args.protocol_id,
            use_llm=args.use_llm,
            llm_client=llm_client,
        )

    _print_report(report)

    output_dir = Path(args.output_dir)
    output_path = output_dir / f"report_{args.protocol_id}_{image_path.stem}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        console.print(f"\n[bold red]No se pudo guardar el informe:[/bold red] {exc}")
        return 1

    console.print(f"\nInforme guardado en: {output_path}")
    return 0 if report["verdict"] != "error" else 1


def _build_llm_client(args: argparse.Namespace):
    if args.llm_backend == "ollama":
        return OllamaClient(
            model=args.ollama_model,
            host=args.ollama_host,
            timeout=args.ollama_timeout,
        )
    return MistralClient(model_path=args.model_path)


def _llm_label(args: argparse.Namespace) -> str:
    if args.llm_backend == "ollama":
        return f"activado (ollama:{args.ollama_model})"
    return "activado (llama-cpp)"


VERDICT_LABELS = {
    "valid": "VALIDO",
    "incomplete": "INCOMPLETO",
    "inconsistent": "NO VALIDO",
    "error": "ERROR",
}


def _print_report(report: dict) -> None:
    verdict_colors = {
        "valid": "green",
        "incomplete": "yellow",
        "inconsistent": "red",
        "error": "red",
    }
    verdict = report["verdict"]
    color = verdict_colors.get(verdict, "white")
    label = VERDICT_LABELS.get(verdict, verdict.upper())
    console.print(
        Panel(
            f"[bold {color}]{label}[/bold {color}]",
            title="Veredicto",
            expand=False,
        )
    )

    ocr = report.get("ocr", {})
    signatures = report.get("signatures", {})
    checks = report.get("rule_checks", {})
    console.print(f"\nConfianza OCR: {ocr.get('confidence', 0.0):.1f}%")
    console.print(f"Bloques OCR: {ocr.get('blocks', 0)}")
    console.print(f"Firmas detectadas: {signatures.get('found', 0)}")
    console.print(f"Verificaciones: {checks.get('passed', 0)}/{checks.get('total', 0)}")
    console.print(f"Puntaje ponderado: {checks.get('weighted_score', 0.0):.1f}/100")

    issues = checks.get("issues", [])
    if issues:
        table = Table(title="Incidencias")
        table.add_column("ID")
        table.add_column("Severidad")
        table.add_column("Zona")
        table.add_column("Mensaje")
        for issue in issues:
            table.add_row(
                str(issue.get("id", "")),
                str(issue.get("severity", "")),
                str(issue.get("zone", "") or ""),
                str(issue.get("message", "")),
            )
        console.print(table)

    semantic_checks = report.get("semantic_checks", [])
    if semantic_checks:
        table = Table(title="Validaciones semanticas")
        table.add_column("ID")
        table.add_column("Estado")
        table.add_column("Justificacion")
        for item in semantic_checks:
            result = item.get("result", {})
            if item.get("skipped"):
                status = "omitida"
            else:
                status = str(result.get("coherent"))
            table.add_row(
                str(item.get("rule_id", "")),
                status,
                str(result.get("justification", "")),
            )
        console.print(table)

    if report.get("errors"):
        console.print("\n[bold red]Errores:[/bold red]")
        for error in report["errors"]:
            console.print(f"  - {error}")


if __name__ == "__main__":
    raise SystemExit(main())
