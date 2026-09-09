"""
Evalua el pipeline OCR + reglas sobre el lote sintetico.

Uso:
    venv\Scripts\python.exe scripts\evaluate_synthetic.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.ocr.extractor import detect_checkboxes, detect_signatures, extract_text_blocks
from src.rules import ProtocolLoader, RuleEngine


DEFAULT_GROUND_TRUTH = ROOT_DIR / "data" / "synthetic" / "ground_truth.json"
DEFAULT_RESULTS_DIR = ROOT_DIR / "data" / "results"

console = Console()


def load_ground_truth(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_document_path(entry: dict) -> Path:
    if "png_path" in entry:
        return ROOT_DIR / entry["png_path"]
    raise ValueError("La entrada de ground truth no contiene 'png_path'.")


def serialize_issue(issue) -> dict:
    return {
        "rule_id": issue.rule_id,
        "severity": issue.severity,
        "zone": issue.zone,
        "message": issue.message,
        "details": issue.details,
    }


def evaluate_documents(
    protocol_id: str,
    ground_truth_path: Path,
) -> dict:
    loader = ProtocolLoader()
    protocol = loader.get_protocol(protocol_id)
    if protocol is None:
        raise ValueError(f"No se encontro el protocolo: {protocol_id}")

    engine = RuleEngine(protocol)
    ground_truth = load_ground_truth(ground_truth_path)

    summary_rows = []
    issue_counter: Counter[str] = Counter()
    expected_counter: Counter[str] = Counter()
    predicted_counter: Counter[str] = Counter()

    total = 0
    correct = 0

    for doc_id, expected in ground_truth.items():
        document_path = resolve_document_path(expected)
        total += 1
        expected_verdict = expected["expected_verdict"]
        expected_counter[expected_verdict] += 1

        try:
            ocr_result = extract_text_blocks(str(document_path))
            signatures = detect_signatures(str(document_path))
            checkboxes = detect_checkboxes(str(document_path))
            validation = engine.validate(
                ocr_result,
                signatures=signatures,
                checkboxes=checkboxes,
            )
            predicted_verdict = validation.verdict
            predicted_counter[predicted_verdict] += 1
            is_correct = expected_verdict == predicted_verdict
            if is_correct:
                correct += 1

            for issue in validation.issues:
                issue_counter[issue.rule_id] += 1

            summary_rows.append(
                {
                    "doc_id": doc_id,
                    "path": str(document_path.relative_to(ROOT_DIR)),
                    "quality_profile": expected.get("quality_profile", ""),
                    "deficiencies": expected.get("deficiencies", []),
                    "expected_verdict": expected_verdict,
                    "predicted_verdict": predicted_verdict,
                    "correct": is_correct,
                    "ocr_confidence": round(ocr_result.avg_confidence, 2),
                    "signatures_found": len(signatures),
                    "checkboxes_found": len(checkboxes),
                    "checks_passed": validation.checks_passed,
                    "checks_total": validation.checks_total,
                    "score": round(validation.score, 4),
                    "issues": [serialize_issue(issue) for issue in validation.issues],
                    "error": None,
                }
            )
        except Exception as exc:
            predicted_counter["error"] += 1
            summary_rows.append(
                {
                    "doc_id": doc_id,
                    "path": str(document_path.relative_to(ROOT_DIR)),
                    "quality_profile": expected.get("quality_profile", ""),
                    "deficiencies": expected.get("deficiencies", []),
                    "expected_verdict": expected_verdict,
                    "predicted_verdict": "error",
                    "correct": False,
                    "ocr_confidence": 0.0,
                    "signatures_found": 0,
                    "checkboxes_found": 0,
                    "checks_passed": 0,
                    "checks_total": 0,
                    "score": 0.0,
                    "issues": [],
                    "error": str(exc),
                }
            )

    accuracy = correct / total if total else 0.0
    return {
        "timestamp": datetime.now().isoformat(),
        "protocol_id": protocol_id,
        "ground_truth_path": str(ground_truth_path.relative_to(ROOT_DIR)),
        "documents_total": total,
        "documents_correct": correct,
        "accuracy": round(accuracy, 4),
        "expected_counts": dict(expected_counter),
        "predicted_counts": dict(predicted_counter),
        "issue_frequency": dict(issue_counter.most_common()),
        "documents": summary_rows,
    }


def write_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"synthetic_eval_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return output_path


def print_summary(report: dict) -> None:
    console.print("[bold]Evaluacion sintetica OCR + reglas[/bold]")
    console.print(f"Protocolo: {report['protocol_id']}")
    console.print(f"Documentos: {report['documents_correct']}/{report['documents_total']} correctos")
    console.print(f"Accuracy: {report['accuracy']:.2%}")

    table = Table(title="Resultado por documento")
    table.add_column("Documento")
    table.add_column("Calidad")
    table.add_column("Esperado")
    table.add_column("Predicho")
    table.add_column("OCR")
    table.add_column("Checks")
    table.add_column("OK")

    for item in report["documents"]:
        table.add_row(
            item["doc_id"],
            item["quality_profile"] or "-",
            item["expected_verdict"],
            item["predicted_verdict"],
            f"{item['ocr_confidence']:.1f}",
            f"{item['checks_passed']}/{item['checks_total']}",
            "X" if item["correct"] else "-",
        )

    console.print(table)

    if report["issue_frequency"]:
        issues_table = Table(title="Frecuencia de incidencias")
        issues_table.add_column("Rule ID")
        issues_table.add_column("Veces")
        for rule_id, count in report["issue_frequency"].items():
            issues_table.add_row(rule_id, str(count))
        console.print(issues_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el lote sintetico con OCR + reglas.")
    parser.add_argument("--protocol-id", default="CI-001", help="ID del protocolo a usar.")
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Ruta al archivo ground_truth.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directorio donde guardar el informe JSON.",
    )
    args = parser.parse_args()

    report = evaluate_documents(
        protocol_id=args.protocol_id,
        ground_truth_path=Path(args.ground_truth),
    )
    output_path = write_report(report, Path(args.output_dir))
    print_summary(report)
    console.print(f"\nInforme guardado en: {output_path}")


if __name__ == "__main__":
    main()
