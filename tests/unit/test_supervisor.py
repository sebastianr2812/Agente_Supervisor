from src.agent import supervisor
from src.rules.engine import ValidationResult


class FakeDocumentOCR:
    avg_confidence = 88.5
    image_path = "fake.png"

    def __init__(self) -> None:
        self.zones = {
            "header": (
                "Northwood Community Hospital. Patient: Jane Doe DOB: 05/14/1985"
            ),
            "body": (
                "Chief complaint: severe abdominal pain and fever. "
                "Medical history: diagnosed with hypertension, care team notes "
                "reviewed by attending physician Dr. Emily Chen, MD."
            ),
            "legal": "",
            "signature": "",
        }
        self.full_text = " ".join(self.zones.values())
        self.blocks = [object(), object(), object()]

    def get_text_by_zone(self, zone: str) -> str:
        return self.zones.get(zone, "")


class FakeLLMClient:
    def check_coherence(self, text_a: str, text_b: str, relation: str) -> dict:
        return {"coherent": True, "justification": "consistente"}


class FailingLLMClient:
    def check_coherence(self, text_a: str, text_b: str, relation: str) -> dict:
        raise RuntimeError("libreria nativa incompatible")


def test_supervise_document_runs_without_real_ocr_or_llm(monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "extract_text_blocks", lambda path, lang="spa": FakeDocumentOCR())
    monkeypatch.setattr(
        supervisor,
        "detect_signatures",
        lambda path: [{"x": 10, "y": 10, "width": 100, "height": 30, "area": 2000}],
    )
    monkeypatch.setattr(
        supervisor,
        "detect_checkboxes",
        lambda path: [{"x": 1, "y": 2, "width": 20, "height": 20, "checked": True}],
    )

    report = supervisor.supervise_document("fake.png", "CN-001", use_llm=False)

    assert report["verdict"] == "valid"
    assert report["ocr"]["confidence"] == 88.5
    assert report["signatures"]["found"] == 1
    assert report["semantic_checks"][0]["skipped"] is True


def test_llm_validation_step_runs_when_enabled() -> None:
    protocol = {
        "campos_obligatorios": [
            {"nombre_zona": "a", "zona_documento": "body"},
            {"nombre_zona": "b", "zona_documento": "body"},
        ],
        "reglas_coherencia": [
            {
                "id_regla": "RCH-X",
                "tipo": "texto_coherente",
                "campo_origen": "a",
                "campo_destino": "b",
                "condicion": "A debe ser coherente con B",
            }
        ],
    }

    state = {
        "protocol": protocol,
        "protocol_id": "P",
        "ocr_result": FakeDocumentOCR(),
        "use_llm": True,
        "_llm_client": FakeLLMClient(),
        "errors": [],
    }

    result = supervisor.llm_validation_step(state)

    assert result["llm_validations"][0]["skipped"] is False
    assert result["llm_validations"][0]["result"]["coherent"] is True


def test_llm_validation_failure_is_reported_without_pipeline_error() -> None:
    protocol = {
        "campos_obligatorios": [
            {"nombre_zona": "a", "zona_documento": "body"},
            {"nombre_zona": "b", "zona_documento": "body"},
        ],
        "reglas_coherencia": [
            {
                "id_regla": "RCH-X",
                "tipo": "texto_coherente",
                "campo_origen": "a",
                "campo_destino": "b",
                "condicion": "A debe ser coherente con B",
            }
        ],
    }

    state = {
        "protocol": protocol,
        "protocol_id": "P",
        "ocr_result": FakeDocumentOCR(),
        "use_llm": True,
        "_llm_client": FailingLLMClient(),
        "errors": [],
    }

    result = supervisor.llm_validation_step(state)

    assert result["errors"] == []
    assert result["llm_validations"][0]["error"] is True
    assert "LLM no disponible" in result["llm_error"]


def test_generate_report_marks_semantic_failure_as_inconsistent() -> None:
    state = {
        "document_path": "fake.png",
        "protocol_id": "P",
        "protocol": {"version": "1"},
        "ocr_result": FakeDocumentOCR(),
        "signatures": [],
        "checkboxes": [],
        "rule_validation": ValidationResult(
            verdict="valid",
            checks_passed=1,
            checks_total=1,
        ),
        "llm_validations": [
            {
                "rule_id": "RCH-X",
                "skipped": False,
                "result": {"coherent": False, "justification": "no cuadra"},
            }
        ],
        "use_llm": True,
        "errors": [],
    }

    result = supervisor.generate_report_step(state)

    assert result["report"]["verdict"] == "inconsistent"
