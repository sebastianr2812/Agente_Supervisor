import json

from src.llm.mistral_client import MistralClient
from src.llm.ollama_client import OllamaClient
from src.llm.utils import clip_text, parse_json_response


class StubMistralClient(MistralClient):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def _generate_text(self, prompt, max_tokens=256, temperature=0.1, stop=None) -> str:
        return self.response_text


def test_parse_json_response_accepts_markdown_fence() -> None:
    raw = """```json
{"verdict": "conforme", "justification": "ok"}
```"""
    result = parse_json_response(raw)
    assert result["verdict"] == "conforme"


def test_parse_json_response_extracts_json_from_surrounding_text() -> None:
    raw = 'Respuesta final: {"coherent": true, "justification": "consistente"} Gracias.'
    result = parse_json_response(raw)
    assert result["coherent"] is True


def test_clip_text_truncates_without_crashing() -> None:
    assert clip_text("abcdef", 4) == "abcd"
    assert clip_text("abc", 10) == "abc"


def test_mistral_validate_semantic_returns_parsed_json() -> None:
    client = StubMistralClient(
        '{"verdict": "no_conforme", "justification": "falta fecha"}'
    )
    result = client.validate_semantic(
        extracted_text="Texto de ejemplo",
        rule_description="Debe contener fecha",
        document_type="consentimiento_informado",
    )
    assert result["verdict"] == "no_conforme"


def test_mistral_validate_semantic_returns_fallback_on_bad_json() -> None:
    client = StubMistralClient("respuesta libre no parseable")
    result = client.validate_semantic(
        extracted_text="Texto de ejemplo",
        rule_description="Debe contener fecha",
        document_type="consentimiento_informado",
    )
    assert result["verdict"] == "indeterminado"
    assert "Error al parsear" in result["justification"]


def test_mistral_check_coherence_returns_parsed_json() -> None:
    client = StubMistralClient(
        '{"coherent": false, "justification": "los riesgos no corresponden"}'
    )
    result = client.check_coherence(
        text_a="Procedimiento A",
        text_b="Riesgos B",
        relation="Los riesgos deben corresponder al procedimiento",
    )
    assert result["coherent"] is False


def test_ollama_check_coherence_returns_parsed_json(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self) -> bytes:
            body = {
                "response": json.dumps(
                    {"coherent": True, "justification": "consistente"}
                )
            }
            return json.dumps(body).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 300
        return FakeResponse()

    monkeypatch.setattr("src.llm.ollama_client.urlopen", fake_urlopen)

    client = OllamaClient(model="mistral:7b")
    result = client.check_coherence(
        text_a="Procedimiento A",
        text_b="Riesgos A",
        relation="Los riesgos deben corresponder al procedimiento",
    )

    assert result["coherent"] is True


def test_ollama_parse_failure_returns_fallback(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self) -> bytes:
            return json.dumps({"response": "respuesta libre"}).encode("utf-8")

    monkeypatch.setattr(
        "src.llm.ollama_client.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    client = OllamaClient(model="mistral:7b")
    result = client.check_coherence("A", "B", "relacion")

    assert result["coherent"] is None
    assert "Error al parsear" in result["justification"]
