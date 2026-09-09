"""
Cliente para Mistral-7B-Instruct ejecutado localmente con llama.cpp.
Se encarga del razonamiento semantico sobre el texto extraido.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
from typing import Any, Optional

from .utils import clip_text, parse_json_response


_LLAMA_UNRAISABLE_FILTER_INSTALLED = False


class MistralClient:
    """Cliente para inferencia local con Mistral-7B."""

    def __init__(
        self,
        model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        lazy_load: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._model: Any = None

        if not lazy_load:
            self._ensure_model_loaded()

    def validate_semantic(
        self,
        extracted_text: str,
        rule_description: str,
        document_type: str,
    ) -> dict[str, Any]:
        """
        Evalua semanticamente si el texto cumple una regla.
        """
        prompt = f"""[INST] Eres un auditor de documentacion sanitaria. Tu tarea es verificar si el siguiente texto extraido de un documento tipo "{document_type}" cumple con la regla de validacion indicada.

TEXTO EXTRAIDO:
{clip_text(extracted_text, 2000)}

REGLA DE VALIDACION:
{rule_description}

Analiza cuidadosamente el texto y responde EXCLUSIVAMENTE con un JSON valido con esta estructura:
{{"verdict": "conforme" o "no_conforme" o "indeterminado", "justification": "explicacion breve en espanol"}}

Responde solo con el JSON, sin texto adicional. [/INST]"""

        raw_text = self._generate_text(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["[INST]"],
        )
        fallback = {
            "verdict": "indeterminado",
            "justification": f"Error al parsear respuesta del modelo: {raw_text[:200]}",
        }
        return self._parse_response(raw_text, fallback)

    def check_coherence(self, text_a: str, text_b: str, relation: str) -> dict[str, Any]:
        """Verifica coherencia entre dos secciones del documento."""
        prompt = f"""[INST] Eres un auditor de documentacion sanitaria. Debes verificar la coherencia entre dos secciones de un documento clinico.

SECCION A:
{clip_text(text_a, 1000)}

SECCION B:
{clip_text(text_b, 1000)}

RELACION ESPERADA: {relation}

Responde con JSON:
{{"coherent": true o false o null, "justification": "explicacion breve en espanol"}}

Solo el JSON, sin texto adicional. [/INST]"""

        raw_text = self._generate_text(
            prompt,
            max_tokens=256,
            temperature=0.1,
        )
        fallback = {
            "coherent": None,
            "justification": f"Error de parseo: {raw_text[:200]}",
        }
        return self._parse_response(raw_text, fallback)

    def _generate_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        stop: Optional[list[str]] = None,
    ) -> str:
        model = self._ensure_model_loaded()
        response = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False,
        )
        return response["choices"][0]["text"].strip()

    def _ensure_model_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo GGUF de Mistral en: {self.model_path}"
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python no esta instalado. Instala la dependencia "
                "para usar Mistral local."
            ) from exc

        _install_llama_cpp_cleanup_filter()

        try:
            if self.verbose:
                self._model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                )
            else:
                with contextlib.redirect_stderr(io.StringIO()):
                    self._model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=self.n_ctx,
                        n_threads=self.n_threads,
                        n_gpu_layers=self.n_gpu_layers,
                        verbose=self.verbose,
                    )
        except OSError as exc:
            raise RuntimeError(
                "No se pudo cargar Mistral con llama-cpp-python. "
                "En Windows, el error 0xc000001d suele indicar que la wheel "
                "instalada usa instrucciones de CPU no soportadas por el equipo. "
                "Reinstala llama-cpp-python con una build compatible."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar Mistral con llama-cpp-python: {exc}"
            ) from exc

        return self._model

    def _parse_response(self, raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = parse_json_response(raw_text)
        except Exception:
            return fallback

        if isinstance(parsed, dict):
            return parsed
        return fallback


def _install_llama_cpp_cleanup_filter() -> None:
    """
    Oculta un aviso secundario de llama-cpp-python cuando falla la carga nativa.

    Algunas builds dejan un objeto LlamaModel parcialmente inicializado y Python
    imprime un AttributeError en __del__. El error util ya se reporta arriba.
    """
    global _LLAMA_UNRAISABLE_FILTER_INSTALLED
    if _LLAMA_UNRAISABLE_FILTER_INSTALLED or not hasattr(sys, "unraisablehook"):
        return

    original_hook = sys.unraisablehook

    def filtered_hook(unraisable) -> None:
        exc = unraisable.exc_value
        target = repr(unraisable.object)
        if (
            isinstance(exc, AttributeError)
            and "sampler" in str(exc)
            and "LlamaModel.__del__" in target
        ):
            return
        original_hook(unraisable)

    sys.unraisablehook = filtered_hook
    _LLAMA_UNRAISABLE_FILTER_INSTALLED = True
