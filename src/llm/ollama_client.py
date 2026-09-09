"""
Cliente para modelos locales servidos por Ollama.
"""
from __future__ import annotations

import json
import unicodedata
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .utils import clip_text, normalize_coherent_field, parse_json_response


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


# Verificacion determinista de respaldo para check_allergy_contraindication
# (hallazgo 2026-09-08): evaluado sobre MED-001-ES, el chequeo LLM abierto
# tenia recall 100% (detectaba las 3 contraindicaciones reales del corpus)
# pero precision ~37.5% -- inventaba con seguridad relaciones farmacologicas
# falsas (p.ej. "Salbutamol es derivado directo de AINEs", "Atorvastatina
# comparte alergeno con AINEs via HMG-CoA reductasa") cada vez que cualquier
# alergia aparecia junto a cualquier medicamento.
#
# Esta tabla NO reemplaza al LLM ni decide "contraindicado: true/false"
# directamente -- solo AJUSTA LA CONFIANZA de lo que el LLM ya senalo (ver
# uso en check_allergy_contraindication). Si la relacion coincide con una
# familia farmacologica real y verificable, se marca "high" (bloqueo duro,
# igual que antes). Si no coincide con ninguna relacion conocida, se
# degrada a "low": el pipeline (_combined_verdict en supervisor.py) ya trata
# "confidence: low" como señal indeterminada, no como fallo duro -- asi que
# la alerta queda registrada pero no fuerza un veredicto "inconsistent" sin
# corroboracion. Esto evita el peligro contrario: silenciar por completo
# una contraindicacion real con un medicamento que no este en esta lista
# (un gate rigido de todo-o-nada convertiria ese caso en un falso negativo,
# el error mas grave posible en un chequeo de seguridad clinica).
#
# Cobertura: los 10 medicamentos y 7 alergias del vocabulario cerrado de
# generate_synthetic_corpus.py (MEDICAMENTOS / ALERGIAS_POOL), mas algunos
# sinonimos reales de la misma familia para no quedar sobreajustada a esos
# 10 nombres exactos. Para documentos reales con farmacos fuera de esta
# lista haria falta una ontologia farmacologica completa (p.ej.
# clasificacion ATC) o una base de interacciones externa -- ver limitaciones
# en el informe.
KNOWN_ALLERGY_DRUG_FAMILIES: dict[str, set[str]] = {
    "penicilina": {"amoxicilina", "ampicilina", "penicilina"},
    "aines": {
        "ibuprofeno", "naproxeno", "aspirina", "acido acetilsalicilico",
        "diclofenaco", "ketorolaco", "dexketoprofeno", "meloxicam",
        "indometacina", "piroxicam",
    },
    "acido acetilsalicilico": {
        "ibuprofeno", "naproxeno", "diclofenaco", "ketorolaco",
        "dexketoprofeno", "meloxicam", "aspirina",
    },
    "sulfamidas": {
        "sulfametoxazol", "trimetoprim", "sulfasalazina", "sulfadiazina",
    },
    # Sin correlato farmacologico oral conocido en este corpus: cualquier
    # medicamento que el LLM señale aqui se trata como no corroborado.
    "latex": set(),
    "contraste yodado": set(),
    "ninguna conocida": set(),
}


def _allergy_match_confidence(alergia_text: str, medicamentos_relacionados: list[str]) -> str:
    """Devuelve 'high' si AL MENOS UNO de los medicamentos senalados por el
    LLM coincide con una familia farmacologica conocida para la alergia
    declarada; 'low' en caso contrario (alergia no reconocida, o ninguno de
    los medicamentos senalados tiene relacion verificable).

    Se usa "al menos uno" en vez de "todos" a proposito: el LLM a veces
    acierta el medicamento realmente relacionado pero ademas menciona uno
    extra sin relacion real en la misma respuesta (p.ej. alergia "AINEs"
    con relacionados=["Ibuprofeno", "Paracetamol"] -- Ibuprofeno es un AINE
    real, Paracetamol no). Exigir que TODOS coincidieran descartaba estos
    aciertos parciales junto con el ruido; con "al menos uno", el
    medicamento verificable sigue corroborando la contraindicacion con
    confianza alta, sin que el acompañante incorrecto la anule."""
    if not medicamentos_relacionados:
        return "high"
    alergia_norm = _strip_accents(alergia_text)
    known_drugs: set[str] = set()
    matched_any_category = False
    for category, drugs in KNOWN_ALLERGY_DRUG_FAMILIES.items():
        if category in alergia_norm:
            matched_any_category = True
            known_drugs |= drugs
    if not matched_any_category:
        return "low"
    return "high" if any(
        _strip_accents(med) in known_drugs for med in medicamentos_relacionados if med
    ) else "low"


class OllamaClient:
    """Cliente HTTP minimo para usar Ollama como backend LLM local."""

    def __init__(
        self,
        # Pinned (2026-09-09): antes "mistral:latest" -- verificado mismo
        # digest exacto que "mistral:7b" (6577803aa9a0), sin descarga nueva.
        # Mismo principio de reproducibilidad ya aplicado a QwenVLClient
        # (ver hallazgo sobre pinning de modelos).
        model: str = "mistral:7b",
        host: str = "http://localhost:11434",
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def validate_semantic(
        self,
        extracted_text: str,
        rule_description: str,
        document_type: str,
    ) -> dict[str, Any]:
        prompt = f"""Eres un auditor de documentacion sanitaria. Verifica si el siguiente texto extraido de un documento tipo "{document_type}" cumple con la regla indicada.

TEXTO EXTRAIDO:
{clip_text(extracted_text, 2000)}

REGLA:
{rule_description}

Responde exclusivamente con JSON valido:
{{"verdict": "conforme" o "no_conforme" o "indeterminado", "justification": "explicacion breve en espanol"}}"""

        raw_text = self._generate(prompt)
        fallback = {
            "verdict": "indeterminado",
            "justification": f"Error al parsear respuesta de Ollama: {raw_text[:200]}",
        }
        return self._parse_response(raw_text, fallback)

    def check_coherence(self, text_a: str, text_b: str, relation: str) -> dict[str, Any]:
        prompt = f"""Eres un auditor de documentacion sanitaria. Verifica la coherencia entre dos secciones de un documento clinico.

SECCION A:
{clip_text(text_a, 1000)}

SECCION B:
{clip_text(text_b, 1000)}

RELACION ESPERADA:
{relation}

Responde "false" (no coherente) si encuentras una contradiccion o incompatibilidad
real entre A y B (p. ej. fechas imposibles, datos del paciente distintos, un
tratamiento que no corresponde al motivo indicado, valores incompatibles entre si).
El hecho de que A y B mencionen temas clinicos distintos NO es, por si solo, una
incoherencia: un antecedente medico (ej. hipertension) puede coexistir sin
contradiccion con un motivo de consulta distinto (ej. dolor lumbar), y un sintoma
puede tener causas no relacionadas con los antecedentes del paciente sin que eso
sea una incoherencia - solo marca "false" si la relacion esperada arriba realmente
no se cumple.

No respondas "null" solo porque falten datos como la edad, el sexo u otros
antecedentes del paciente -esos datos no son necesarios para juzgar la relacion
esperada arriba-. Usa "null" si, con la informacion que SI tienes en A y B,
genuinamente no puedes decidir con confianza.

Ademas de "coherent", indica tu nivel de confianza:
- "high": la contradiccion (o la coherencia) es evidente y estas seguro.
- "medium": hay una senal razonable en ese sentido, pero no es del todo concluyente.
- "low": no tienes evidencia suficiente para decidir con seguridad.

Importante: "coherent" y "confidence" son dos preguntas distintas. Si dudas
entre coherente y no coherente, esa duda va en "confidence" (usa "medium" o
"low" ahi), NO en "coherent". "coherent" debe ser exactamente true o false
-tu mejor respuesta aunque no estes seguro-, salvo que genuinamente falten
datos en A o B para evaluar la relacion esperada, en cuyo caso usa null.

Responde exclusivamente con JSON valido, usando true/false/null como LITERALES
JSON (sin comillas, no como texto):
{{"coherent": true o false o null, "confidence": "high" o "medium" o "low", "justification": "explicacion breve en espanol"}}"""

        raw_text = self._generate(prompt)
        fallback = {
            "coherent": None,
            "confidence": "low",
            "justification": f"Error al parsear respuesta de Ollama: {raw_text[:200]}",
        }
        return self._parse_response(raw_text, fallback)

    def check_allergy_contraindication(self, alergia_text: str, medicamentos_text: str) -> dict[str, Any]:
        """
        Comprobacion de contraindicacion alergia/medicamento, descompuesta
        como tarea de ENUMERACION en vez de juicio abierto de si/no -- la
        misma receta validada en check_labeled_date_present (hallazgo 18):
        pedirle al modelo que liste candidatos observables, no que emita un
        veredicto directo, y resolver la logica final (hay o no
        contraindicacion) en Python comparando la lista devuelta contra los
        medicamentos realmente presentes en el texto, en vez de confiar en
        que el modelo mismo declare "contraindicado: true/false".
        """
        prompt = f"""Eres un farmaceutico clinico revisando una lista de medicacion.

ALERGIA DECLARADA DEL PACIENTE:
{clip_text(alergia_text, 300)}

MEDICAMENTOS PRESCRITOS:
{clip_text(medicamentos_text, 1000)}

Tarea: lista TODOS los medicamentos de la lista anterior que pertenezcan a la
MISMA familia farmacologica o sean un derivado directo del alergeno
declarado (ejemplos objetivos de familia farmacologica: Amoxicilina y
Ampicilina son penicilinas; Ibuprofeno y Naproxeno son AINEs
antiinflamatorios). Si la alergia declarada es "Ninguna conocida" o ningun
medicamento de la lista pertenece a esa familia, la lista debe quedar
vacia -- no incluyas un medicamento solo porque trate una dolencia
relacionada, unicamente por pertenecer a la misma familia o ser un
derivado quimico directo del alergeno.

Responde EXCLUSIVAMENTE con JSON valido, sin texto adicional:
{{"medicamentos_relacionados": ["nombre1", "nombre2"] o [], "justification": "explicacion breve en espanol"}}"""

        raw_text = self._generate(prompt)
        fallback = {
            "medicamentos_relacionados": [],
            "justification": f"Error al parsear respuesta de Ollama: {raw_text[:200]}",
        }
        result = self._parse_response(raw_text, fallback)
        if not isinstance(result.get("medicamentos_relacionados"), list):
            result["medicamentos_relacionados"] = []
        # FIX (2026-09-08): ver KNOWN_ALLERGY_DRUG_FAMILIES arriba. La
        # confianza ya no es un "high" fijo -- se degrada a "low" cuando la
        # relacion que propone el LLM no coincide con ninguna familia
        # farmacologica conocida y verificable.
        result["confidence"] = _allergy_match_confidence(alergia_text, result["medicamentos_relacionados"])
        return result

    def translate_to_spanish(self, text: str) -> str:
        """Traduce una frase corta al espanol (sin formato JSON, texto plano)."""
        if not text.strip():
            return text

        prompt = (
            "You are a translation engine, not an assistant. Translate the sentence "
            "below into Spanish. Output ONLY the raw Spanish translation on a single "
            "line. Do NOT add quotes, notes, alternatives, explanations, disclaimers, "
            "or any text other than the translation itself.\n\n"
            f"SENTENCE: {text}\n\nSPANISH TRANSLATION:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 80, "stop": ["\n"]},
        }
        request = Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        if "error" in body:
            raise RuntimeError(f"Ollama error: {body['error']}")

        return str(body.get("response", "")).strip().strip('"')

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                # temperature 0.0 (no 0.1): un auditor de cumplimiento no
                # deberia dar un veredicto distinto para el mismo documento
                # en dos ejecuciones -- se detecto una variacion asi en la
                # evaluacion integral (mismo documento, mismo prompt, un
                # "coherent: true" y luego un "false" en otra corrida).
                "temperature": 0.0,
                "num_predict": 256,
            },
        }
        request = Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise RuntimeError(
                "Ollama no respondio antes del timeout. En CPU la primera carga "
                "del modelo puede tardar varios minutos; prueba de nuevo o usa "
                "un modelo mas pequeno."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "No se pudo conectar con Ollama. Verifica que el servicio este "
                "activo y que el modelo exista con 'ollama list'."
            ) from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama devolvio una respuesta no JSON: {raw_body[:200]}") from exc

        if "error" in body:
            raise RuntimeError(f"Ollama error: {body['error']}")

        return str(body.get("response", "")).strip()

    def _parse_response(self, raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = parse_json_response(raw_text)
        except Exception:
            return fallback

        if isinstance(parsed, dict):
            return normalize_coherent_field(parsed)
        return fallback
