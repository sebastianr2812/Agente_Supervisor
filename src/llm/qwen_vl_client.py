"""
Cliente para Qwen2.5-VL servido localmente via Ollama.

Reemplaza la version anterior basada en transformers/HuggingFace: esa via
descarga los pesos directamente desde HuggingFace, cuya CDN de archivos
grandes esta bloqueada en esta red. Ollama sirve el mismo modelo desde su
propio registro sin ese problema.

Analisis visual directo sobre la imagen del documento: no depende de la
calidad del OCR, pero es mucho mas lento en CPU (~2 minutos por imagen sin
GPU), por eso se ofrece como paso opcional "lento" y no como parte del
pipeline por defecto.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .utils import normalize_coherent_field, parse_json_response


class QwenVLClient:
    """Cliente HTTP minimo para usar Qwen2.5-VL (via Ollama) como analizador visual."""

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        host: str = "http://localhost:11434",
        timeout: int = 300,
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def check_document_coherence(
        self,
        image_path: str,
        document_type: str,
    ) -> dict[str, Any]:
        """
        Analiza la imagen completa del documento y evalua si su contenido es
        internamente coherente (fechas, secciones, firmas, datos del
        paciente), independientemente de lo que haya extraido el OCR.
        """
        prompt = (
            f"You are a healthcare document auditor. Look at this scanned "
            f'"{document_type}" document image.\n\n'
            "Check specifically for these things only:\n"
            "1. Patient name/ID is identical every time it appears in the "
            "document.\n"
            "2. Dates are logically possible. The patient's date of birth "
            "is ALWAYS much earlier than the document date - that is "
            "normal and expected, NOT an error. Only flag a date if it is "
            "impossible (e.g. a birth date in the future, or the document "
            "dated before the patient was born).\n"
            "3. Medication or diagnosis mentioned is not self-contradictory.\n"
            "4. A signature is present somewhere on the page if the "
            "document type requires one.\n\n"
            "Only report a problem you can point to directly in the image "
            "and are confident about. If you are not sure, do not report "
            "it. Never list the same issue more than once.\n\n"
            'JSON only: {"coherent": true/false, "confidence": "high/medium/low", '
            '"issues": ["..."], "justification": "..."}'
        )

        raw_text = self._generate(image_path, prompt)
        fallback = {
            "coherent": None,
            "confidence": "low",
            "issues": [],
            "justification": f"Could not parse Qwen response: {raw_text[:200]}",
        }
        return self._parse_response(raw_text, fallback)

    def classify_document_verdict(
        self,
        image_path: str,
        document_type: str,
        requirements_summary: str,
    ) -> dict[str, Any]:
        """
        Pide a Qwen2.5-VL un veredicto directo de 3 clases (valido /
        incompleto / incoherente) mirando unicamente la imagen, sin OCR ni
        motor de reglas. A diferencia de check_document_coherence (un
        chequeo binario auxiliar pensado para combinarse con las reglas),
        este metodo mide la capacidad del componente visual de forma
        aislada frente al mismo objetivo de 3 clases que el resto del
        pipeline, para poder comparar su kappa directamente contra
        "reglas" y "reglas+LLM" sin que la logica de combinacion limite lo
        que el modelo puede demostrar.

        `requirements_summary` es una descripcion breve, en lenguaje
        natural, de los campos que el protocolo exige para este tipo de
        documento (generada a partir de las descripciones de
        campos_obligatorios del YAML), para que el modelo tenga un
        criterio de "completo" equivalente al que usan las reglas.
        """
        birth_date_note = (
            " A date of birth being present and much earlier than the "
            "document date is normal and counts as found, NOT an error."
            if "nacimiento" in requirements_summary.lower()
            else ""
        )
        items = [it.strip() for it in requirements_summary.split(";") if it.strip()]
        numbered_items = "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))

        prompt = (
            f"You are a healthcare document auditor. Look at this scanned "
            f'"{document_type}" document image.\n\n'
            f"This document type must contain exactly these items, no "
            f"others:\n{numbered_items}\n\n"
            "Step 1: for EACH numbered item above, look carefully at the "
            "image and decide if it is clearly present, quoting or "
            "describing exactly what you see for it. Do not consider "
            "anything beyond this numbered list, even if you believe other "
            "information is normally expected in this type of "
            f"document.{birth_date_note}\n\n"
            "Step 2: now go through EXACTLY these 5 consistency checks, one "
            "by one. For each, answer yes (consistent) or no (contradiction "
            "found), with brief evidence. Skip a check only if it plainly "
            "does not apply to this document type (e.g. no medication is "
            "listed at all), and mark it 'not applicable'. Do NOT report a "
            "problem you are not highly confident about - if unsure, answer "
            "'yes' (consistent).\n"
            "  a. Dates: do all dates make sense TOGETHER? A birth date "
            "being much earlier than the document date is NORMAL - answer "
            "'yes' for that. Only answer 'no' if a date is impossible (e.g. "
            "in the future) or the SAME field shows two different values in "
            "two places.\n"
            "  b. Patient identity: is the patient's name/ID EXACTLY the "
            "same everywhere it appears? Answer 'no' only if you see two "
            "clearly DIFFERENT names or ID numbers for what should be the "
            "same patient.\n"
            "  c. Medication vs allergy: does any listed medication "
            "obviously contradict a stated allergy? For this check, answer "
            "'no' only for a direct, unambiguous contradiction (e.g. the "
            "exact allergen is prescribed). If there is a plausible medical "
            "reason it could still be appropriate, or you cannot be fully "
            "sure, answer 'uncertain' instead of guessing.\n"
            "  d. Vital signs / lab values: are they within plausible "
            "ranges and consistent with each other? Answer 'no' only for "
            "values that are physically impossible. If a combination looks "
            "unusual but could have a legitimate clinical explanation, "
            "answer 'uncertain' instead of guessing.\n"
            "  e. Diagnosis/reason for visit vs treatment/plan: does the "
            "treatment plan correspond to the stated diagnosis or reason "
            "for visit? Answer 'no' ONLY if the treatment is for a "
            "completely different, unrelated condition (see example "
            "below). If the treatment could plausibly be part of a broader "
            "workup, a precaution, or a differential diagnosis for the "
            "stated reason (even if not the most obvious first step), "
            "answer 'uncertain' instead of 'no' - a real doctor may "
            "reasonably order tests beyond the obvious first guess, and "
            "you do not have the clinical training to rule that out with "
            "confidence.\n\n"
            "Example of a genuine inconsistency for check (e): reason for "
            "visit is 'sprained ankle', but the treatment plan describes "
            "'starting chemotherapy for lung cancer' - completely unrelated "
            "body system and condition, so check (e) = 'no'. Do NOT answer "
            "'no' just because a test targets a different body part than "
            "the main symptom (e.g. chest pain with a neurological test "
            "ordered as part of a broader workup is 'uncertain', not "
            "'no') - only unrelated-condition cases this extreme count as "
            "'no'.\n\n"
            "Step 3: give a final verdict based on steps 1 and 2:\n"
            '- "incomplete": at least one numbered item from step 1 was '
            "not found in the image (this takes priority over everything "
            "else).\n"
            '- "inconsistent": all items from step 1 were found, and at '
            'least one check in step 2 (a-e) was answered "no".\n'
            '- "valid": all items from step 1 were found, no check was '
            'answered "no", regardless of any "uncertain" answers.\n\n'
            "Separately, set \"requires_review\" to true if any check "
            "c, d or e was answered \"uncertain\" (even if the verdict "
            "ended up \"valid\") - this flags cases that need a human "
            "clinical expert to confirm, rather than forcing you to guess.\n\n"
            'JSON only: {"checklist": [{"item": "1", "found": true or '
            'false, "evidence": "what you see or \'not found\'"}, ...one '
            'entry per numbered item...], "consistency_checks": [{"check": '
            '"a", "consistent": true, false, "uncertain", or "not '
            'applicable", "evidence": "..."}, ...one entry for each of '
            'a-e...], "verdict": "valid" or "incomplete" or "inconsistent", '
            '"requires_review": true or false, "confidence": "high/medium/low"}'
        )

        raw_text = self._generate(image_path, prompt)
        fallback = {
            "verdict": None,
            "confidence": "low",
            "checklist": [],
            "consistency_checks": [],
            "requires_review": None,
            "reason": f"Could not parse Qwen response: {raw_text[:200]}",
        }
        return self._parse_response(raw_text, fallback)

    # Signos de alarma de alta urgencia comprobados por check_urgency_coherence,
    # mapeados a la clave booleana que se le pide al modelo en el paso 1.
    _ALARM_FLAGS = {
        "tiene_dolor_toracico": "dolor toracico",
        "tiene_dificultad_respirar_subita": "dificultad para respirar subita",
        "tiene_alteracion_consciencia": "alteracion de la consciencia",
        "tiene_sangrado_abundante": "sangrado abundante",
        "tiene_signos_acv": "signos de accidente cerebrovascular",
        "tiene_fiebre_alta_con_sepsis": "fiebre muy alta con sepsis",
    }

    def check_urgency_coherence(self, image_path: str) -> dict[str, Any]:
        """
        Comprueba si el motivo de consulta de un documento clinico (nota
        clinica, etc.) tiene un nivel de urgencia coherente con el contexto
        que describen los antecedentes, en 2 llamadas separadas en vez de
        una sola.

        Se probo primero con una unica llamada que le pedia al modelo
        "detectar signos de alarma en el motivo, comparar con los
        antecedentes, y decidir" en un solo paso: la deteccion de signos de
        alarma en si era inestable (el modelo alucinaba un signo que no
        estaba en el texto, o dejaba de detectar uno real, segun cambios
        minimos e irrelevantes en el formato del prompt), y la logica
        condicional "si no hay signos, es trivialmente coherente" no se
        aplicaba de forma fiable ni siquiera separando el razonamiento en
        pasos dentro de la misma respuesta. Descomponerlo en 2 preguntas
        independientes y acotadas -- (1) listar SI/NO para cada signo de
        alarma posible, uno por uno, en vez de pedir una unica eleccion
        abierta; (2) solo si el paso 1 encontro alguno, preguntar aparte si
        los antecedentes lo explican -- y dejar la logica "si esta vacio,
        es coherente" en Python en vez de en el modelo, se probo estable en
        los casos verificados manualmente (funciona igual sobre la pagina
        completa que sobre un recorte de la zona relevante, sin necesitar
        recortar la imagen segun el OCR).
        """
        paso1_prompt = (
            "Mira esta imagen de un documento clinico. Busca UNICAMENTE el "
            "texto del MOTIVO DE CONSULTA (ignora el resto de la imagen).\n\n"
            "Primero transcribe que dice el motivo de consulta. Luego "
            "responde SI o NO para cada signo de alarma, comprobando "
            "literalmente si esa palabra o algo equivalente aparece en el "
            "texto que transcribiste (no asumas nada que no este "
            "escrito):\n\n"
            "JSON en este orden exacto:\n"
            '{"que_dice_el_motivo": "...", '
            '"tiene_dolor_toracico": true o false, '
            '"tiene_dificultad_respirar_subita": true o false, '
            '"tiene_alteracion_consciencia": true o false, '
            '"tiene_sangrado_abundante": true o false, '
            '"tiene_signos_acv": true o false, '
            '"tiene_fiebre_alta_con_sepsis": true o false}'
        )
        raw1 = self._generate(image_path, paso1_prompt)
        fallback1 = {"que_dice_el_motivo": "", **{k: False for k in self._ALARM_FLAGS}}
        result1 = self._parse_response(raw1, fallback1)

        signos_detectados = [
            label for key, label in self._ALARM_FLAGS.items()
            if result1.get(key) is True
        ]

        if not signos_detectados:
            return {
                "coherent": True,
                "confidence": "high",
                "signos_detectados": [],
                "justification": (
                    "El motivo de consulta no contiene ningun signo de "
                    "alarma de alta urgencia, no hay nada que comparar "
                    "contra los antecedentes."
                ),
            }

        signos_texto = ", ".join(signos_detectados)
        paso2_prompt = (
            f'Mira esta imagen de un documento clinico. El motivo de '
            f'consulta indica: "{signos_texto}".\n\n'
            "Busca UNICAMENTE el texto de ANTECEDENTES / HISTORIA CLINICA "
            "(ignora el resto de la imagen).\n\n"
            "Primero transcribe que dicen los antecedentes. Luego responde: "
            "los antecedentes mencionan algo que razonablemente explique o "
            "se relacione con ese signo de alarma (ej. una enfermedad "
            "cardiaca, cardiovascular, pulmonar, o algun antecedente "
            "relevante)? Si los antecedentes solo hablan de temas no "
            "relacionados (ej. revisiones de otra especialidad sin "
            "conexion, o el paciente es descrito como sano sin nada "
            "relevante), responde false.\n\n"
            "JSON en este orden exacto:\n"
            '{"que_dicen_los_antecedentes": "...", '
            '"explica_el_signo_de_alarma": true o false}'
        )
        raw2 = self._generate(image_path, paso2_prompt)
        fallback2 = {"que_dicen_los_antecedentes": "", "explica_el_signo_de_alarma": None}
        result2 = self._parse_response(raw2, fallback2)
        explicado = result2.get("explica_el_signo_de_alarma")

        return {
            "coherent": bool(explicado) if isinstance(explicado, bool) else None,
            "confidence": "medium" if isinstance(explicado, bool) else "low",
            "signos_detectados": signos_detectados,
            "justification": (
                f"Motivo de consulta con signo(s) de alarma ({signos_texto}); "
                f"antecedentes: {result2.get('que_dicen_los_antecedentes', '')}"
            ),
        }

    def check_labeled_date_present(
        self,
        image_path: str,
        accept_label_keywords: list[str],
        reject_label_keywords: list[str],
    ) -> dict[str, Any]:
        """
        Comprueba si existe en la imagen una fecha de un tipo concreto (ej.
        "fecha del servicio/muestra", distinta de "fecha de emision del
        informe"), sin pedirle al modelo que haga esa distincion el mismo.

        Se probo primero pidiendole directamente "encuentra la fecha X, NO
        la fecha Y" en una sola pregunta: el modelo percibe bien las fechas
        (las lee correctamente, con confianza alta) pero no aplica de forma
        fiable la exclusion -- devolvia "encontrado" usando la fecha del
        tipo que se le pidio ignorar, en documentos donde la fecha
        correcta genuinamente no existe. Es el mismo patron ya visto con
        Mistral (hallazgo 15) y con check_urgency_coherence: estos modelos
        no siguen bien una instruccion de "ignora/excluye X", aunque
        perciban X correctamente.

        La solucion es la misma que en esos casos: no pedirle al modelo
        ninguna exclusion. Se le pide unicamente una enumeracion cruda de
        TODAS las fechas visibles junto con su etiqueta o contexto cercano
        (una tarea sin ambiguedad, verificada de forma estable), y la
        decision de cual cuenta como "la fecha correcta" se hace despues en
        Python, por coincidencia de palabras clave en la etiqueta devuelta.
        """
        prompt = (
            "Look at this scanned healthcare document image. List EVERY "
            "date that appears anywhere on the page, along with the label "
            "or nearby text that identifies what each date refers to. Do "
            "not skip any date, even if you think it might not be the one "
            "being asked about elsewhere -- just list everything you "
            "see.\n\n"
            'JSON only: {"dates_found": [{"date": "...", '
            '"label_or_context": "..."}]}'
        )
        raw = self._generate(image_path, prompt)
        fallback = {"dates_found": []}
        parsed = self._parse_response(raw, fallback)
        dates_found = parsed.get("dates_found") or []

        accepted = []
        rejected = []
        for entry in dates_found:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label_or_context", "")).lower()
            # Se prioriza "accept" sobre "reject" cuando ambos aparecen en
            # la misma etiqueta: el modelo a veces fusiona dos campos
            # fisicamente distintos en un solo texto de contexto (ej.
            # "Fecha de recogida: Fecha de emision: ..." cuando ambas
            # fechas aparecen juntas en la pagina) -- si la palabra clave
            # de aceptacion SI aparece, cuenta como encontrada aunque el
            # texto fusionado tambien mencione la palabra de rechazo.
            if any(kw.lower() in label for kw in accept_label_keywords):
                accepted.append(entry)
            elif any(kw.lower() in label for kw in reject_label_keywords):
                rejected.append(entry)

        return {
            "found": bool(accepted),
            "confidence": "high" if (accepted or rejected) else "low",
            "matched": accepted,
            "rejected_as_wrong_type": rejected,
            "all_dates_found": dates_found,
        }

    def list_patient_identifiers(self, image_path: str) -> dict[str, Any]:
        """
        Enumera todos los numeros de identificacion de paciente (numero de
        historia clinica/NHC, numero de expediente, etc.) visibles en la
        imagen, junto con su etiqueta o contexto cercano.

        Mismo principio que check_labeled_date_present: enumeracion cruda
        de TODO lo que se ve, sin pedirle al modelo que decida cual es "el
        identificador correcto" o si dos valores corresponden al mismo
        paciente -- esa comparacion se hace despues en Python (normalizando
        y contando valores distintos), evitando el mismo patron de fallo ya
        documentado (los modelos no siguen bien instrucciones de comparar/
        excluir, aunque perciban los datos individuales correctamente).
        """
        prompt = (
            "Look at this scanned healthcare document image. List EVERY "
            "patient identification number that appears anywhere on the "
            "page (medical record number, NHC, chart/file number, patient "
            "ID, etc.), along with the label or nearby text that identifies "
            "what each number refers to. Do not skip any, even if you "
            "think two of them refer to the same patient -- just list "
            "everything you see, exactly as written on the page.\n\n"
            'JSON only: {"identifiers_found": [{"value": "...", '
            '"label_or_context": "..."}]}'
        )
        raw_text = self._generate(image_path, prompt)
        fallback = {"identifiers_found": []}
        return self._parse_response(raw_text, fallback)

    def verify_document_zone(
        self,
        image_path: str,
        zone_description: str,
        check_type: str = "presence",
    ) -> dict[str, Any]:
        """Verifica una zona especifica del documento (firma, casilla, legibilidad)."""
        prompts_by_type = {
            "presence": (
                f"Look at this scanned healthcare document image. "
                f"Can you identify {zone_description}? "
                'Respond ONLY with JSON: {"found": true or false, '
                '"confidence": "high/medium/low", "details": "what you see"}'
            ),
            "checkbox": (
                f"Look at the checkboxes in this scanned document image. "
                f"Is the checkbox for '{zone_description}' marked/checked? "
                'Respond ONLY with JSON: {"checked": true or false, '
                '"confidence": "high/medium/low", "details": "what you see"}'
            ),
        }
        prompt = prompts_by_type.get(check_type, prompts_by_type["presence"])
        raw_text = self._generate(image_path, prompt)
        fallback = {
            "found": None,
            "confidence": "low",
            "details": f"Could not parse Qwen response: {raw_text[:200]}",
        }
        return self._parse_response(raw_text, fallback)

    def _generate(self, image_path: str, prompt: str, _retry_count: int = 0) -> str:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
            "options": {
                # temperature 0.0, misma razon que en ollama_client.py: un
                # veredicto de cumplimiento no deberia variar entre
                # ejecuciones para el mismo documento.
                "temperature": 0.0,
                "num_predict": 1000,
                "num_ctx": self.num_ctx,
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
                "Ollama no respondio antes del timeout. Qwen2.5-VL en CPU puede "
                "tardar varios minutos por imagen."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                "No se pudo conectar con Ollama. Verifica que el servicio este "
                "activo y que 'qwen2.5vl:7b' este descargado."
            ) from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama devolvio una respuesta no JSON: {raw_body[:200]}") from exc

        if "error" in body:
            raise RuntimeError(f"Ollama error: {body['error']}")

        text = str(body.get("response", "")).strip()

        # El proceso del modelo se degrada tras varias llamadas largas
        # seguidas (prompts extensos como classify_document_verdict) y
        # empieza a devolver respuestas vacias con "done": false, sin
        # lanzar ningun error HTTP. Se detecto de forma reproducible al
        # evaluar el corpus completo. En vez de descartar el documento
        # como abstencion, se intenta una recarga del modelo (equivalente
        # a `ollama stop`) y se reintenta -- hasta 2 veces (3 intentos en
        # total), no solo 1: se observo durante la validacion de
        # check_labeled_date_present que un solo reintento a veces no
        # bastaba (2 de 30 documentos necesitaron un segundo reintento
        # para dejar de devolver vacio), aunque la logica en si era
        # correcta una vez que el modelo respondia con contenido real.
        if not text and _retry_count < 2:
            try:
                subprocess.run(
                    ["ollama", "stop", self.model],
                    capture_output=True, timeout=30, check=False,
                )
            except Exception:
                pass
            return self._generate(image_path, prompt, _retry_count=_retry_count + 1)

        return text

    def _parse_response(self, raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = parse_json_response(raw_text)
        except Exception:
            return fallback

        if isinstance(parsed, dict):
            return normalize_coherent_field(parsed)
        return fallback
