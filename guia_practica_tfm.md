# Guía práctica paso a paso — TFM Propuesta 10
## Agente supervisor de documentación sanitaria con OCR y LLM locales

---

## Fase 0: Preparación del entorno (Día 1-2)

### 0.1. Crear el repositorio Git

```bash
mkdir agente-supervisor-sanitario
cd agente-supervisor-sanitario
git init

# Crear estructura de directorios
mkdir -p src/{ocr,rules,llm,agent}
mkdir -p configs/protocolos
mkdir -p data/{raw,processed,results,synthetic}
mkdir -p tests/{unit,integration}
mkdir -p docs
mkdir -p notebooks

# Crear archivos __init__.py
touch src/__init__.py src/ocr/__init__.py src/rules/__init__.py
touch src/llm/__init__.py src/agent/__init__.py

# Crear .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
venv/
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
models/
*.gguf
.DS_Store
EOF

touch data/raw/.gitkeep data/processed/.gitkeep

git add .
git commit -m "feat: estructura inicial del proyecto"
```

### 0.2. Crear entorno virtual Python

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Crear requirements.txt
cat > requirements.txt << 'EOF'
# OCR y procesamiento de imagen
pytesseract==0.3.13
opencv-python==4.9.0.80
Pillow==10.4.0
docling==2.15.0

# LLM local
llama-cpp-python==0.3.4
transformers==4.48.0
torch==2.5.1
accelerate==1.2.0
qwen-vl-utils==0.0.8

# Orquestación del agente
langgraph==0.2.60
langchain-core==0.3.28

# Protocolos y configuración
PyYAML==6.0.2
pydantic==2.10.3
jsonschema==4.23.0

# Evaluación
scikit-learn==1.6.0
pandas==2.2.3
numpy==1.26.4

# Testing y desarrollo
pytest==8.3.4
pytest-cov==6.0.0
python-dotenv==1.0.1
rich==13.9.4
EOF

pip install -r requirements.txt
```

### 0.3. Instalar Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-spa
# Verificar:
tesseract --version
tesseract --list-langs  # Debe incluir 'spa'
```

**Windows:**
- Descargar instalador de: https://github.com/UB-Mannheim/tesseract/wiki
- Instalar con el paquete de idioma español
- Añadir al PATH: `C:\Program Files\Tesseract-OCR`

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 0.4. Descargar modelos LLM

```bash
mkdir -p models

# Mistral-7B-Instruct cuantizado (Q4_K_M ≈ 4.4 GB)
# Opción A: Usando huggingface-cli
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
    mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --local-dir models/

# Opción B: Descarga directa
wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Qwen2.5-VL (se descarga automáticamente con transformers)
# Se descargará al ejecutar el código por primera vez
# El modelo Qwen2.5-VL-7B-Instruct pesa ~16 GB en FP16
# Para ahorro de VRAM se puede usar Qwen2.5-VL-3B-Instruct (~6 GB)
```

---

## Fase 1: Pipeline OCR (Día 3-5)

### 1.1. Módulo de preprocesamiento de imagen

Crear `src/ocr/preprocess.py`:

```python
"""
Módulo de preprocesamiento de imágenes de documentos sanitarios.
Aplica correcciones geométricas y mejora de calidad antes del OCR.
"""
import cv2
import numpy as np
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Carga una imagen en escala de grises."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises si es necesario."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def deskew(img: np.ndarray) -> np.ndarray:
    """
    Corrige la inclinación del documento usando la transformada de Hough.
    Detecta líneas horizontales y calcula el ángulo de rotación necesario.
    """
    gray = to_grayscale(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:  # Solo líneas casi horizontales
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    if abs(median_angle) < 0.1:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def binarize_sauvola(img: np.ndarray, window_size: int = 25,
                      k: float = 0.2) -> np.ndarray:
    """
    Binarización adaptativa con el método de Sauvola.
    Mejor que Otsu para documentos con iluminación variable.

    T(x,y) = mean(x,y) * (1 + k * (std(x,y) / R - 1))
    donde R = max(std) = 128 para imágenes de 8 bits.
    """
    gray = to_grayscale(img)
    gray = gray.astype(np.float64)

    # Calcular media y desviación estándar local
    mean = cv2.blur(gray, (window_size, window_size))
    mean_sq = cv2.blur(gray ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))

    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray >= threshold] = 255

    return binary


def remove_noise(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Elimina ruido con filtro de mediana."""
    return cv2.medianBlur(img, kernel_size)


def preprocess_document(image_path: str) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento.
    Retorna imagen binaria lista para OCR.
    """
    img = load_image(image_path)
    img = deskew(img)
    img = binarize_sauvola(img)
    img = remove_noise(img)
    return img
```

### 1.2. Módulo de extracción OCR

Crear `src/ocr/extractor.py`:

```python
"""
Módulo de extracción de texto con Tesseract OCR.
Produce un documento estructurado con zonas semánticas.
"""
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .preprocess import preprocess_document, load_image, to_grayscale


@dataclass
class TextBlock:
    """Un bloque de texto extraído con su ubicación y confianza."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    zone: str = "body"  # header, body, legal, signature, checkbox


@dataclass
class DocumentOCR:
    """Resultado estructurado de la extracción OCR."""
    blocks: list[TextBlock] = field(default_factory=list)
    full_text: str = ""
    avg_confidence: float = 0.0
    image_path: str = ""

    def get_text_by_zone(self, zone: str) -> str:
        return " ".join(b.text for b in self.blocks if b.zone == zone)


def extract_text_blocks(image_path: str, lang: str = "spa") -> DocumentOCR:
    """
    Extrae texto con datos de posición y confianza.
    """
    # Preprocesar imagen
    processed = preprocess_document(image_path)

    # Extraer con Tesseract a nivel de bloque
    data = pytesseract.image_to_data(
        processed, lang=lang, output_type=Output.DICT
    )

    blocks = []
    img_height = processed.shape[0]

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = float(data['conf'][i])

        if not text or conf < 0:
            continue

        block = TextBlock(
            text=text,
            x=data['left'][i],
            y=data['top'][i],
            width=data['width'][i],
            height=data['height'][i],
            confidence=conf,
        )

        # Clasificación básica por posición vertical
        relative_y = block.y / img_height
        if relative_y < 0.15:
            block.zone = "header"
        elif relative_y > 0.85:
            block.zone = "signature"
        elif relative_y > 0.70:
            block.zone = "legal"
        else:
            block.zone = "body"

        blocks.append(block)

    full_text = pytesseract.image_to_string(processed, lang=lang)

    avg_conf = (sum(b.confidence for b in blocks) / len(blocks)
                if blocks else 0.0)

    return DocumentOCR(
        blocks=blocks,
        full_text=full_text,
        avg_confidence=avg_conf,
        image_path=image_path,
    )


def detect_signatures(image_path: str) -> list[dict]:
    """
    Detecta zonas con posibles firmas manuscritas.
    Busca áreas con alta densidad de trazos no textuales.
    """
    img = load_image(image_path)
    gray = to_grayscale(img)

    # Binarizar
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Buscar contornos grandes en zona inferior del documento
    h, w = binary.shape
    roi = binary[int(h*0.7):, :]  # Solo parte inferior

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    signatures = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # Firma: área razonable, no muy cuadrada (no es casilla)
        aspect = cw / max(ch, 1)
        if 500 < area < 50000 and 1.5 < aspect < 8:
            signatures.append({
                "x": x, "y": y + int(h*0.7),
                "width": cw, "height": ch,
                "area": area
            })

    return signatures


def detect_checkboxes(image_path: str) -> list[dict]:
    """
    Detecta casillas de verificación y determina si están marcadas.
    """
    img = load_image(image_path)
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

    checkboxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)

        # Casilla: cuadrada, tamaño típico 15-40px
        if 0.8 < aspect < 1.2 and 15 < cw < 50 and 15 < ch < 50:
            # Determinar si está marcada: densidad de píxeles negros
            roi = binary[y:y+ch, x:x+cw]
            fill_ratio = np.sum(roi > 0) / (cw * ch)
            is_checked = fill_ratio > 0.3

            checkboxes.append({
                "x": x, "y": y,
                "width": cw, "height": ch,
                "checked": is_checked,
                "fill_ratio": round(fill_ratio, 3)
            })

    return checkboxes
```

### 1.3. Probar el OCR con un documento de ejemplo

Crear `notebooks/01_test_ocr.py`:

```python
"""
Script de prueba para el pipeline OCR.
Usa un documento de ejemplo para verificar que todo funciona.
"""
from src.ocr.extractor import extract_text_blocks, detect_signatures, detect_checkboxes
from rich.console import Console
from rich.table import Table

console = Console()

# Prueba con un documento (cambia la ruta)
image_path = "data/synthetic/ejemplo_consentimiento_01.png"

console.print("[bold]1. Extracción de texto[/bold]")
result = extract_text_blocks(image_path)
console.print(f"  Bloques extraídos: {len(result.blocks)}")
console.print(f"  Confianza media: {result.avg_confidence:.1f}%")
console.print(f"  Texto completo ({len(result.full_text)} chars):")
console.print(result.full_text[:500])

console.print("\n[bold]2. Texto por zonas[/bold]")
for zone in ["header", "body", "legal", "signature"]:
    text = result.get_text_by_zone(zone)
    if text:
        console.print(f"  [{zone}]: {text[:100]}...")

console.print("\n[bold]3. Detección de firmas[/bold]")
sigs = detect_signatures(image_path)
console.print(f"  Firmas detectadas: {len(sigs)}")

console.print("\n[bold]4. Detección de casillas[/bold]")
checks = detect_checkboxes(image_path)
table = Table(title="Casillas detectadas")
table.add_column("Posición")
table.add_column("Marcada")
table.add_column("Fill ratio")
for cb in checks:
    table.add_row(
        f"({cb['x']}, {cb['y']})",
        "✓" if cb['checked'] else "✗",
        str(cb['fill_ratio'])
    )
console.print(table)
```

---

## Fase 2: Protocolos de validación YAML (Día 5-7)

### 2.1. Esquema de protocolo

Crear `configs/schema_protocolo.yaml`:

```yaml
# Esquema que deben seguir todos los protocolos de validación
version: "1.0"
description: "Esquema base para protocolos de validación documental"

estructura:
  metadatos:
    - id_protocolo        # Identificador único
    - version             # Versión del protocolo
    - proceso_sanitario   # Proceso al que aplica
    - tipo_documento      # Tipo de documento que valida
    - fecha_vigencia      # Desde cuándo aplica

  campos_obligatorios:    # Lista de zonas que deben tener contenido
    - nombre_zona         # Nombre identificativo
    - tipo_contenido      # texto | firma | casilla | fecha
    - zona_documento      # header | body | legal | signature
    - descripcion         # Qué se espera encontrar

  reglas_contenido:       # Patrones que deben aparecer
    - id_regla
    - zona_documento
    - tipo                # regex | keyword | semantic
    - patron              # Expresión regular o palabras clave
    - obligatoria         # true | false
    - mensaje_error       # Mensaje si no se cumple

  reglas_coherencia:      # Relaciones entre campos
    - id_regla
    - tipo                # fecha_posterior | campo_presente_si | texto_coherente
    - campo_origen
    - campo_destino
    - condicion
    - mensaje_error
```

### 2.2. Ejemplo: Protocolo de Consentimiento Informado

Crear `configs/protocolos/consentimiento_informado.yaml`:

```yaml
id_protocolo: "CI-001"
version: "1.0"
proceso_sanitario: "general"
tipo_documento: "consentimiento_informado"
fecha_vigencia: "2024-01-01"
base_legal: "Ley 41/2002, Art. 8"

campos_obligatorios:
  - nombre_zona: "identificacion_paciente"
    tipo_contenido: "texto"
    zona_documento: "header"
    descripcion: "Nombre completo y DNI/NIE del paciente"

  - nombre_zona: "identificacion_profesional"
    tipo_contenido: "texto"
    zona_documento: "header"
    descripcion: "Nombre y número de colegiado del profesional"

  - nombre_zona: "descripcion_procedimiento"
    tipo_contenido: "texto"
    zona_documento: "body"
    descripcion: "Descripción clara del procedimiento a realizar"

  - nombre_zona: "riesgos_generales"
    tipo_contenido: "texto"
    zona_documento: "body"
    descripcion: "Riesgos generales del procedimiento"

  - nombre_zona: "riesgos_especificos"
    tipo_contenido: "texto"
    zona_documento: "body"
    descripcion: "Riesgos específicos personalizados"

  - nombre_zona: "alternativas"
    tipo_contenido: "texto"
    zona_documento: "body"
    descripcion: "Alternativas al procedimiento propuesto"

  - nombre_zona: "declaracion_comprension"
    tipo_contenido: "texto"
    zona_documento: "legal"
    descripcion: "Declaración de que el paciente comprende la información"

  - nombre_zona: "firma_paciente"
    tipo_contenido: "firma"
    zona_documento: "signature"
    descripcion: "Firma del paciente o representante legal"

  - nombre_zona: "firma_profesional"
    tipo_contenido: "firma"
    zona_documento: "signature"
    descripcion: "Firma del profesional sanitario"

  - nombre_zona: "fecha_firma"
    tipo_contenido: "fecha"
    zona_documento: "signature"
    descripcion: "Fecha de firma del documento"

reglas_contenido:
  - id_regla: "RC-001"
    zona_documento: "body"
    tipo: "keyword"
    patron: ["riesgo", "complicación", "complicacion"]
    obligatoria: true
    mensaje_error: "No se encontró mención de riesgos en el cuerpo del documento"

  - id_regla: "RC-002"
    zona_documento: "body"
    tipo: "keyword"
    patron: ["alternativa", "opción", "opcion"]
    obligatoria: true
    mensaje_error: "No se encontró mención de alternativas al procedimiento"

  - id_regla: "RC-003"
    zona_documento: "legal"
    tipo: "keyword"
    patron: ["voluntariamente", "libremente", "consentimiento"]
    obligatoria: true
    mensaje_error: "No se encontró declaración de voluntariedad"

  - id_regla: "RC-004"
    zona_documento: "header"
    tipo: "regex"
    patron: "\\d{8}[A-Z]"
    obligatoria: true
    mensaje_error: "No se encontró DNI/NIE válido en la cabecera"

  - id_regla: "RC-005"
    zona_documento: "signature"
    tipo: "regex"
    patron: "\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}"
    obligatoria: true
    mensaje_error: "No se encontró fecha de firma"

reglas_coherencia:
  - id_regla: "RCH-001"
    tipo: "campo_presente_si"
    campo_origen: "firma_paciente"
    campo_destino: "fecha_firma"
    condicion: "Si hay firma del paciente, debe haber fecha"
    mensaje_error: "Existe firma del paciente pero no se encontró la fecha"

  - id_regla: "RCH-002"
    tipo: "texto_coherente"
    campo_origen: "descripcion_procedimiento"
    campo_destino: "riesgos_especificos"
    condicion: "Los riesgos deben ser coherentes con el procedimiento descrito"
    mensaje_error: "Los riesgos específicos no parecen corresponderse con el procedimiento"
```

### 2.3. Motor de reglas

Crear `src/rules/engine.py`:

```python
"""
Motor de validación basado en reglas YAML.
Carga protocolos y evalúa documentos contra sus reglas.
"""
import yaml
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from src.ocr.extractor import DocumentOCR


@dataclass
class ValidationIssue:
    """Una incidencia detectada en la validación."""
    rule_id: str
    severity: str  # "error" | "warning"
    message: str
    zone: Optional[str] = None
    details: Optional[str] = None


@dataclass
class ValidationResult:
    """Resultado de la validación de un documento."""
    verdict: str  # "valid" | "incomplete" | "inconsistent"
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0

    @property
    def score(self) -> float:
        if self.checks_total == 0:
            return 0.0
        return self.checks_passed / self.checks_total


class ProtocolLoader:
    """Carga y valida protocolos YAML."""

    def __init__(self, protocols_dir: str = "configs/protocolos"):
        self.protocols_dir = Path(protocols_dir)
        self.protocols = {}

    def load_all(self):
        for yaml_file in self.protocols_dir.glob("*.yaml"):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                protocol = yaml.safe_load(f)
            self.protocols[protocol['id_protocolo']] = protocol
        return self.protocols

    def get_protocol(self, protocol_id: str) -> dict:
        if not self.protocols:
            self.load_all()
        return self.protocols.get(protocol_id)


class RuleEngine:
    """Evalúa un documento OCR contra un protocolo de validación."""

    def __init__(self, protocol: dict):
        self.protocol = protocol

    def validate(self, doc: DocumentOCR,
                 signatures: list = None,
                 checkboxes: list = None) -> ValidationResult:
        issues = []
        checks_passed = 0
        checks_total = 0

        # 1. Verificar campos obligatorios
        for campo in self.protocol.get('campos_obligatorios', []):
            checks_total += 1
            zone_text = doc.get_text_by_zone(campo['zona_documento'])

            if campo['tipo_contenido'] == 'firma':
                # Verificar presencia de firma en la zona
                has_signature = signatures and len(signatures) > 0
                if has_signature:
                    checks_passed += 1
                else:
                    issues.append(ValidationIssue(
                        rule_id=f"CAMPO-{campo['nombre_zona']}",
                        severity="error",
                        message=f"Campo obligatorio ausente: {campo['descripcion']}",
                        zone=campo['zona_documento'],
                    ))
            elif campo['tipo_contenido'] == 'texto':
                if zone_text.strip():
                    checks_passed += 1
                else:
                    issues.append(ValidationIssue(
                        rule_id=f"CAMPO-{campo['nombre_zona']}",
                        severity="error",
                        message=f"Campo obligatorio vacío: {campo['descripcion']}",
                        zone=campo['zona_documento'],
                    ))

        # 2. Verificar reglas de contenido
        for regla in self.protocol.get('reglas_contenido', []):
            checks_total += 1
            zone_text = doc.get_text_by_zone(
                regla['zona_documento']
            ).lower()

            found = False
            if regla['tipo'] == 'keyword':
                found = any(kw.lower() in zone_text
                           for kw in regla['patron'])
            elif regla['tipo'] == 'regex':
                found = bool(re.search(regla['patron'], zone_text))

            if found:
                checks_passed += 1
            elif regla.get('obligatoria', True):
                issues.append(ValidationIssue(
                    rule_id=regla['id_regla'],
                    severity="error",
                    message=regla['mensaje_error'],
                    zone=regla['zona_documento'],
                ))

        # 3. Determinar veredicto
        errors = [i for i in issues if i.severity == "error"]
        if not errors:
            verdict = "valid"
        elif len(errors) <= 2:
            verdict = "incomplete"
        else:
            verdict = "inconsistent"

        return ValidationResult(
            verdict=verdict,
            issues=issues,
            checks_passed=checks_passed,
            checks_total=checks_total,
        )
```

---

## Fase 3: Módulo LLM local (Día 7-10)

### 3.1. Interfaz Mistral-7B con llama.cpp

Crear `src/llm/mistral_client.py`:

```python
"""
Cliente para Mistral-7B-Instruct ejecutado localmente con llama.cpp.
Se encarga del razonamiento semántico sobre el texto extraído.
"""
from llama_cpp import Llama
from pathlib import Path


class MistralClient:
    """Cliente para inferencia local con Mistral-7B."""

    def __init__(self, model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                 n_ctx: int = 4096, n_threads: int = 4):
        self.model = Llama(
            model_path=str(Path(model_path)),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )

    def validate_semantic(self, extracted_text: str,
                          rule_description: str,
                          document_type: str) -> dict:
        """
        Evalúa semánticamente si el texto cumple una regla.
        Retorna: {"verdict": "conforme|no_conforme|indeterminado",
                  "justification": "..."}
        """
        prompt = f"""[INST] Eres un auditor de documentación sanitaria. Tu tarea es verificar si el siguiente texto extraído de un documento tipo "{document_type}" cumple con la regla de validación indicada.

TEXTO EXTRAÍDO:
{extracted_text[:2000]}

REGLA DE VALIDACIÓN:
{rule_description}

Analiza cuidadosamente el texto y responde EXCLUSIVAMENTE con un JSON válido con esta estructura:
{{"verdict": "conforme" o "no_conforme" o "indeterminado", "justification": "explicación breve en español"}}

Responde solo con el JSON, sin texto adicional. [/INST]"""

        response = self.model(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["[INST]", "\n\n"],
        )

        text = response['choices'][0]['text'].strip()

        # Parsear respuesta
        import json
        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError:
            return {
                "verdict": "indeterminado",
                "justification": f"Error al parsear respuesta del modelo: {text[:200]}"
            }

    def check_coherence(self, text_a: str, text_b: str,
                        relation: str) -> dict:
        """Verifica coherencia entre dos secciones del documento."""
        prompt = f"""[INST] Eres un auditor de documentación sanitaria. Debes verificar la coherencia entre dos secciones de un documento clínico.

SECCIÓN A:
{text_a[:1000]}

SECCIÓN B:
{text_b[:1000]}

RELACIÓN ESPERADA: {relation}

¿Son coherentes estas dos secciones según la relación descrita? Responde con JSON:
{{"coherent": true o false, "justification": "explicación breve en español"}}

Solo el JSON, sin texto adicional. [/INST]"""

        response = self.model(
            prompt,
            max_tokens=256,
            temperature=0.1,
        )

        text = response['choices'][0]['text'].strip()
        import json
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"coherent": None, "justification": f"Error de parseo: {text[:200]}"}
```

### 3.2. Interfaz Qwen2.5-VL

Crear `src/llm/qwen_vl_client.py`:

```python
"""
Cliente para Qwen2.5-VL ejecutado localmente.
Procesa directamente imágenes de documentos para verificación visual.
"""
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import json


class QwenVLClient:
    """Cliente para inferencia local con Qwen2.5-VL."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Carga el modelo. Usa 3B para hardware limitado, 7B si tienes
        GPU con 16+ GB VRAM.
        """
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def verify_document_zone(self, image_path: str,
                              zone_description: str,
                              check_type: str = "presence") -> dict:
        """
        Verifica una zona del documento usando visión directa.

        Args:
            image_path: Ruta a la imagen del documento
            zone_description: Qué buscar (ej: "firma del paciente")
            check_type: "presence" | "readability" | "checkbox"

        Returns:
            {"found": bool, "confidence": str, "details": str}
        """
        prompts_by_type = {
            "presence": (
                f"Analiza esta imagen de un documento sanitario. "
                f"¿Puedes identificar {zone_description}? "
                f"Responde con JSON: "
                f'{{"found": true/false, "confidence": "alta/media/baja", '
                f'"details": "descripción de lo que ves"}}'
            ),
            "readability": (
                f"Analiza la legibilidad del texto en esta imagen de documento sanitario. "
                f"¿El texto de {zone_description} es legible? "
                f"Responde con JSON: "
                f'{{"readable": true/false, "confidence": "alta/media/baja", '
                f'"extracted_text": "texto que puedes leer", "issues": "problemas encontrados"}}'
            ),
            "checkbox": (
                f"Analiza las casillas de verificación en esta imagen de documento sanitario. "
                f"¿La casilla correspondiente a '{zone_description}' está marcada? "
                f"Responde con JSON: "
                f'{{"checked": true/false, "confidence": "alta/media/baja", '
                f'"details": "descripción de lo que ves"}}'
            ),
        }

        prompt = prompts_by_type.get(check_type, prompts_by_type["presence"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        response_text = self.processor.decode(
            generated, skip_special_tokens=True
        ).strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {
                "found": None,
                "confidence": "baja",
                "details": f"Respuesta no parseable: {response_text[:200]}"
            }

    def extract_and_validate(self, image_path: str,
                              validation_rules: list[str]) -> dict:
        """
        Pipeline completo: extrae texto de imagen y valida reglas.
        Usa Qwen2.5-VL como alternativa al pipeline OCR+Mistral.
        """
        rules_text = "\n".join(f"- {r}" for r in validation_rules)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": (
                        "Eres un auditor de documentación sanitaria. Analiza este documento "
                        "escaneado y verifica las siguientes reglas de validación:\n\n"
                        f"{rules_text}\n\n"
                        "Para cada regla, indica si se cumple o no y por qué. "
                        "Responde con JSON:\n"
                        '{"results": [{"rule": "texto de la regla", '
                        '"compliant": true/false, "justification": "..."}], '
                        '"overall_verdict": "valid/incomplete/inconsistent"}'
                    )},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
        )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        response_text = self.processor.decode(
            generated, skip_special_tokens=True
        ).strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"error": f"Respuesta no parseable: {response_text[:300]}"}
```

---

## Fase 4: Agente supervisor con LangGraph (Día 10-14)

### 4.1. Definición del grafo de estados

Crear `src/agent/supervisor.py`:

```python
"""
Agente supervisor que orquesta el pipeline completo.
Implementado como grafo de estados con LangGraph.
"""
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from typing import Optional, Any
import json
from datetime import datetime

from src.ocr.extractor import extract_text_blocks, detect_signatures, detect_checkboxes
from src.rules.engine import RuleEngine, ProtocolLoader, ValidationResult
from src.llm.mistral_client import MistralClient


@dataclass
class SupervisorState:
    """Estado del agente supervisor durante el procesamiento."""
    document_path: str = ""
    protocol_id: str = ""

    # Resultados intermedios
    ocr_result: Any = None
    signatures: list = field(default_factory=list)
    checkboxes: list = field(default_factory=list)
    rule_validation: Optional[ValidationResult] = None
    llm_validations: list = field(default_factory=list)

    # Resultado final
    verdict: str = ""
    report: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def ocr_step(state: dict) -> dict:
    """Paso 1: Extraer texto con OCR."""
    try:
        ocr_result = extract_text_blocks(state["document_path"])
        signatures = detect_signatures(state["document_path"])
        checkboxes = detect_checkboxes(state["document_path"])
        return {
            **state,
            "ocr_result": ocr_result,
            "signatures": signatures,
            "checkboxes": checkboxes,
        }
    except Exception as e:
        return {**state, "errors": state.get("errors", []) + [f"OCR error: {e}"]}


def rule_validation_step(state: dict) -> dict:
    """Paso 2: Validar contra reglas del protocolo."""
    if state.get("errors"):
        return state

    try:
        loader = ProtocolLoader()
        protocol = loader.get_protocol(state["protocol_id"])
        if not protocol:
            return {**state, "errors": [f"Protocolo no encontrado: {state['protocol_id']}"]}

        engine = RuleEngine(protocol)
        result = engine.validate(
            state["ocr_result"],
            state.get("signatures", []),
            state.get("checkboxes", []),
        )
        return {**state, "rule_validation": result}
    except Exception as e:
        return {**state, "errors": state.get("errors", []) + [f"Rule error: {e}"]}


def llm_validation_step(state: dict) -> dict:
    """Paso 3: Validación semántica con LLM."""
    if state.get("errors"):
        return state

    try:
        client = MistralClient()
        loader = ProtocolLoader()
        protocol = loader.get_protocol(state["protocol_id"])

        validations = []
        # Solo pasar al LLM las reglas de coherencia
        for regla in protocol.get("reglas_coherencia", []):
            if regla["tipo"] == "texto_coherente":
                text_a = state["ocr_result"].get_text_by_zone("body")
                text_b = state["ocr_result"].get_text_by_zone("body")
                result = client.check_coherence(text_a, text_b, regla["condicion"])
                validations.append({
                    "rule_id": regla["id_regla"],
                    "result": result,
                })

        return {**state, "llm_validations": validations}
    except Exception as e:
        return {**state, "errors": state.get("errors", []) + [f"LLM error: {e}"]}


def generate_report_step(state: dict) -> dict:
    """Paso 4: Generar informe final."""
    rule_result = state.get("rule_validation")
    llm_results = state.get("llm_validations", [])

    # Combinar veredictos
    if state.get("errors"):
        verdict = "error"
    elif rule_result and rule_result.verdict == "valid" and \
         all(v["result"].get("coherent", True) for v in llm_results):
        verdict = "valid"
    elif rule_result and rule_result.verdict == "inconsistent":
        verdict = "inconsistent"
    else:
        verdict = "incomplete"

    report = {
        "timestamp": datetime.now().isoformat(),
        "document": state["document_path"],
        "protocol": state["protocol_id"],
        "verdict": verdict,
        "ocr_confidence": state["ocr_result"].avg_confidence if state.get("ocr_result") else 0,
        "signatures_found": len(state.get("signatures", [])),
        "checkboxes": [
            {"position": f"({c['x']},{c['y']})", "checked": c["checked"]}
            for c in state.get("checkboxes", [])
        ],
        "rule_checks": {
            "passed": rule_result.checks_passed if rule_result else 0,
            "total": rule_result.checks_total if rule_result else 0,
            "issues": [
                {"id": i.rule_id, "severity": i.severity, "message": i.message}
                for i in (rule_result.issues if rule_result else [])
            ],
        },
        "semantic_checks": llm_results,
        "errors": state.get("errors", []),
    }

    return {**state, "verdict": verdict, "report": report}


def build_supervisor_graph():
    """Construye el grafo del agente supervisor."""
    graph = StateGraph(dict)

    graph.add_node("ocr", ocr_step)
    graph.add_node("rules", rule_validation_step)
    graph.add_node("llm", llm_validation_step)
    graph.add_node("report", generate_report_step)

    graph.set_entry_point("ocr")
    graph.add_edge("ocr", "rules")
    graph.add_edge("rules", "llm")
    graph.add_edge("llm", "report")
    graph.add_edge("report", END)

    return graph.compile()


def supervise_document(document_path: str, protocol_id: str) -> dict:
    """
    Punto de entrada principal.
    Ejecuta el pipeline completo sobre un documento.
    """
    graph = build_supervisor_graph()
    initial_state = {
        "document_path": document_path,
        "protocol_id": protocol_id,
        "errors": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state["report"]
```

### 4.2. Script de ejecución

Crear `run_supervisor.py`:

```python
"""
Script para ejecutar el agente supervisor sobre un documento.
Uso: python run_supervisor.py <imagen> <protocolo_id>
"""
import sys
import json
from rich.console import Console
from rich.panel import Panel
from src.agent.supervisor import supervise_document

console = Console()

if len(sys.argv) < 3:
    console.print("[red]Uso: python run_supervisor.py <imagen> <protocolo_id>[/red]")
    console.print("Ejemplo: python run_supervisor.py data/synthetic/ci_01.png CI-001")
    sys.exit(1)

image_path = sys.argv[1]
protocol_id = sys.argv[2]

console.print(f"\n[bold]Agente Supervisor de Documentación Sanitaria[/bold]")
console.print(f"Documento: {image_path}")
console.print(f"Protocolo: {protocol_id}\n")

with console.status("Procesando documento..."):
    report = supervise_document(image_path, protocol_id)

# Mostrar resultado
verdict_colors = {"valid": "green", "incomplete": "yellow", "inconsistent": "red", "error": "red"}
color = verdict_colors.get(report["verdict"], "white")

console.print(Panel(
    f"[bold {color}]{report['verdict'].upper()}[/bold {color}]",
    title="Veredicto",
    expand=False,
))

console.print(f"\nConfianza OCR: {report['ocr_confidence']:.1f}%")
console.print(f"Firmas detectadas: {report['signatures_found']}")
console.print(f"Verificaciones: {report['rule_checks']['passed']}/{report['rule_checks']['total']}")

if report["rule_checks"]["issues"]:
    console.print("\n[bold]Incidencias:[/bold]")
    for issue in report["rule_checks"]["issues"]:
        icon = "❌" if issue["severity"] == "error" else "⚠️"
        console.print(f"  {icon} [{issue['id']}] {issue['message']}")

# Guardar informe completo
output_path = f"data/results/report_{protocol_id}_{image_path.split('/')[-1].split('.')[0]}.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
console.print(f"\nInforme guardado en: {output_path}")
```

---

## Fase 5: Generar documentos sintéticos de prueba (Día 8-10, en paralelo)

### 5.1. Generador de consentimientos informados

Crear `scripts/generate_synthetic.py`:

```python
"""
Genera documentos sanitarios sintéticos para evaluación.
Incluye variantes con deficiencias controladas para el ground truth.
"""
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import random
import os

# pip install reportlab


def generate_consent_form(output_path: str, deficiencies: list = None):
    """
    Genera un consentimiento informado en PDF.

    deficiencies: lista de deficiencias a introducir:
        - "no_firma_paciente"
        - "no_fecha"
        - "sin_riesgos"
        - "sin_alternativas"
        - "dni_incompleto"
        - "texto_ilegible"
    """
    deficiencies = deficiencies or []
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # --- CABECERA ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2*cm, "CONSENTIMIENTO INFORMADO")

    c.setFont("Helvetica", 10)
    c.drawString(2*cm, height - 3*cm, "Hospital Universitario de Ejemplo")
    c.drawString(2*cm, height - 3.5*cm, "Servicio de Cirugía General")

    # Datos del paciente
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, height - 5*cm, "DATOS DEL PACIENTE")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, height - 5.7*cm, "Nombre: García López, María del Carmen")

    if "dni_incompleto" in deficiencies:
        c.drawString(2*cm, height - 6.2*cm, "DNI: 1234____")
    else:
        c.drawString(2*cm, height - 6.2*cm, "DNI: 12345678A")

    # Datos del profesional
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, height - 7.2*cm, "PROFESIONAL RESPONSABLE")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, height - 7.9*cm, "Dr. Martínez Ruiz, Antonio - Col. 28/12345")

    # Descripción del procedimiento
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, height - 9.2*cm, "DESCRIPCIÓN DEL PROCEDIMIENTO")
    c.setFont("Helvetica", 9)
    text = ("Se le propone la realización de una intervención quirúrgica "
            "consistente en colecistectomía laparoscópica para la extracción "
            "de la vesícula biliar debido a colelitiasis sintomática.")
    c.drawString(2*cm, height - 10*cm, text[:90])
    c.drawString(2*cm, height - 10.5*cm, text[90:])

    # Riesgos
    if "sin_riesgos" not in deficiencies:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2*cm, height - 11.8*cm, "RIESGOS")
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, height - 12.5*cm,
                     "Riesgos generales: infección, hemorragia, complicaciones anestésicas.")
        c.drawString(2*cm, height - 13*cm,
                     "Riesgos específicos: lesión de vía biliar, conversión a cirugía abierta.")

    # Alternativas
    if "sin_alternativas" not in deficiencies:
        c.setFont("Helvetica-Bold", 10)
        y_alt = height - 14.3*cm if "sin_riesgos" not in deficiencies else height - 11.8*cm
        c.drawString(2*cm, y_alt, "ALTERNATIVAS")
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, y_alt - 0.7*cm,
                     "Alternativa: tratamiento conservador con dieta y opción de vigilancia.")

    # Declaración
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, height - 17*cm, "DECLARACIÓN DEL PACIENTE")
    c.setFont("Helvetica", 9)
    c.drawString(2*cm, height - 17.7*cm,
                 "Declaro que he sido informado/a de forma clara y comprensible sobre")
    c.drawString(2*cm, height - 18.2*cm,
                 "el procedimiento, sus riesgos y alternativas, y otorgo mi consentimiento")
    c.drawString(2*cm, height - 18.7*cm,
                 "voluntariamente y libremente.")

    # Firmas
    c.setFont("Helvetica", 10)
    if "no_fecha" not in deficiencies:
        c.drawString(2*cm, height - 20.5*cm, "Fecha: 15/03/2025")

    c.drawString(2*cm, height - 22*cm, "Firma del paciente:")
    if "no_firma_paciente" not in deficiencies:
        # Simular una firma con líneas
        c.setStrokeColorRGB(0, 0, 0.5)
        c.setLineWidth(1.5)
        import math
        x_start = 2*cm
        y_base = height - 23*cm
        for i in range(50):
            x = x_start + i * 1.5
            y = y_base + math.sin(i * 0.3) * 5 + random.uniform(-2, 2)
            if i == 0:
                c.line(x, y, x+1, y)
            else:
                c.line(x-1.5, prev_y, x, y)
            prev_y = y

    c.drawString(10*cm, height - 22*cm, "Firma del profesional:")
    # Siempre incluir firma del profesional
    c.setStrokeColorRGB(0, 0, 0)
    c.line(10*cm, height - 23.2*cm, 14*cm, height - 23*cm)
    c.line(14*cm, height - 23*cm, 11*cm, height - 23.5*cm)

    c.save()


if __name__ == "__main__":
    os.makedirs("data/synthetic", exist_ok=True)

    # Documentos correctos
    for i in range(5):
        generate_consent_form(f"data/synthetic/ci_correcto_{i+1:02d}.pdf")

    # Documentos con deficiencias específicas
    deficiency_sets = [
        (["no_firma_paciente"], "sin_firma"),
        (["no_fecha"], "sin_fecha"),
        (["sin_riesgos"], "sin_riesgos"),
        (["sin_alternativas"], "sin_alternativas"),
        (["dni_incompleto"], "dni_incompleto"),
        (["no_firma_paciente", "no_fecha"], "sin_firma_ni_fecha"),
        (["sin_riesgos", "sin_alternativas"], "sin_riesgos_ni_alternativas"),
    ]

    for defects, label in deficiency_sets:
        generate_consent_form(f"data/synthetic/ci_defecto_{label}.pdf",
                             deficiencies=defects)

    # Ground truth
    import json
    ground_truth = {}
    for i in range(5):
        ground_truth[f"ci_correcto_{i+1:02d}"] = {"expected_verdict": "valid", "deficiencies": []}

    for defects, label in deficiency_sets:
        ground_truth[f"ci_defecto_{label}"] = {"expected_verdict": "incomplete", "deficiencies": defects}

    with open("data/synthetic/ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Generados {5 + len(deficiency_sets)} documentos sintéticos")
    print("Ground truth guardado en data/synthetic/ground_truth.json")
```

---

## Fase 6: Evaluación (Semanas 11-14)

### 6.1. Script de evaluación

Crear `scripts/evaluate.py`:

```python
"""
Evaluación del sistema contra el ground truth.
Calcula métricas: sensibilidad, especificidad, Cohen's kappa.
"""
import json
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    cohen_kappa_score, accuracy_score
)
from src.agent.supervisor import supervise_document
from rich.console import Console
from rich.table import Table

console = Console()

# Cargar ground truth
with open("data/synthetic/ground_truth.json") as f:
    ground_truth = json.load(f)

y_true = []
y_pred = []
results = []

for doc_name, expected in ground_truth.items():
    doc_path = f"data/synthetic/{doc_name}.pdf"
    console.print(f"Procesando: {doc_name}...")

    report = supervise_document(doc_path, "CI-001")

    y_true.append(expected["expected_verdict"])
    y_pred.append(report["verdict"])
    results.append({
        "document": doc_name,
        "expected": expected["expected_verdict"],
        "predicted": report["verdict"],
        "correct": expected["expected_verdict"] == report["verdict"],
    })

# Métricas
console.print("\n[bold]RESULTADOS DE EVALUACIÓN[/bold]\n")

table = Table(title="Detalle por documento")
table.add_column("Documento")
table.add_column("Esperado")
table.add_column("Predicho")
table.add_column("Correcto")
for r in results:
    color = "green" if r["correct"] else "red"
    table.add_row(r["document"], r["expected"], r["predicted"],
                  f"[{color}]{'✓' if r['correct'] else '✗'}[/{color}]")
console.print(table)

# Métricas globales
console.print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.2%}")
console.print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.3f}")
console.print(f"\n{classification_report(y_true, y_pred)}")
```

---

## Resumen: ¿Qué hacer y cuándo?

| Semana | Qué hacer | Entregable |
|--------|-----------|------------|
| 6-7 | Crear repo, instalar Tesseract, escribir módulo OCR | `src/ocr/` funcionando |
| 7-8 | Escribir protocolos YAML, motor de reglas | `configs/` + `src/rules/` |
| 8-9 | Instalar Mistral + Qwen2.5-VL, escribir clientes LLM | `src/llm/` funcionando |
| 9-10 | Integrar agente con LangGraph, generar docs sintéticos | Agente ejecutable |
| 10 | **ENTREGA 2**: Memoria + demo del pipeline | Borrador intermedio |
| 11-12 | Evaluación formal con métricas | Resultados cuantitativos |
| 13-14 | Completar memoria, revisión cruzada | **ENTREGA 3**: Final |
