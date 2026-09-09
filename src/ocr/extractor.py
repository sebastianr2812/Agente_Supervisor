"""
MÃ³dulo de extracciÃ³n de texto con Tesseract OCR.
Produce un documento estructurado con zonas semÃ¡nticas.
"""
import os
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .preprocess import preprocess_document, load_image, to_grayscale

# Cambia esta ruta si lo instalaste en otra carpeta
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Se fuerza (no setdefault) para que nunca dependa de si la terminal/IDE que
# ejecuta este proceso tiene o no la variable de entorno cargada correctamente.
# pytesseract hace "from os import environ", por lo que este cambio se refleja
# automaticamente en el proceso hijo que lanza tesseract.exe.
# Se usa la carpeta tessdata que instala Tesseract por defecto (ya trae
# eng.traineddata y osd.traineddata), en vez de una copia aparte en AppData.
os.environ["TESSDATA_PREFIX"] = r'C:\Program Files\Tesseract-OCR\tessdata'

HEADER_MAX_Y = 0.38
BODY_MAX_Y = 0.70
LEGAL_MAX_Y = 0.90

# Zonificacion v2 (heuristica multi-senal): la v1 clasificaba un bloque OCR
# solo por su fraccion vertical fija en la pagina, lo que el profesor senalo
# como no generalizable a firmas laterales, cabeceras ampliadas o disenos
# institucionales distintos. La v2 combina tres senales, en este orden de
# prioridad:
#   1. Contenido: si el texto del bloque contiene una palabra caracteristica
#      de firma/visado o de aviso legal, se clasifica por esa palabra sin
#      importar su posicion (resuelve el caso "firma lateral").
#   2. Hueco vertical adaptativo: el limite cabecera/cuerpo ya no es una
#      fraccion fija (0.38) sino el mayor hueco en blanco detectado en el
#      tercio superior de la pagina, propio de cada documento (resuelve el
#      caso "cabecera ampliada"). Los limites cuerpo/legal/firma siguen
#      siendo fracciones fijas: no se observo en la evaluacion un patron de
#      hueco igual de fiable para esas fronteras, y mantenerlas fijas evita
#      introducir una fuente de error nueva sin evidencia que la respalde.
#   3. Conciencia de columnas: NO implementada en esta version. Un formulario
#      a dos columnas seguiria zonificandose solo por Y. Se documenta como
#      limitacion conocida y linea de mejora, en vez de forzar una
#      implementacion no evaluada.
# El emparejamiento por palabra se hace a nivel de la palabra OCR individual
# (no de frase completa), porque Tesseract entrega bloques palabra a palabra
# y los terminos mas diagnosticos (firma, sello, visado, confidencial, lopd,
# rgpd) suelen aparecer como palabras sueltas en el texto extraido.
SIGNATURE_KEYWORDS = ("firma", "firmado", "firmante", "sello", "visado")
LEGAL_KEYWORDS = ("confidencial", "confidencialidad", "lopd", "rgpd")

# Hueco minimo (px a 150 DPI) para considerar que un salto vertical entre
# bloques es una frontera de seccion real y no el espaciado normal entre
# lineas de un mismo parrafo.
MIN_HEADER_GAP_PX = 30
# Rango donde se acepta un limite de cabecera adaptativo. Fuera de este
# rango se descarta el hueco encontrado y se usa HEADER_MAX_Y de respaldo,
# para evitar confundir un hueco grande en mitad del cuerpo con el fin de
# la cabecera.
ADAPTIVE_HEADER_MIN = 0.15
ADAPTIVE_HEADER_MAX = 0.60


@dataclass
class TextBlock:
    """Un bloque de texto extraido con su ubicacion y confianza."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    zone: str = "body"


@dataclass
class DocumentOCR:
    """Resultado estructurado de la extraccion OCR."""
    blocks: list[TextBlock] = field(default_factory=list)
    full_text: str = ""
    avg_confidence: float = 0.0
    image_path: str = ""

    def get_text_by_zone(self, zone: str) -> str:
        return " ".join(b.text for b in self.blocks if b.zone == zone)


def _content_zone_override(text: str) -> Optional[str]:
    """
    Clasifica un bloque por su contenido cuando contiene una palabra
    caracteristica de firma/visado o de aviso legal, sin importar su
    posicion. Devuelve None si no hay coincidencia (se usa la posicion).
    """
    t = text.lower().strip(" .,:;()[]")
    if not t:
        return None
    for kw in SIGNATURE_KEYWORDS:
        if kw in t:
            return "signature"
    for kw in LEGAL_KEYWORDS:
        if kw in t:
            return "legal"
    return None


def _adaptive_header_boundary(y_positions: list[int], img_height: int) -> float:
    """
    Calcula el limite cabecera/cuerpo como el mayor hueco vertical entre
    bloques de texto dentro de la mitad superior de la pagina, en vez de
    asumir siempre la fraccion fija HEADER_MAX_Y. Si no hay un hueco claro
    (contenido denso desde el principio, o muy pocos bloques), se devuelve
    HEADER_MAX_Y como valor de respaldo.
    """
    if not y_positions:
        return HEADER_MAX_Y

    candidate_ys = sorted({y for y in y_positions if y / img_height <= 0.60})
    if len(candidate_ys) < 3:
        return HEADER_MAX_Y

    gaps = [(b - a, (a + b) / 2) for a, b in zip(candidate_ys, candidate_ys[1:])]
    best_gap, boundary_y = max(gaps, key=lambda g: g[0])

    if best_gap < MIN_HEADER_GAP_PX:
        return HEADER_MAX_Y

    boundary = boundary_y / img_height
    if not (ADAPTIVE_HEADER_MIN <= boundary <= ADAPTIVE_HEADER_MAX):
        return HEADER_MAX_Y

    return boundary


def classify_zone(relative_y: float, header_boundary: float = HEADER_MAX_Y) -> str:
    """
    Clasifica un bloque OCR en una zona logica del documento por posicion.
    `header_boundary` permite pasar el limite adaptativo calculado por
    `_adaptive_header_boundary` en vez del valor fijo HEADER_MAX_Y; se usa
    solo cuando `_content_zone_override` no determino una zona por
    contenido (ver extract_text_blocks).
    """
    if relative_y < header_boundary:
        return "header"
    if relative_y < BODY_MAX_Y:
        return "body"
    if relative_y < LEGAL_MAX_Y:
        return "legal"
    return "signature"


def extract_text_blocks(image_path: str, lang: str = "spa") -> DocumentOCR:
    """Extrae texto con datos de posicion y confianza, y lo zonifica con
    la heuristica multi-senal (contenido + hueco adaptativo + posicion,
    ver comentario junto a SIGNATURE_KEYWORDS)."""
    processed = preprocess_document(image_path)

    data = pytesseract.image_to_data(
        processed, lang=lang, output_type=Output.DICT
    )

    img_height = processed.shape[0]

    raw_words = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = float(data['conf'][i])
        if not text or conf < 0:
            continue
        raw_words.append({
            "text": text,
            "x": data['left'][i],
            "y": data['top'][i],
            "width": data['width'][i],
            "height": data['height'][i],
            "confidence": conf,
        })

    header_boundary = _adaptive_header_boundary(
        [w["y"] for w in raw_words], img_height
    )

    blocks = []
    for w in raw_words:
        block = TextBlock(
            text=w["text"], x=w["x"], y=w["y"],
            width=w["width"], height=w["height"], confidence=w["confidence"],
        )

        override = _content_zone_override(w["text"])
        if override:
            block.zone = override
        else:
            block.zone = classify_zone(w["y"] / img_height, header_boundary)

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
    Detecta posibles firmas manuscritas en todo el documento.

    CAMBIOS v4: se elimino la restriccion a un ROI vertical fijo (v2/v3
    buscaban solo en el 35% inferior, asumiendo firma siempre al pie de
    pagina). Esa asuncion no se cumple en formularios con firma en columna
    lateral o a media altura, y el motor de reglas evalua la deteccion
    contra la zona esperada del protocolo (zona_documento), no contra la
    posicion en la que se busco: restringir la busqueda aqui solo podia
    producir falsos negativos, nunca aportaba precision adicional que la
    zona del protocolo no aportara ya. Se mantienen sin cambios los
    filtros de forma (area, aspect ratio, complejidad, densidad) que son
    los que realmente distinguen una firma de otros trazos.

    CAMBIOS v2 (vigentes):
    - Umbral de area mas permisivo (200 - 80000)
    - Aspect ratio mas amplio (1.2 - 12)
    - Filtra contornos que son lineas rectas (separadores)
    - Anade analisis de complejidad del contorno
    """
    img = load_image(image_path)
    gray = to_grayscale(img)

    # Binarizar con Otsu
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape

    # Buscar en todo el documento (ver CAMBIOS v4 en el docstring)
    y_start = 0
    roi = binary[y_start:, :]
 
    # Eliminar lÃ­neas horizontales rectas (separadores, lÃ­neas de firma)
    # Crear kernel horizontal largo
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    detected_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN,
                                       horizontal_kernel, iterations=1)
    # Restar las lÃ­neas rectas del ROI
    roi_clean = cv2.subtract(roi, detected_lines)
 
    # TambiÃ©n eliminar lÃ­neas verticales
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_vlines = cv2.morphologyEx(roi_clean, cv2.MORPH_OPEN,
                                        vertical_kernel, iterations=1)
    roi_clean = cv2.subtract(roi_clean, detected_vlines)
 
    # Dilatar ligeramente para conectar trazos de firma cercanos
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi_dilated = cv2.dilate(roi_clean, dilate_kernel, iterations=2)
 
    contours, _ = cv2.findContours(roi_dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
 
    signatures = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        rect_area = cw * ch
        if rect_area == 0:
            continue
 
        aspect = cw / max(ch, 1)
        # Densidad del contorno dentro del bounding box
        density = area / rect_area
        # Complejidad: perÃ­metroÂ² / Ã¡rea (mÃ¡s alto = mÃ¡s complejo = mÃ¡s probable firma)
        perimeter = cv2.arcLength(cnt, True)
        complexity = (perimeter ** 2) / max(area, 1)
 
        # CRITERIOS DE FIRMA:
        # - Ãrea entre 1000 y 80000 pÃ­xeles
        # - Aspect ratio entre 2.0 y 12
        # - Ancho mÃ­nimo 80px
        # - Altura mÃ­nima 10px
        # - Complejidad > 30
        # - Densidad < 0.6
        if (1000 < area < 80000 and
            2.0 < aspect < 12 and
            cw > 80 and ch > 10 and
            complexity > 30 and
            density < 0.6):
            signatures.append({
                "x": x,
                "y": y + y_start,
                "width": cw,
                "height": ch,
                "area": int(area),
                "aspect_ratio": round(aspect, 2),
                "complexity": round(complexity, 1),
                "density": round(density, 3),
            })
 
    # Ordenar por Ã¡rea (las firmas mÃ¡s grandes primero)
    signatures.sort(key=lambda s: s["area"], reverse=True)
 
    return signatures
 
 
def detect_checkboxes(image_path: str) -> list[dict]:
    """
    Detecta casillas de verificacion en todo el documento y determina si
    estan marcadas.

    CAMBIOS v4: se elimino la restriccion a la mitad inferior del
    documento. La asuncion original ("las casillas de consentimiento
    estan abajo") no se cumple para checklists cuyos items ocupan buena
    parte del cuerpo del documento (p. ej. PREOP-001, donde el bloque de
    verificacion empieza bien antes del 50% de la pagina): con esa
    restriccion, la deteccion fallaba sistematicamente (0% de recall)
    incluso en las plantillas de desarrollo, no solo en las mas dificiles.
    Los filtros de forma (cuadratura, tamano, vertices, extension) siguen
    siendo los que evitan falsos positivos, no la posicion en la pagina.

    CAMBIOS v2 (vigentes):
    - Rango de tamano ajustado: 18-35px (las casillas reales son ~20px)
    - Filtra por cuadratura mas estricta (0.85 - 1.15)
    - Verifica que la casilla tenga bordes claros (4 lados)
    - Excluye contornos que son parte de letras/texto
    """
    img = load_image(image_path)
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape

    # Buscar en todo el documento (ver CAMBIOS v4 en el docstring)
    y_start = 0
    roi = binary[y_start:, :]
 
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
 
    checkboxes = []
    for i, cnt in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        area = cv2.contourArea(cnt)
        rect_area = cw * ch
 
        if rect_area == 0:
            continue
 
        # CRITERIOS DE CASILLA:
        # 1. Cuadrada (aspect ratio 0.85 - 1.15)
        # 2. TamaÃ±o tÃ­pico: 18-35 px (a 150 DPI, las casillas son ~20px)
        # 3. El contorno debe parecerse a un rectÃ¡ngulo
        #    (approxPolyDP con pocos vÃ©rtices)
        if not (0.85 < aspect < 1.15 and 18 < cw < 35 and 18 < ch < 35):
            continue
 
        # Verificar que el contorno es aproximadamente rectangular
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices = len(approx)
 
        # Un cuadrado tiene 4 vÃ©rtices (+/- 1 por imperfecciÃ³n)
        if num_vertices < 3 or num_vertices > 6:
            continue
 
        # Verificar que la extensiÃ³n (Ã¡rea contorno / Ã¡rea bounding box)
        # es alta (> 0.7 para formas rectangulares)
        extent = area / rect_area
        if extent < 0.5:
            continue
 
        # Determinar si estÃ¡ marcada: analizar densidad INTERIOR
        # Reducir el ROI para excluir los bordes de la casilla
        margin = 4
        inner_y1 = max(0, y + margin)
        inner_y2 = min(roi.shape[0], y + ch - margin)
        inner_x1 = max(0, x + margin)
        inner_x2 = min(roi.shape[1], x + cw - margin)
 
        inner_roi = roi[inner_y1:inner_y2, inner_x1:inner_x2]
        if inner_roi.size == 0:
            continue
 
        inner_fill = np.sum(inner_roi > 0) / inner_roi.size
        is_checked = inner_fill > 0.15  # Umbral bajo porque la X no llena todo
 
        checkboxes.append({
            "x": x,
            "y": y + y_start,
            "width": cw,
            "height": ch,
            "checked": is_checked,
            "fill_ratio": round(inner_fill, 3),
            "vertices": num_vertices,
            "extent": round(extent, 3),
        })
 
    # Eliminar duplicados cercanos (mismo checkbox detectado 2 veces)
    filtered = []
    for cb in checkboxes:
        is_duplicate = False
        for existing in filtered:
            if (abs(cb["x"] - existing["x"]) < 10 and
                abs(cb["y"] - existing["y"]) < 10):
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(cb)
 
    # Ordenar por posiciÃ³n Y (de arriba a abajo)
    filtered.sort(key=lambda c: c["y"])
 
    return filtered
