"""
Modulo de preprocesamiento de imagenes de documentos sanitarios.
Aplica correcciones geometricas y mejora de calidad antes del OCR.

FIX 1 (deskew): la funcion deskew() fallaba con
"cannot unpack non-iterable numpy.int32 object" porque
cv2.HoughLinesP puede devolver las lineas con forma (N, 4) o
(N, 1, 4) segun la version de opencv-python instalada. Se
normaliza con .reshape(-1)[:4] para que funcione en ambos casos.

FIX 2 (remove_noise, 2026-07-21): la evaluacion integral sobre el
corpus sintetico revelo que el filtro de mediana (medianBlur, k=3)
aplicado DESPUES de la binarizacion de Sauvola destruye
sistematicamente el texto en fuentes normales/pequenas (<=11-12pt a
150 DPI). El motivo: en una imagen ya binarizada, un trazo de texto
fino (1-2 px de ancho) queda rodeado mayoritariamente por pixeles de
fondo en una ventana 3x3, por lo que la mediana lo sustituye por
blanco, borrando el caracter. Se comprobo empiricamente (OCR con
Tesseract + idioma "spa" sobre un documento sintetico de prueba):

  - Sin remove_noise(): 5/6 palabras clave del protocolo detectadas
    correctamente (paciente, motivo de consulta, antecedentes,
    medico, hospital).
  - Con remove_noise() (comportamiento original): 0-1/6 palabras
    clave detectadas; el texto se convierte en ruido irreconocible
    salvo en titulos grandes en negrita (>=14pt).

Por tanto, remove_noise() se ha retirado del pipeline por defecto
en preprocess_document(). La funcion se conserva por si se necesita
para escaneos reales con ruido de sal y pimienta genuino (polvo,
grano de escaner), pero debe aplicarse con cautela y, preferiblemente,
con un kernel mas pequeno o un filtro que preserve mejor los bordes
(p. ej. cv2.fastNlMeansDenoising) en vez de medianBlur con k=3.
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
    Corrige la inclinacion del documento usando la transformada de Hough.
    Detecta lineas horizontales y calcula el angulo de rotacion necesario.
    """
    gray = to_grayscale(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return img

    angles = []
    for line in lines:
        # cv2.HoughLinesP puede devolver cada linea como shape (4,)
        # o como shape (1, 4) segun la version de opencv-python.
        # reshape(-1) normaliza ambos casos a un vector plano de 4 valores.
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:  # Solo lineas casi horizontales
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
    Binarizacion adaptativa con el metodo de Sauvola.
    Mejor que Otsu para documentos con iluminacion variable.

    T(x,y) = mean(x,y) * (1 + k * (std(x,y) / R - 1))
    donde R = max(std) = 128 para imagenes de 8 bits.
    """
    gray = to_grayscale(img)
    gray = gray.astype(np.float64)

    # Calcular media y desviacion estandar local
    mean = cv2.blur(gray, (window_size, window_size))
    mean_sq = cv2.blur(gray ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    R = 128.0
    threshold = mean * (1.0 + k * (std / R - 1.0))

    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray >= threshold] = 255

    return binary


def remove_noise(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Elimina ruido con filtro de mediana.

    ADVERTENCIA (ver docstring del modulo): aplicar esta funcion sobre
    una imagen ya binarizada con texto en fuente normal/pequena
    (<=11-12pt a 150 DPI) puede borrar los trazos por completo. Solo
    se recomienda usarla sobre escaneos reales con ruido de sal y
    pimienta verificado, idealmente sobre la imagen en escala de
    grises ANTES de binarizar, y evaluando el impacto en la tasa de
    deteccion de campos antes de activarla en produccion.
    """
    return cv2.medianBlur(img, kernel_size)


def upscale(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """
    Sobremuestrea la imagen antes de binarizar.

    FIX 3 (upscale, 2026-09-07): el corpus sintetico se genera a 150 DPI
    (ver generate_synthetic_corpus.py), por debajo de los ~300 DPI que la
    documentacion de Tesseract recomienda para texto de fuente normal/
    pequena. A 150 DPI, un digito de una fuente de 11-12pt mide apenas
    unos pocos pixeles de alto, el margen exacto donde un caracter
    empieza a fusionarse con el vecino o a confundirse con una letra de
    forma similar -- precisamente el patron de fallo (digitos fusionados,
    "OS/O7I2028" en vez de "05/07/2028", separadores perdidos) que motivo
    gran parte de la tolerancia a errores anadida al motor de reglas esta
    sesion. Interpolacion cubica (INTER_CUBIC) ANTES de binarizar, para
    aprovechar el gradiente de grises original en vez de reescalar bordes
    ya binarizados.

    Verificado de forma dirigida sobre documentos con valores previamente
    ilegibles (ver hallazgo en apuntes_hallazgos.md): factor=2.0 (~300
    DPI equivalente) recupera la fecha de nacimiento completa en varios
    casos donde antes solo se leia un bloque de digitos fusionados sin
    separadores, a un costo de ~1.2s/documento adicional en CPU.
    """
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def preprocess_document(image_path: str, denoise: bool = False,
                         upscale_factor: float = 2.0) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento.
    Retorna imagen binaria lista para OCR.

    El parametro `denoise` esta desactivado por defecto: la evaluacion
    integral sobre el corpus sintetico demostro que remove_noise() con
    kernel_size=3 aplicado tras la binarizacion destruye el texto en fuentes
    normales, reduciendo la tasa de deteccion de campos de forma
    generalizada (ver detalle en el docstring del modulo). Se deja
    como parametro opcional para poder reactivarlo de forma
    controlada y medir su impacto real en escaneos con ruido
    genuino, en vez de aplicarlo incondicionalmente.

    `upscale_factor` (ver upscale() arriba): activado por defecto en 2.0
    tras verificar mejoras concretas en digitos ilegibles a la resolucion
    original de 150 DPI del corpus. Pasar 1.0 (o None/0) para desactivarlo
    y reproducir el comportamiento previo a este cambio.
    """
    img = load_image(image_path)
    if upscale_factor and upscale_factor != 1.0:
        img = upscale(img, upscale_factor)
    img = deskew(img)
    img = binarize_sauvola(img)
    if denoise:
        img = remove_noise(img)
    return img
