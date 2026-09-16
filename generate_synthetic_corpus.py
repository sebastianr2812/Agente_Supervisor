#!/usr/bin/env python3
"""
generate_synthetic_corpus.py - Generador del corpus sintetico de evaluacion
para PathwayGuard.

Genera documentos clinicos sinteticos en espanol como imagenes PNG,
con defectos controlados y ground truth completo para evaluar el agente
supervisor de extremo a extremo.

Estructura (v5, diseno de 230 documentos):
  - 5 familias documentales, 230 documentos en total (entre 36 y 50 por
    familia segun cuantos estados adicionales de FAMILY_ONLY_STATES le
    aplican), repartidos en 115 de desarrollo (layout A/B, cosmetico) +
    115 de evaluacion final (layout C, cabecera ampliada y firma lateral).
    Ver SUBSTATE_PLAN (base comun a las 5 familias) y FAMILY_ONLY_STATES
    (estados adicionales que solo aplican a un subconjunto de familias).
  - Ground truth JSON con: veredicto esperado, campos, firmas, casillas,
    defectos, particion (dev/eval) y plantilla (A/B/C) de cada documento
  - Imagenes PNG a 150 DPI simulando documentos escaneados

Estados (granulares, se agregan a 3 clases de veredicto vía
_compute_expected_verdict / gt["expected_verdict"]):
  valid            - Todos los campos presentes y correctos            -> valid
  incomplete_1     - 1 campo no critico ausente                        -> incomplete
  incomplete_2     - 2+ campos ausentes (al menos 1 critico)           -> incomplete
  inconsistent_sem - Campos presentes pero semanticamente contradictorios -> inconsistent
  inconsistent_mul - Multiples problemas combinados                    -> inconsistent

Layout "C" (solo particion eval, ver DocumentGenerator.draw_header_extension
y draw_signature_block/_draw_lateral_signature): a diferencia de A/B, que
comparten las mismas fracciones de zona que HEADER_MAX_Y/BODY_MAX_Y/
LEGAL_MAX_Y de src/ocr/extractor.py, layout C amplia la cabecera y mueve la
firma a una columna lateral, precisamente para que la evaluacion final mida
generalizacion en vez de repetir el ajuste del desarrollo.

Version 3 (2026-07-21): rediseno de densidad/realismo tras feedback directo
del usuario sobre la v2 ("muy feo, no se ven reales, mucho espacio en
blanco, no veo firmas ni nada claro, la letra muy pequena, no parece un
documento de verdad, mucha desproporcion de espacios y muy poca info").
Cambios respecto a v2:

  1. Tipografia mas grande en toda la plantilla (titulo 18->22, texto de
     cuerpo/etiquetas 11->12, pie 9->10) y margen reducido (0.7"->0.55")
     para aprovechar mas superficie de pagina.
  2. Membrete ampliado: se anade linea de direccion/telefono/CIF del
     centro (elemento estandar de cualquier membrete hospitalario real) y
     una caja de "datos del documento" (numero, fecha de emision, pagina)
     en la esquina superior derecha.
  3. ELIMINADO el patron "y = max(y, ZONA_FIN)" que dejaba grandes huecos
     en blanco cuando el contenido natural terminaba antes del limite de
     zona. Los huecos que aun quedan (necesarios para que un campo caiga
     dentro de la zona Y correcta segun la heuristica de zonificacion del
     pipeline real) ahora se rellenan con contenido real:
       - En la zona de cabecera: una caja de "observaciones/alergias"
         con lineas de puntos (elemento habitual en formularios sanitarios
         impresos para anotacion manuscrita), no espacio vacio.
       - En la zona legal: una caja de aviso legal/confidencialidad con
         texto real (LOPD 3/2018, Ley 41/2002) dimensionada para ocupar
         el hueco en vez de una linea + espacio en blanco.
  4. Firma y sello del facultativo responsable anadidos a las 5 familias
     (antes solo el checklist preoperatorio tenia firma). El trazo de
     firma es mas grande, mas grueso y de un tono tinta oscuro (antes
     2px gris apenas visible; ahora 3px, mayor tamano). Se omite en los
     estados de mayor gravedad (incomplete_2, inconsistent_mul) para
     mantener una senal de ground truth de firma tambien en esas 4
     familias, no solo en PREOP.
  5. Una seccion de contenido adicional por familia (constantes vitales
     y plan de tratamiento en notas clinicas, observaciones facultativas
     en informes de laboratorio, alergias/instrucciones en listas de
     medicacion, servicio de destino/motivo de ingreso en admision, tipo
     de intervencion/observaciones de enfermeria en el checklist
     preoperatorio) para aumentar la densidad de informacion real de cada
     documento.

  Se mantiene todo lo de v2: membrete con logo y color de marca por
  hospital, tablas con bordes y zebra, sellos rotados, 2 variantes de
  plantilla (layout A/B), formatos de fecha variables y realismo de
  escaneo (ruido + sombra + rotacion leve) aplicado de forma consistente
  a las 5 familias.

  IMPORTANTE - hallazgo de compatibilidad con el pipeline OCR real: el
  modulo de binarizacion Sauvola de PathwayGuard (src/ocr/preprocess.py)
  asume texto oscuro sobre fondo claro. Se probo empiricamente que un
  encabezado de tabla con texto BLANCO sobre fondo de color solido se
  pierde por completo tras la binarizacion (0% de palabras clave
  detectadas), mientras que el mismo encabezado con fondo de color CLARO
  (tintado, mezclado con blanco) y texto oscuro se lee correctamente. Por
  eso todos los elementos de color de este generador usan fondos tintados
  claros con texto oscuro, nunca texto claro sobre fondo oscuro solido.
  El tono de "tinta" usado para las firmas (azul-negro muy oscuro) se
  mantiene igual de oscuro que el negro puro en escala de grises, para no
  reabrir ese mismo problema.

Uso:
  python generate_synthetic_corpus.py [--output-dir DIR] [--seed S]

Requisitos: Pillow, numpy (para el realismo de escaneo)
"""

import argparse
import hashlib
import json
import math
import os
import random
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuracion de pagina (A4 a 150 DPI)
# ---------------------------------------------------------------------------
DPI = 150
PAGE_W = int(8.27 * DPI)   # 1240 px
PAGE_H = int(11.69 * DPI)  # 1753 px
MARGIN = int(0.55 * DPI)   # 82 px (v2 usaba 0.7" / 105px; se reduce para
                           # aprovechar mas superficie util de la pagina)

# Zonas verticales (coherentes con la heuristica de PathwayGuard - NO TOCAR
# estos ratios, deben seguir alineados con HEADER_MAX_Y del motor real)
HEADER_END = int(PAGE_H * 0.35)     # 35% superior
BODY_END = int(PAGE_H * 0.68)       # 68%
LEGAL_END = int(PAGE_H * 0.88)      # 88%
# signature: 88-100%

# Colores
BLACK = (0, 0, 0)
DARK_GRAY = (60, 60, 60)
GRAY = (120, 120, 120)
LIGHT_GRAY = (200, 200, 200)
WHITE = (255, 255, 255)
LINE_COLOR = (180, 180, 180)
INK_COLOR = (18, 18, 55)  # tinta de firma: azul-negro oscuro, tan oscuro
                          # como el negro puro en escala de grises

# Paleta de colores de marca (una por hospital, deterministica)
ACCENT_COLORS = [
    (21, 82, 140),   # azul institucional
    (18, 110, 78),   # verde sanitario
    (140, 40, 50),   # granate
    (70, 60, 130),   # violeta
    (15, 100, 105),  # teal
    (150, 100, 20),  # ocre
]

# ---------------------------------------------------------------------------
# Pools de datos realistas en espanol
# ---------------------------------------------------------------------------
HOSPITALES = [
    "Hospital Universitario La Paz",
    "Hospital General de Valencia",
    "Hospital Clinico San Carlos",
    "Hospital Virgen del Rocio",
    "Hospital Universitario Central de Asturias",
    "Hospital Regional de Malaga",
    "Hospital Universitario de Salamanca",
    "Complejo Hospitalario de Navarra",
]

# Direccion / telefono / CIF ficticios pero con formato real, uno por
# hospital (mismo indice que HOSPITALES) para el membrete ampliado.
HOSPITAL_DIRECCIONES = [
    ("Paseo de la Castellana 261, 28046 Madrid", "91 727 70 00", "Q2812345F"),
    ("Av. del Tres de Marzo s/n, 46014 Valencia", "96 197 20 00", "Q4611223G"),
    ("Calle del Prof. Martin Lagos s/n, 28040 Madrid", "91 330 30 00", "Q2833445H"),
    ("Av. Manuel Siurot s/n, 41013 Sevilla", "95 501 20 00", "Q4155667J"),
    ("Av. de Roma s/n, 33011 Oviedo", "98 510 61 00", "Q3355889K"),
    ("Av. Carlos Haya s/n, 29010 Malaga", "95 129 00 00", "Q2966778L"),
    ("Paseo San Vicente 58, 37007 Salamanca", "92 329 11 00", "Q3777990M"),
    ("Calle Irunlarrea 3, 31008 Pamplona", "84 842 21 00", "Q3188001N"),
]

SERVICIOS = [
    "Servicio de Medicina Interna",
    "Servicio de Cardiologia",
    "Servicio de Traumatologia",
    "Servicio de Cirugia General",
    "Servicio de Neumologia",
    "Servicio de Neurologia",
    "Servicio de Pediatria",
    "Servicio de Urgencias",
]

NOMBRES = [
    "Maria Garcia Lopez", "Juan Martinez Rodriguez", "Ana Fernandez Ruiz",
    "Carlos Sanchez Perez", "Laura Gonzalez Diaz", "Pedro Hernandez Moreno",
    "Isabel Jimenez Alvarez", "Miguel Torres Romero", "Carmen Ruiz Navarro",
    "Francisco Lopez Serrano", "Elena Martin Castillo", "Jose Diaz Herrera",
    "Sofia Moreno Gutierrez", "Antonio Alvarez Molina", "Patricia Perez Ortega",
]

MEDICOS = [
    ("Dr. Roberto Vega Martinez", "Col. 28/12345"),
    ("Dra. Elena Soler Prieto", "Col. 46/67890"),
    ("Dr. Alejandro Ramos Gil", "Col. 33/11223"),
    ("Dra. Lucia Navarro Ruiz", "Col. 08/44556"),
    ("Dr. Fernando Iglesias Blanco", "Col. 50/77889"),
    ("Dra. Marta Delgado Cruz", "Col. 28/99012"),
]

MOTIVOS_CONSULTA = [
    "Dolor toracico de 2 horas de evolucion, irradiado a brazo izquierdo, "
    "acompanado de sudoracion y nauseas.",
    "Disnea progresiva de 3 dias de evolucion con tos productiva y fiebre "
    "de 38.5 grados.",
    "Cefalea intensa de inicio subito, asociada a rigidez nucal y "
    "fotofobia. Sin antecedentes previos similares.",
    "Dolor abdominal en fosa iliaca derecha de 12 horas de evolucion, con "
    "nauseas y febricula.",
    "Mareo y perdida de equilibrio de 48 horas de evolucion, con "
    "acufenos bilaterales.",
    "Lumbalgia mecanica de 5 dias sin mejoria con analgesicos habituales, "
    "con irradiacion a miembro inferior derecho.",
]

# Cada entrada es (texto_antecedentes, medicacion_habitual_coherente): 3 de
# las 5 entradas YA mencionan un tratamiento farmacologico activo (por
# realismo clinico, no por casualidad) -- si "Medicacion habitual" se
# declarara "Ninguna" de forma incondicional para todos los estados (como en
# una version anterior), esas 3 entradas quedarian internamente
# contradictorias en CUALQUIER estado que las usara al azar (incluido
# "valid"), no solo en inconsistent_medtexto. La segunda posicion de cada
# tupla es la unica declaracion de "Medicacion habitual" coherente con esos
# antecedentes; inconsistent_medtexto es el UNICO estado que fuerza
# "Ninguna" a proposito, pisando esta correspondencia.
ANTECEDENTES = [
    ("Antecedentes: Hipertension arterial en tratamiento con enalapril 10 mg. "
     "Diabetes mellitus tipo 2 diagnosticada en 2018, controlada con metformina. "
     "Sin alergias medicamentosas conocidas.", "Enalapril 10 mg, Metformina"),
    ("Antecedentes: Asma bronquial desde la infancia. Exfumador desde 2020. "
     "Apendicectomia en 2015. Sin alergias conocidas.", "Ninguna"),
    ("Antecedentes: Sin patologias previas relevantes. No toma medicacion "
     "habitual. Alergia a penicilina documentada.", "Ninguna"),
    ("Antecedentes: Fibrilacion auricular paroxistica, anticoagulado con "
     "apixaban. Dislipemia en tratamiento con atorvastatina 40 mg. "
     "Colecistectomia en 2019.", "Apixaban, Atorvastatina 40 mg"),
    ("Antecedentes: Hipotiroidismo en tratamiento con levotiroxina 75 mcg. "
     "Hernia discal L4-L5 intervenida en 2021. Sin alergias.", "Levotiroxina 75 mcg"),
]

PLANES_TRATAMIENTO = [
    "Se solicita analitica completa y radiografia de torax. Se pauta "
    "tratamiento sintomatico y se cita para revision en 48 horas.",
    "Se inicia tratamiento antibiotico empirico y se solicita interconsulta "
    "a Neumologia. Control clinico en 72 horas.",
    "Se deriva a Neurologia de forma preferente y se solicita TC craneal "
    "urgente para descartar patologia intracraneal.",
    "Se solicita ecografia abdominal y se mantiene en observacion con "
    "dieta absoluta hasta valoracion por Cirugia.",
    "Se pauta tratamiento rehabilitador y analgesia escalonada. Revision "
    "en consultas externas en 2 semanas.",
]

MEDICAMENTOS = [
    ("Metformina", "850 mg", "cada 12 horas", "oral"),
    ("Enalapril", "10 mg", "cada 24 horas", "oral"),
    ("Atorvastatina", "40 mg", "cada 24 horas (noche)", "oral"),
    ("Omeprazol", "20 mg", "cada 24 horas (antes del desayuno)", "oral"),
    ("Paracetamol", "1 g", "cada 8 horas si dolor", "oral"),
    ("Ibuprofeno", "600 mg", "cada 8 horas con alimentos", "oral"),
    ("Apixaban", "5 mg", "cada 12 horas", "oral"),
    ("Levotiroxina", "75 mcg", "cada 24 horas en ayunas", "oral"),
    ("Salbutamol", "100 mcg", "2 inhalaciones si disnea", "inhalada"),
    ("Amoxicilina", "500 mg", "cada 8 horas durante 7 dias", "oral"),
]

ALERGIAS_POOL = [
    "Penicilina", "Sulfamidas", "AINEs", "Latex", "Ninguna conocida",
    "Contraste yodado", "Acido acetilsalicilico",
]

# Pares alergia -> medicamento de MEDICAMENTOS que pertenece a la misma
# familia farmacologica (contraindicacion real, no inventada): Amoxicilina
# es una penicilina; Ibuprofeno es un AINE. Usado solo por
# inconsistent_alergia en MED-001-ES -- es la unica incoherencia del corpus
# sin respaldo determinista, pensada para medir si el LLM detecta una
# contraindicacion real usando conocimiento farmacologico general.
CONTRAINDICACIONES = {
    "Penicilina": "Amoxicilina",
    "AINEs": "Ibuprofeno",
    # Acido acetilsalicilico (aspirina) es quimicamente un AINE, por lo que
    # tambien contraindica Ibuprofeno -- incluido para que la exclusion de
    # colisiones accidentales (ver mas abajo) lo cubra igual que a "AINEs".
    "Acido acetilsalicilico": "Ibuprofeno",
}

RESULTADOS_LAB = [
    ("Glucosa", "105", "mg/dL", "70-110"),
    ("Hemoglobina", "13.2", "g/dL", "12.0-16.0"),
    ("Leucocitos", "8.500", "/uL", "4.000-11.000"),
    ("Plaquetas", "245.000", "/uL", "150.000-400.000"),
    ("Creatinina", "0.9", "mg/dL", "0.7-1.3"),
    ("Urea", "38", "mg/dL", "15-45"),
    ("Colesterol total", "215", "mg/dL", "<200"),
    ("Trigliceridos", "142", "mg/dL", "<150"),
    ("TSH", "2.8", "mUI/L", "0.4-4.0"),
    ("PCR", "12.5", "mg/L", "<5.0"),
    ("Sodio", "140", "mEq/L", "135-145"),
    ("Potasio", "4.2", "mEq/L", "3.5-5.0"),
]

ITEMS_PREOP = [
    "Identidad del paciente verificada",
    "Sitio quirurgico marcado",
    "Consentimiento informado firmado",
    "Alergias conocidas revisadas",
    "Ayuno de 8 horas confirmado",
    "Riesgo de via aerea dificil evaluado",
    "Profilaxis antibiotica administrada",
    "Hemograma preoperatorio revisado",
    "Grupo sanguineo y pruebas cruzadas",
    "Reserva de sangre confirmada",
]

TIPOS_INTERVENCION = [
    "Colecistectomia laparoscopica", "Protesis total de rodilla",
    "Apendicectomia", "Hernioplastia inguinal", "Artroscopia de hombro",
    "Cirugia de cataratas", "Reseccion transuretral",
]


# ---------------------------------------------------------------------------
# Utilidades de dibujo
# ---------------------------------------------------------------------------

def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Intenta cargar una fuente del sistema; si no, usa la por defecto."""
    font_candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def hospital_color(hospital: str) -> tuple:
    """Color de marca deterministico por nombre de hospital (mismo hospital
    -> mismo color en todos los documentos, como una identidad visual real)."""
    h = int(hashlib.md5(hospital.encode("utf-8")).hexdigest(), 16)
    return ACCENT_COLORS[h % len(ACCENT_COLORS)]


def hospital_address(hospital: str) -> tuple:
    """Direccion/telefono/CIF deterministicos por hospital (mismo indice que
    HOSPITALES; si no se encuentra, se deriva de forma deterministica)."""
    try:
        idx = HOSPITALES.index(hospital)
        return HOSPITAL_DIRECCIONES[idx]
    except ValueError:
        h = int(hashlib.md5(hospital.encode("utf-8")).hexdigest(), 16)
        return HOSPITAL_DIRECCIONES[h % len(HOSPITAL_DIRECCIONES)]


def tint(color: tuple, factor: float = 0.85) -> tuple:
    """Mezcla un color con blanco para obtener un tono claro de fondo.

    IMPORTANTE: nunca usar el color solido como fondo con texto claro encima;
    la binarizacion Sauvola del pipeline real pierde por completo el texto
    blanco sobre fondo oscuro (verificado empiricamente). Los fondos de color
    de este generador SIEMPRE deben ser tintes claros (factor >= 0.75) con
    texto oscuro/del color de marca encima, nunca al reves.
    """
    return tuple(int(c + (255 - c) * factor) for c in color)


def draw_line(draw: ImageDraw.ImageDraw, y: int, x1: int = MARGIN, x2: int = PAGE_W - MARGIN,
             color=LINE_COLOR, width: int = 1):
    """Dibuja una linea horizontal."""
    draw.line([(x1, y), (x2, y)], fill=color, width=width)


def draw_text_block(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
                    font: ImageFont.FreeTypeFont, max_width: int = None,
                    color=BLACK, line_spacing: int = 4) -> int:
    """Dibuja texto con word wrap. Retorna la Y final."""
    if max_width is None:
        max_width = PAGE_W - 2 * MARGIN
    char_width = font.size * 0.55  # Estimacion
    chars_per_line = max(10, int(max_width / char_width))
    lines = []
    for paragraph in text.split('\n'):
        wrapped = textwrap.wrap(paragraph, width=chars_per_line) or ['']
        lines.extend(wrapped)

    for line in lines:
        draw.text((x, y), line, fill=color, font=font)
        y += font.size + line_spacing
    return y


def draw_label_value(draw: ImageDraw.ImageDraw, label: str, value: str,
                     x: int, y: int, label_font, value_font,
                     label_w: int = 210, label_color=DARK_GRAY) -> int:
    """Dibuja un par etiqueta: valor. Retorna Y final."""
    draw.text((x, y), label, fill=label_color, font=label_font)
    draw.text((x + label_w, y), value, fill=BLACK, font=value_font)
    return y + value_font.size + 8


def draw_checkbox(draw: ImageDraw.ImageDraw, x: int, y: int, checked: bool,
                  size: int = 24, color=BLACK):
    """Dibuja una casilla de verificacion."""
    draw.rectangle([(x, y), (x + size, y + size)], outline=color, width=2)
    if checked:
        m = 5
        draw.line([(x + m, y + m), (x + size - m, y + size - m)], fill=color, width=3)
        draw.line([(x + m, y + size - m), (x + size - m, y + m)], fill=color, width=3)


def draw_signature(draw: ImageDraw.ImageDraw, x: int, y: int, width: int = 210,
                   height: int = 46, seed: int = 42, color=INK_COLOR):
    """Dibuja una firma simulada (trazo ondulado tipo cursiva manuscrita).

    v3: trazo mas grande y mas grueso que en v2 (antes 160x30 a 2px, apenas
    visible; ahora 210x46 a 3px con doble pasada para dar cuerpo de tinta)
    en respuesta directa al feedback de que las firmas no se veian claras.
    """
    rng = random.Random(seed)
    points = []
    n_points = rng.randint(16, 26)
    for i in range(n_points):
        px = x + int(width * i / (n_points - 1))
        py = y + rng.randint(-height // 2, height // 2)
        points.append((px, py))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=3)
        # Segunda pasada con leve offset para dar cuerpo a la tinta
        draw.line([(points[i][0], points[i][1] + 1), (points[i + 1][0], points[i + 1][1] + 1)],
                  fill=color, width=2)
    # Rubrica final (rasgo de cierre tipico de una firma manuscrita)
    fx, fy = points[-1]
    draw.line([(fx, fy), (fx + 14, fy - height // 2)], fill=color, width=3)
    draw.line([(fx + 14, fy - height // 2), (fx + 14, fy + height // 3)], fill=color, width=3)


def draw_logo(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, color: tuple):
    """Dibuja un logo simple (cruz medica dentro de un circulo)."""
    draw.ellipse([x, y, x + size, y + size], outline=color, width=3)
    cx, cy = x + size // 2, y + size // 2
    arm = size // 3
    thick = max(3, size // 7)
    draw.rectangle([cx - thick // 2, cy - arm, cx + thick // 2, cy + arm], fill=color)
    draw.rectangle([cx - arm, cy - thick // 2, cx + arm, cy + thick // 2], fill=color)


def draw_header_band(draw: ImageDraw.ImageDraw, color: tuple, height: int = 10):
    """Banda de color superior (identidad de marca)."""
    draw.rectangle([0, 0, PAGE_W, height], fill=color)


def draw_doc_info_box(draw: ImageDraw.ImageDraw, y_top: int, doc_id: str,
                      fecha_emision: str, tipo_doc: str):
    """
    Caja de "datos del documento" en la esquina superior derecha (numero de
    documento, fecha de emision, pagina, tipo) -- elemento administrativo
    presente en la practica totalidad de la documentacion clinica impresa
    real, y que ademas ayuda a llenar de forma legitima la zona de cabecera
    en vez de dejarla en blanco.
    """
    box_w, box_h = 250, 66
    bx = PAGE_W - MARGIN - box_w
    by = y_top
    font_l = get_font(9)
    draw.rectangle([bx, by, bx + box_w, by + box_h], outline=LINE_COLOR, width=1)
    draw.text((bx + 9, by + 6), f"Documento nº: {doc_id}", fill=DARK_GRAY, font=font_l)
    draw.text((bx + 9, by + 22), f"Fecha emision: {fecha_emision}", fill=DARK_GRAY, font=font_l)
    draw.text((bx + 9, by + 38), "Pagina 1 de 1", fill=DARK_GRAY, font=font_l)
    draw.text((bx + 9, by + 54), tipo_doc[:34], fill=DARK_GRAY, font=font_l)


def fill_notes_gap(draw: ImageDraw.ImageDraw, y: int, target_y: int, x: int, width: int,
                   color: tuple, font_label, title: str = "OBSERVACIONES / ALERGIAS") -> int:
    """
    Si queda un hueco antes del limite de zona siguiente, en vez de dejarlo
    en blanco se dibuja una caja con lineas de puntos para anotacion
    manuscrita -- un elemento estandar de los formularios sanitarios
    impresos reales. Mantiene la alineacion de zonas que necesita la
    heuristica de zonificacion del pipeline (el contenido siguiente debe
    empezar en target_y), pero sustituye el espacio vacio por un elemento
    de formulario reconocible.
    """
    height = target_y - y - 10
    if height < 30:
        return target_y
    draw.rectangle([x, y, x + width, y + height], outline=LINE_COLOR, width=1)
    draw.text((x + 10, y + 7), title, fill=color, font=font_label)
    line_y = y + font_label.size + 22
    while line_y < y + height - 8:
        draw.line([(x + 16, line_y), (x + width - 16, line_y)], fill=(216, 216, 216), width=1)
        line_y += 24
    return target_y


LEGAL_DISCLAIMER = (
    # IMPORTANTE: este texto se dibuja SIN CONDICION en todos los documentos
    # (via draw_legal_box), asi que NO debe contener ninguna de las palabras
    # clave (patrones_esperados) que los protocolos YAML usan para detectar
    # campos obligatorios -- en particular "paciente", "medico"/"dr."/"dra.",
    # "hospital"/"centro", "antecedentes", "resultado", "firma", etc. Una
    # version anterior de este aviso incluia la palabra "paciente" dos veces,
    # lo que anulaba silenciosamente la ausencia deliberada del campo
    # identificacion_paciente en los estados incomplete/inconsistent de las
    # 5 familias (el motor de reglas encontraba "paciente" en este texto y
    # daba el campo por presente). Verificado contra los 5 YAML de
    # configs/protocolos_es/: este texto no contiene ninguno de sus
    # patrones_esperados.
    "Este documento contiene informacion sanitaria de caracter confidencial, "
    "protegida por la Ley Organica 3/2018 de Proteccion de Datos Personales "
    "y garantia de los derechos digitales, y por la normativa sanitaria "
    "vigente. Queda prohibida su reproduccion, cesion o divulgacion a "
    "terceros sin autorizacion expresa. Documento generado con fines de "
    "evaluacion academica: todos los datos son ficticios."
)


def _wrapped_line_count(text: str, font, max_width: int) -> int:
    """Cuenta cuantas lineas ocupara `text` al envolverlo a `max_width` con
    `font`, sin dibujar nada (para poder dimensionar cajas de antemano)."""
    char_width = font.size * 0.55
    chars_per_line = max(10, int(max_width / char_width))
    n = 0
    for paragraph in text.split('\n'):
        n += len(textwrap.wrap(paragraph, width=chars_per_line) or [''])
    return n


def draw_legal_box(draw: ImageDraw.ImageDraw, y: int, target_y: int, x: int, width: int,
                   font_label, font_small, color: tuple, extra_line: str = None) -> int:
    """
    Caja de aviso legal / confidencialidad con texto real (no relleno).
    Sustituye al antiguo patron "y = max(y, LEGAL_END); linea; texto de pie"
    que dejaba un bloque de canvas vacio cuando el cuerpo del documento
    terminaba antes de la zona legal.

    La caja tiene una altura de texto FIJA (calculada a partir del propio
    aviso legal, no estirada artificialmente). Si aun asi queda hueco hasta
    `target_y` (la zona legal del documento puede empezar muy por debajo del
    contenido real, segun la heuristica de zonificacion del pipeline), el
    resto de la caja se rellena con lineas de puntos para "notas adicionales"
    -- igual que fill_notes_gap -- en lugar de dejar un bloque de canvas
    vacio dentro del propio recuadro (el problema detectado en la v3 inicial:
    una caja enorme con 3 lineas de texto arriba y cientos de pixeles en
    blanco debajo, que seguia leyendose como "espacio desperdiciado").
    """
    n_lines = _wrapped_line_count(LEGAL_DISCLAIMER, font_small, width - 20)
    text_h = n_lines * (font_small.size + 3)
    extra_h = (font_small.size + 6) if extra_line else 0
    content_h = 8 + font_label.size + 8 + text_h + extra_h + 10
    height = max(content_h, target_y - y if target_y - y > 0 else 0, 100)

    draw.rectangle([x, y, x + width, y + height], outline=LINE_COLOR, width=1)
    draw.text((x + 10, y + 8), "AVISO LEGAL Y CONFIDENCIALIDAD", fill=color, font=font_label)
    ty = draw_text_block(draw, LEGAL_DISCLAIMER, x + 10, y + 8 + font_label.size + 8,
                         font_small, max_width=width - 20, color=GRAY, line_spacing=3)
    if extra_line:
        draw.text((x + 10, ty + 2), extra_line, fill=GRAY, font=font_small)
        ty += font_small.size + 6

    # Si sobra espacio hasta el final de la caja, se rellena con lineas de
    # puntos para notas manuscritas en vez de dejarlo en blanco.
    remaining = (y + height) - (ty + 10)
    if remaining > 30:
        draw.line([(x + 10, ty + 8), (x + width - 10, ty + 8)], fill=(210, 210, 210), width=1)
        draw.text((x + 10, ty + 14), "Notas adicionales:", fill=GRAY, font=font_small)
        line_y = ty + 14 + font_small.size + 10
        while line_y < y + height - 8:
            draw.line([(x + 16, line_y), (x + width - 16, line_y)], fill=(220, 220, 220), width=1)
            line_y += 22

    return y + height + 14


def draw_page_footer(draw: ImageDraw.ImageDraw, hospital: str, doc_id: str,
                     show_center: bool = True):
    """
    Pie de pagina fijo en el margen inferior (elemento estandar de cualquier
    impresion hospitalaria real: nombre del centro, identificador del
    documento y aviso de generacion electronica). Se dibuja siempre en la
    misma posicion absoluta, independientemente de cuanto contenido tenga el
    documento, para que la parte inferior de la pagina nunca quede vacia.

    `show_center=False` omite el nombre del hospital: se usa cuando el
    estado del documento omite deliberadamente identificacion_centro (solo
    ocurre en MED-001-ES incomplete_1/incomplete_2), para no reintroducir
    por la puerta de atras la palabra "hospital"/"centro" que el motor de
    reglas busca -- ver nota en LEGAL_DISCLAIMER sobre el mismo problema.
    """
    font_f = get_font(9)
    y = PAGE_H - MARGIN + 8
    draw_line(draw, y - 10, color=(225, 225, 225), width=1)
    if show_center:
        text = f"{hospital}  |  Documento generado electronicamente  |  {doc_id}"
    else:
        text = f"Documento generado electronicamente  |  {doc_id}"
    draw.text((MARGIN, y), text, fill=(160, 160, 160), font=font_f)


def draw_table(draw: ImageDraw.ImageDraw, x: int, y: int, col_widths: list,
               headers: list, rows: list, font_header, font_body,
               accent_color: tuple, zebra: bool = True) -> int:
    """
    Dibuja una tabla con bordes reales y encabezado con tinte de color.

    Cada celda de `rows` puede ser un string, o una tupla (texto, color) para
    resaltar valores (p. ej. resultados fuera de rango en rojo).

    NOTA: el encabezado usa fondo CLARO (tinte del color de marca) con texto
    del color de marca encima, nunca texto claro sobre fondo solido oscuro
    -- ver docstring de `tint()`.
    """
    total_w = sum(col_widths)
    row_h = font_body.size + 16
    header_h = font_header.size + 16
    header_bg = tint(accent_color, 0.82)
    zebra_bg = tint(accent_color, 0.94)

    draw.rectangle([x, y, x + total_w, y + header_h], fill=header_bg)
    cx = x
    for w, h in zip(col_widths, headers):
        draw.text((cx + 9, y + 8), h, fill=accent_color, font=font_header)
        cx += w

    yy = y + header_h
    for i, row in enumerate(rows):
        if zebra and i % 2 == 1:
            draw.rectangle([x, yy, x + total_w, yy + row_h], fill=zebra_bg)
        cx = x
        for w, val in zip(col_widths, row):
            color = val[1] if isinstance(val, tuple) else BLACK
            text = val[0] if isinstance(val, tuple) else str(val)
            draw.text((cx + 9, yy + 8), text, fill=color, font=font_body)
            cx += w
        yy += row_h

    draw.rectangle([x, y, x + total_w, yy], outline=accent_color, width=1)
    cx = x
    for w in col_widths[:-1]:
        cx += w
        draw.line([(cx, y), (cx, yy)], fill=(210, 210, 210), width=1)
    yy2 = y + header_h
    for i in range(len(rows)):
        draw.line([(x, yy2), (x + total_w, yy2)], fill=(222, 222, 222), width=1)
        yy2 += row_h

    return yy + 14


def draw_stamp(img: Image.Image, cx: int, cy: int, lines: list, color: tuple,
              rng: random.Random, radius_x: int = 95, radius_y: int = 52,
              font_size: int = 11):
    """
    Dibuja un sello oficial ovalado y rotado ("VALIDADO", etc.) sobre una
    capa RGBA independiente, compuesta luego sobre el documento.

    Se coloca deliberadamente en la zona de firma/legal (llamar solo con
    coordenadas en esa zona) para no interferir con el OCR de campos
    obligatorios en header/body.
    """
    layer_size = (radius_x * 2 + 30, radius_y * 2 + 30)
    layer = Image.new("RGBA", layer_size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    lx, ly = layer_size[0] // 2, layer_size[1] // 2
    ld.ellipse([lx - radius_x, ly - radius_y, lx + radius_x, ly + radius_y],
               outline=color + (200,), width=3)
    ld.ellipse([lx - radius_x + 6, ly - radius_y + 6, lx + radius_x - 6, ly + radius_y - 6],
               outline=color + (120,), width=1)
    font_stamp = get_font(font_size, bold=True)
    total_h = len(lines) * (font_size + 5)
    start_y = ly - total_h // 2
    for i, line in enumerate(lines):
        bbox = ld.textbbox((0, 0), line, font=font_stamp)
        tw = bbox[2] - bbox[0]
        ld.text((lx - tw // 2, start_y + i * (font_size + 5)), line,
                fill=color + (200,), font=font_stamp)

    angle = rng.uniform(-14, 14)
    rotated = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    img.paste(rotated, (cx - rotated.width // 2, cy - rotated.height // 2), rotated)


def apply_scan_realism(img: Image.Image, rng: random.Random,
                       intensity: str = "light") -> Image.Image:
    """
    Simula artefactos de escaneo real de forma OCR-segura:
      1. Sombra de iluminacion desigual de baja frecuencia (simula tapa de
         escaner o luz ambiental no uniforme).
      2. Ruido fino tipo grano de escaner (leve; niveles mas altos degradan
         la deteccion de campos, ver notas de calibracion en el modulo).
      3. Rotacion leve (el deskew() del pipeline real la corrige).

    Los niveles de ruido se calibraron empiricamente con OCR real
    (Tesseract + idioma "spa") para no destruir la legibilidad de fuentes
    de 11-12pt a 150 DPI, que es el tamano critico para los campos
    obligatorios.
    """
    import numpy as np

    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[0], arr.shape[1]

    grad_res = 8
    shade_range = {"light": 6, "medium": 10, "heavy": 16}.get(intensity, 6)
    low_res = np.random.RandomState(rng.randint(0, 99999)).uniform(
        -shade_range, shade_range, (grad_res, grad_res)
    )
    shade = Image.fromarray(low_res.astype(np.float32), mode='F').resize((w, h), Image.BICUBIC)
    shade_arr = np.array(shade)[:, :, None]
    arr = arr + shade_arr

    noise_level = {"light": 2.5, "medium": 4.0, "heavy": 7.0}.get(intensity, 2.5)
    noise = np.random.RandomState(rng.randint(0, 99999)).normal(0, noise_level, arr.shape)
    arr = arr + noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img_out = Image.fromarray(arr)

    angle_max = {"light": 0.5, "medium": 0.7, "heavy": 1.0}.get(intensity, 0.5)
    angle = rng.uniform(-angle_max, angle_max)
    img_out = img_out.rotate(angle, fillcolor=(255, 255, 255), expand=False,
                             resample=Image.BICUBIC)

    return img_out


# Alias retrocompatible (nombre usado en la v1 del script)
def add_scan_artifacts(img: Image.Image, rng: random.Random,
                       intensity: str = "light") -> Image.Image:
    return apply_scan_realism(img, rng, intensity)


# ---------------------------------------------------------------------------
# Generadores de documentos por familia
# ---------------------------------------------------------------------------

DATE_FORMATS = ["numeric_slash", "numeric_dash", "long_es"]
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def generate_date(rng: random.Random, year: int = 2026, fmt: str = None) -> str:
    """Genera una fecha aleatoria en uno de varios formatos en espanol."""
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    fmt = fmt or rng.choice(DATE_FORMATS)
    if fmt == "numeric_dash":
        return f"{day:02d}-{month:02d}-{year}"
    if fmt == "long_es":
        return f"{day} de {MESES_ES[month - 1]} de {year}"
    return f"{day:02d}/{month:02d}/{year}"


def generate_nhc(rng: random.Random) -> str:
    """Genera un NHC ficticio."""
    return f"NHC-{rng.randint(100000, 999999)}"


def generate_dni(rng: random.Random) -> str:
    """Genera un DNI ficticio."""
    num = rng.randint(10000000, 99999999)
    letras = "TRWAGMYFPDXBNJZSQVHLCKE"
    return f"{num}{letras[num % 23]}"


def generate_dob(rng: random.Random) -> str:
    """Genera fecha de nacimiento."""
    year = rng.randint(1940, 2005)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{day:02d}/{month:02d}/{year}"


class DocumentGenerator:
    """Clase base para generacion de documentos."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.font_title = get_font(22, bold=True)
        self.font_subtitle = get_font(15, bold=True)
        self.font_addr = get_font(10)
        self.font_label = get_font(12, bold=True)
        self.font_normal = get_font(12)
        self.font_small = get_font(10)

    def create_canvas(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new('RGB', (PAGE_W, PAGE_H), WHITE)
        draw = ImageDraw.Draw(img)
        return img, draw

    def draw_masthead(self, img, draw, hospital: str, subtitle: str,
                      layout: str, doc_id: str = "", fecha_emision: str = "",
                      tipo_doc: str = "") -> tuple:
        """
        Dibuja el membrete (banda de color + logo + nombre + direccion) y
        una caja de datos del documento en la esquina superior derecha.
        Retorna (color_de_marca, y_final). `layout` (\"A\"/\"B\") varia la
        posicion del logo y el grosor de la banda para dar variedad visual
        entre documentos.
        """
        color = hospital_color(hospital)
        direccion, telefono, cif = hospital_address(hospital)
        y = MARGIN

        if layout == "A":
            draw_header_band(draw, color, height=14)
            draw_logo(draw, MARGIN, y + 14, 52, color)
            draw.text((MARGIN + 66, y + 12), hospital, fill=BLACK, font=self.font_title)
            draw.text((MARGIN + 66, y + 40), subtitle, fill=GRAY, font=self.font_subtitle)
            draw.text((MARGIN + 66, y + 63), f"{direccion}  |  Tel. {telefono}  |  CIF {cif}",
                      fill=GRAY, font=self.font_addr)
            y += 96
        else:
            draw_header_band(draw, color, height=18)
            draw.text((MARGIN, y + 20), hospital, fill=BLACK, font=self.font_title)
            draw_logo(draw, PAGE_W - MARGIN - 52, y + 16, 52, color)
            draw.text((MARGIN, y + 48), subtitle, fill=GRAY, font=self.font_subtitle)
            draw.text((MARGIN, y + 71), f"{direccion}  |  Tel. {telefono}  |  CIF {cif}",
                      fill=GRAY, font=self.font_addr)
            y += 96

        if doc_id:
            draw_doc_info_box(draw, MARGIN, doc_id, fecha_emision, tipo_doc)

        draw_line(draw, y, color=tint(color, 0.55), width=2)
        y += 16
        return color, y

    def draw_header_extension(self, draw, y: int, color: tuple) -> int:
        """
        Extension de cabecera (layout "C", solo particion de evaluacion).

        Anade una caja adicional de "acreditacion / control interno del
        centro" tras el membrete normal, empujando hacia abajo el resto del
        contenido (identificacion de paciente, motivo, etc. segun cada
        familia). El objetivo es que campos con zona_documento="header" en
        los protocolos queden fisicamente por debajo de HEADER_MAX_Y
        (0.38), fuera del rango que la heuristica geometrica de
        PathwayGuard considera cabecera, para poder medir el error de
        zonificacion en un diseno no ajustado a esos umbrales.
        """
        box_top = y
        box_h = 168
        draw.rectangle([MARGIN, box_top, PAGE_W - MARGIN, box_top + box_h],
                       outline=tint(color, 0.6), width=1)
        draw.text((MARGIN + 14, box_top + 12), "ACREDITACION Y CONTROL INTERNO",
                  fill=color, font=self.font_label)
        lines = [
            "Codigo de centro sanitario: verificado segun registro autonomico vigente.",
            "Unidad de calidad documental: revision periodica de historias clinicas activa.",
            "Circuito de custodia: documento generado y archivado bajo el protocolo interno",
            "de gestion documental del centro, con trazabilidad de acceso registrada.",
        ]
        ly = box_top + 42
        for line in lines:
            draw.text((MARGIN + 14, ly), line, fill=DARK_GRAY, font=self.font_small)
            ly += 20
        return box_top + box_h + 16

    def draw_signature_block(self, img, draw, x: int, y: int, width: int,
                             nombre: str, col: str, color: tuple,
                             label: str = "Firma y sello del facultativo responsable:",
                             include: bool = True, layout: str = "A") -> int:
        """
        Bloque de firma del facultativo (v3): mas grande y mas oscuro que en
        v2, presente en las 5 familias documentales (antes solo en el
        checklist preoperatorio). Se omite deliberadamente en los estados
        de mayor gravedad para mantener una senal de ground truth de firma
        tambien en estas 4 familias.

        Si layout == "C" (particion de evaluacion), la firma se dibuja en
        una columna lateral a media altura de la pagina en vez de en la
        banda inferior habitual, para poner a prueba si la deteccion de
        firmas y la zonificacion generalizan a una posicion no estandar
        (caso "firma lateral" senalado como limitacion de la heuristica).
        El valor de "y" devuelto no cambia respecto al de entrada, porque
        el bloque lateral se dibuja fuera del flujo vertical normal del
        documento y no debe desplazar el contenido posterior.
        """
        if layout == "C":
            self._draw_lateral_signature(img, draw, nombre, col, color, include)
            return y

        draw.text((x, y), label, fill=color, font=self.font_label)
        y += 26
        if include:
            draw_signature(draw, x + 12, y + 28, width=220, height=48,
                           seed=self.rng.randint(0, 999999))
            draw_stamp(img, x + width - 110, y + 32, ["VALIDADO", "CONFORME"], color,
                      self.rng, radius_x=82, radius_y=44, font_size=11)
            y += 62
            draw_line(draw, y, x, x + 240)
            y += 6
            draw.text((x, y), f"{nombre}  -  {col}", fill=DARK_GRAY, font=self.font_small)
            y += 18
        else:
            draw.text((x, y + 6), "[ Firma pendiente de validacion ]", fill=(190, 30, 30),
                      font=self.font_normal)
            y += 30
        return y

    def _draw_lateral_signature(self, img, draw, nombre: str, col: str,
                                color: tuple, include: bool = True) -> None:
        """
        Bloque de firma en columna lateral derecha, a media altura de la
        pagina (layout "C"). Se dibuja como overlay independiente del flujo
        vertical normal del documento: no reserva espacio ni desplaza el
        resto del contenido, igual que ocurriria en un formulario real con
        una columna lateral de visado/firma junto al cuerpo principal.
        """
        lat_w = 250
        x = PAGE_W - MARGIN - lat_w
        y = int(PAGE_H * 0.42)

        draw.rectangle([x - 10, y - 10, x + lat_w + 10, y + 130],
                       outline=tint(color, 0.6), width=1)
        draw.text((x, y), "Visado:", fill=color, font=self.font_label)
        y += 22
        if include:
            draw_signature(draw, x + 6, y + 24, width=lat_w - 20, height=40,
                           seed=self.rng.randint(0, 999999))
            y += 52
            draw_line(draw, y, x, x + lat_w - 10)
            y += 6
            draw.text((x, y), nombre[:22], fill=DARK_GRAY, font=self.font_small)
            y += 16
            draw.text((x, y), col, fill=DARK_GRAY, font=self.font_small)
        else:
            draw.text((x, y + 4), "[ Pendiente ]", fill=(190, 30, 30),
                      font=self.font_small)


class ClinicalNoteGenerator(DocumentGenerator):
    """Genera notas clinicas (CN-001-ES)."""

    def generate(self, state: str, variant: int, layout: str | None = None) -> tuple[Image.Image, dict]:
        img, draw = self.create_canvas()
        gt = self._init_ground_truth(state, variant)
        doc_id = f"CN-001-ES_{state}_v{variant:02d}"
        layout = layout or self.rng.choice(["A", "B"])

        hospital = self.rng.choice(HOSPITALES)
        servicio = self.rng.choice(SERVICIOS)
        paciente = self.rng.choice(NOMBRES)
        nhc = generate_nhc(self.rng)
        dob = generate_dob(self.rng)
        fecha = generate_date(self.rng)
        medico, col = self.rng.choice(MEDICOS)
        motivo = self.rng.choice(MOTIVOS_CONSULTA)
        antecedentes, medicacion_habitual = self.rng.choice(ANTECEDENTES)
        plan = self.rng.choice(PLANES_TRATAMIENTO)
        vitales = (f"TA {self.rng.randint(110,140)}/{self.rng.randint(60,90)} mmHg  |  "
                  f"FC {self.rng.randint(60,100)} lpm  |  "
                  f"Temp {self.rng.uniform(36.0,37.5):.1f} C  |  "
                  f"Sat O2 {self.rng.randint(95,100)}%")

        omit_fields = set()
        semantic_issues = []

        if state == "incomplete_1":
            omit_fields.add("equipo_asistencial")
        elif state == "incomplete_2":
            omit_fields.add("motivo_consulta")
            omit_fields.add("equipo_asistencial")
        elif state == "inconsistent_sem":
            motivo = "Dolor toracico agudo de 3 horas de evolucion con sudoracion."
            antecedentes = ("Antecedentes: Paciente sano sin patologias previas. "
                           "Consulta por revision rutinaria de dermatologia. "
                           "Ultima revision oftalmologica hace 2 meses.")
            medicacion_habitual = "Ninguna"
            semantic_issues.append("motivo_consulta vs historia_medica: incoherencia semantica")
        elif state == "inconsistent_mul":
            omit_fields.add("identificacion_paciente")
            motivo = "Fractura de femur tras caida accidental."
            antecedentes = ("Antecedentes: Diagnosticado de migranas cronicas. "
                           "Seguimiento por otorrinolaringologia.")
            medicacion_habitual = "Ninguna"
            semantic_issues.append("motivo_consulta vs historia_medica: incoherencia semantica")
            semantic_issues.append("identificacion_paciente ausente")
        elif state == "inconsistent_temporal":
            # Fecha de nacimiento generada DESPUES de la fecha del documento
            # (imposible: el paciente no habria nacido aun).
            dob = f"{self.rng.randint(1,28):02d}/{self.rng.randint(1,12):02d}/{2026 + self.rng.randint(1,3)}"
            semantic_issues.append("fecha_nacimiento posterior a la fecha del documento")
        elif state == "inconsistent_identidad":
            semantic_issues.append("identificador de paciente distinto en la caja de auditoria")
        elif state == "inconsistent_medtexto":
            # Se declara "Medicacion habitual: Ninguna" pero los antecedentes
            # mencionan un tratamiento farmacologico activo -- coherencia
            # entre un campo declarado y texto libre, no entre dos campos
            # de texto libre como en inconsistent_sem.
            antecedentes = ("Antecedentes: Hipertension arterial en tratamiento con "
                            "Enalapril desde hace 3 anos. Sin alergias conocidas.")
            medicacion_habitual = "Ninguna"
            semantic_issues.append("medicacion_habitual 'Ninguna' contradicha por tratamiento activo en antecedentes")

        # IMPORTANTE: el bloque de firma decorativo reimprime "Dr./Dra. X -
        # Col. Y", que satisface literalmente los patrones de
        # equipo_asistencial (["medico","dr.","dra.","colegiado","col."]).
        # Se ata a la MISMA omision que el campo, no al nombre del estado --
        # una version anterior excluia la firma solo en incomplete_2 /
        # inconsistent_mul, dejandola activa en incomplete_1 pese a que ese
        # estado tambien omite equipo_asistencial, lo que anulaba la prueba
        # de campo ausente (el motor de reglas encontraba "medico"/"dr." en
        # la firma y daba el campo por presente).
        include_signature = "equipo_asistencial" not in omit_fields

        color, y = self.draw_masthead(img, draw, hospital, servicio, layout,
                                      doc_id=doc_id, fecha_emision=fecha, tipo_doc="Nota clinica")
        if layout == "C":
            y = self.draw_header_extension(draw, y, color)

        draw.text((MARGIN, y), "NOTA CLINICA", fill=BLACK, font=self.font_subtitle)
        y += 26
        y = draw_label_value(draw, "Fecha de expedicion:", fecha, MARGIN, y, self.font_label, self.font_normal, label_w=260)

        if "identificacion_paciente" not in omit_fields:
            y = draw_label_value(draw, "Paciente:", paciente, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "NHC:", nhc, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "Fecha de nacimiento:", dob, MARGIN, y, self.font_label, self.font_normal, label_w=260)
            gt["fields"]["identificacion_paciente"] = True
        else:
            gt["fields"]["identificacion_paciente"] = False

        gt["fields"]["identificacion_centro"] = True

        draw_line(draw, y + 5)
        y += 16

        y = fill_notes_gap(draw, y, HEADER_END + 10, MARGIN, PAGE_W - 2 * MARGIN, color,
                           self.font_label, title="ALERGIAS / OBSERVACIONES DE TRIAJE")

        if "motivo_consulta" not in omit_fields:
            draw.text((MARGIN, y), "MOTIVO DE CONSULTA", fill=color, font=self.font_label)
            y += 20
            y = draw_text_block(draw, motivo, MARGIN, y, self.font_normal)
            y += 12
            gt["fields"]["motivo_consulta"] = True
        else:
            gt["fields"]["motivo_consulta"] = False

        draw.text((MARGIN, y), "ANTECEDENTES / HISTORIA CLINICA", fill=color, font=self.font_label)
        y += 20
        y = draw_text_block(draw, antecedentes, MARGIN, y, self.font_normal)
        y += 12
        gt["fields"]["historia_medica"] = True

        y = draw_label_value(draw, "Medicacion habitual:", medicacion_habitual, MARGIN, y, self.font_label, self.font_normal, label_w=220)
        gt["fields"]["medicacion_habitual"] = True

        draw.text((MARGIN, y), "CONSTANTES VITALES", fill=color, font=self.font_label)
        y += 20
        draw.text((MARGIN, y), vitales, fill=BLACK, font=self.font_normal)
        y += 24

        draw.text((MARGIN, y), "PLAN Y TRATAMIENTO", fill=color, font=self.font_label)
        y += 20
        y = draw_text_block(draw, plan, MARGIN, y, self.font_normal)
        y += 10

        if "equipo_asistencial" not in omit_fields:
            draw.text((MARGIN, y), "MEDICO RESPONSABLE", fill=color, font=self.font_label)
            y += 20
            draw.text((MARGIN, y), f"{medico}  -  {col}", fill=BLACK, font=self.font_normal)
            y += 22
            gt["fields"]["equipo_asistencial"] = True
        else:
            gt["fields"]["equipo_asistencial"] = False

        # NOTA (v3): a diferencia del checklist preoperatorio, ninguna regla
        # de CN-001-ES exige que la firma caiga en la zona "signature" (no
        # hay campo obligatorio con zona_documento: signature en este
        # protocolo), asi que la caja legal NO se estira artificialmente
        # hasta LEGAL_END -- se dimensiona solo con su contenido real, para
        # evitar el bloque de lineas de puntos desproporcionadamente largo
        # que resultaba de forzar el limite de zona sin necesidad.
        legal_extra = f"Fecha del documento: {fecha}"
        if state == "inconsistent_identidad":
            nhc_auditoria = generate_nhc(self.rng)
            while nhc_auditoria == nhc:
                nhc_auditoria = generate_nhc(self.rng)
            legal_extra += f"  |  NHC verificado en admision: {nhc_auditoria}"

        y += 8
        y = draw_legal_box(draw, y, y, MARGIN, PAGE_W - 2 * MARGIN,
                           self.font_label, self.font_small, color,
                           extra_line=legal_extra)

        y = self.draw_signature_block(img, draw, MARGIN, y, PAGE_W - 2 * MARGIN,
                                      medico, col, color, include=include_signature,
                                      layout=layout)

        gt["signatures"] = {"expected": 1, "present": 1 if include_signature else 0}
        gt["checkboxes"] = {"expected": 0, "present": 0, "checked": 0}
        gt["semantic_issues"] = semantic_issues
        gt["expected_verdict"] = self._compute_expected_verdict(state)

        draw_page_footer(draw, hospital, doc_id,
                         show_center="identificacion_centro" not in omit_fields)
        return img, gt

    def _init_ground_truth(self, state: str, variant: int) -> dict:
        return {
            "family": "CN-001-ES",
            "family_name": "notas_clinicas",
            "state": state,
            "variant": variant,
            "fields": {},
            "signatures": {},
            "checkboxes": {},
            "semantic_issues": [],
            "expected_verdict": "",
        }

    def _compute_expected_verdict(self, state: str) -> str:
        if state == "valid":
            return "valid"
        elif state in ("incomplete_1", "incomplete_2"):
            return "incomplete"
        else:
            return "inconsistent"


class DiagnosticReportGenerator(DocumentGenerator):
    """Genera informes diagnosticos y de laboratorio (DLR-001-ES)."""

    def generate(self, state: str, variant: int, layout: str | None = None) -> tuple[Image.Image, dict]:
        img, draw = self.create_canvas()
        gt = {
            "family": "DLR-001-ES", "family_name": "informes_diagnosticos",
            "state": state, "variant": variant, "fields": {},
            "signatures": {}, "checkboxes": {}, "semantic_issues": [],
            "expected_verdict": "",
        }
        doc_id = f"DLR-001-ES_{state}_v{variant:02d}"
        layout = layout or self.rng.choice(["A", "B"])

        hospital = self.rng.choice(HOSPITALES)
        paciente = self.rng.choice(NOMBRES)
        nhc = generate_nhc(self.rng)
        fecha = generate_date(self.rng)
        medico, col = self.rng.choice(MEDICOS)
        resultados = self.rng.sample(RESULTADOS_LAB, k=min(6, len(RESULTADOS_LAB)))

        omit_fields = set()
        semantic_issues = []
        fuera_de_rango = False

        if state == "incomplete_1":
            omit_fields.add("medico_solicitante")
        elif state == "incomplete_2":
            omit_fields.add("identificacion_paciente")
            omit_fields.add("medico_solicitante")
        elif state == "inconsistent_sem":
            resultados = [("Prueba A", "positivo", "", ""),
                         ("Prueba B", "negativo", "", ""),
                         ("Prueba C", "indeterminado", "", "")]
            semantic_issues.append("resultados sin formato estandar de laboratorio")
        elif state == "inconsistent_mul":
            # v5: antes omitia fecha_servicio + identificacion_paciente (2
            # campos ausentes, sin ninguna otra senal), lo que lo hacia
            # indistinguible en naturaleza de incomplete_2 -- solo cambiaba
            # que campos faltaban. Ahora combina 1 campo ausente con un
            # defecto de contenido real y detectable de forma deterministica
            # (formato_numerico_requerido sobre "resultados", igual que en
            # inconsistent_sem), para que la etiqueta "inconsistent" refleje
            # una incoherencia real y no solo una convencion de nombre.
            omit_fields.add("identificacion_paciente")
            resultados = [("Prueba A", "positivo", "", ""),
                         ("Prueba B", "negativo", "", ""),
                         ("Prueba C", "indeterminado", "", "")]
            semantic_issues.append("identificacion_paciente ausente")
            semantic_issues.append("resultados sin formato estandar de laboratorio")
        elif state == "inconsistent_identidad":
            semantic_issues.append("identificador de paciente distinto en la caja de auditoria")

        # Ver nota equivalente en ClinicalNoteGenerator: la firma no puede
        # atarse al nombre del estado, debe atarse al campo que realmente
        # omite (medico_solicitante), o reintroduce "medico"/"dr."/"dra."
        # por la puerta de atras cuando ese campo deberia estar ausente.
        include_signature = "medico_solicitante" not in omit_fields

        color, y = self.draw_masthead(
            img, draw, hospital, "Servicio de Laboratorio - Informe de Resultados", layout,
            doc_id=doc_id, fecha_emision=fecha, tipo_doc="Informe diagnostico"
        )
        if layout == "C":
            y = self.draw_header_extension(draw, y, color)
        gt["fields"]["identificacion_centro"] = True

        if "identificacion_paciente" not in omit_fields:
            y = draw_label_value(draw, "Paciente:", paciente, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "NHC:", nhc, MARGIN, y, self.font_label, self.font_normal)
            gt["fields"]["identificacion_paciente"] = True
        else:
            gt["fields"]["identificacion_paciente"] = False

        if "medico_solicitante" not in omit_fields:
            y = draw_label_value(draw, "Medico solicitante:", f"{medico} ({col})",
                               MARGIN, y, self.font_label, self.font_normal, label_w=250)
            gt["fields"]["medico_solicitante"] = True
        else:
            gt["fields"]["medico_solicitante"] = False

        if "fecha_servicio" not in omit_fields:
            y = draw_label_value(draw, "Fecha de recogida:", fecha, MARGIN, y, self.font_label, self.font_normal, label_w=250)
            gt["fields"]["fecha_servicio"] = True
        else:
            gt["fields"]["fecha_servicio"] = False

        draw_line(draw, y + 5); y += 16

        y = fill_notes_gap(draw, y, HEADER_END + 10, MARGIN, PAGE_W - 2 * MARGIN, color,
                           self.font_label, title="MUESTRA / CONDICIONES DE EXTRACCION")

        draw.text((MARGIN, y), "RESULTADOS DE DETERMINACIONES", fill=color, font=self.font_label)
        y += 24

        col_widths = [250, 140, 120, 170]
        headers = ["Determinacion", "Resultado", "Unidad", "Ref."]
        table_rows = []
        for nombre, valor, unidad, ref in resultados:
            estado_color = BLACK
            try:
                val = float(valor.replace('.', '').replace(',', '.'))
                if ref and '-' in ref:
                    lo, hi = ref.split('-')
                    lo, hi = float(lo.replace('.', '').replace(',', '.')), float(hi.replace('.', '').replace(',', '.'))
                    if not (lo <= val <= hi):
                        estado_color = (190, 30, 30)
                        fuera_de_rango = True
                elif ref.startswith('<'):
                    lim = float(ref[1:].replace('.', '').replace(',', '.'))
                    if not (val < lim):
                        estado_color = (190, 30, 30)
                        fuera_de_rango = True
            except (ValueError, TypeError):
                pass
            table_rows.append([nombre, (valor, estado_color), unidad, ref])

        y = draw_table(draw, MARGIN, y, col_widths, headers, table_rows,
                       self.font_label, self.font_normal, color)

        gt["fields"]["resultados"] = True

        y += 8
        draw.text((MARGIN, y), "OBSERVACIONES DEL FACULTATIVO", fill=color, font=self.font_label)
        y += 20
        obs = ("Se observan valores fuera del rango de referencia; se recomienda "
              "repetir la determinacion en 2 semanas y valoracion clinica." if fuera_de_rango
              else "Todos los valores se encuentran dentro de los rangos de referencia "
                   "esperados para el perfil solicitado.")
        y = draw_text_block(draw, obs, MARGIN, y, self.font_normal)

        # NOTA (v3): DLR-001-ES no exige zona "signature"; caja legal
        # dimensionada solo con su contenido real (ver nota en CN).
        legal_extra = f"Fecha del informe: {fecha}"
        if state == "inconsistent_identidad":
            nhc_auditoria = generate_nhc(self.rng)
            while nhc_auditoria == nhc:
                nhc_auditoria = generate_nhc(self.rng)
            legal_extra += f"  |  NHC verificado en admision: {nhc_auditoria}"

        y += 8
        y = draw_legal_box(draw, y, y, MARGIN, PAGE_W - 2 * MARGIN,
                           self.font_label, self.font_small, color,
                           extra_line=legal_extra)

        y = self.draw_signature_block(img, draw, MARGIN, y, PAGE_W - 2 * MARGIN,
                                      medico, col, color, include=include_signature,
                                      layout=layout)

        gt["signatures"] = {"expected": 1, "present": 1 if include_signature else 0}
        gt["checkboxes"] = {"expected": 0, "present": 0, "checked": 0}
        gt["semantic_issues"] = semantic_issues
        gt["expected_verdict"] = "valid" if state == "valid" else ("incomplete" if state.startswith("incomplete") else "inconsistent")

        draw_page_footer(draw, hospital, doc_id,
                         show_center="identificacion_centro" not in omit_fields)
        return img, gt


class MedicationListGenerator(DocumentGenerator):
    """Genera listas de medicacion (MED-001-ES)."""

    def generate(self, state: str, variant: int, layout: str | None = None) -> tuple[Image.Image, dict]:
        img, draw = self.create_canvas()
        gt = {
            "family": "MED-001-ES", "family_name": "lista_medicacion",
            "state": state, "variant": variant, "fields": {},
            "signatures": {}, "checkboxes": {}, "semantic_issues": [],
            "expected_verdict": "",
        }
        doc_id = f"MED-001-ES_{state}_v{variant:02d}"
        layout = layout or self.rng.choice(["A", "B"])

        hospital = self.rng.choice(HOSPITALES)
        paciente = self.rng.choice(NOMBRES)
        nhc = generate_nhc(self.rng)
        dob = generate_dob(self.rng)
        fecha = generate_date(self.rng)
        medico, col = self.rng.choice(MEDICOS)
        alergia = self.rng.choice(ALERGIAS_POOL)
        # Excluye del sorteo el medicamento que contradiria la alergia
        # elegida (ver CONTRAINDICACIONES): sin esto, una alergia a
        # "Penicilina"/"AINEs" podia coincidir por puro azar con
        # "Amoxicilina"/"Ibuprofeno" en la lista de medicamentos de
        # CUALQUIER estado (no solo inconsistent_alergia), creando una
        # contraindicacion clinica REAL que el LLM detectaba correctamente
        # pero que el ground truth etiquetaba como "valid"/"incomplete" --
        # un defecto del corpus, no un falso positivo del modelo.
        medicamentos_pool = [
            m for m in MEDICAMENTOS if m[0] != CONTRAINDICACIONES.get(alergia)
        ]
        meds = self.rng.sample(medicamentos_pool, k=min(self.rng.randint(3, 6), len(medicamentos_pool)))

        omit_fields = set()
        semantic_issues = []
        include_doses = True
        include_signature = state not in ("incomplete_2", "inconsistent_mul")

        if state == "incomplete_1":
            omit_fields.add("identificacion_centro")
        elif state == "incomplete_2":
            omit_fields.add("identificacion_paciente")
            omit_fields.add("identificacion_centro")
        elif state == "inconsistent_sem":
            include_doses = False
            semantic_issues.append("medicamentos listados sin dosis ni posologia")
        elif state == "inconsistent_mul":
            omit_fields.add("identificacion_paciente")
            include_doses = False
            semantic_issues.append("identificacion_paciente ausente")
            semantic_issues.append("medicamentos sin dosis")
        elif state == "inconsistent_temporal":
            dob = f"{self.rng.randint(1,28):02d}/{self.rng.randint(1,12):02d}/{2026 + self.rng.randint(1,3)}"
            semantic_issues.append("fecha_nacimiento posterior a la fecha del documento")
        elif state == "inconsistent_identidad":
            semantic_issues.append("identificador de paciente distinto en la caja de auditoria")
        elif state == "inconsistent_alergia":
            # Alergia declarada perteneciente a la misma familia
            # farmacologica que un medicamento prescrito -- contraindicacion
            # real, pensada especificamente para probar el LLM (ver hallazgo
            # de contraindicacion clinica): ninguna regla determinista
            # respalda esta comprobacion a proposito.
            alergia, med_contraindicado = self.rng.choice(list(CONTRAINDICACIONES.items()))
            otros = [m for m in MEDICAMENTOS if m[0] != med_contraindicado]
            meds = self.rng.sample(otros, k=min(self.rng.randint(2, 4), len(otros)))
            meds.append(next(m for m in MEDICAMENTOS if m[0] == med_contraindicado))
            self.rng.shuffle(meds)
            semantic_issues.append(f"medicamento {med_contraindicado} contraindicado con alergia declarada a {alergia}")

        if "identificacion_centro" not in omit_fields:
            color, y = self.draw_masthead(img, draw, hospital, "Servicio de Farmacia", layout,
                                          doc_id=doc_id, fecha_emision=fecha, tipo_doc="Lista de medicacion")
            gt["fields"]["identificacion_centro"] = True
        else:
            color = hospital_color(hospital)
            draw_header_band(draw, color, height=14)
            y = MARGIN
            draw.text((MARGIN, y), "LISTA DE MEDICACION ACTIVA", fill=BLACK, font=self.font_title)
            y += 44
            draw_line(draw, y, color=tint(color, 0.55), width=2); y += 16
            gt["fields"]["identificacion_centro"] = False

        if layout == "C":
            y = self.draw_header_extension(draw, y, color)

        if "identificacion_paciente" not in omit_fields:
            y = draw_label_value(draw, "Paciente:", paciente, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "NHC:", nhc, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "Fecha de nacimiento:", dob, MARGIN, y, self.font_label, self.font_normal, label_w=260)
            gt["fields"]["identificacion_paciente"] = True
        else:
            gt["fields"]["identificacion_paciente"] = False

        y = draw_label_value(draw, "Fecha de revision:", fecha, MARGIN, y, self.font_label, self.font_normal, label_w=260)
        y = draw_label_value(draw, "Alergias conocidas:", alergia, MARGIN, y, self.font_label, self.font_normal, label_w=260)
        draw_line(draw, y + 5); y += 16

        y = fill_notes_gap(draw, y, HEADER_END + 10, MARGIN, PAGE_W - 2 * MARGIN, color,
                           self.font_label, title="INTERACCIONES A VIGILAR")

        draw.text((MARGIN, y), "MEDICACION ACTIVA", fill=color, font=self.font_label)
        y += 26

        if include_doses:
            col_widths = [240, 110, 280, 120]
            headers = ["Medicamento", "Dosis", "Posologia", "Via"]
            rows = [[nombre, dosis, posologia, via] for nombre, dosis, posologia, via in meds]
            y = draw_table(draw, MARGIN, y, col_widths, headers, rows,
                          self.font_label, self.font_normal, color)
        else:
            for i, (nombre, dosis, posologia, via) in enumerate(meds, 1):
                draw.text((MARGIN, y), f"{i}. {nombre}", fill=BLACK, font=self.font_label)
                y += 18
                draw.text((MARGIN + 25, y), "(sin informacion de dosis)", fill=GRAY, font=self.font_normal)
                y += 20

        gt["fields"]["medicamentos"] = True

        # NOTA (v3): renombrado desde "INSTRUCCIONES PARA EL PACIENTE" -- esa
        # version contenia la palabra "paciente" sin condicion, anulando el
        # estado inconsistent_mul de esta familia (que omite deliberadamente
        # identificacion_paciente). Ver nota en LEGAL_DISCLAIMER.
        y += 8
        draw.text((MARGIN, y), "RECOMENDACIONES DE ADMINISTRACION", fill=color, font=self.font_label)
        y += 20
        y = draw_text_block(draw, "Tomar la medicacion segun la posologia indicada. No suspender "
                            "ningun tratamiento sin consultar previamente con su medico. Ante "
                            "cualquier reaccion adversa, acudir al servicio de Urgencias.",
                            MARGIN, y, self.font_normal)

        # NOTA (v3): MED-001-ES no exige zona "signature"; caja legal
        # dimensionada solo con su contenido real (ver nota en CN).
        y += 8
        legal_extra = f"Ultima actualizacion: {fecha}"
        if state == "inconsistent_identidad":
            nhc_auditoria = generate_nhc(self.rng)
            while nhc_auditoria == nhc:
                nhc_auditoria = generate_nhc(self.rng)
            legal_extra += f"  |  NHC verificado en admision: {nhc_auditoria}"

        y = draw_legal_box(draw, y, y, MARGIN, PAGE_W - 2 * MARGIN,
                           self.font_label, self.font_small, color,
                           extra_line=legal_extra)

        y = self.draw_signature_block(img, draw, MARGIN, y, PAGE_W - 2 * MARGIN,
                                      medico, col, color, include=include_signature,
                                      layout=layout)

        gt["signatures"] = {"expected": 1, "present": 1 if include_signature else 0}
        gt["checkboxes"] = {"expected": 0, "present": 0, "checked": 0}
        gt["semantic_issues"] = semantic_issues
        gt["expected_verdict"] = "valid" if state == "valid" else ("incomplete" if state.startswith("incomplete") else "inconsistent")

        # Esta es la UNICA familia que omite identificacion_centro
        # (incomplete_1/incomplete_2): sin este flag el pie de pagina
        # reintroducia el nombre del hospital sin condicion y anulaba esa
        # prueba. Ver nota en draw_page_footer / LEGAL_DISCLAIMER.
        draw_page_footer(draw, hospital, doc_id,
                         show_center="identificacion_centro" not in omit_fields)
        return img, gt


class AdmissionFormGenerator(DocumentGenerator):
    """Genera formularios de admision (ADM-001-ES)."""

    def generate(self, state: str, variant: int, layout: str | None = None) -> tuple[Image.Image, dict]:
        img, draw = self.create_canvas()
        gt = {
            "family": "ADM-001-ES", "family_name": "formulario_admision",
            "state": state, "variant": variant, "fields": {},
            "signatures": {}, "checkboxes": {}, "semantic_issues": [],
            "expected_verdict": "",
        }
        doc_id = f"ADM-001-ES_{state}_v{variant:02d}"
        layout = layout or self.rng.choice(["A", "B"])

        hospital = self.rng.choice(HOSPITALES)
        servicio = self.rng.choice(SERVICIOS)
        paciente = self.rng.choice(NOMBRES)
        nhc = generate_nhc(self.rng)
        dob = generate_dob(self.rng)
        dni = generate_dni(self.rng)
        fecha = generate_date(self.rng)
        telefono = f"+34 {self.rng.randint(600,699)} {self.rng.randint(100,999)} {self.rng.randint(100,999)}"
        medico, col = self.rng.choice(MEDICOS)
        cama = f"Planta {self.rng.randint(2,8)} - Cama {self.rng.randint(1,30):02d}"
        motivo_ingreso = self.rng.choice(MOTIVOS_CONSULTA)
        antecedentes, medicacion_habitual = self.rng.choice(ANTECEDENTES)

        omit_fields = set()
        semantic_issues = []
        include_signature = state not in ("incomplete_2", "inconsistent_mul")

        if state == "incomplete_1":
            omit_fields.add("contacto")
        elif state == "incomplete_2":
            omit_fields.add("identificacion_paciente")
            omit_fields.add("contacto")
        elif state == "inconsistent_sem":
            semantic_issues.append("signos vitales con valores incompatibles entre si")
        elif state == "inconsistent_mul":
            # v5: antes omitia identificacion_paciente + signos_vitales (2
            # campos ausentes, sin ninguna otra senal) -- igual que en DLR,
            # eso lo hacia indistinguible en naturaleza de incomplete_2.
            # Ahora combina 1 campo ausente con signos vitales PRESENTES
            # pero fisiologicamente imposibles (mismos valores que
            # inconsistent_sem), detectable por la regla deterministica
            # RCH-ADM-ES-003 (rango_fisiologico) sin depender de LLM/VL.
            omit_fields.add("identificacion_paciente")
            semantic_issues.append("identificacion_paciente ausente")
            semantic_issues.append("signos vitales con valores incompatibles entre si")
        elif state == "inconsistent_temporal":
            dob = f"{self.rng.randint(1,28):02d}/{self.rng.randint(1,12):02d}/{2026 + self.rng.randint(1,3)}"
            semantic_issues.append("fecha_nacimiento posterior a la fecha del documento")
        elif state == "inconsistent_identidad":
            semantic_issues.append("identificador de paciente distinto en la caja de auditoria")
        elif state == "inconsistent_medtexto":
            antecedentes = ("Hipertension arterial en tratamiento con Enalapril desde "
                           "hace 3 anos. Sin alergias conocidas.")
            medicacion_habitual = "Ninguna"
            semantic_issues.append("medicacion_habitual 'Ninguna' contradicha por tratamiento activo en antecedentes")

        color, y = self.draw_masthead(img, draw, hospital, f"{servicio} - Formulario de Admision", layout,
                                      doc_id=doc_id, fecha_emision=fecha, tipo_doc="Formulario de admision")
        if layout == "C":
            y = self.draw_header_extension(draw, y, color)
        gt["fields"]["identificacion_centro"] = True

        draw.text((MARGIN, y), "DATOS DEL PACIENTE", fill=color, font=self.font_label)
        y += 22

        if "identificacion_paciente" not in omit_fields:
            y = draw_label_value(draw, "Nombre completo:", paciente, MARGIN, y, self.font_label, self.font_normal, label_w=230)
            y = draw_label_value(draw, "NHC:", nhc, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "DNI:", dni, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "Fecha de nacimiento:", dob, MARGIN, y, self.font_label, self.font_normal, label_w=260)
            gt["fields"]["identificacion_paciente"] = True
        else:
            gt["fields"]["identificacion_paciente"] = False

        y = draw_label_value(draw, "Fecha de ingreso:", fecha, MARGIN, y, self.font_label, self.font_normal, label_w=230)
        y = draw_label_value(draw, "Servicio / Cama:", cama, MARGIN, y, self.font_label, self.font_normal, label_w=230)

        if "contacto" not in omit_fields:
            y = draw_label_value(draw, "Telefono de contacto:", telefono, MARGIN, y, self.font_label, self.font_normal, label_w=260)
            gt["fields"]["contacto"] = True
        else:
            gt["fields"]["contacto"] = False

        draw_line(draw, y + 5); y += 16

        y = fill_notes_gap(draw, y, HEADER_END + 10, MARGIN, PAGE_W - 2 * MARGIN, color,
                           self.font_label, title="OBSERVACIONES DE ADMISION")

        draw.text((MARGIN, y), "MOTIVO DE INGRESO", fill=color, font=self.font_label)
        y += 20
        y = draw_text_block(draw, motivo_ingreso, MARGIN, y, self.font_normal)
        y += 12

        if "signos_vitales_o_historial" not in omit_fields:
            draw.text((MARGIN, y), "SIGNOS VITALES AL INGRESO", fill=color, font=self.font_label)
            y += 22

            if state in ("inconsistent_sem", "inconsistent_mul"):
                vitales = [
                    ("Tension arterial:", "350/200 mmHg"),
                    ("Frecuencia cardiaca:", "12 lpm"),
                    ("Temperatura:", "28.0 C"),
                    ("Saturacion O2:", "102%"),
                ]
            else:
                vitales = [
                    ("Tension arterial:", f"{self.rng.randint(110,140)}/{self.rng.randint(60,90)} mmHg"),
                    ("Frecuencia cardiaca:", f"{self.rng.randint(60,100)} lpm"),
                    ("Temperatura:", f"{self.rng.uniform(36.0, 37.5):.1f} C"),
                    ("Saturacion O2:", f"{self.rng.randint(95,100)}%"),
                ]

            for label, valor in vitales:
                y = draw_label_value(draw, label, valor, MARGIN + 20, y, self.font_label, self.font_normal, label_w=260)

            y += 10
            draw.text((MARGIN, y), "ANTECEDENTES RELEVANTES", fill=color, font=self.font_label)
            y += 20
            y = draw_text_block(draw, antecedentes, MARGIN, y, self.font_normal)
            y += 10
            y = draw_label_value(draw, "Medicacion habitual:", medicacion_habitual, MARGIN, y, self.font_label, self.font_normal, label_w=220)
            gt["fields"]["signos_vitales_o_historial"] = True
            gt["fields"]["medicacion_habitual"] = True
        else:
            gt["fields"]["signos_vitales_o_historial"] = False
            gt["fields"]["medicacion_habitual"] = False

        # NOTA (v3): ADM-001-ES no exige zona "signature"; caja legal
        # dimensionada solo con su contenido real (ver nota en CN).
        legal_extra = f"Fecha de registro: {fecha}"
        if state == "inconsistent_identidad":
            nhc_auditoria = generate_nhc(self.rng)
            while nhc_auditoria == nhc:
                nhc_auditoria = generate_nhc(self.rng)
            legal_extra += f"  |  NHC verificado en admision: {nhc_auditoria}"

        y += 8
        y = draw_legal_box(draw, y, y, MARGIN, PAGE_W - 2 * MARGIN,
                           self.font_label, self.font_small, color,
                           extra_line=legal_extra)

        if layout == "B":
            draw_stamp(img, PAGE_W - MARGIN - 130, y + 60, ["ADMISION", "CONFORME"], color, self.rng,
                      radius_x=90, radius_y=48)

        y = self.draw_signature_block(img, draw, MARGIN, y, PAGE_W - 2 * MARGIN,
                                      medico, col, color,
                                      label="Firma y sello del responsable de admision:",
                                      include=include_signature, layout=layout)

        gt["signatures"] = {"expected": 1, "present": 1 if include_signature else 0}
        gt["checkboxes"] = {"expected": 0, "present": 0, "checked": 0}
        gt["semantic_issues"] = semantic_issues
        gt["expected_verdict"] = "valid" if state == "valid" else ("incomplete" if state.startswith("incomplete") else "inconsistent")

        draw_page_footer(draw, hospital, doc_id,
                         show_center="identificacion_centro" not in omit_fields)
        return img, gt


class PreopChecklistGenerator(DocumentGenerator):
    """Genera checklists preoperatorios (PREOP-001-ES)."""

    def generate(self, state: str, variant: int, layout: str | None = None) -> tuple[Image.Image, dict]:
        img, draw = self.create_canvas()
        gt = {
            "family": "PREOP-001-ES", "family_name": "checklist_preoperatorio",
            "state": state, "variant": variant, "fields": {},
            "signatures": {}, "checkboxes": {}, "semantic_issues": [],
            "expected_verdict": "",
        }
        doc_id = f"PREOP-001-ES_{state}_v{variant:02d}"
        layout = layout or self.rng.choice(["A", "B"])

        hospital = self.rng.choice(HOSPITALES)
        paciente = self.rng.choice(NOMBRES)
        nhc = generate_nhc(self.rng)
        dob = generate_dob(self.rng)
        fecha = generate_date(self.rng)
        cirujano, col = self.rng.choice(MEDICOS)
        items = self.rng.sample(ITEMS_PREOP, k=min(7, len(ITEMS_PREOP)))
        intervencion = self.rng.choice(TIPOS_INTERVENCION)

        omit_fields = set()
        semantic_issues = []
        include_signature = True
        all_checked = True

        if state == "incomplete_1":
            omit_fields.add("responsable_clinico")
        elif state == "incomplete_2":
            omit_fields.add("firma_paciente")
            include_signature = False
            omit_fields.add("responsable_clinico")
        elif state == "inconsistent_sem":
            all_checked = False
            semantic_issues.append("items criticos del checklist marcados como No")
        elif state == "inconsistent_mul":
            omit_fields.add("identificacion_paciente")
            include_signature = False
            omit_fields.add("firma_paciente")
            all_checked = False
            semantic_issues.append("identificacion_paciente ausente")
            semantic_issues.append("firma_paciente ausente")
            semantic_issues.append("items criticos sin verificar")
        elif state == "inconsistent_temporal":
            dob = f"{self.rng.randint(1,28):02d}/{self.rng.randint(1,12):02d}/{2026 + self.rng.randint(1,3)}"
            semantic_issues.append("fecha_nacimiento posterior a la fecha del documento")
        elif state == "inconsistent_identidad":
            semantic_issues.append("identificador de paciente distinto en la caja de auditoria")

        color, y = self.draw_masthead(img, draw, hospital, "Lista de Verificacion Preoperatoria", layout,
                                      doc_id=doc_id, fecha_emision=fecha, tipo_doc="Checklist preoperatorio")
        if layout == "C":
            y = self.draw_header_extension(draw, y, color)
        gt["fields"]["identificacion_centro"] = True

        if "identificacion_paciente" not in omit_fields:
            y = draw_label_value(draw, "Paciente:", paciente, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "NHC:", nhc, MARGIN, y, self.font_label, self.font_normal)
            y = draw_label_value(draw, "Fecha de nacimiento:", dob, MARGIN, y, self.font_label, self.font_normal, label_w=260)
            gt["fields"]["identificacion_paciente"] = True
        else:
            gt["fields"]["identificacion_paciente"] = False

        y = draw_label_value(draw, "Fecha programada:", fecha, MARGIN, y, self.font_label, self.font_normal, label_w=260)
        y = draw_label_value(draw, "Tipo de intervencion:", intervencion, MARGIN, y, self.font_label, self.font_normal, label_w=260)

        if "responsable_clinico" not in omit_fields:
            y = draw_label_value(draw, "Cirujano responsable:", f"{cirujano} ({col})",
                               MARGIN, y, self.font_label, self.font_normal, label_w=270)
            gt["fields"]["responsable_clinico"] = True
        else:
            gt["fields"]["responsable_clinico"] = False

        draw_line(draw, y + 5); y += 16

        y = fill_notes_gap(draw, y, HEADER_END + 10, MARGIN, PAGE_W - 2 * MARGIN, color,
                           self.font_label, title="OBSERVACIONES DE ENFERMERIA")

        draw.text((MARGIN, y), "ITEMS DE VERIFICACION", fill=color, font=self.font_label)
        y += 24

        n_checkboxes = len(items)
        n_checked = 0

        for i, item in enumerate(items):
            checked = all_checked or (i < 3)
            draw_checkbox(draw, MARGIN, y, checked, color=color if checked else DARK_GRAY)
            respuesta = "Si" if checked else "No"
            draw.text((MARGIN + 34, y + 3), f"{item}: {respuesta}",
                     fill=BLACK, font=self.font_normal)
            y += 30
            if checked:
                n_checked += 1

        gt["fields"]["items_checklist"] = True
        gt["checkboxes"] = {
            "expected": n_checkboxes,
            "present": n_checkboxes,
            "checked": n_checked,
        }

        legal_extra = f"Fecha: {fecha}"
        if state == "inconsistent_identidad":
            nhc_auditoria = generate_nhc(self.rng)
            while nhc_auditoria == nhc:
                nhc_auditoria = generate_nhc(self.rng)
            legal_extra += f"  |  NHC verificado en admision: {nhc_auditoria}"

        y += 8
        y = draw_legal_box(draw, y, LEGAL_END, MARGIN, PAGE_W - 2 * MARGIN,
                           self.font_label, self.font_small, color,
                           extra_line=legal_extra)

        if layout == "A":
            draw_stamp(img, PAGE_W - MARGIN - 130, y + 55, ["QUIROFANO", "VERIFICADO"], color, self.rng,
                      radius_x=90, radius_y=48)

        if layout == "C":
            # Firma del paciente en columna lateral en vez de la banda
            # inferior habitual: este es el campo critico de PREOP-001
            # (zona_documento="signature", peso 3), por lo que es el caso
            # mas relevante para medir si el detector de firmas y la
            # zonificacion generalizan a una posicion no estandar.
            self._draw_lateral_signature(img, draw, paciente, "Firma paciente",
                                         color, include=include_signature)
            gt["fields"]["firma_paciente"] = include_signature
            gt["signatures"] = {"expected": 1, "present": 1 if include_signature else 0}
        elif include_signature:
            draw.text((MARGIN, y), "Firma del paciente (consentimiento):", fill=color, font=self.font_label)
            y += 26
            draw_signature(draw, MARGIN + 12, y + 28, width=220, height=48,
                           seed=self.rng.randint(0, 999999))
            y += 62
            draw_line(draw, y, MARGIN + 12, MARGIN + 260)
            y += 6
            draw.text((MARGIN + 12, y), paciente, fill=DARK_GRAY, font=self.font_small)
            y += 16
            draw.text((MARGIN + 12, y), f"Fecha: {fecha}", fill=DARK_GRAY, font=self.font_small)
            y += 18
            gt["fields"]["firma_paciente"] = True
            gt["signatures"] = {"expected": 1, "present": 1}
        else:
            draw.text((MARGIN, y + 6), "[ Firma del paciente pendiente ]", fill=(190, 30, 30), font=self.font_normal)
            y += 30
            gt["fields"]["firma_paciente"] = False
            gt["signatures"] = {"expected": 1, "present": 0}

        # IMPORTANTE: la etiqueta original "Firma y sello del cirujano
        # responsable:" contenia literalmente "cirujano" y "responsable",
        # que son 2 de los 5 patrones de responsable_clinico
        # (["cirujano","anestesiologo","dr.","dra.","responsable"]) -- se
        # imprimia SIEMPRE, incluso cuando incomplete_1 omite justo ese
        # campo (el include forzado `or state == "incomplete_1"` ademas
        # imprimia el nombre real del cirujano con "Dra."/"Dr." debajo). El
        # resultado era que responsable_clinico nunca podia probarse como
        # ausente para PREOP. Se renombra la etiqueta para no usar esas
        # palabras y se ata el include a la MISMA omision del campo.
        y += 10
        y = self.draw_signature_block(img, draw, MARGIN, y, PAGE_W - 2 * MARGIN,
                                      cirujano, col, color,
                                      label="Validacion y sello facultativo:",
                                      include="responsable_clinico" not in omit_fields)

        gt["semantic_issues"] = semantic_issues
        gt["expected_verdict"] = "valid" if state == "valid" else ("incomplete" if state.startswith("incomplete") else "inconsistent")

        draw_page_footer(draw, hospital, doc_id,
                         show_center="identificacion_centro" not in omit_fields)
        return img, gt


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

FAMILIES = {
    "CN-001-ES": ClinicalNoteGenerator,
    "DLR-001-ES": DiagnosticReportGenerator,
    "MED-001-ES": MedicationListGenerator,
    "ADM-001-ES": AdmissionFormGenerator,
    "PREOP-001-ES": PreopChecklistGenerator,
}

STATES = [
    "valid", "incomplete_1", "incomplete_2", "inconsistent_sem", "inconsistent_mul",
    "inconsistent_temporal", "inconsistent_identidad",
    "inconsistent_medtexto", "inconsistent_alergia",
]

# inconsistent_temporal / inconsistent_identidad: universales, aplican a las
# 5 familias (todas dibujan fecha de nacimiento + fecha del documento, y
# todas usan draw_legal_box). inconsistent_medtexto / inconsistent_alergia:
# solo tienen sentido donde existe el campo correspondiente (historia
# clinica en texto libre para medtexto; lista de medicamentos + alergias
# para alergia) -- ver FAMILY_ONLY_STATES mas abajo, que restringe estos
# dos ultimos a sus familias aplicables en vez de generarlos en las 5.

# Diseno base comun a las 5 familias (particion dev-eval): cada substate
# granular (usado internamente por cada generador para decidir que campo
# omitir o que incoherencia introducir) se reparte en (dev, eval). Los
# totales por familia dan 18 dev + 18 eval = 36 documentos base, y a nivel
# de veredicto de 3 clases (valid/incomplete/inconsistent) cada particion
# queda balanceada por familia. FAMILY_ONLY_STATES anade documentos extra
# solo a las familias donde aplica cada estado adicional (ver mas abajo),
# lo que hace que el total real por familia varie entre 36 y 50 y el
# corpus completo llegue a 230 documentos (115 dev + 115 eval).
# La particion NO se resuelve por variante de layout cosmetico (A/B, que
# comparten exactamente las mismas fracciones de zona que la heuristica de
# PathwayGuard): los documentos "eval" se generan siempre en layout "C"
# (cabecera ampliada + firma lateral), la unica plantilla del corpus que no
# fue disenada para coincidir con HEADER_MAX_Y/BODY_MAX_Y/LEGAL_MAX_Y. Esto
# evita el sesgo de circularidad senalado en el diseno v3 (el ground truth
# ya no se construye "para que el campo caiga en la zona correcta segun la
# heuristica real") y permite medir el error de zonificacion en un diseno
# no ajustado a esos umbrales, tal como exige la evaluacion.
SUBSTATE_PLAN = {
    "valid":                  (5, 5),
    "incomplete_1":           (3, 2),
    "incomplete_2":           (2, 3),
    "inconsistent_sem":       (3, 2),
    "inconsistent_mul":       (2, 3),
    "inconsistent_identidad": (3, 3),
}

# Estados que solo aplican a un subconjunto de familias (no todas tienen el
# campo necesario para que el defecto tenga sentido). El estado se genera
# SOLO para las familias listadas aqui, con el (dev, eval) indicado; para
# cualquier familia no listada, se omite por completo (equivalente a que no
# exista en absoluto para esa familia, no a (0, 0) documentos "vacios").
FAMILY_ONLY_STATES = {
    # DLR-001-ES no dibuja fecha de nacimiento (solo NHC) -- el chequeo
    # temporal requiere dob, asi que se aplica a las otras 4 familias. Con
    # densidad (4,4) en vez de (3,3) para cerrar el corpus en 230
    # documentos exactos y dar algo mas de resolucion estadistica al tipo
    # con mas dependencia de la calidad del OCR.
    "inconsistent_temporal": {
        "CN-001-ES": (4, 4), "MED-001-ES": (4, 4),
        "ADM-001-ES": (4, 4), "PREOP-001-ES": (4, 4),
    },
    "inconsistent_medtexto": {"ADM-001-ES": (3, 3), "CN-001-ES": (3, 3)},
    "inconsistent_alergia": {"MED-001-ES": (3, 3)},
}


def _plan_for(state: str, family_id: str) -> tuple[int, int] | None:
    """Devuelve (dev, eval) para `state` en `family_id`, o None si ese
    estado no aplica a esa familia (ver FAMILY_ONLY_STATES)."""
    if state in FAMILY_ONLY_STATES:
        return FAMILY_ONLY_STATES[state].get(family_id)
    return SUBSTATE_PLAN[state]


def generate_corpus(output_dir: str, seed: int = 42,
                    scan_artifacts: bool = True, scan_intensity: str = "light"):
    """
    Genera el corpus completo: 230 documentos (115 dev + 115 eval; entre 36
    y 50 documentos por familia segun los estados adicionales de
    FAMILY_ONLY_STATES que le apliquen -- ver SUBSTATE_PLAN mas arriba).

    Args:
        output_dir: Directorio de salida
        seed: Semilla para reproducibilidad
        scan_artifacts: Si True (por defecto), aplica realismo de escaneo
            (ruido + sombra + rotacion leve) a las 5 familias de forma
            consistente.
        scan_intensity: "light" (por defecto, calibrado para no danar la
            deteccion de campos), "medium" o "heavy".
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    total = sum(
        sum(_plan_for(state, family_id) or (0, 0))
        for family_id in FAMILIES
        for state in STATES
    )

    ground_truth = {
        "metadata": {
            "generator": "generate_synthetic_corpus.py",
            "version": "5.0",
            "seed": seed,
            "substate_plan": {k: {"dev": v[0], "eval": v[1]} for k, v in SUBSTATE_PLAN.items()},
            "family_only_states": {
                state: {fam: {"dev": v[0], "eval": v[1]} for fam, v in plans.items()}
                for state, plans in FAMILY_ONLY_STATES.items()
            },
            "total_documents": total,
            "generated_at": datetime.now().isoformat(),
            "families": list(FAMILIES.keys()),
            "states": STATES,
            "partitions": {
                "dev": {"layouts": ["A", "B"], "purpose": "ajuste de umbrales, regex, pesos y prompts"},
                "eval": {"layouts": ["C"], "purpose": "evaluacion final congelada; cabecera ampliada y firma lateral, no vista durante el desarrollo"},
            },
            "scan_artifacts": scan_artifacts,
            "scan_intensity": scan_intensity if scan_artifacts else None,
        },
        "documents": [],
    }

    doc_idx = 0

    for family_id, generator_class in FAMILIES.items():
        family_dir = images_dir / family_id
        family_dir.mkdir(exist_ok=True)

        for state in STATES:
            plan_tuple = _plan_for(state, family_id)
            if plan_tuple is None:
                continue
            dev_count, eval_count = plan_tuple
            plan = [("dev", None) for _ in range(dev_count)] + \
                   [("eval", "C") for _ in range(eval_count)]

            for v, (partition, forced_layout) in enumerate(plan, start=1):
                doc_idx += 1
                # hash() de Python esta aleatorizado por proceso para strings
                # desde 3.3 (PYTHONHASHSEED) salvo que se fije explicitamente:
                # usar hash() aqui hacia que "--seed 42" NO fuera realmente
                # reproducible entre ejecuciones (cada regeneracion del
                # corpus producia contenido distinto pese a la semilla).
                # hashlib.md5 sobre bytes es estable entre procesos y
                # versiones de Python.
                doc_key = f"{family_id}_{state}_{v}".encode("utf-8")
                doc_hash = int(hashlib.md5(doc_key).hexdigest(), 16)
                doc_seed = seed + doc_hash % 100000
                doc_rng = random.Random(doc_seed)
                generator = generator_class(doc_rng)

                layout = forced_layout if forced_layout else doc_rng.choice(["A", "B"])
                img, gt = generator.generate(state, v, layout=layout)

                if scan_artifacts:
                    img = apply_scan_realism(img, doc_rng, scan_intensity)

                filename = f"{family_id}_{state}_v{v:02d}.png"
                filepath = family_dir / filename
                img.save(str(filepath), "PNG")

                with open(filepath, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

                gt["doc_id"] = f"{family_id}_{state}_v{v:02d}"
                gt["filename"] = filename
                gt["filepath"] = str(filepath.relative_to(output_path))
                gt["file_hash"] = file_hash
                gt["partition"] = partition
                gt["template"] = layout

                ground_truth["documents"].append(gt)

                pct = doc_idx / total * 100
                print(f"  [{doc_idx:3d}/{total}] ({pct:5.1f}%) [{partition}/{layout}] {filename}")

    gt_path = output_path / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Corpus generado: {total} documentos")
    print(f"  Familias: {len(FAMILIES)}")
    print(f"  Estados: {len(STATES)}")
    print(f"  Realismo de escaneo: {'Si (' + scan_intensity + ')' if scan_artifacts else 'No'}")
    print(f"  Directorio: {output_path}")
    print(f"  Ground truth: {gt_path}")
    print(f"{'='*60}")

    breakdown = {}
    for doc in ground_truth["documents"]:
        key = (doc["partition"], doc["expected_verdict"])
        breakdown[key] = breakdown.get(key, 0) + 1
    print(f"\nDistribucion de veredictos esperados por particion:")
    for (partition, verdict), count in sorted(breakdown.items()):
        print(f"  {partition:5s} / {verdict:12s}: {count}")

    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Genera corpus sintetico para evaluacion de PathwayGuard"
    )
    parser.add_argument(
        "--output-dir", default="data/synthetic_corpus",
        help="Directorio de salida (default: data/synthetic_corpus)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para reproducibilidad"
    )
    parser.add_argument(
        "--no-scan-artifacts", action="store_true",
        help="Desactivar el realismo de escaneo (por defecto esta activado)"
    )
    parser.add_argument(
        "--scan-intensity", choices=["light", "medium", "heavy"], default="light",
        help="Intensidad del realismo de escaneo (default: light, calibrado para no danar el OCR)"
    )

    args = parser.parse_args()
    scan_artifacts = not args.no_scan_artifacts

    total = sum(
        sum(_plan_for(state, family_id) or (0, 0))
        for family_id in FAMILIES
        for state in STATES
    )
    print(f"Generando corpus sintetico...")
    print(f"  Output: {args.output_dir}")
    print(f"  Diseno: {total} documentos (base 42/familia + estados adicionales segun familia)")
    print(f"  Seed: {args.seed}")
    print(f"  Realismo de escaneo: {'Si (' + args.scan_intensity + ')' if scan_artifacts else 'No'}")
    print()

    generate_corpus(args.output_dir, args.seed, scan_artifacts, args.scan_intensity)


if __name__ == "__main__":
    main()
