# -*- coding: utf-8 -*-
"""
Helpers para editar TFM_PathwayGuard.docx preservando estilos exactos.
"""
import copy
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def set_paragraph_text(paragraph, new_text):
    """Reemplaza el texto de un parrafo de un solo run, conservando su estilo."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(new_text)
        return
    runs[0].text = new_text
    for r in runs[1:]:
        r.text = ""


def insert_paragraph_after(paragraph, text="", style=None):
    """Inserta un nuevo parrafo INMEDIATAMENTE DESPUES del parrafo dado,
    devolviendo el objeto Paragraph nuevo (para poder encadenar)."""
    new_p_elem = copy.deepcopy(paragraph._p)
    # limpiar todos los runs del clon
    for r in new_p_elem.findall(qn('w:r')):
        new_p_elem.remove(r)
    paragraph._p.addnext(new_p_elem)
    new_para = Paragraph(new_p_elem, paragraph._parent)
    if style is not None:
        new_para.style = style
    else:
        new_para.style = paragraph.style
    if text:
        new_para.add_run(text)
    return new_para


def insert_table_after(paragraph, rows_data, style_name="Table Grid"):
    """Inserta una tabla nueva inmediatamente despues del parrafo dado.
    rows_data: lista de listas (filas de strings). Devuelve el objeto Table.
    Tambien inserta un parrafo vacio Normal justo despues de la tabla (Word
    exige que un documento no termine ni encadene tablas sin un parrafo
    intermedio en algunos casos, y facilita insertar mas contenido despues).
    """
    doc = paragraph.part.document
    n_rows = len(rows_data)
    n_cols = len(rows_data[0]) if rows_data else 1
    table = doc.add_table(rows=n_rows, cols=n_cols)
    if style_name:
        table.style = style_name
    for i, row in enumerate(rows_data):
        for j, val in enumerate(row):
            table.cell(i, j).text = str(val)
    # mover la tabla (que add_table() puso al final del documento) a la
    # posicion correcta, justo despues del parrafo de referencia.
    tbl_elem = table._tbl
    tbl_elem.getparent().remove(tbl_elem)
    paragraph._p.addnext(tbl_elem)
    return table


def paragraph_after_table(table, parent_doc):
    """Inserta SIEMPRE un parrafo NUEVO y vacio (estilo Normal) justo
    despues de la tabla dada, y lo devuelve. Nunca reutiliza el parrafo
    que ya exista despues de la tabla en el documento original -- ese
    parrafo pertenece al contenido preexistente (ej. el siguiente
    encabezado) y no debe tratarse como un hueco disponible. Usar SIEMPRE
    esto (nunca un parrafo de una celda de la tabla, ni asumir que el
    siguiente elemento del XML esta libre) como referencia para insertar
    contenido despues de una tabla.
    """
    from docx.oxml import OxmlElement
    tbl_elem = table._tbl
    new_p_elem = OxmlElement('w:p')
    tbl_elem.addnext(new_p_elem)
    new_para = Paragraph(new_p_elem, parent_doc)
    new_para.style = parent_doc.styles['Normal']
    return new_para


def find_paragraph_by_text(doc, text_substring, start=0):
    for i in range(start, len(doc.paragraphs)):
        if text_substring in doc.paragraphs[i].text:
            return i
    return -1
