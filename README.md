# Supervisor de documentos clínicos (PathwayGuard)

Agente de supervisión documental sanitaria: OCR + motor de reglas ponderado + razonamiento con LLM local.

## 1. Instalación

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Dependencias de sistema (no se instalan con pip)

- **Tesseract OCR 5.x**: https://github.com/UB-Mannheim/tesseract/wiki (instalador para Windows).
  La ruta al ejecutable se configura en `src/ocr/extractor.py` (`pytesseract.pytesseract.tesseract_cmd`).
- **Ollama** (opcional, solo si se activa la validación semántica con LLM): https://ollama.com
  ```powershell
  ollama pull mistral:7b
  ollama pull qwen2.5vl:7b   # opcional, solo para el análisis visual profundo
  ```
  Debe estar sirviendo en `http://localhost:11434` (por defecto).

## 2. Descargar el dataset ClinOCR-Bench (opcional)

El proyecto se evaluó contra **ClinOCR-Bench**, un benchmark público de 384 imágenes clínicas
escaneadas con transcripción de referencia. No se incluye en este repositorio por su tamaño (~123 MB).
No es necesario para ejecutar la aplicación (`app.py`) con tus propios documentos, pero sí para
reproducir los benchmarks de evaluación OCR.

Para descargarlo:

```powershell
Invoke-WebRequest -Uri "https://github.com/ClinOCR-Bench/ClinOCR-Bench/releases/download/v1.0/ClinOCR-Bench-v1.0.zip" -OutFile "ClinOCR-Bench-v1.0.zip"
Expand-Archive -Path "ClinOCR-Bench-v1.0.zip" -DestinationPath "data\clinocr_bench\raw"
```

Al terminar debe quedar la carpeta `data\clinocr_bench\raw\ClinOCR-Bench\` con las subcarpetas
`scans\` (imágenes) y `ground_truth\` (transcripciones de referencia).

Licencia y detalles del dataset: https://github.com/ClinOCR-Bench/ClinOCR-Bench

## 3. Verificar la instalación

```powershell
.\venv\Scripts\python.exe -m pytest tests\ -q
```

Debe mostrar `18 passed`.

## 4. Ejecutar la aplicación

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Se abre en `http://localhost:8501`. Selecciona un tipo de documento, sube una imagen o PDF
escaneado y pulsa "Validate document".

## 5. Ejecutar desde línea de comandos

```powershell
.\venv\Scripts\python.exe run_supervisor.py <ruta_imagen> <protocolo_id>
.\venv\Scripts\python.exe run_supervisor.py <ruta_imagen> <protocolo_id> --use-llm
```

Por defecto usa `configs/protocolos_es`, el conjunto en español evaluado en la memoria.
Protocolos disponibles: `CN-001-ES` (notas clínicas), `DLR-001-ES` (informes diagnósticos y de
laboratorio), `MED-001-ES` (listas de medicación), `ADM-001-ES` (formularios de admisión),
`PREOP-001-ES` (formularios preoperatorios).

Para ejecutar contra los protocolos en inglés usados en la prueba de integración sobre
ClinOCR-Bench (sección 5.11.1 de la memoria), añade `--protocols-dir configs/protocolos`
y usa los IDs sin sufijo (`CN-001`, `DLR-001`, `MED-001`, `ADM-001`, `PREOP-001`).
