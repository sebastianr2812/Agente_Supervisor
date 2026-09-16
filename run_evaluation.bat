@echo off
REM ============================================================
REM  PathwayGuard - Evaluacion Integral
REM  Ejecutar desde la raiz del proyecto
REM ============================================================

echo ============================================================
echo  PASO 1: Generar corpus sintetico (230 documentos: 115 desarrollo + 115 evaluacion)
echo ============================================================
python generate_synthetic_corpus.py --output-dir data/synthetic_corpus_v2 --seed 42
if errorlevel 1 (echo ERROR en generacion & pause & exit /b 1)

echo.
echo ============================================================
echo  PASO 2: Evaluacion sobre la particion de evaluacion (115 docs) - solo reglas
echo ============================================================
python evaluate_synthetic_corpus.py --corpus-dir data/synthetic_corpus_v2 --protocols-dir configs/protocolos_es --partition eval --output-dir data/results/synthetic
if errorlevel 1 (echo ERROR en evaluacion & pause & exit /b 1)

echo.
echo ============================================================
echo  PASO 3: Ablacion completa (reglas / reglas+LLM / reglas+LLM+VL)
echo  NOTA: requiere Ollama corriendo con mistral:7b (y qwen2.5vl:7b para VL).
echo  reglas+LLM+VL sobre las 115 imagenes puede tardar varias horas en CPU
echo  (ver memoria, seccion de rendimiento: ~94-160s/documento en esa config).
echo ============================================================
set /p ABLATION="Ejecutar ablacion completa sobre las 115 imagenes de evaluacion? (S/N): "
if /i "%ABLATION%"=="S" (
    python evaluate_synthetic_corpus.py --corpus-dir data/synthetic_corpus_v2 --protocols-dir configs/protocolos_es --partition eval --output-dir data/results/synthetic --ablation --vl-limit 115
)

echo.
echo ============================================================
echo  PASO 4: Generar figuras (a partir del resultado mas completo
echo  disponible: reglas+LLM+VL si se ejecuto la ablacion en el
echo  Paso 3, si no reglas+LLM, si no reglas solas)
echo ============================================================
set "RESULTS_FILE="
for /f "delims=" %%f in ('dir /b /o-d "data\results\synthetic\eval_reglas+LLM+VL_*.json" 2^>nul') do (
    if not defined RESULTS_FILE set "RESULTS_FILE=%%f"
)
if defined RESULTS_FILE goto :visualize

for /f "delims=" %%f in ('dir /b /o-d "data\results\synthetic\eval_reglas+LLM_*.json" 2^>nul') do (
    if not defined RESULTS_FILE set "RESULTS_FILE=%%f"
)
if defined RESULTS_FILE goto :visualize

for /f "delims=" %%f in ('dir /b /o-d "data\results\synthetic\eval_reglas_*.json" 2^>nul') do (
    if not defined RESULTS_FILE set "RESULTS_FILE=%%f"
)

:visualize
if not defined RESULTS_FILE (
    echo No se encontro ningun archivo de resultados en data\results\synthetic\
) else (
    echo Visualizando data\results\synthetic\%RESULTS_FILE% ...
    python visualize_evaluation.py --results-file "data\results\synthetic\%RESULTS_FILE%" --output-dir figures
)

echo.
echo ============================================================
echo  COMPLETADO
echo  - Corpus: data/synthetic_corpus_v2/ (230 documentos PNG + ground_truth.json)
echo  - Resultados: data/results/synthetic/
echo  - Figuras: figures/
echo ============================================================
pause
