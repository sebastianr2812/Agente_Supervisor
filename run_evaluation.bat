@echo off
REM ============================================================
REM  PathwayGuard - Evaluacion Integral
REM  Ejecutar desde la raiz del proyecto
REM ============================================================

echo ============================================================
echo  PASO 1: Generar corpus sintetico (75 documentos)
echo ============================================================
python generate_synthetic_corpus.py --output-dir data/synthetic_corpus --variants 3 --seed 42
if errorlevel 1 (echo ERROR en generacion & pause & exit /b 1)

echo.
echo ============================================================
echo  PASO 2: Evaluacion - Solo reglas
echo ============================================================
python evaluate_synthetic_corpus.py --corpus-dir data/synthetic_corpus --protocols-dir configs/protocolos_es --output-dir data/results/synthetic
if errorlevel 1 (echo ERROR en evaluacion & pause & exit /b 1)

echo.
echo ============================================================
echo  PASO 3: Evaluacion con ablacion (reglas / reglas+LLM / reglas+VL)
echo  NOTA: Requiere Ollama corriendo con mistral:7b-instruct y qwen2.5-vl:3b
echo ============================================================
set /p ABLATION="Ejecutar ablacion con LLM? (S/N): "
if /i "%ABLATION%"=="S" (
    python evaluate_synthetic_corpus.py --corpus-dir data/synthetic_corpus --protocols-dir configs/protocolos_es --output-dir data/results/synthetic --ablation
)

echo.
echo ============================================================
echo  PASO 4: Generar figuras
echo ============================================================
for %%f in (data\results\synthetic\eval_reglas_*.json) do (
    echo Visualizando %%f ...
    python visualize_evaluation.py --results-file "%%f" --output-dir figures
)

echo.
echo ============================================================
echo  COMPLETADO
echo  - Corpus: data/synthetic_corpus/ (75 documentos PNG + ground_truth.json)
echo  - Resultados: data/results/synthetic/
echo  - Figuras: figures/
echo ============================================================
pause
