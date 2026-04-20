@echo off
REM Full contamination detection pipeline.
REM Run from the contamination_detection\ directory.
REM Prerequisites: pip install -r requirements.txt
REM                Copy .env.example to .env and set OPENAI_API_KEY

setlocal
cd /d "%~dp0"

echo === Step 0: Download datasets ===
python scripts\00_load_datasets.py
if errorlevel 1 goto :error

echo.
echo === Step 1: N-gram filter (C_lex detection) ===
python scripts\01_ngram_filter.py
if errorlevel 1 goto :error

echo.
echo === Step 2: Embedding retrieval (C_sem candidates) ===
python scripts\02_embedding_retrieval.py
if errorlevel 1 goto :error

echo.
echo === Step 3: LLM judge (C_sem verification) ===
python scripts\03_llm_judge.py
if errorlevel 1 goto :error

echo.
echo === Step 4: Build clean control set ===
python scripts\04_build_clean_set.py
if errorlevel 1 goto :error

echo.
echo === Step 5: Validate and report ===
python scripts\05_validate_and_report.py --no-spotcheck
if errorlevel 1 goto :error

echo.
echo === Done! ===
echo Key outputs:
echo   results\final_summary.csv
echo   data\output\*_c_lex.jsonl
echo   data\output\*_c_sem.jsonl
echo   data\output\clean.jsonl
goto :eof

:error
echo.
echo ERROR: step failed. Check output above.
exit /b 1
