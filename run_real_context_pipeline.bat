@echo off
setlocal

set DATA_FILE=data\contextual_examples_real_weak.csv
set MODEL_DIR=models\hieroglyph-context-role-logreg
set BATCH_SIZE=8
set CPU_THREADS=2
set REFRESH=0

if "%~1"=="--refresh" set REFRESH=1
if "%~1"=="/refresh" set REFRESH=1
if not "%~1"=="" if "%REFRESH%"=="0" (
    echo Unknown argument: %~1
    echo Usage:
    echo   run_real_context_pipeline.bat
    echo   run_real_context_pipeline.bat --refresh
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python, then reopen Command Prompt and run this file again.
    exit /b 1
)

if exist "%DATA_FILE%" if "%REFRESH%"=="0" (
    echo Step 1 of 3: Existing weak-labeled real dataset found at %DATA_FILE%. Skipping download and labeling.
    echo To rebuild it, run: run_real_context_pipeline.bat --refresh
) else (
    echo Step 1 of 3: Downloading real corpus data and weak-labeling high-confidence examples...
    if "%REFRESH%"=="1" (
        python -m src.real_context_data --refresh
    ) else (
        python -m src.real_context_data
    )
    if errorlevel 1 exit /b %errorlevel%
)

echo Step 2 of 3: Training the context-aware role classifier from real weak labels...
python -m src.context_train --data-file %DATA_FILE% --model-dir %MODEL_DIR% --batch-size %BATCH_SIZE% --cpu-threads %CPU_THREADS%
if errorlevel 1 exit /b %errorlevel%

echo Step 3 of 3: Starting the contextual Gradio demo...
python app_context.py

endlocal
