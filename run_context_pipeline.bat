@echo off
setlocal

set DATA_FILE=data\contextual_examples.csv
set MODEL_DIR=models\hieroglyph-context-role-logreg
set BATCH_SIZE=8
set CPU_THREADS=2

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python, then reopen Command Prompt and run this file again.
    exit /b 1
)

if not exist "%DATA_FILE%" (
    echo Context data file was not found: %DATA_FILE%
    exit /b 1
)

echo Step 1 of 2: Training the context-aware role classifier...
python -m src.context_train --data-file %DATA_FILE% --model-dir %MODEL_DIR% --batch-size %BATCH_SIZE% --cpu-threads %CPU_THREADS%
if errorlevel 1 exit /b %errorlevel%

echo Step 2 of 2: Starting the contextual Gradio demo...
python app_context.py

endlocal
