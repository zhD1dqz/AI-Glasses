@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing. Please ask for environment setup.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "tools\annotate_hands.py" %*
set "ANNOTATION_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %ANNOTATION_EXIT%
