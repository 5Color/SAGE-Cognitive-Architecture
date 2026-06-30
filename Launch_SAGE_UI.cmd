@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  SAGE Local Control Panel UI
echo ========================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Streamlit is not installed.
    echo Installing Streamlit into the current Python environment...
    python -m pip install streamlit
    if errorlevel 1 (
        echo Failed to install Streamlit.
        echo Please run: python -m pip install streamlit
        pause
        exit /b 1
    )
)

echo Starting UI...
echo Open: http://127.0.0.1:7860
python tools\run_sage_ui.py
pause
endlocal
