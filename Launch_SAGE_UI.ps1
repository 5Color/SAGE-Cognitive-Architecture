$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "========================================"
Write-Host " SAGE Local Control Panel UI"
Write-Host "========================================"
Write-Host ""

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit is not installed."
    Write-Host "Installing Streamlit into the current Python environment..."
    python -m pip install streamlit
}

Write-Host "Starting UI..."
Write-Host "Open: http://127.0.0.1:7860"
python tools\run_sage_ui.py
