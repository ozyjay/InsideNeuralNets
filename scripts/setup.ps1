[CmdletBinding()]
param(
    [string]$PythonBin = $env:PYTHON_BIN
)

$ErrorActionPreference = "Stop"

if (-not $PythonBin) {
    $PythonCommand = Get-Command python3, python, py -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $PythonCommand) {
        throw "Python was not found. Install Python or add it to PATH."
    }
    $PythonBin = $PythonCommand.Source
}

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = if ($IsWindows) {
    Join-Path $VenvDir "Scripts/python.exe"
} else {
    Join-Path $VenvDir "bin/python"
}

Set-Location $ProjectRoot

Write-Host "Setting up InsideNeuralNets"
Write-Host "Project: $ProjectRoot"
Write-Host "Virtual environment: $VenvDir"

if ($env:VIRTUAL_ENV -and ($env:VIRTUAL_ENV -ne $VenvDir)) {
    Write-Warning "Another virtual environment is active: $($env:VIRTUAL_ENV)"
    Write-Host "Deactivate it first if this is not intentional. Continuing with project .venv."
}

if (-not (Test-Path -Path $VenvDir -PathType Container)) {
    Write-Host "Creating local virtual environment..."
    & $PythonBin -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment with '$PythonBin'."
    }
} else {
    Write-Host "Using existing local virtual environment."
}

if (-not (Test-Path -Path $VenvPython -PathType Leaf)) {
    throw "Virtual environment Python was not found at $VenvPython"
}

# Safety check: all installs below must go into the project venv, not global/user site-packages.
$PipPrefix = & $VenvPython -m pip --version
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect pip inside the virtual environment."
}
Write-Host "Using pip: $PipPrefix"

if (-not ($PipPrefix -like "*$VenvDir*")) {
    throw "pip does not appear to point inside $VenvDir. Aborting to avoid global installs."
}

Write-Host "Upgrading pip inside the virtual environment..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

Write-Host "Installing project requirements inside .venv..."
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

Write-Host "Running local setup check..."
& $VenvPython scripts/check_setup.py
if ($LASTEXITCODE -ne 0) {
    throw "Setup check failed."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Activate the environment with:"
Write-Host "  . .venv/bin/Activate.ps1"
Write-Host ""
Write-Host "Run the demo with:"
Write-Host "  pwsh -NoProfile -File scripts/run_dev.ps1"
Write-Host ""
Write-Host "Run the fullscreen booth with:"
Write-Host "  pwsh -NoProfile -File scripts/run_booth.ps1"
Write-Host ""
Write-Host "Or explicitly:"
Write-Host "  $VenvPython -m uvicorn app:app --host 127.0.0.1 --port 3450 --log-level warning --no-access-log"
