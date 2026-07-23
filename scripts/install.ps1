[CmdletBinding()]
param(
    [string]$PythonBin = $env:PYTHON_BIN,
    [switch]$SkipModelWeights
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$SetupScript = Join-Path $ScriptDir "setup.ps1"
Set-Location $ProjectRoot

Write-Host "Installing InsideNeuralNets"
Write-Host "Project: $ProjectRoot"
Write-Host ""

$SetupArguments = @{}
if ($PythonBin) {
    $SetupArguments["PythonBin"] = $PythonBin
}

& $SetupScript @SetupArguments

$VenvPython = if ($IsWindows) {
    Join-Path $ProjectRoot ".venv/Scripts/python.exe"
} else {
    Join-Path $ProjectRoot ".venv/bin/python"
}
if (-not (Test-Path -Path $VenvPython -PathType Leaf)) {
    throw "Virtual environment Python was not found at $VenvPython"
}

if ($SkipModelWeights) {
    Write-Warning "Skipping pretrained model downloads. Live predictions require weights to be cached separately."
} else {
    Write-Host ""
    Write-Host "Downloading and validating pretrained model weights..."
    & $VenvPython scripts/cache_models.py
    if ($LASTEXITCODE -ne 0) {
        throw "One or more pretrained models could not be prepared. Check the network connection and retry."
    }
}

Write-Host ""
Write-Host "InsideNeuralNets is installed."
Write-Host ""
Write-Host "Windowed rehearsal:"
Write-Host "  pwsh -NoProfile -File scripts/run_booth.ps1 -Windowed"
Write-Host ""
Write-Host "Fullscreen booth:"
Write-Host "  pwsh -NoProfile -File scripts/run_booth.ps1"
