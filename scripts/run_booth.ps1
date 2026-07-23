[CmdletBinding()]
param(
    [switch]$Windowed,
    [string]$Browser,
    [string]$ListenHost,
    [ValidateRange(1, 65535)]
    [int]$Port
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$VenvPythonCandidates = @(
    (Join-Path $ProjectRoot ".venv/Scripts/python.exe"),
    (Join-Path $ProjectRoot ".venv/bin/python")
)
$PythonBin = $VenvPythonCandidates |
    Where-Object { Test-Path -Path $_ -PathType Leaf } |
    Select-Object -First 1

if (-not $PythonBin) {
    $PythonCommand = Get-Command python3, python, py -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $PythonCommand) {
        throw "Python was not found. Run scripts/setup.ps1 first or add Python to PATH."
    }
    $PythonBin = $PythonCommand.Source
}

$BoothArguments = @("scripts/run_booth.py")
if ($Windowed) {
    $BoothArguments += "--windowed"
}
if ($Browser) {
    $BoothArguments += @("--browser", $Browser)
}
if ($ListenHost) {
    $BoothArguments += @("--host", $ListenHost)
}
if ($PSBoundParameters.ContainsKey("Port")) {
    $BoothArguments += @("--port", $Port)
}

& $PythonBin @BoothArguments
exit $LASTEXITCODE
