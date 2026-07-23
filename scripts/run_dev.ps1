[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$EnvPath = Join-Path $ProjectRoot ".env"
if (Test-Path -Path $EnvPath -PathType Leaf) {
    Get-Content $EnvPath | ForEach-Object {
        $Line = $_.Trim()
        if (-not $Line -or $Line.StartsWith("#") -or -not $Line.Contains("=")) {
            return
        }

        $Parts = $Line.Split("=", 2)
        $Name = $Parts[0].Trim()
        $Value = $Parts[1].Trim().Trim('"').Trim("'")
        if ($Name) {
            Set-Item -Path "Env:$Name" -Value $Value
        }
    }
}

$HostName = if ($env:FRONTEND_HOST) { $env:FRONTEND_HOST } else { "127.0.0.1" }
$Port = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { "3450" }

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

& $PythonBin scripts/stop_dev.py --host $HostName --port $Port --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Previous-process cleanup failed."
}

& $PythonBin -m uvicorn app:app --host $HostName --port $Port --log-level warning --no-access-log
exit $LASTEXITCODE
