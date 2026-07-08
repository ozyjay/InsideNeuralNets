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

$VenvPython = Join-Path $ProjectRoot ".venv/bin/python"
if (Test-Path -Path $VenvPython -PathType Leaf) {
    $PythonBin = $VenvPython
} else {
    $PythonBin = "python3"
}

& $PythonBin scripts/stop_dev.py --host $HostName --port $Port --project-root $ProjectRoot
exit $LASTEXITCODE
