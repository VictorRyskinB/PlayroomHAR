# PlayroomHAR installer / updater — internal logic.
# Do not run this file directly: double-click install.bat instead.
#
# Installs the Playroom Action Annotator into %LOCALAPPDATA%\PlayroomHAR:
#   1. Finds Python 3.9+ (installs Python 3.12 via winget if missing)
#   2. Downloads the latest code from GitHub (master branch)
#   3. Creates a virtual environment and installs requirements
#   4. Pre-downloads the default YOLO model so first run works offline
#   5. Creates a desktop shortcut
#
# Safe to re-run: an existing install is updated in place (venv is kept).

$ErrorActionPreference = "Stop"

$RepoOwner  = "VictorRyskinB"
$RepoName   = "PlayroomHAR"
$Branch     = "master"
$ZipUrl     = "https://github.com/$RepoOwner/$RepoName/archive/refs/heads/$Branch.zip"
$InstallDir = Join-Path $env:LOCALAPPDATA "PlayroomHAR"

Write-Host ""
Write-Host "=== PlayroomHAR installer ===" -ForegroundColor Cyan
Write-Host "Install location: $InstallDir"
Write-Host ""

# ---------------------------------------------------------------- Python ----
$VersionProbe = "import sys;print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))"

function Test-PythonCommand([string]$Command) {
    # Returns the command if it runs Python >= 3.9, otherwise $null.
    # stderr is redirected at the cmd.exe level to avoid PowerShell error records.
    $v = cmd /c "$Command -c `"$VersionProbe`" 2>nul"
    if ($LASTEXITCODE -ne 0 -or -not $v) { return $null }
    $parts = "$v".Trim() -split "\."
    if ($parts.Count -lt 2) { return $null }
    $major = [int]$parts[0]; $minor = [int]$parts[1]
    if ($major -eq 3 -and $minor -ge 9) { return $Command }
    return $null
}

function Find-Python {
    foreach ($cand in @("py -3.12", "py -3.11", "py -3.13", "py -3", "python")) {
        $found = Test-PythonCommand $cand
        if ($found) { return $found }
    }
    # Direct paths, for the case where Python was just installed and PATH
    # has not refreshed in this session.
    foreach ($ver in @("312", "311", "313", "310")) {
        $direct = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$ver\python.exe"
        if (Test-Path $direct) {
            $found = Test-PythonCommand "`"$direct`""
            if ($found) { return $found }
        }
    }
    return $null
}

Write-Host "[1/5] Looking for Python 3.9+ ..."
$Python = Find-Python
if (-not $Python) {
    Write-Host "      Python not found. Installing Python 3.12 via winget (this may take a few minutes)..."
    winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    $Python = Find-Python
    if (-not $Python) {
        throw "Python was installed but could not be located. Please close this window, re-run install.bat, and if it still fails install Python 3.12 manually from python.org."
    }
}
Write-Host "      Using: $Python"

# ------------------------------------------------------------ Download ------
Write-Host "[2/5] Downloading latest PlayroomHAR code from GitHub ..."
$TmpZip = Join-Path $env:TEMP "PlayroomHAR-src.zip"
$TmpDir = Join-Path $env:TEMP "PlayroomHAR-src"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $ZipUrl -OutFile $TmpZip -UseBasicParsing
if (Test-Path $TmpDir) { Remove-Item -Recurse -Force $TmpDir }
Expand-Archive -Path $TmpZip -DestinationPath $TmpDir
$SrcDir = Join-Path $TmpDir "$RepoName-$Branch"
if (-not (Test-Path (Join-Path $SrcDir "main.py"))) {
    throw "Downloaded archive does not look like PlayroomHAR (main.py missing)."
}
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Path (Join-Path $SrcDir "*") -Destination $InstallDir -Recurse -Force
Remove-Item -Force $TmpZip
Remove-Item -Recurse -Force $TmpDir
Write-Host "      Code copied to $InstallDir"

# ---------------------------------------------------------------- Venv ------
Write-Host "[3/5] Setting up Python environment (first install downloads ~1 GB, please wait) ..."
$VenvDir = Join-Path $InstallDir ".venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    cmd /c "$Python -m venv `"$VenvDir`""
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
}
& $VenvPy -m pip install --upgrade pip --quiet
& $VenvPy -m pip install -r (Join-Path $InstallDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed. Check your internet connection and re-run install.bat." }

# ------------------------------------------------------------- Model --------
Write-Host "[4/5] Pre-downloading the default YOLO model ..."
Push-Location $InstallDir
try {
    & $VenvPy -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    if ($LASTEXITCODE -ne 0) { Write-Host "      (Model pre-download failed - the app will download it on first use instead.)" }
} finally {
    Pop-Location
}

# ---------------------------------------------------- Shortcut + launcher ---
Write-Host "[5/5] Creating desktop shortcut ..."

$DebugBat = Join-Path $InstallDir "PlayroomHAR-debug.bat"
@"
@echo off
rem Runs PlayroomHAR with a visible console so errors can be read.
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py
pause
"@ | Out-File -FilePath $DebugBat -Encoding ascii

$WshShell = New-Object -ComObject WScript.Shell
$Desktop  = $WshShell.SpecialFolders.Item("Desktop")
$Shortcut = $WshShell.CreateShortcut((Join-Path $Desktop "PlayroomHAR.lnk"))
$Shortcut.TargetPath       = Join-Path $VenvDir "Scripts\pythonw.exe"
$Shortcut.Arguments        = "main.py"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description      = "Playroom Action Annotator"
$Shortcut.Save()

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "Launch PlayroomHAR from the desktop shortcut."
Write-Host "If the app does not open, run PlayroomHAR-debug.bat in $InstallDir to see the error."
Write-Host "To update to the latest version later, just run install.bat again."
