@echo off
title PlayroomHAR Installer
rem This is the file to run. It executes installer-core.ps1 (the actual
rem install logic) - from disk if it is next to this file, otherwise
rem fetched from GitHub.

if exist "%~dp0installer-core.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer-core.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-Expression (Invoke-RestMethod 'https://raw.githubusercontent.com/VictorRyskinB/PlayroomHAR/master/installer-core.ps1')"
)

echo.
pause
