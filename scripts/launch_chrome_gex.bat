@echo off
REM Launch Chrome with remote debugging so Claude can screenshot GEX Suite
REM Usage: scripts\launch_chrome_gex.bat

set PORT=9223
set PROFILE=%USERPROFILE%\chrome-debug-gex
set URL=https://gexsuite.com

set "CHROME="
if exist "%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"       set "CHROME=%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"
if exist "%PROGRAMFILES(x86)%\Google\Chrome\Application\chrome.exe"  set "CHROME=%PROGRAMFILES(x86)%\Google\Chrome\Application\chrome.exe"
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"        set "CHROME=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"

if "%CHROME%"=="" (
    echo Chrome not found. Install Google Chrome first.
    pause
    exit /b 1
)

echo Starting Chrome on debug port %PORT%...
echo Navigate to GEX Suite and log in, then Claude can read your heatmap.
echo.
start "" "%CHROME%" --remote-debugging-port=%PORT% --user-data-dir="%PROFILE%" %URL%
