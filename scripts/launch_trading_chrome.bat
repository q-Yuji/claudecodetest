@echo off
REM Launch everything needed for a trading session:
REM   1. IBKR Client Portal Gateway  (localhost:5000)
REM   2. Chrome with TradingView + GEX Suite on debug port 9222
REM
REM Usage: scripts\launch_trading_chrome.bat
REM After it opens: log in to IBKR at https://localhost:5000, then log in to TV + GEX Suite.

set PORT=9222
set PROFILE=%USERPROFILE%\chrome-trading
set JAVA_HOME=C:\java21\jdk-21.0.11+10-jre
set PATH=%JAVA_HOME%\bin;%PATH%

set "CHROME="
if exist "%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"       set "CHROME=%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"
if exist "%PROGRAMFILES(x86)%\Google\Chrome\Application\chrome.exe"  set "CHROME=%PROGRAMFILES(x86)%\Google\Chrome\Application\chrome.exe"
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"        set "CHROME=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"

if "%CHROME%"=="" (
    echo Chrome not found. Install Google Chrome and try again.
    pause
    exit /b 1
)

echo [1/2] Starting IBKR Client Portal Gateway on localhost:5000...
start "IBKR Gateway" /D "C:\ibkr-gateway" cmd /c "bin\run.bat root\conf.yaml"
echo     Open https://localhost:5000 in Chrome and log in with your IBKR credentials.
echo.

echo [2/2] Starting Chrome on debug port %PORT%...
echo     Log in to TradingView and GEX Suite when it opens.
echo.
start "" "%CHROME%" --remote-debugging-port=%PORT% --remote-allow-origins=http://localhost:%PORT% --user-data-dir="%PROFILE%" --no-first-run --new-window "https://localhost:5000" "https://www.tradingview.com/chart/" "https://gexsuite.com"

echo Done. Log in to IBKR (first tab), then TradingView and GEX Suite.
echo Run python morning_brief.py when ready.
