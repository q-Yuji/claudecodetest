@echo off
REM Launch Chrome for a trading session:
REM   - TradingView (tab 1) on debug port 9222  ← MCP connects here
REM   - GEX Suite   (tab 2) on the same port    ← gex.py reads this tab
REM
REM Usage: scripts\launch_trading_chrome.bat
REM Run this once at the start of each session, then log in to both sites.

set PORT=9222
set PROFILE=%USERPROFILE%\chrome-trading

set "CHROME="
if exist "%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"       set "CHROME=%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"
if exist "%PROGRAMFILES(x86)%\Google\Chrome\Application\chrome.exe"  set "CHROME=%PROGRAMFILES(x86)%\Google\Chrome\Application\chrome.exe"
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"        set "CHROME=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"

if "%CHROME%"=="" (
    echo Chrome not found. Install Google Chrome and try again.
    pause
    exit /b 1
)

echo Starting Chrome on debug port %PORT%...
echo Opening TradingView + GEX Suite. Log in to both sites.
echo.
start "" "%CHROME%" --remote-debugging-port=%PORT% --user-data-dir="%PROFILE%" --no-first-run --new-window "https://www.tradingview.com/chart/" "https://gexsuite.com"

echo Done. Chrome is open. Log in to TradingView and GEX Suite.
echo Claude can now see both via the MCP and gex.py.
