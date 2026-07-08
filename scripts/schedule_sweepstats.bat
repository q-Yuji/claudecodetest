@echo off
rem Register the "SweepStats Daily" scheduled task: every weekday at 18:30
rem (after the 16:00 NY close with margin) run the SweepStats daily pipeline
rem so the dataset compounds with zero manual effort. Re-running this script
rem is safe (/F overwrites the existing task).
rem
rem schtasks has no working-directory option, so the task command cd's into
rem the repo first.

schtasks /Create /TN "SweepStats Daily" /TR "cmd /c cd /d C:\Users\lucap\.vscode-shared\claudecodetest && python -m backtest.daily_stats_run" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:30 /F
