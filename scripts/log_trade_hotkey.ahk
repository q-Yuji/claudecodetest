#Requires AutoHotkey v2.0
#SingleInstance Force

; F11 -- switch to the VS Code window running Claude Code, focus its
; active terminal, and submit the "log that trade" trigger phrase.
;
; Note: F11 is the standard fullscreen-toggle key in Chrome/TradingView and
; VS Code. Binding it here means it stops toggling fullscreen anywhere while
; this script is running.
;
; Fragility note: "Focus Terminal" focuses whichever terminal tab is currently
; active in VS Code. If you keep other terminal tabs open (dev servers, etc.),
; make sure the Claude Code tab is the last one you used, or this will type
; into the wrong terminal.
F11:: {
    if WinExist("claudecodetest ahk_exe Code.exe")
        WinActivate
    else if WinExist("ahk_exe Code.exe")
        WinActivate
    else
        return

    if !WinWaitActive(, , 2)
        return

    Send("^+p")
    Sleep(200)
    SendText("Focus Terminal")
    Sleep(150)
    Send("{Enter}")
    Sleep(250)
    SendText("log that trade")
    Send("{Enter}")
}
