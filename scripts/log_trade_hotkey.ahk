#Requires AutoHotkey v2.0
#SingleInstance Force

; Ctrl+Alt+L -- switch to the VS Code window running Claude Code, focus its
; active terminal, and submit the "log that trade" trigger phrase.
;
; Fragility note: "Focus Terminal" focuses whichever terminal tab is currently
; active in VS Code. If you keep other terminal tabs open (dev servers, etc.),
; make sure the Claude Code tab is the last one you used, or this will type
; into the wrong terminal.
^!l:: {
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
