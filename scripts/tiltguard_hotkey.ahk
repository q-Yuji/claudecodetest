#Requires AutoHotkey v2.0
#SingleInstance Force

; F9 -- arm a TiltGuard cooldown: launches the fullscreen countdown overlay
; (python -m tiltguard.main arm) on all monitors with your rules text on it.
; Press it the moment you get stopped out, before the revenge trade forms.
;
; Install: create a shortcut to this script in the Startup folder
; (Win+R -> shell:startup), same as scripts/log_trade_hotkey.ahk (F11).
;
; The overlay is dismissible only by its countdown ending or by typing the
; unlock sentence from tiltguard/config.json. Duration/rules live there too.
;
; F9 was chosen because F11 is taken by the trade-logging hotkey and
; F10/F12 have common conflicts (menu focus / browser dev tools).
F9:: {
    Run('python -m tiltguard.main arm', 'C:\Users\lucap\.vscode-shared\claudecodetest', 'Hide')
}
