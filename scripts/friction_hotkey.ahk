#Requires AutoHotkey v2.0
#SingleInstance Force

; F8 -- log an "ugh, this again" friction moment from any app: pops a small
; input box (does NOT switch windows -- unlike the F11 trade-logging hotkey,
; capture happens right where you are), and appends one line to
; journal/frictions.md with the timestamp and the active window title.
; Esc or an empty box logs nothing. Part of the one-week friction-journal
; product-discovery experiment.
;
; Install: create a shortcut to this script in the Startup folder
; (Win+R -> shell:startup), same as scripts/log_trade_hotkey.ahk (F11).
F8:: {
    title := WinGetTitle("A")
    if (StrLen(title) > 60)
        title := SubStr(title, 1, 60)

    ib := InputBox("What's the friction? (one line)", "Friction journal", "w520 h130")
    if (ib.Result != "OK" || Trim(ib.Value) = "")
        return

    line := "- " . FormatTime(, "yyyy-MM-dd HH:mm") . " | [" . title . "] | "
        . Trim(ib.Value) . "`n"
    FileAppend(line, "C:\Users\lucap\.vscode-shared\claudecodetest\journal\frictions.md", "UTF-8-RAW")
}
