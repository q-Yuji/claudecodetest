"""Fullscreen cooldown overlay for TiltGuard.

Covers every attached monitor with a topmost, near-black countdown window
showing the user's own rules text. Dismissible only by the countdown
reaching zero or by typing the exact unlock sentence into the entry box --
not by a click, not by Escape.

This is a friction device, not a jail: it never intercepts Ctrl+Alt+Del,
never grabs the keyboard globally, and the process can always be killed
from Task Manager.
"""

import ctypes
import time
import tkinter as tk
from ctypes import wintypes

BG = "#0d0d0d"
FG = "#e8e8e8"
DIM = "#8a8a8a"
ACCENT = "#ff5c5c"


def _set_dpi_aware() -> None:
    # EnumDisplayMonitors returns physical pixels; without DPI awareness
    # tkinter geometry is in scaled pixels and the overlay would leave a
    # strip of screen uncovered on DPI-scaled monitors.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _monitor_rects() -> list[tuple[int, int, int, int]]:
    """Return (left, top, width, height) for every attached display."""
    rects: list[tuple[int, int, int, int]] = []
    proc_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def _collect(_hmon, _hdc, lprect, _lparam):
        r = lprect.contents
        rects.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, proc_type(_collect), 0)
    return rects or [(0, 0, 1920, 1080)]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def run_overlay(minutes: float, rules_text: list[str], unlock_sentence: str) -> None:
    """Show the overlay on all monitors; block until countdown ends or unlock."""
    _set_dpi_aware()
    root = tk.Tk()
    root.withdraw()

    deadline = time.monotonic() + minutes * 60.0
    target = _normalize(unlock_sentence)
    windows: list[tk.Toplevel] = []
    countdown_labels: list[tk.Label] = []
    first_entry: tk.Entry | None = None

    def try_unlock(entry: tk.Entry) -> None:
        if _normalize(entry.get()) == target:
            root.destroy()
        else:
            entry.delete(0, tk.END)

    for x, y, w, h in _monitor_rects():
        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.configure(bg=BG)
        win.attributes("-topmost", True)
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        windows.append(win)

        tk.Label(win, text="COOLDOWN", font=("Segoe UI", 22, "bold"),
                 fg=DIM, bg=BG).pack(pady=(int(h * 0.08), 0))
        lbl = tk.Label(win, text="--:--", font=("Segoe UI", 110, "bold"),
                       fg=ACCENT, bg=BG)
        lbl.pack(pady=(8, int(h * 0.04)))
        countdown_labels.append(lbl)

        for line in rules_text:
            tk.Label(win, text=line, font=("Segoe UI", 24),
                     fg=FG, bg=BG).pack(pady=5)

        tk.Label(win, text="Unlock early by typing, exactly:",
                 font=("Segoe UI", 13), fg=DIM, bg=BG).pack(pady=(int(h * 0.05), 2))
        tk.Label(win, text=f'"{unlock_sentence}"',
                 font=("Segoe UI", 13, "italic"), fg=DIM, bg=BG).pack(pady=(0, 8))
        entry = tk.Entry(win, font=("Segoe UI", 15), width=70, justify="center",
                         bg="#1a1a1a", fg=FG, insertbackground=FG, relief="flat")
        entry.pack(ipady=7)
        entry.bind("<Return>", lambda _e, e=entry: try_unlock(e))
        if first_entry is None:
            first_entry = entry

    def tick() -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            root.destroy()
            return
        m, s = divmod(int(remaining + 0.999), 60)
        for lbl in countdown_labels:
            lbl.config(text=f"{m:02d}:{s:02d}")
        root.after(200, tick)

    def reassert_topmost() -> None:
        # TradingView also uses topmost windows; keep winning every 2s.
        for win in windows:
            win.lift()
            win.attributes("-topmost", True)
        root.after(2000, reassert_topmost)

    windows[0].focus_force()
    if first_entry is not None:
        first_entry.focus_set()
    tick()
    reassert_topmost()
    root.mainloop()
