"""situation_room.py — NQ Situation Room (SweepStats product wrapper, v1 draft).

Renders results/situation_room.html: a self-contained dark war-room page
assembling the repo's regenerated JSONs (morning_brief, gex_levels,
session_stats_summary) into one fixed 1440px screen. Read-only over its
inputs; never crashes on a missing/stale/malformed input — panels degrade
to "NO FEED" or dim with a STALE chip instead.

Run:
  python situation_room.py [--png] [--open]

--png renders results/situation_room.png with a fresh headless Chrome
(never the port-9222 trading Chrome). Spec: blueprints/situation-room.md;
the extra "TODAY'S SCRIPT" row comes from the roadmap
(blueprints/situation-room-roadmap.md, feature 1).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import webbrowser
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT_HTML = RESULTS / "situation_room.html"
OUT_PNG = RESULTS / "situation_room.png"

ET = ZoneInfo("America/New_York")
STALE_HOURS = 24.0

CHROME_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / r"Google\Chrome\Application\chrome.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / r"Google\Chrome\Application\chrome.exe",
    Path(os.environ.get("LocalAppData", "")) / r"Google\Chrome\Application\chrome.exe",
]

# ---------------------------------------------------------------- loading

def _age_hours(ts: datetime) -> float:
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now()
    return max(0.0, (now - ts).total_seconds() / 3600.0)


def load(name: str, ts_key: str) -> tuple[dict | None, float | None]:
    """Read a results/ JSON. Returns (data|None, age_hours|None)."""
    try:
        data = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None, None
    except (OSError, ValueError):
        return None, None
    age = None
    raw = data.get(ts_key)
    if isinstance(raw, str):
        try:
            age = _age_hours(datetime.fromisoformat(raw))
        except ValueError:
            pass
    return data, age


# ------------------------------------------------------------- formatting

def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def fnum(x: float, dp: int = 2) -> str:
    return f"{x:,.{dp}f}"


def chg_cls(x: float) -> str:
    return "up" if x > 0 else "dn" if x < 0 else "flat"


def age_chip(age: float | None) -> str:
    if age is None:
        return ""
    if age > STALE_HOURS:
        return f'<span class="chip stale">STALE · {round(age)}H AGO</span>'
    label = "LIVE" if age < 1 else f"{round(age)}H AGO"
    return f'<span class="chip">{label}</span>'


def panel(title: str, body: str, age: float | None = None,
          extra_chip: str = "", cls: str = "") -> str:
    stale = age is not None and age > STALE_HOURS
    return (
        f'<section class="panel {cls}">'
        f'<div class="phead"><span class="ptitle">{esc(title)}</span>'
        f'<span class="chips">{extra_chip}{age_chip(age)}</span></div>'
        f'<div class="pbody{" stale-dim" if stale else ""}">{body}</div>'
        f"</section>"
    )


NOFEED = '<div class="nofeed">NO FEED</div>'

# ----------------------------------------------------------------- panels

def build_tape(brief: dict | None, age: float | None) -> str:
    if brief is None:
        return panel("Tape", NOFEED)
    ctx = brief.get("ctx", {}) or {}
    parts = []
    nq = ctx.get("NQ")
    if isinstance(nq, dict):
        c = float(nq.get("chg_pct") or 0)
        parts.append(
            '<div class="tape-nq mono"><span class="sym">NQ</span>'
            f'<span class="px {chg_cls(c)}">{fnum(float(nq.get("last") or 0))}</span>'
            f'<span class="chg {chg_cls(c)}">{c:+.2f}%</span></div>')
    for sym in ("ES", "SPY", "QQQ", "VIX"):
        q = ctx.get(sym)
        if not isinstance(q, dict):
            continue
        c = float(q.get("chg_pct") or 0)
        parts.append(
            f'<div class="tape-q mono"><span class="sym">{sym}</span>'
            f'<span class="px">{fnum(float(q.get("last") or 0))}</span>'
            f'<span class="chg {chg_cls(c)}">{c:+.2f}%</span></div>')
    chips = []
    for sym, s in (brief.get("sectors") or {}).items():
        if not isinstance(s, dict):
            continue
        c = float(s.get("chg_pct") or 0)
        chips.append(f'<span class="schip mono {chg_cls(c)}">{esc(sym)} {c:+.1f}%</span>')
    body = ('<div class="tape">' + "".join(parts)
            + '<div class="sectors">' + "".join(chips) + "</div></div>")
    return panel("Tape", body, age)


_STRENGTH_RE = re.compile(r"\s*\[(\++)\]\s*")


def _level_cls(name: str) -> str:
    low = name.lower()
    if "call wall" in low:
        return "cwall"
    if "put wall" in low:
        return "pwall"
    if "flip" in low:
        return "flip"
    if low.startswith(("γ", "gamma")):
        return "gamma"
    if low.startswith(("asia", "london")):
        return "amd"
    return "corr"


def _ladder_row(name: str, price: float) -> str:
    m = _STRENGTH_RE.search(name)
    dots = ""
    if m:
        dots = '<span class="dots">' + "●" * len(m.group(1)) + "</span>"
        name = _STRENGTH_RE.sub("", name)
    return (f'<div class="lrow {_level_cls(name)}"><span class="lname">{esc(name)}</span>'
            f'{dots}<span class="lpx mono">{fnum(price)}</span></div>')


# correlated/gamma rows kept nearest to LAST; walls/flips/AMD always shown
LADDER_NEAREST = 16


def build_ladder(gex: dict | None, age: float | None) -> str:
    if gex is None:
        return panel("GEX Level Ladder", NOFEED)
    last = gex.get("current_price")
    last = float(last) if isinstance(last, (int, float)) else None
    rows: list[tuple[float, str, str]] = []  # (price, cls, html)
    for lv in gex.get("levels") or []:
        try:
            p, name = float(lv["price"]), str(lv["name"])
            rows.append((p, _level_cls(_STRENGTH_RE.sub("", name)), _ladder_row(name, p)))
        except (KeyError, TypeError, ValueError):
            continue
    amd = gex.get("session_amd") or {}
    for key, label in (("asia_high", "ASIA HIGH"), ("asia_low", "ASIA LOW"),
                       ("london_high", "LONDON HIGH"), ("london_low", "LONDON LOW")):
        v = amd.get(key)
        if isinstance(v, (int, float)):
            rows.append((float(v), "amd", _ladder_row(label, float(v))))
    if last is not None:
        rows.append((last, "last",
                     f'<div class="lrow last"><span class="lname">► LAST'
                     f'</span><span class="lpx mono">{fnum(last)}</span></div>'))
    rows.sort(key=lambda r: r[0], reverse=True)

    # keep the panel compact: majors always, minors only near price
    keep = {id(r) for r in rows if r[1] not in ("corr", "gamma")}
    if last is not None:
        minors = sorted((r for r in rows if id(r) not in keep),
                        key=lambda r: abs(r[0] - last))
        keep |= {id(r) for r in minors[:LADDER_NEAREST]}
    else:
        keep = {id(r) for r in rows}

    parts, skipping = [], False
    for r in rows:
        if id(r) in keep:
            parts.append(r[2])
            skipping = False
        elif not skipping:
            parts.append('<div class="lrow gap"><span class="lname">···</span></div>')
            skipping = True

    sym = esc(gex.get("symbol") or "")
    body = f'<div class="ladder">{"".join(parts)}</div>'
    return panel(f"GEX Level Ladder · {sym}", body, age)


def _buckets_html(summary: dict | None) -> str:
    out = ""
    for rng, b in ((summary or {}).get("time_buckets") or {}).items():
        if not isinstance(b, dict):
            continue
        out += (
            f'<div class="bucket" data-range="{esc(rng)}">'
            f'<div class="brange mono">{esc(rng)}</div>'
            f'<div class="bfake mono">{float(b.get("fakeout_pct") or 0):.0f}%</div>'
            f'<div class="bsub">n={int(b.get("touches") or 0)} · '
            f'+{float(b.get("fakeout_median_mfe60_pts") or 0):.0f} pts/60m</div></div>')
    if not out:
        return ""
    return ('<div class="subhead bsep">FAKEOUT % BY TIME OF FIRST TOUCH</div>'
            f'<div class="buckets">{out}</div>')


def build_clock(summary: dict | None) -> str:
    rows = "".join(
        f'<div class="crow" id="sess-{k}"><span class="cname">{n}</span>'
        f'<span class="cwin mono">{w}</span>'
        f'<span class="cstate mono" id="state-{k}">—</span>'
        f'<span class="ceta mono" id="eta-{k}"></span></div>'
        for k, n, w in (("asia", "ASIA", "18:00–00:00"),
                        ("london", "LONDON", "02:00–05:00"),
                        ("ny", "NEW YORK", "09:30–16:00")))
    body = (f'<div class="clockbig mono" id="et-big">--:--:--</div>'
            f'<div class="clocksub">EASTERN TIME</div>{rows}'
            + _buckets_html(summary))
    return panel("Session Clock", body)


def _hero_numbers(summary: dict) -> tuple[int, int, int]:
    ft = summary.get("first_touch") or {}
    touched = sum(int(v.get("touched") or 0) for v in ft.values() if isinstance(v, dict))
    fake = sum(int(v.get("fakeout") or 0) for v in ft.values() if isinstance(v, dict))
    pct = round(100.0 * fake / touched) if touched else 0
    return pct, fake, touched


def build_hero(summary: dict | None, age: float | None) -> str:
    if summary is None:
        return panel("SweepStats Hero", NOFEED)
    pct, fake, touched = _hero_numbers(summary)
    s = summary.get("sample") or {}
    body = (
        f'<div class="hero"><div class="heropct mono">{pct}<span>%</span></div>'
        f'<div class="herolabel">of first NY touches of Asia/London levels are fakeouts</div>'
        f'<div class="herosub mono">n={touched} episodes · {esc(s.get("sessions", "?"))} sessions<br>'
        f'{esc(s.get("from", "?"))} → {esc(s.get("to", "?"))}</div></div>')
    return panel("SweepStats", body, age)


_SWEEP_KEYS = {"asia high": "asia_high", "asia low": "asia_low",
               "none": "none", "both": "both"}
_BUCKET_TITLES = {"asia_high": "LONDON SWEEPS ASIA HIGH",
                  "asia_low": "LONDON SWEEPS ASIA LOW",
                  "none": "NO LONDON SWEEP",
                  "both": "LONDON SWEEPS BOTH SIDES"}
_LEVEL_LABELS = {"asia_high": "ASIA HIGH", "asia_low": "ASIA LOW",
                 "london_high": "LONDON HIGH", "london_low": "LONDON LOW"}


def build_script(gex: dict | None, summary: dict | None) -> str:
    amd = (gex or {}).get("session_amd") or {}
    if not amd or summary is None:
        return panel("Today's Script", NOFEED)
    sweep_raw = str(amd.get("london_sweep") or "none")
    key = _SWEEP_KEYS.get(sweep_raw.lower(), "none")
    lm = (summary.get("london_manipulation") or {}).get(key) or {}
    days = int(lm.get("days") or 0)
    up_pct = float(lm.get("ny_up_pct") or 0)
    med = float(lm.get("median_ny_change_pts") or 0)
    low_n = ' <span class="lown">(LOW SAMPLE)</span>' if days < 5 else ""
    bias = str(amd.get("day_bias") or "").upper()
    bias_chip = (f'<span class="chip {"biasup" if bias == "BULLISH" else "biasdn"}">'
                 f'BIAS {esc(bias)}</span>') if bias else ""

    # date chip: flag when the script is from a previous ET trading day
    d = str(amd.get("date") or "")
    date_chip = ""
    if d and d != datetime.now(ET).date().isoformat():
        date_chip = f'<span class="chip stale">SCRIPT FROM {esc(d)}</span>'

    left = (
        f'<div class="scriptpat mono">{esc(_BUCKET_TITLES[key])}</div>'
        f'<div class="scripthist">→ NY closed <span class="{"up" if up_pct >= 50 else "dn"}">'
        f'UP {up_pct:.0f}%</span> of days · median NY move '
        f'<span class="mono {chg_cls(med)}">{med:+.1f} pts</span> · n={days} days{low_n}</div>')

    ft = summary.get("first_touch") or {}
    trows = ""
    for k, label in _LEVEL_LABELS.items():
        price = amd.get(k)
        st = ft.get(k) or {}
        if not isinstance(price, (int, float)) or not st:
            continue
        trows += (
            f'<div class="srow"><span class="slvl">{label}</span>'
            f'<span class="mono spx">{fnum(float(price))}</span>'
            f'<span class="mono sfake">{float(st.get("fakeout_pct") or 0):.0f}% FAKEOUT</span>'
            f'<span class="mono srun">stops run ≈{float(st.get("fakeout_median_overshoot_pts") or 0):.0f} pts past</span>'
            f'<span class="mono sbnc up">bounce +{float(st.get("fakeout_median_mfe60_pts") or 0):.0f} pts/60m</span></div>')
    right = (f'<div class="scriptlvls"><div class="subhead">IF FIRST TOUCH TODAY →</div>'
             f'{trows}</div>')

    n = (summary.get("sample") or {}).get("sessions", "?")
    body = f'<div class="script"><div class="scriptleft">{left}</div>{right}</div>'
    return panel(f"Today's Script — overnight pattern vs {esc(n)}-session history",
                 body, extra_chip=date_chip + bias_chip)


def build_first_touch(summary: dict | None, age: float | None) -> str:
    if summary is None:
        return panel("First-Touch Board", NOFEED)
    rows = ""
    for k, label in _LEVEL_LABELS.items():
        st = (summary.get("first_touch") or {}).get(k)
        if not isinstance(st, dict):
            continue
        pct = float(st.get("fakeout_pct") or 0)
        rows += (
            f'<div class="ftrow"><span class="ftname">{label}</span>'
            f'<span class="ftbar"><span class="ftfill" style="width:{pct:.1f}%"></span></span>'
            f'<span class="mono ftpct">{pct:.0f}%</span>'
            f'<span class="mono ftsub">{int(st.get("touched") or 0)} touches</span>'
            f'<span class="mono ftsub up">+{float(st.get("fakeout_median_mfe60_pts") or 0):.0f} pts/60m</span>'
            f'<span class="mono ftsub dn">risk {float(st.get("fakeout_median_mae120_pts") or 0):.0f} pts</span></div>')
    legend = ('<div class="ftlegend">FAKEOUT % OF FIRST NY TOUCH · MEDIAN REVERSAL '
              '· STOP-RUN RISK (MAE 120M)</div>')
    return panel("First-Touch Board", legend + rows, age)


def build_matrix(summary: dict | None, age: float | None) -> str:
    if summary is None:
        return panel("London Manipulation Matrix", NOFEED)
    cells = ""
    for k, title in _BUCKET_TITLES.items():
        b = (summary.get("london_manipulation") or {}).get(k)
        if not isinstance(b, dict):
            continue
        days = int(b.get("days") or 0)
        up = float(b.get("ny_up_pct") or 0)
        med = float(b.get("median_ny_change_pts") or 0)
        low_n = '<span class="lown"> (LOW SAMPLE)</span>' if days < 5 else ""
        cells += (
            f'<div class="mcell"><div class="mtitle">{title}{low_n}</div>'
            f'<div class="mstats mono"><span>n={days}</span>'
            f'<span class="{"up" if up >= 50 else "dn"}">NY UP {up:.0f}%</span>'
            f'<span class="{chg_cls(med)}">{med:+.1f} pts</span></div></div>')
    return panel("London Manipulation Matrix", f'<div class="matrix">{cells}</div>', age)


def build_wire(brief: dict | None, age: float | None) -> str:
    if brief is None:
        return panel("Wire", NOFEED)
    items = ""
    for n in (brief.get("news") or [])[:6]:
        try:
            hhmm = parsedate_to_datetime(n["date"]).astimezone(ET).strftime("%H:%M")
        except (KeyError, TypeError, ValueError):
            hhmm = "--:--"
        items += (f'<div class="wrow"><span class="mono wtime">{hhmm}</span>'
                  f'<a href="{esc(n.get("url") or "#")}" target="_blank" rel="noopener">'
                  f'{esc(n.get("title") or "")}</a></div>')
    cal = ""
    for ev in brief.get("calendar") or []:
        if not isinstance(ev, dict):
            continue
        cal += (f'<span class="calchip mono">{esc(ev.get("time") or "")} '
                f'{esc(ev.get("title") or "")} · {esc(ev.get("impact") or "")}'
                f'{" · f:" + esc(ev["forecast"]) if ev.get("forecast") else ""}</span>')
    if cal:
        cal = f'<div class="cal">{cal}</div>'
    return panel("Wire", items + cal, age)


# ------------------------------------------------------------------ page

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0d10;color:#c9d4de;font:13px/1.45 "Segoe UI",system-ui,sans-serif;
  background-image:
    repeating-linear-gradient(0deg,rgba(255,255,255,.015) 0 1px,transparent 1px 24px),
    repeating-linear-gradient(90deg,rgba(255,255,255,.015) 0 1px,transparent 1px 24px)}
a{color:#c9d4de;text-decoration:none}a:hover{color:#ffb454}
.mono{font-family:"Cascadia Mono","Consolas",monospace}
.up{color:#3ddc84}.dn{color:#ff5f56}.flat{color:#5c6a78}
.wrap{width:1440px;margin:0 auto;padding:14px;display:flex;flex-direction:column;gap:10px}
.panel{background:#10151b;border:1px solid #1e2833;border-radius:4px;padding:10px 12px}
.phead{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.ptitle{font-family:"Cascadia Mono","Consolas",monospace;font-size:11px;
  text-transform:uppercase;letter-spacing:.12em;color:#5c6a78}
.ptitle::before{content:"▮ ";color:#ffb454}
.chips{display:flex;gap:6px}
.chip{font:10px "Cascadia Mono","Consolas",monospace;letter-spacing:.08em;color:#5c6a78;
  border:1px solid #1e2833;border-radius:3px;padding:2px 6px;white-space:nowrap}
.chip.stale{color:#ffb454;border-color:#ffb454}
.chip.biasup{color:#3ddc84;border-color:#3ddc84}.chip.biasdn{color:#ff5f56;border-color:#ff5f56}
.stale-dim{opacity:.55}
.nofeed{color:#5c6a78;font:18px "Cascadia Mono","Consolas",monospace;
  letter-spacing:.35em;text-align:center;padding:44px 0}
/* header */
header.panel{display:flex;justify-content:space-between;align-items:center;position:relative;
  padding:14px 16px;overflow:hidden}
header.panel::before{content:"";position:absolute;inset:0 0 auto 0;height:5px;
  background:repeating-linear-gradient(90deg,rgba(255,180,84,.35) 0 6px,transparent 6px 12px)}
.wordmark{font:18px "Cascadia Mono","Consolas",monospace;letter-spacing:.35em;color:#c9d4de}
.wordmark b{color:#ffb454;font-weight:normal}
.hclocks{display:flex;gap:22px;font-family:"Cascadia Mono","Consolas",monospace}
.hclocks .lbl{color:#5c6a78;font-size:10px;letter-spacing:.12em;display:block}
.hclocks .val{font-size:17px}
.hright{text-align:right;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.08em}
.hright .n{color:#ffb454}
/* tape */
.tape{display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.tape .sym{color:#5c6a78;font-size:11px;letter-spacing:.12em;margin-right:7px}
.tape-nq .px{font-size:26px}.tape-nq .chg{font-size:14px;margin-left:8px}
.tape-q .px{font-size:15px}.tape-q .chg{font-size:11px;margin-left:6px}
.sectors{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.schip{font-size:10.5px;border:1px solid #1e2833;border-radius:3px;padding:2px 6px}
/* row2 grid */
.row2{display:grid;grid-template-columns:2fr 1fr 1fr;gap:10px;align-items:stretch}
/* ladder */
.ladder{display:flex;flex-direction:column}
.lrow{display:flex;align-items:baseline;gap:8px;padding:1px 6px;border-left:2px solid transparent}
.lrow .lname{color:#5c6a78;font-size:11.5px;letter-spacing:.04em}
.lrow .lpx{margin-left:auto;font-size:12.5px;color:#c9d4de}
.lrow.corr .lpx{color:#5c6a78}
.lrow.gamma{border-left-color:#4dd0e1}.lrow.gamma .lname{color:#4dd0e1}
.lrow.cwall{border-left-color:#3ddc84}.lrow.cwall .lname{color:#3ddc84;font-weight:600}
.lrow.pwall{border-left-color:#ff5f56}.lrow.pwall .lname{color:#ff5f56;font-weight:600}
.lrow.flip{border-left-color:#ffb454}.lrow.flip .lname{color:#ffb454;font-weight:600}
.lrow.amd{border-left-color:#4dd0e1}.lrow.amd .lname{color:#4dd0e1;font-weight:600}
.lrow.amd .lpx{color:#4dd0e1}
.lrow.last{background:#1a2129;border:1px solid #ffb454;border-radius:3px;margin:2px 0;padding:3px 6px}
.lrow.last .lname{color:#ffb454;font-weight:600}.lrow.last .lpx{color:#ffb454;font-size:14px}
.lrow.gap .lname{color:#2b3844;letter-spacing:4px}
.dots{color:#ffb454;font-size:9px;letter-spacing:2px}
/* clock */
.clockbig{font-size:34px;text-align:center;margin-top:6px}
.clocksub{text-align:center;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.3em;margin-bottom:16px}
.crow{display:flex;align-items:baseline;gap:8px;padding:9px 4px;border-top:1px solid #1e2833}
.cname{font:12px "Cascadia Mono","Consolas",monospace;letter-spacing:.12em}
.cwin{color:#5c6a78;font-size:10.5px}
.cstate{margin-left:auto;font-size:11px;letter-spacing:.1em}
.cstate.open{color:#3ddc84}.cstate.closed{color:#5c6a78}
.ceta{color:#5c6a78;font-size:10.5px}
/* row2 panels stretch; hero centers in its column */
.row2 .panel{display:flex;flex-direction:column}
.row2 .pbody{flex:1;display:flex;flex-direction:column}
.row2 .pbody>.hero{margin:auto 0}
/* hero */
.hero{text-align:center;padding-top:10px}
.heropct{font-size:88px;color:#ffb454;line-height:1}
.heropct span{font-size:40px}
.herolabel{margin:10px auto 8px;max-width:230px;color:#c9d4de;font-size:13px}
.herosub{color:#5c6a78;font-size:10.5px;letter-spacing:.05em}
/* script */
.script{display:grid;grid-template-columns:1fr 1.4fr;gap:18px}
.scriptpat{font-size:22px;letter-spacing:.14em;color:#ffb454;margin:4px 0 8px}
.scripthist{font-size:13px;color:#c9d4de}
.lown{color:#ff5f56;font-size:10px;letter-spacing:.08em}
.subhead{font:10px "Cascadia Mono","Consolas",monospace;color:#5c6a78;
  letter-spacing:.14em;margin-bottom:5px}
.srow{display:flex;gap:14px;align-items:baseline;padding:3px 0;border-top:1px solid #1e2833;font-size:12px}
.srow .slvl{width:96px;color:#4dd0e1;font:11px "Cascadia Mono","Consolas",monospace;letter-spacing:.06em}
.srow .spx{width:80px;color:#c9d4de}
.srow .sfake{width:104px;color:#ffb454}
.srow .srun{color:#5c6a78}
.srow .sbnc{margin-left:auto}
.bsep{margin-top:16px}
.buckets{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.bucket{border:1px solid #1e2833;border-radius:3px;padding:6px 8px;text-align:center}
.bucket.active{border-color:#ffb454}
.bucket.active .brange{color:#ffb454}
.brange{font-size:10px;color:#5c6a78;letter-spacing:.1em}
.bfake{font-size:20px;color:#c9d4de}
.bsub{font-size:10px;color:#5c6a78}
/* row4 */
.row4{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.ftlegend{font:9.5px "Cascadia Mono","Consolas",monospace;color:#5c6a78;
  letter-spacing:.08em;margin-bottom:6px}
.ftrow{display:flex;align-items:center;gap:10px;padding:7px 0;border-top:1px solid #1e2833}
.ftname{width:100px;font:11px "Cascadia Mono","Consolas",monospace;
  color:#4dd0e1;letter-spacing:.06em}
.ftbar{flex:1;height:8px;background:#1e2833;border-radius:2px;overflow:hidden}
.ftfill{display:block;height:100%;background:#ffb454}
.ftpct{width:44px;text-align:right;color:#ffb454;font-size:13px}
.ftsub{width:92px;text-align:right;color:#5c6a78;font-size:10.5px}
.matrix{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.mcell{border:1px solid #1e2833;border-radius:3px;padding:9px 10px}
.mtitle{font:10.5px "Cascadia Mono","Consolas",monospace;color:#5c6a78;letter-spacing:.08em;margin-bottom:7px}
.mstats{display:flex;justify-content:space-between;font-size:13px}
/* wire */
.wrow{display:flex;gap:12px;padding:4px 0;border-top:1px solid #1e2833;font-size:12.5px}
.wrow:first-child{border-top:none}
.wtime{color:#ffb454;font-size:11px}
.cal{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}
.calchip{font-size:10.5px;color:#4dd0e1;border:1px solid #1e2833;border-radius:3px;padding:2px 8px}
/* footer */
footer{display:flex;justify-content:space-between;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.08em;padding:2px 4px 10px}
footer .amber{color:#ffb454}
"""

JS = """
function etParts(){
  var p=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour12:false,
    hour:'2-digit',minute:'2-digit',second:'2-digit'}).formatToParts(new Date());
  var o={};p.forEach(function(x){o[x.type]=x.value});
  if(o.hour==='24')o.hour='00';return o;}
function utcStr(){
  return new Intl.DateTimeFormat('en-GB',{timeZone:'UTC',hour12:false,
    hour:'2-digit',minute:'2-digit',second:'2-digit'}).format(new Date());}
function fmtEta(mins){var h=Math.floor(mins/60),m=mins%60;
  return (h?h+'H ':'')+m+'M';}
var SESS=[['asia',1080,1440],['london',120,300],['ny',570,960]];
function tick(){
  var o=etParts(),et=o.hour+':'+o.minute+':'+o.second;
  var els=document.querySelectorAll('.js-et');for(var i=0;i<els.length;i++)els[i].textContent=et;
  var b=document.getElementById('et-big');if(b)b.textContent=et;
  var u=document.getElementById('hdr-utc');if(u)u.textContent=utcStr();
  var mins=parseInt(o.hour,10)*60+parseInt(o.minute,10);
  SESS.forEach(function(s){
    var open=mins>=s[1]&&mins<s[2];
    var st=document.getElementById('state-'+s[0]),eta=document.getElementById('eta-'+s[0]);
    if(!st)return;
    st.textContent=open?'OPEN':'CLOSED';st.className='cstate mono '+(open?'open':'closed');
    eta.textContent=open?('closes '+fmtEta(s[2]-mins)):('opens '+fmtEta((s[1]-mins+1440)%1440));
  });
  var bks=document.querySelectorAll('.bucket');
  for(var j=0;j<bks.length;j++){
    var r=bks[j].getAttribute('data-range').split('-');
    var a=r[0].split(':'),z=r[1].split(':');
    var lo=parseInt(a[0],10)*60+parseInt(a[1],10),hi=parseInt(z[0],10)*60+parseInt(z[1],10);
    bks[j].className='bucket'+((mins>=lo&&mins<hi)?' active':'');
  }
}
tick();setInterval(tick,1000);
"""


def build_page(brief, brief_age, gex, gex_age, summary, summary_age) -> str:
    n_sessions = "—"
    if summary:
        n_sessions = str((summary.get("sample") or {}).get("sessions", "—"))
    built = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    header = (
        '<header class="panel">'
        '<div class="wordmark">NQ <b>SITUATION ROOM</b></div>'
        '<div class="hclocks">'
        '<div><span class="lbl">UTC</span><span class="val" id="hdr-utc">--:--:--</span></div>'
        '<div><span class="lbl">NEW YORK</span><span class="val js-et">--:--:--</span></div>'
        "</div>"
        f'<div class="hright">SWEEPSTATS DATA · <span class="n">n={esc(n_sessions)} SESSIONS</span>'
        f"<br>BUILD {esc(built)}</div>"
        "</header>")

    def src_age(label: str, age: float | None) -> str:
        if age is None:
            return f'{label} <span class="amber">NO FEED</span>'
        if age > STALE_HOURS:
            return f'{label} <span class="amber">{round(age)}H STALE</span>'
        return f"{label} {round(age)}H"

    ages = " · ".join((src_age("BRIEF", brief_age), src_age("GEX", gex_age),
                       src_age("STATS", summary_age)))
    footer = (
        "<footer>"
        f"<span>{ages}</span>"
        "<span>Past frequency ≠ future probability. Not financial advice.</span>"
        "<span>SweepStats — NQ session-level statistics</span>"
        "</footer>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>NQ Situation Room</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        + header
        + build_tape(brief, brief_age)
        + '<div class="row2">'
        + build_ladder(gex, gex_age)
        + build_clock(summary)
        + build_hero(summary, summary_age)
        + "</div>"
        + build_script(gex, summary)
        + '<div class="row4">'
        + build_first_touch(summary, summary_age)
        + build_matrix(summary, summary_age)
        + "</div>"
        + build_wire(brief, brief_age)
        + footer
        + f"<script>{JS}</script></div></body></html>")


# ------------------------------------------------------------------- png

def _find_chrome() -> Path | None:
    for p in CHROME_CANDIDATES:
        if p.is_file():
            return p
    return None


def render_png() -> bool:
    """Render with a fresh headless Chrome — never the port-9222 trading one."""
    chrome = _find_chrome()
    if chrome is None:
        print("  WARNING: chrome.exe not found — skipping PNG render")
        return False
    subprocess.run(
        [str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--screenshot={OUT_PNG.resolve()}", "--window-size=1440,1680",
         OUT_HTML.resolve().as_uri()],
        check=True, capture_output=True, timeout=120)
    return True


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Render the NQ Situation Room page")
    ap.add_argument("--png", action="store_true", help="also render situation_room.png")
    ap.add_argument("--open", action="store_true", help="open the HTML in a browser")
    args = ap.parse_args()

    brief, brief_age = load("morning_brief.json", "generated")
    gex, gex_age = load("gex_levels.json", "timestamp")
    summary, summary_age = load("session_stats_summary.json", "generated")

    OUT_HTML.write_text(
        build_page(brief, brief_age, gex, gex_age, summary, summary_age),
        encoding="utf-8")
    print(f"HTML -> {OUT_HTML}")

    if args.png and render_png():
        print(f"PNG  -> {OUT_PNG}")
    if args.open:
        webbrowser.open(OUT_HTML.resolve().as_uri())


if __name__ == "__main__":
    main()
