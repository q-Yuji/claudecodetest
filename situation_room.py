"""situation_room.py — NQ Situation Room (SweepStats product wrapper, v2 UI).

Renders results/situation_room.html: a self-contained dark war-room page
assembling the repo's regenerated JSONs (morning_brief, gex_levels,
session_stats_summary) into one fixed 1440px screen. Read-only over its
inputs; never crashes on a missing/stale/malformed input — panels degrade
to "NO FEED" or dim with a STALE chip instead.

v2 UI (2026-07-10): SitDeck-style depth and interactivity — collapsible
panels, clickable ladder rows that open per-level drawers (stats for
session levels, interpretation for GEX level types), tabbed stats deck,
ring-gauge hero, 24h session timeline, wire show-more. Everything is
inline vanilla JS; opening the HTML with #png in the URL expands all
sections for screenshot capture.

Run:
  python situation_room.py [--png] [--open] [--public]

--public renders the product edition (results/situation_room_public.*):
price-derived session levels only, zero GEX Suite data.
Spec: blueprints/situation-room.md + blueprints/situation-room-roadmap.md.
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
          extra_chip: str = "", cls: str = "", nofeed: bool = False,
          collapsible: bool = True) -> str:
    """Panel card. LED reflects feed state; header click collapses."""
    if nofeed:
        led, body = "red", '<div class="nofeed">NO FEED</div>'
    elif age is not None and age > STALE_HOURS:
        led = "amber"
    else:
        led = "green"
    stale = age is not None and age > STALE_HOURS
    head_cls = "phead tgl" if collapsible else "phead"
    chev = '<span class="chev">▾</span>' if collapsible else ""
    return (
        f'<section class="panel {cls}">'
        f'<div class="{head_cls}"><span class="ptitle"><span class="led {led}"></span>'
        f'{esc(title)}</span>'
        f'<span class="chips">{extra_chip}{age_chip(age)}{chev}</span></div>'
        f'<div class="pbody{" stale-dim" if stale else ""}">{body}</div>'
        f"</section>"
    )


# ----------------------------------------------------------------- panels

def build_tape(brief: dict | None, age: float | None) -> str:
    if brief is None:
        return panel("Tape", "", nofeed=True)
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

# Interpretation drawers for GEX level types (from the GEX Suite reference)
_TYPE_INFO = [
    ("call wall 0dte", "Same-day-expiry call wall — highly reactive intraday "
                       "magnet/barrier. Resets each session."),
    ("put wall 0dte", "Same-day-expiry put wall — highly reactive intraday "
                      "floor. Resets each session."),
    ("gamma flip 0dte", "Same-day regime line — the most reactive chop/trend "
                        "boundary intraday."),
    ("call wall", "Highest call-side gamma concentration. Dealer hedging sells "
                  "into rallies here; a clean break flips it to support."),
    ("put wall", "Highest put-side gamma concentration — the structural floor. "
                 "A failed put wall in bearish tape accelerates downside."),
    ("gamma flip", "Regime line. Above: positive gamma — chop, mean reversion, "
                   "fade the edges. Below: negative gamma — trend, momentum, "
                   "breakdowns accelerate."),
]


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


def _type_drawer(name: str, strength: int) -> str:
    low = name.lower()
    text = ""
    for key, desc in _TYPE_INFO:
        if key in low:
            text = desc
            break
    else:
        if low.startswith(("γ", "gamma")):
            text = ("Ranked gamma reaction zone — Level 1 strongest. Price "
                    "tends to step between these levels.")
        elif low.startswith("qqq"):
            text = ("Correlated level projected from QQQ options onto NQ. "
                    "Strongest at confluence with gamma levels.")
    if strength:
        text += (f' Flagged {"+" * strength} — indicator strength marker; '
                 f'+++ is maximum conviction.')
    return text


_LEVEL_LABELS = {"asia_high": "ASIA HIGH", "asia_low": "ASIA LOW",
                 "london_high": "LONDON HIGH", "london_low": "LONDON LOW"}


def _stats_drawer(key: str, ft: dict) -> str:
    st = ft.get(key) or {}
    if not st:
        return ""
    cells = (
        ("TOUCHED", f'{int(st.get("touched") or 0)}/{int(st.get("days_in_sample") or 0)} days'),
        ("FAKEOUT", f'{float(st.get("fakeout_pct") or 0):.0f}%'),
        ("STOP-RUN DEPTH", f'≈{float(st.get("fakeout_median_overshoot_pts") or 0):.0f} pts past'),
        ("MED. BOUNCE 60M", f'+{float(st.get("fakeout_median_mfe60_pts") or 0):.0f} pts'),
        ("MED. BOUNCE 120M", f'+{float(st.get("fakeout_median_mfe120_pts") or 0):.0f} pts'),
        ("ADVERSE 120M", f'{float(st.get("fakeout_median_mae120_pts") or 0):.0f} pts'),
        ("BOUNCE ≥30PTS", f'{float(st.get("fakeout_mfe60_ge30pts_pct") or 0):.0f}% of fakeouts'),
        ("IF IT BREAKS", f'+{float(st.get("break_median_mfe60_pts") or 0):.0f} pts/60m'),
    )
    grid = "".join(f'<div class="dcell"><span>{k}</span>'
                   f'<b class="mono">{v}</b></div>' for k, v in cells)
    return f'<div class="dgrid">{grid}</div>'


class _Drawers:
    """Allocates unique drawer ids and pairs rows with hidden detail divs."""

    def __init__(self) -> None:
        self.n = 0

    def row(self, row_html_open: str, inner: str, drawer: str) -> str:
        if not drawer:
            return row_html_open.replace("{D}", "") + inner + "</div>"
        self.n += 1
        did = f"d{self.n}"
        opener = row_html_open.replace("{D}", f' data-d="{did}"')
        return (opener + f'<span class="rchev">▸</span>{inner}</div>'
                + f'<div class="drawer" id="{did}" hidden>{drawer}</div>')


LADDER_NEAREST = 16


def build_ladder(gex: dict | None, summary: dict | None,
                 age: float | None) -> str:
    if gex is None:
        return panel("GEX Level Ladder", "", nofeed=True)
    ft = (summary or {}).get("first_touch") or {}
    last = gex.get("current_price")
    last = float(last) if isinstance(last, (int, float)) else None
    dw = _Drawers()

    rows: list[tuple[float, str, str]] = []  # (price, cls, html)
    for lv in gex.get("levels") or []:
        try:
            p, raw = float(lv["price"]), str(lv["name"])
        except (KeyError, TypeError, ValueError):
            continue
        m = _STRENGTH_RE.search(raw)
        strength = len(m.group(1)) if m else 0
        name = _STRENGTH_RE.sub("", raw)
        cls = _level_cls(name)
        dots = f'<span class="dots">{"●" * strength}</span>' if strength else ""
        inner = (f'<span class="lname">{esc(name)}</span>{dots}'
                 f'<span class="lpx mono">{fnum(p)}</span>')
        rows.append((p, cls, dw.row(f'<div class="lrow {cls}"{{D}}>', inner,
                                    _type_drawer(name, strength))))
    amd = gex.get("session_amd") or {}
    for key, label in _LEVEL_LABELS.items():
        v = amd.get(key)
        if isinstance(v, (int, float)):
            inner = (f'<span class="lname">{label}</span>'
                     f'<span class="lpx mono">{fnum(float(v))}</span>')
            rows.append((float(v), "amd",
                         dw.row('<div class="lrow amd"{D}>', inner,
                                _stats_drawer(key, ft))))
    if last is not None:
        rows.append((last, "last",
                     f'<div class="lrow last"><span class="lname">► LAST'
                     f'</span><span class="lpx mono">{fnum(last)}</span></div>'))
    rows.sort(key=lambda r: r[0], reverse=True)

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
    hint = '<div class="lnote">click a level for its playbook</div>'
    return panel(f"GEX Level Ladder · {sym}",
                 f'<div class="ladder">{"".join(parts)}</div>{hint}', age)


def build_public_ladder(gex: dict | None, brief: dict | None,
                        summary: dict | None, age: float | None) -> str:
    """Product-edition ladder: price-derived session levels only (no GEX
    Suite data), each annotated with its SweepStats first-touch odds."""
    amd = (gex or {}).get("session_amd") or {}
    ft = (summary or {}).get("first_touch") or {}
    nq = ((brief or {}).get("ctx") or {}).get("NQ") or {}
    last = nq.get("last") if isinstance(nq.get("last"), (int, float)) \
        else (gex or {}).get("current_price")
    if not amd and last is None:
        return panel("Session Liquidity Ladder", "", nofeed=True)
    dw = _Drawers()

    rows: list[tuple[float, str]] = []
    for key, label in _LEVEL_LABELS.items():
        v = amd.get(key)
        if not isinstance(v, (int, float)):
            continue
        st = ft.get(key) or {}
        odds = ""
        if st:
            odds = (f'<span class="lodds mono">{float(st.get("fakeout_pct") or 0):.0f}% FAKE '
                    f'· stops run ≈{float(st.get("fakeout_median_overshoot_pts") or 0):.0f} pts '
                    f'· +{float(st.get("fakeout_median_mfe60_pts") or 0):.0f}/60m</span>')
        inner = (f'<span class="lname">{label}</span>{odds}'
                 f'<span class="lpx mono">{fnum(float(v))}</span>')
        rows.append((float(v), dw.row('<div class="lrow amd roomy"{D}>', inner,
                                      _stats_drawer(key, ft))))
    pc = nq.get("prev_close")
    if isinstance(pc, (int, float)):
        rows.append((float(pc),
                     f'<div class="lrow corr roomy"><span class="lname">PREV CLOSE</span>'
                     f'<span class="lpx mono">{fnum(float(pc))}</span></div>'))
    if isinstance(last, (int, float)):
        rows.append((float(last),
                     f'<div class="lrow last roomy"><span class="lname">► LAST'
                     f'</span><span class="lpx mono">{fnum(float(last))}</span></div>'))
    rows.sort(key=lambda r: r[0], reverse=True)
    note = ('<div class="lnote">levels computed from price data · odds from '
            'the SweepStats dataset · click a level for its full playbook</div>')
    return panel("Session Liquidity Ladder · NQ",
                 f'<div class="ladder">{"".join(r[1] for r in rows)}</div>{note}', age)


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
    # 24h ET timeline: Asia 18–24, London 2–5, NY 9:30–16
    blocks = "".join(
        f'<div class="tlblock {c}" style="left:{lo / 14.4:.2f}%;'
        f'width:{(hi - lo) / 14.4:.2f}%"></div>'
        for c, lo, hi in (("tl-asia", 1080, 1440), ("tl-london", 120, 300),
                          ("tl-ny", 570, 960)))
    ticks = "".join(f'<span class="tltick" style="left:{h / .24:.1f}%">'
                    f'{h:02d}</span>' for h in (0, 6, 12, 18))
    timeline = (f'<div class="tlwrap"><div class="tl">{blocks}'
                f'<div class="tlneedle" id="tl-needle"></div></div>'
                f'<div class="tlticks">{ticks}</div></div>')
    rows = "".join(
        f'<div class="crow" id="sess-{k}"><span class="cname">{n}</span>'
        f'<span class="cwin mono">{w}</span>'
        f'<span class="cstate mono" id="state-{k}">—</span>'
        f'<span class="ceta mono" id="eta-{k}"></span></div>'
        for k, n, w in (("asia", "ASIA", "18:00–00:00"),
                        ("london", "LONDON", "02:00–05:00"),
                        ("ny", "NEW YORK", "09:30–16:00")))
    body = (f'<div class="clockbig mono" id="et-big">--:--:--</div>'
            f'<div class="clocksub">EASTERN TIME</div>{timeline}{rows}'
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
        return panel("SweepStats", "", nofeed=True)
    pct, fake, touched = _hero_numbers(summary)
    s = summary.get("sample") or {}
    # ring gauge: r=70 → circumference ≈ 439.82
    dash = 439.82 * pct / 100.0
    ring = (
        '<svg class="ring" viewBox="0 0 170 170" width="170" height="170">'
        '<circle cx="85" cy="85" r="70" fill="none" stroke="rgba(255,180,84,.12)"'
        ' stroke-width="10"/>'
        f'<circle cx="85" cy="85" r="70" fill="none" stroke="#ffb454"'
        f' stroke-width="10" stroke-linecap="round"'
        f' stroke-dasharray="{dash:.1f} 439.82"'
        ' transform="rotate(-90 85 85)"/>'
        f'<text x="85" y="80" text-anchor="middle" class="ringpct">{pct}%</text>'
        f'<text x="85" y="102" text-anchor="middle" class="ringsub">FAKEOUT</text>'
        "</svg>")
    body = (
        f'<div class="hero bracket">{ring}'
        f'<div class="herolabel">of first NY touches of Asia/London levels are fakeouts</div>'
        f'<div class="herosub mono">n={touched} episodes · {esc(s.get("sessions", "?"))} sessions<br>'
        f'{esc(s.get("from", "?"))} → {esc(s.get("to", "?"))}</div>'
        f'<button class="linkbtn" id="hero-more">VIEW BREAKDOWN ▾</button></div>')
    return panel("SweepStats", body, age)


_SWEEP_KEYS = {"asia high": "asia_high", "asia low": "asia_low",
               "none": "none", "both": "both"}
_BUCKET_TITLES = {"asia_high": "LONDON SWEEPS ASIA HIGH",
                  "asia_low": "LONDON SWEEPS ASIA LOW",
                  "none": "NO LONDON SWEEP",
                  "both": "LONDON SWEEPS BOTH SIDES"}


def build_script(gex: dict | None, summary: dict | None) -> str:
    amd = (gex or {}).get("session_amd") or {}
    if not amd or summary is None:
        return panel("Today's Script", "", nofeed=True)
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

    d = str(amd.get("date") or "")
    date_chip = ""
    if d and d != datetime.now(ET).date().isoformat():
        date_chip = f'<span class="chip stale">SCRIPT FROM {esc(d)}</span>'

    left = (
        f'<div class="verdict bracket">'
        f'<div class="scriptpat mono">{esc(_BUCKET_TITLES[key])}</div>'
        f'<div class="scripthist">→ NY closed <span class="{"up" if up_pct >= 50 else "dn"}">'
        f'UP {up_pct:.0f}%</span> of days · median NY move '
        f'<span class="mono {chg_cls(med)}">{med:+.1f} pts</span> · n={days} days{low_n}</div>'
        f"</div>")

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
    body = f'<div class="script">{left}{right}</div>'
    return panel(f"Today's Script — overnight pattern vs {esc(n)}-session history",
                 body, extra_chip=date_chip + bias_chip, cls="accent")


def _first_touch_pane(summary: dict) -> str:
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
    return legend + rows


def _matrix_pane(summary: dict) -> str:
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
    return f'<div class="matrix">{cells}</div>'


def build_stats_deck(summary: dict | None, age: float | None) -> str:
    if summary is None:
        return panel("Stats Deck", "", nofeed=True)
    body = (
        '<div class="tabbar">'
        '<button class="tab active" data-pane="pane-ft">FIRST-TOUCH BOARD</button>'
        '<button class="tab" data-pane="pane-lm">LONDON MATRIX</button>'
        "</div>"
        f'<div class="tabpane active" id="pane-ft">'
        f'<div class="pane-title">FIRST-TOUCH BOARD</div>{_first_touch_pane(summary)}</div>'
        f'<div class="tabpane" id="pane-lm">'
        f'<div class="pane-title">LONDON MATRIX</div>{_matrix_pane(summary)}</div>')
    return panel("Stats Deck", body, age, cls="deck")


def build_wire(brief: dict | None, age: float | None) -> str:
    if brief is None:
        return panel("Wire", "", nofeed=True)
    news = (brief.get("news") or [])[:6]
    items = ""
    for i, n in enumerate(news):
        try:
            hhmm = parsedate_to_datetime(n["date"]).astimezone(ET).strftime("%H:%M")
        except (KeyError, TypeError, ValueError):
            hhmm = "--:--"
        more = " wmore" if i >= 3 else ""
        items += (f'<div class="wrow{more}"><span class="mono wtime">{hhmm}</span>'
                  f'<a href="{esc(n.get("url") or "#")}" target="_blank" rel="noopener">'
                  f'{esc(n.get("title") or "")}</a></div>')
    toggle = ""
    if len(news) > 3:
        toggle = (f'<button class="linkbtn" id="wire-more">SHOW ALL '
                  f'({len(news)}) ▾</button>')
    cal = ""
    for ev in brief.get("calendar") or []:
        if not isinstance(ev, dict):
            continue
        cal += (f'<span class="calchip mono">{esc(ev.get("time") or "")} '
                f'{esc(ev.get("title") or "")} · {esc(ev.get("impact") or "")}'
                f'{" · f:" + esc(ev["forecast"]) if ev.get("forecast") else ""}</span>')
    if cal:
        cal = f'<div class="cal">{cal}</div>'
    return panel("Wire", items + toggle + cal, age)


# ------------------------------------------------------------------ page

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0d10;color:#c9d4de;font:13px/1.45 "Segoe UI",system-ui,sans-serif;
  background-image:
    radial-gradient(1200px 500px at 50% -100px,rgba(77,208,225,.05),transparent),
    repeating-linear-gradient(0deg,rgba(255,255,255,.015) 0 1px,transparent 1px 24px),
    repeating-linear-gradient(90deg,rgba(255,255,255,.015) 0 1px,transparent 1px 24px)}
a{color:#c9d4de;text-decoration:none}a:hover{color:#ffb454}
button{font:inherit;color:inherit;background:none;border:none;cursor:pointer}
.mono{font-family:"Cascadia Mono","Consolas",monospace}
.up{color:#3ddc84}.dn{color:#ff5f56}.flat{color:#5c6a78}
.wrap{width:1440px;margin:0 auto;padding:14px;display:flex;flex-direction:column;gap:12px}
/* ---- panel chrome: depth ---- */
.panel{background:linear-gradient(180deg,#121821,#0e1319);
  border:1px solid #1e2833;border-radius:4px;padding:10px 12px;position:relative;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.045),0 10px 28px rgba(0,0,0,.42);
  transition:border-color .15s ease}
.panel:hover{border-color:#2c3d4e}
.panel.accent{border-color:#3a3324}
.panel.accent:hover{border-color:#ffb454}
.phead{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;
  min-height:24px}
.phead.tgl{cursor:pointer;user-select:none}
.ptitle{font-family:"Cascadia Mono","Consolas",monospace;font-size:11px;
  text-transform:uppercase;letter-spacing:.12em;color:#5c6a78;display:flex;
  align-items:center;gap:8px}
.led{width:6px;height:6px;border-radius:50%;flex:none}
.led.green{background:#3ddc84;box-shadow:0 0 6px rgba(61,220,132,.8);
  animation:pulse 2.4s ease-in-out infinite}
.led.amber{background:#ffb454;box-shadow:0 0 6px rgba(255,180,84,.8)}
.led.red{background:#ff5f56;box-shadow:0 0 6px rgba(255,95,86,.8)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.chips{display:flex;gap:6px;align-items:center}
.chip{font:10px "Cascadia Mono","Consolas",monospace;letter-spacing:.08em;color:#5c6a78;
  border:1px solid #1e2833;border-radius:3px;padding:2px 6px;white-space:nowrap}
.chip.stale{color:#ffb454;border-color:#775a2e}
.chip.biasup{color:#3ddc84;border-color:#2b6647}.chip.biasdn{color:#ff5f56;border-color:#743732}
.chev{color:#5c6a78;font-size:10px;transition:transform .15s ease;margin-left:2px}
.panel.collapsed .chev{transform:rotate(-90deg)}
.panel.collapsed .pbody{display:none}
.stale-dim{opacity:.55}
.nofeed{color:#5c6a78;font:18px "Cascadia Mono","Consolas",monospace;
  letter-spacing:.35em;text-align:center;padding:44px 0}
.subhead{font:10px "Cascadia Mono","Consolas",monospace;color:#5c6a78;
  letter-spacing:.14em;margin-bottom:5px}
.linkbtn{font:10px "Cascadia Mono","Consolas",monospace;letter-spacing:.12em;
  color:#ffb454;padding:6px 0 2px;display:block}
.linkbtn:hover{text-shadow:0 0 8px rgba(255,180,84,.5)}
/* corner brackets */
.bracket{position:relative}
.bracket::before,.bracket::after{content:"";position:absolute;width:12px;height:12px;
  border:1px solid rgba(255,180,84,.55)}
.bracket::before{top:0;left:0;border-right:none;border-bottom:none}
.bracket::after{bottom:0;right:0;border-left:none;border-top:none}
/* ---- header ---- */
header.panel{display:flex;justify-content:space-between;align-items:center;
  padding:14px 16px;overflow:hidden}
header.panel::before{content:"";position:absolute;inset:0 0 auto 0;height:5px;
  background:repeating-linear-gradient(90deg,rgba(255,180,84,.35) 0 6px,transparent 6px 12px)}
.wordmark{font:18px "Cascadia Mono","Consolas",monospace;letter-spacing:.35em;color:#c9d4de}
.wordmark b{color:#ffb454;font-weight:normal;text-shadow:0 0 14px rgba(255,180,84,.35)}
.cursorblk{display:inline-block;width:9px;height:16px;background:#ffb454;
  margin-left:6px;vertical-align:-2px;animation:blink 1.1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.hclocks{display:flex;gap:22px;font-family:"Cascadia Mono","Consolas",monospace}
.hclocks .lbl{color:#5c6a78;font-size:10px;letter-spacing:.12em;display:block}
.hclocks .val{font-size:17px}
.hright{text-align:right;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.08em}
.hright .n{color:#ffb454}
/* ---- tape ---- */
.tape{display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.tape .sym{color:#5c6a78;font-size:11px;letter-spacing:.12em;margin-right:7px}
.tape-nq .px{font-size:26px;text-shadow:0 0 18px rgba(61,220,132,.25)}
.tape-nq .px.dn{text-shadow:0 0 18px rgba(255,95,86,.25)}
.tape-nq .chg{font-size:14px;margin-left:8px}
.tape-q .px{font-size:15px}.tape-q .chg{font-size:11px;margin-left:6px}
.sectors{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.schip{font-size:10.5px;border:1px solid #1e2833;border-radius:3px;padding:2px 6px;
  transition:border-color .15s ease}
.schip:hover{border-color:#2c3d4e}
/* ---- row grids ---- */
.row2{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;align-items:stretch}
.row2 .panel{display:flex;flex-direction:column}
.row2 .pbody{flex:1;display:flex;flex-direction:column}
.row2 .pbody>.hero{margin:auto 0}
.row4{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}
/* ---- ladder ---- */
.ladder{display:flex;flex-direction:column}
.lrow{display:flex;align-items:baseline;gap:8px;padding:2px 6px;
  border-left:2px solid transparent;border-radius:3px;min-height:21px}
.lrow[data-d]{cursor:pointer}
.lrow[data-d]:hover{background:rgba(255,255,255,.035)}
.rchev{color:#39485a;font-size:9px;flex:none;transition:transform .15s ease}
.lrow.open .rchev{transform:rotate(90deg);color:#ffb454}
.lrow .lname{color:#5c6a78;font-size:11.5px;letter-spacing:.04em}
.lrow .lpx{margin-left:auto;font-size:12.5px;color:#c9d4de}
.lrow.corr .lpx{color:#5c6a78}
.lrow.gamma{border-left-color:#4dd0e1}.lrow.gamma .lname{color:#4dd0e1}
.lrow.cwall{border-left-color:#3ddc84}.lrow.cwall .lname{color:#3ddc84;font-weight:600}
.lrow.pwall{border-left-color:#ff5f56}.lrow.pwall .lname{color:#ff5f56;font-weight:600}
.lrow.flip{border-left-color:#ffb454}.lrow.flip .lname{color:#ffb454;font-weight:600}
.lrow.amd{border-left-color:#4dd0e1}.lrow.amd .lname{color:#4dd0e1;font-weight:600}
.lrow.amd .lpx{color:#4dd0e1}
.lrow.last{background:linear-gradient(90deg,rgba(255,180,84,.12),rgba(255,180,84,.03));
  border:1px solid #ffb454;border-radius:3px;margin:2px 0;padding:4px 8px;
  box-shadow:0 0 14px rgba(255,180,84,.12)}
.lrow.last .lname{color:#ffb454;font-weight:600}.lrow.last .lpx{color:#ffb454;font-size:14px}
.lrow.gap .lname{color:#2b3844;letter-spacing:4px}
.lrow.roomy{padding:9px 8px;font-size:14px}
.lrow.roomy .lname{font-size:13px}.lrow.roomy .lpx{font-size:15px}
.lodds{margin-left:14px;font-size:10.5px;color:#5c6a78;letter-spacing:.04em}
.lnote{margin-top:10px;font:10px "Cascadia Mono","Consolas",monospace;
  color:#39485a;letter-spacing:.08em}
.dots{color:#ffb454;font-size:9px;letter-spacing:2px}
/* drawers */
.drawer{background:#0b0f14;border:1px solid #1e2833;border-left:2px solid #ffb454;
  border-radius:3px;margin:2px 0 4px 14px;padding:8px 10px;font-size:11.5px;
  color:#8b9aa8;line-height:1.5}
.dgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px 14px}
.dcell{display:flex;flex-direction:column}
.dcell span{font:9px "Cascadia Mono","Consolas",monospace;color:#5c6a78;letter-spacing:.1em}
.dcell b{font-weight:normal;font-size:12px;color:#c9d4de}
/* ---- clock ---- */
.clockbig{font-size:32px;text-align:center;margin-top:4px;
  text-shadow:0 0 22px rgba(201,212,222,.15)}
.clocksub{text-align:center;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.3em;margin-bottom:12px}
.tlwrap{margin:0 2px 12px}
.tl{position:relative;height:14px;background:#0b0f14;border:1px solid #1e2833;
  border-radius:3px;overflow:hidden}
.tlblock{position:absolute;top:2px;bottom:2px;border-radius:2px;opacity:.55}
.tl-asia{background:#4dd0e1}.tl-london{background:#ffb454}.tl-ny{background:#3ddc84}
.tlneedle{position:absolute;top:-1px;bottom:-1px;width:2px;background:#fff;
  box-shadow:0 0 8px rgba(255,255,255,.8);left:0}
.tlticks{position:relative;height:12px;margin-top:2px}
.tltick{position:absolute;font:9px "Cascadia Mono","Consolas",monospace;color:#39485a}
.crow{display:flex;align-items:baseline;gap:8px;padding:8px 4px;border-top:1px solid #1e2833}
.cname{font:12px "Cascadia Mono","Consolas",monospace;letter-spacing:.12em}
.cwin{color:#5c6a78;font-size:10.5px}
.cstate{margin-left:auto;font-size:11px;letter-spacing:.1em}
.cstate.open{color:#3ddc84}.cstate.closed{color:#5c6a78}
.ceta{color:#5c6a78;font-size:10.5px}
.bsep{margin-top:14px}
.buckets{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.bucket{border:1px solid #1e2833;border-radius:3px;padding:6px 8px;text-align:center;
  background:#0b0f14;transition:border-color .15s ease}
.bucket.active{border-color:#ffb454;box-shadow:0 0 12px rgba(255,180,84,.15)}
.bucket.active .brange{color:#ffb454}
.brange{font-size:10px;color:#5c6a78;letter-spacing:.1em}
.bfake{font-size:20px;color:#c9d4de}
.bsub{font-size:10px;color:#5c6a78}
/* ---- hero ---- */
.hero{text-align:center;padding:14px 8px}
.ring{display:block;margin:0 auto 6px;filter:drop-shadow(0 0 18px rgba(255,180,84,.25))}
.ringpct{font:34px "Cascadia Mono","Consolas",monospace;fill:#ffb454}
.ringsub{font:9px "Cascadia Mono","Consolas",monospace;fill:#5c6a78;letter-spacing:.3em}
.herolabel{margin:6px auto;max-width:230px;color:#c9d4de;font-size:13px}
.herosub{color:#5c6a78;font-size:10.5px;letter-spacing:.05em}
.hero .linkbtn{margin:8px auto 0}
/* ---- script ---- */
.script{display:grid;grid-template-columns:1fr 1.4fr;gap:18px}
.verdict{padding:14px 16px;background:#0b0f14;border:1px solid #1e2833;border-radius:3px}
.scriptpat{font-size:22px;letter-spacing:.14em;color:#ffb454;margin:2px 0 8px;
  text-shadow:0 0 16px rgba(255,180,84,.3)}
.scripthist{font-size:13px;color:#c9d4de}
.lown{color:#ff5f56;font-size:10px;letter-spacing:.08em}
.srow{display:flex;gap:14px;align-items:baseline;padding:4px 6px;
  border-top:1px solid #1e2833;font-size:12px;border-radius:3px}
.srow:hover{background:rgba(255,255,255,.035)}
.srow .slvl{width:96px;color:#4dd0e1;font:11px "Cascadia Mono","Consolas",monospace;letter-spacing:.06em}
.srow .spx{width:80px;color:#c9d4de}
.srow .sfake{width:104px;color:#ffb454}
.srow .srun{color:#5c6a78}
.srow .sbnc{margin-left:auto}
/* ---- stats deck ---- */
.tabbar{display:flex;gap:2px;margin-bottom:10px;border-bottom:1px solid #1e2833}
.tab{font:10.5px "Cascadia Mono","Consolas",monospace;letter-spacing:.12em;
  color:#5c6a78;padding:6px 12px;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab:hover{color:#c9d4de}
.tab.active{color:#ffb454;border-bottom-color:#ffb454}
.tabpane{display:none}
.tabpane.active{display:block}
.pane-title{display:none;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.14em;margin:10px 0 4px}
.ftlegend{font:9.5px "Cascadia Mono","Consolas",monospace;color:#5c6a78;
  letter-spacing:.08em;margin-bottom:6px}
.ftrow{display:flex;align-items:center;gap:10px;padding:8px 4px;border-top:1px solid #1e2833;
  border-radius:3px}
.ftrow:hover{background:rgba(255,255,255,.035)}
.ftname{width:100px;font:11px "Cascadia Mono","Consolas",monospace;
  color:#4dd0e1;letter-spacing:.06em}
.ftbar{flex:1;height:8px;background:rgba(255,180,84,.1);border-radius:2px;overflow:hidden}
.ftfill{display:block;height:100%;background:#ffb454;border-radius:0 4px 4px 0}
.ftpct{width:44px;text-align:right;color:#ffb454;font-size:13px}
.ftsub{width:92px;text-align:right;color:#5c6a78;font-size:10.5px}
.matrix{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.mcell{border:1px solid #1e2833;border-radius:3px;padding:9px 10px;background:#0b0f14;
  transition:border-color .15s ease}
.mcell:hover{border-color:#2c3d4e}
.mtitle{font:10.5px "Cascadia Mono","Consolas",monospace;color:#5c6a78;letter-spacing:.08em;margin-bottom:7px}
.mstats{display:flex;justify-content:space-between;font-size:13px}
/* ---- wire ---- */
.wrow{display:flex;gap:12px;padding:5px 4px;border-top:1px solid #1e2833;font-size:12.5px;
  border-radius:3px}
.wrow:first-child{border-top:none}
.wrow:hover{background:rgba(255,255,255,.035)}
.wrow.wmore{display:none}
.wire-open .wrow.wmore{display:flex}
.wtime{color:#ffb454;font-size:11px}
.cal{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}
.calchip{font-size:10.5px;color:#4dd0e1;border:1px solid #1e2833;border-radius:3px;padding:2px 8px}
/* ---- footer ---- */
footer{display:flex;justify-content:space-between;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.08em;padding:2px 4px 10px}
footer .amber{color:#ffb454}
/* ---- expand-all (#png screenshot mode) ---- */
body.expand-all .tabbar{display:none}
body.expand-all .tabpane{display:block}
body.expand-all .pane-title{display:block}
body.expand-all .wrow.wmore{display:flex}
body.expand-all #wire-more,body.expand-all #hero-more{display:none}
body.expand-all .lnote{display:none}
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
  var nd=document.getElementById('tl-needle');
  if(nd)nd.style.left=(mins/1440*100).toFixed(2)+'%';
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

/* collapsible panels */
document.querySelectorAll('.phead.tgl').forEach(function(h){
  h.addEventListener('click',function(e){
    if(e.target.closest('a,button'))return;
    h.parentElement.classList.toggle('collapsed');
  });
});
/* ladder drawers */
document.querySelectorAll('.lrow[data-d]').forEach(function(row){
  row.addEventListener('click',function(){
    var d=document.getElementById(row.getAttribute('data-d'));
    if(!d)return;
    d.hidden=!d.hidden;
    row.classList.toggle('open',!d.hidden);
  });
});
/* tabs */
document.querySelectorAll('.tab').forEach(function(t){
  t.addEventListener('click',function(){
    var bar=t.parentElement,deck=bar.parentElement;
    bar.querySelectorAll('.tab').forEach(function(x){x.classList.remove('active')});
    deck.querySelectorAll('.tabpane').forEach(function(x){x.classList.remove('active')});
    t.classList.add('active');
    var p=document.getElementById(t.getAttribute('data-pane'));
    if(p)p.classList.add('active');
  });
});
/* wire show-more */
var wm=document.getElementById('wire-more');
if(wm){var wmOrig=wm.textContent;
  wm.addEventListener('click',function(){
    var p=wm.closest('.pbody');p.classList.toggle('wire-open');
    wm.textContent=p.classList.contains('wire-open')?'SHOW LESS \\u25B4':wmOrig;
  });}
/* hero -> stats deck */
var hm=document.getElementById('hero-more');
if(hm)hm.addEventListener('click',function(){
  var deck=document.querySelector('.panel.deck');
  if(!deck)return;
  deck.classList.remove('collapsed');
  deck.scrollIntoView({behavior:'smooth',block:'center'});
});
/* screenshot mode */
if(location.hash==='#png')document.body.classList.add('expand-all');
"""


def build_page(brief, brief_age, gex, gex_age, summary, summary_age,
               public: bool = False) -> str:
    n_sessions = "—"
    if summary:
        n_sessions = str((summary.get("sample") or {}).get("sessions", "—"))
    built = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    edition = ('PUBLIC EDITION · PRICE DATA ONLY<br>' if public else "")

    header = (
        '<header class="panel">'
        '<div class="wordmark">NQ <b>SITUATION ROOM</b><span class="cursorblk"></span></div>'
        '<div class="hclocks">'
        '<div><span class="lbl">UTC</span><span class="val" id="hdr-utc">--:--:--</span></div>'
        '<div><span class="lbl">NEW YORK</span><span class="val js-et">--:--:--</span></div>'
        "</div>"
        f'<div class="hright">{edition}SWEEPSTATS DATA · <span class="n">n={esc(n_sessions)} SESSIONS</span>'
        f"<br>BUILD {esc(built)}</div>"
        "</header>")

    def src_age(label: str, age: float | None) -> str:
        if age is None:
            return f'{label} <span class="amber">NO FEED</span>'
        if age > STALE_HOURS:
            return f'{label} <span class="amber">{round(age)}H STALE</span>'
        return f"{label} {round(age)}H"

    ages = " · ".join((src_age("BRIEF", brief_age),
                       src_age("LEVELS" if public else "GEX", gex_age),
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
        + (build_public_ladder(gex, brief, summary, gex_age) if public
           else build_ladder(gex, summary, gex_age))
        + build_clock(summary)
        + build_hero(summary, summary_age)
        + "</div>"
        + build_script(gex, summary)
        + '<div class="row4">'
        + build_stats_deck(summary, summary_age)
        + build_wire(brief, brief_age)
        + "</div>"
        + footer
        + f"<script>{JS}</script></div></body></html>")


# ------------------------------------------------------------------- png

def _find_chrome() -> Path | None:
    for p in CHROME_CANDIDATES:
        if p.is_file():
            return p
    return None


def render_png(html_path: Path, png_path: Path, height: int) -> bool:
    """Render with a fresh headless Chrome — never the port-9222 trading one.

    Appends #png so the page opens in expand-all mode (tabs stacked, wire
    fully shown) for the shareable screenshot.
    """
    chrome = _find_chrome()
    if chrome is None:
        print("  WARNING: chrome.exe not found — skipping PNG render")
        return False
    subprocess.run(
        [str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--screenshot={png_path.resolve()}", f"--window-size=1440,{height}",
         html_path.resolve().as_uri() + "#png"],
        check=True, capture_output=True, timeout=120)
    return True


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description="Render the NQ Situation Room page")
    ap.add_argument("--png", action="store_true", help="also render the PNG")
    ap.add_argument("--open", action="store_true", help="open the HTML in a browser")
    ap.add_argument("--public", action="store_true",
                    help="product edition: price-derived levels only, no GEX data")
    args = ap.parse_args()

    brief, brief_age = load("morning_brief.json", "generated")
    gex, gex_age = load("gex_levels.json", "timestamp")
    summary, summary_age = load("session_stats_summary.json", "generated")

    suffix = "_public" if args.public else ""
    out_html = RESULTS / f"situation_room{suffix}.html"
    out_png = RESULTS / f"situation_room{suffix}.png"

    out_html.write_text(
        build_page(brief, brief_age, gex, gex_age, summary, summary_age,
                   public=args.public),
        encoding="utf-8")
    print(f"HTML -> {out_html}")

    if args.png and render_png(out_html, out_png,
                               height=1330 if args.public else 1680):
        print(f"PNG  -> {out_png}")
    if args.open:
        webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()
