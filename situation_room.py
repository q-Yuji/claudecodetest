"""situation_room.py — NQ Situation Room (SweepStats product wrapper, v3 UI).

Renders results/situation_room.html: a self-contained dark page assembling
the repo's regenerated JSONs (morning_brief, gex_levels,
session_stats_summary). Read-only over its inputs; never crashes on a
missing/stale/malformed input — sections degrade to "NO FEED" or carry a
STALE marker instead.

v3 UI (2026-07-10): editorial terminal, not a widget dashboard — masthead
with double rule, scrolling ticker, numbered sections split by hairlines
on one continuous surface (no cards), condensed display type for verdicts,
mono strictly for data, tables set like a rate sheet. Interactions kept:
collapsible sections, clickable ladder rows with detail drawers, stats
tabs, wire show-more. Opening the HTML with #png expands everything for
screenshot capture.

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


def load_ledger() -> dict | None:
    """data/prop_ledger.json — hand-curated account economics (personal only)."""
    try:
        data = json.loads((ROOT / "data" / "prop_ledger.json")
                          .read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("accounts") else None
    except (OSError, ValueError):
        return None


def load_scoreboard() -> dict | None:
    """Walk-forward grades, computed live from the tracked dataset.

    Stdlib-only import (see backtest/scoreboard.py) — no results/ JSON to
    go stale: the scoreboard is always exactly as fresh as the dataset.
    """
    try:
        from backtest.scoreboard import compute_from_file
        return compute_from_file()
    except Exception:
        return None


# ------------------------------------------------------------- formatting

def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def fnum(x: float, dp: int = 2) -> str:
    return f"{x:,.{dp}f}"


def chg_cls(x: float) -> str:
    return "up" if x > 0 else "dn" if x < 0 else "flat"


def _status(age: float | None, nofeed: bool = False) -> str:
    if nofeed:
        return '<span class="st bad">NO FEED</span>'
    if age is None:
        return ""
    if age > STALE_HOURS:
        return f'<span class="st warn">STALE {round(age)}H</span>'
    label = "LIVE" if age < 1 else f"{round(age)}H AGO"
    return f'<span class="st">{label}</span>'


def section(num: str, title: str, body: str, age: float | None = None,
            right: str = "", anchor: str = "", nofeed: bool = False) -> str:
    """Numbered editorial section: running head + hairline, no card box."""
    if nofeed:
        body = '<div class="nofeed">— NO FEED —</div>'
    stale = age is not None and age > STALE_HOURS
    return (
        f'<section class="sec" id="{anchor}">'
        f'<div class="runhead tgl"><span class="rh-l"><i>{num}</i>{esc(title)}'
        f'<b class="fold">[–]</b></span>'
        f'<span class="rh-r">{right}{_status(age, nofeed)}</span></div>'
        f'<div class="secbody{" stale-dim" if stale else ""}">{body}</div>'
        f"</section>")


# ----------------------------------------------------------------- ticker

def build_ticker(brief: dict | None) -> str:
    if brief is None:
        return ""
    items = []
    ctx = brief.get("ctx") or {}
    for sym in ("NQ", "ES", "SPY", "QQQ", "VIX"):
        q = ctx.get(sym)
        if not isinstance(q, dict):
            continue
        c = float(q.get("chg_pct") or 0)
        arrow = "▲" if c > 0 else "▼" if c < 0 else "—"
        items.append(f'<span class="ti"><b>{sym}</b> {fnum(float(q.get("last") or 0))} '
                     f'<span class="{chg_cls(c)}">{arrow} {abs(c):.2f}%</span></span>')
    for sym, s in (brief.get("sectors") or {}).items():
        if not isinstance(s, dict):
            continue
        c = float(s.get("chg_pct") or 0)
        items.append(f'<span class="ti">{esc(sym)} '
                     f'<span class="{chg_cls(c)}">{c:+.1f}%</span></span>')
    if not items:
        return ""
    run = '<span class="sep">/</span>'.join(items)
    # content doubled so the CSS translate(-50%) loop is seamless
    return (f'<div class="ticker"><div class="tickrun">'
            f'<span class="tickseg">{run}{"<span class=sep>/</span>"}</span>'
            f'<span class="tickseg">{run}{"<span class=sep>/</span>"}</span>'
            f"</div></div>")


# ------------------------------------------------------------------- hero

def _todays_events() -> list[dict]:
    try:
        from data.market_events import events_for
        return events_for(datetime.now(ET).date())
    except Exception:
        return []


_EVENT_LABELS = {"opex": "OPEX", "quad_witching": "QUAD WITCH",
                 "vix_exp": "VIX EXP", "cpi": "CPI", "fomc": "FOMC"}

_SWEEP_KEYS = {"asia high": "asia_high", "asia low": "asia_low",
               "none": "none", "both": "both"}
_PATTERN_TITLES = {"asia_high": "London sweeps Asia High.",
                   "asia_low": "London sweeps Asia Low.",
                   "none": "No London sweep.",
                   "both": "London sweeps both sides."}
_LEVEL_LABELS = {"asia_high": "ASIA HIGH", "asia_low": "ASIA LOW",
                 "london_high": "LONDON HIGH", "london_low": "LONDON LOW"}


def _hero_numbers(summary: dict) -> tuple[int, int, int]:
    ft = summary.get("first_touch") or {}
    touched = sum(int(v.get("touched") or 0) for v in ft.values() if isinstance(v, dict))
    fake = sum(int(v.get("fakeout") or 0) for v in ft.values() if isinstance(v, dict))
    pct = round(100.0 * fake / touched) if touched else 0
    return pct, fake, touched


def build_hero(gex: dict | None, summary: dict | None) -> str:
    amd = (gex or {}).get("session_amd") or {}
    if not amd or summary is None:
        return section("01", "Today's Script", "", anchor="script", nofeed=True)
    key = _SWEEP_KEYS.get(str(amd.get("london_sweep") or "none").lower(), "none")
    lm = (summary.get("london_manipulation") or {}).get(key) or {}
    days = int(lm.get("days") or 0)
    up_pct = float(lm.get("ny_up_pct") or 0)
    med = float(lm.get("median_ny_change_pts") or 0)
    low_n = ' <span class="lown">low sample</span>' if days < 5 else ""
    pct, _, touched = _hero_numbers(summary)
    s = summary.get("sample") or {}

    bias = str(amd.get("day_bias") or "").upper()
    chips = ""
    d = str(amd.get("date") or "")
    if d and d != datetime.now(ET).date().isoformat():
        chips += f'<span class="tag warn">[ SCRIPT FROM {esc(d)} ]</span>'
    if bias:
        chips += (f'<span class="tag {"up" if bias == "BULLISH" else "dn"}">'
                  f'[ BIAS {esc(bias)} ]</span>')

    # regime chip: trend/chop gate from daily closes (backtest.regime)
    lr = summary.get("latest_regime")
    if isinstance(lr, dict):
        reg = str(lr.get("regime") or "")
        cls = {"chop": "warn", "trend_up": "up", "trend_down": "dn"}.get(reg, "")
        chips += (f'<span class="tag {cls}">[ REGIME {esc(reg.replace("_", " ").upper())}'
                  f' · ER {float(lr.get("er") or 0):.2f} ]</span>')

    # flow-event chips: expiry/macro days are regime overrides — say so
    todays_events = _todays_events()
    for ev in todays_events:
        chips += f'<span class="tag warn">[ {esc(ev["label"])} ]</span>'
    flow_note = ""
    if todays_events:
        flow_note = (
            '<p class="dek flownote">' +
            " · ".join(esc(ev["note"]) for ev in todays_events) +
            " — pattern odds take a back seat on flow days.</p>")

    kicker = esc(datetime.now(ET).strftime("%A, %B %d %Y").upper())
    left = (
        f'<div class="kicker mono">{kicker} · OVERNIGHT PATTERN</div>'
        f'<h1 class="verdict">{esc(_PATTERN_TITLES[key])}</h1>'
        f'<p class="dek">On the {days} prior days with this pattern, New York closed '
        f'<span class="{"up" if up_pct >= 50 else "dn"}">up {up_pct:.0f}% of the time</span> '
        f'with a median session move of <span class="mono {chg_cls(med)}">{med:+.1f} pts</span>.{low_n}</p>'
        f'{flow_note}'
        f'<div class="tags mono">{chips}</div>')

    right = (
        f'<div class="bigstat"><span class="bignum">{pct}<i>%</i></span>'
        f'<span class="bigcap mono">FAKEOUT RATE — first NY touches of Asia/London '
        f'levels<br>n={touched} touches · {esc(s.get("sessions", "?"))} sessions · '
        f'{esc(s.get("from", "?"))} → {esc(s.get("to", "?"))}</span></div>')

    # playbook: a rate-sheet table
    ft = summary.get("first_touch") or {}
    rows = ""
    for k, label in _LEVEL_LABELS.items():
        price = amd.get(k)
        st = ft.get(k) or {}
        if not isinstance(price, (int, float)) or not st:
            continue
        rows += (
            f'<tr><td class="mono lvl">{label}</td>'
            f'<td class="mono">{fnum(float(price))}</td>'
            f'<td class="mono amber">{float(st.get("fakeout_pct") or 0):.0f}%</td>'
            f'<td class="mono">≈{float(st.get("fakeout_median_overshoot_pts") or 0):.0f} pts</td>'
            f'<td class="mono up">+{float(st.get("fakeout_median_mfe60_pts") or 0):.0f} pts</td>'
            f'<td class="mono dn">{float(st.get("fakeout_median_mae120_pts") or 0):.0f} pts</td></tr>')
    table = (
        '<table class="sheet"><thead><tr>'
        "<th>IF FIRST TOUCH TODAY</th><th>PRICE</th><th>FAKEOUT</th>"
        "<th>STOPS RUN</th><th>MED. BOUNCE 60M</th><th>RISK 120M</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>")

    n = (summary.get("sample") or {}).get("sessions", "?")
    body = (f'<div class="herogrid"><div>{left}</div>{right}</div>{table}')
    return section("01", f"Today's Script — vs {esc(n)}-session history",
                   body, anchor="script")


# ----------------------------------------------------------------- ladder

_STRENGTH_RE = re.compile(r"\s*\[(\++)\]\s*")

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
        return (opener + f'<span class="rchev">+</span>{inner}</div>'
                + f'<div class="drawer" id="{did}" hidden>{drawer}</div>')


LADDER_NEAREST = 16


def _ladder_html(gex: dict, summary: dict | None) -> str:
    ft = (summary or {}).get("first_touch") or {}
    last = gex.get("current_price")
    last = float(last) if isinstance(last, (int, float)) else None
    dw = _Drawers()

    rows: list[tuple[float, str, str]] = []
    for lv in gex.get("levels") or []:
        try:
            p, raw = float(lv["price"]), str(lv["name"])
        except (KeyError, TypeError, ValueError):
            continue
        m = _STRENGTH_RE.search(raw)
        strength = len(m.group(1)) if m else 0
        name = _STRENGTH_RE.sub("", raw)
        cls = _level_cls(name)
        dots = f'<span class="dots">{"+" * strength}</span>' if strength else ""
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
                     f'<div class="lrow last"><span class="lname">LAST</span>'
                     f'<span class="lpx mono">{fnum(last)}</span></div>'))
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
    return "".join(parts)


def _public_ladder_html(gex: dict | None, brief: dict | None,
                        summary: dict | None) -> str:
    amd = (gex or {}).get("session_amd") or {}
    ft = (summary or {}).get("first_touch") or {}
    nq = ((brief or {}).get("ctx") or {}).get("NQ") or {}
    last = nq.get("last") if isinstance(nq.get("last"), (int, float)) \
        else (gex or {}).get("current_price")
    dw = _Drawers()
    rows: list[tuple[float, str]] = []
    for key, label in _LEVEL_LABELS.items():
        v = amd.get(key)
        if not isinstance(v, (int, float)):
            continue
        st = ft.get(key) or {}
        odds = ""
        if st:
            odds = (f'<span class="lodds mono">{float(st.get("fakeout_pct") or 0):.0f}% fake '
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
                     f'<div class="lrow last roomy"><span class="lname">LAST</span>'
                     f'<span class="lpx mono">{fnum(float(last))}</span></div>'))
    rows.sort(key=lambda r: r[0], reverse=True)
    return "".join(r[1] for r in rows)


# ------------------------------------------------------------- structure

def _buckets_html(summary: dict | None) -> str:
    out = ""
    for rng, b in ((summary or {}).get("time_buckets") or {}).items():
        if not isinstance(b, dict):
            continue
        out += (
            f'<div class="bucket" data-range="{esc(rng)}">'
            f'<span class="brange mono">{esc(rng)}</span>'
            f'<span class="bfake mono">{float(b.get("fakeout_pct") or 0):.0f}%</span>'
            f'<span class="bsub mono">n={int(b.get("touches") or 0)} · '
            f'+{float(b.get("fakeout_median_mfe60_pts") or 0):.0f}/60m</span></div>')
    if not out:
        return ""
    return (f'<div class="colhead mono">FAKEOUT % BY TIME OF FIRST TOUCH</div>'
            f'<div class="buckets">{out}</div>')


def build_structure(gex: dict | None, brief: dict | None, summary: dict | None,
                    age: float | None, public: bool) -> str:
    if public:
        title = "Session Liquidity Map · NQ"
        ladder = _public_ladder_html(gex, brief, summary)
        note = ("levels computed from price data · odds from the SweepStats "
                "dataset · click a level for its playbook")
        if not ladder:
            return section("02", title, "", anchor="structure", nofeed=True)
    else:
        title = f'GEX Level Ladder · {esc((gex or {}).get("symbol") or "NQ")}'
        if gex is None:
            return section("02", title, "", anchor="structure", nofeed=True)
        ladder = _ladder_html(gex, summary)
        note = "click a level for its playbook"

    clock = (
        f'<div class="clockbig mono" id="et-big">--:--:--</div>'
        f'<div class="clocksub mono">EASTERN TIME</div>'
        '<div class="tlwrap"><div class="tl">'
        + "".join(f'<div class="tlblock {c}" style="left:{lo / 14.4:.2f}%;'
                  f'width:{(hi - lo) / 14.4:.2f}%"></div>'
                  for c, lo, hi in (("tl-asia", 1080, 1440),
                                    ("tl-london", 120, 300),
                                    ("tl-ny", 570, 960)))
        + '<div class="tlneedle" id="tl-needle"></div></div>'
        + '<div class="tlticks">'
        + "".join(f'<span class="tltick mono" style="left:{h / .24:.1f}%">{h:02d}</span>'
                  for h in (0, 6, 12, 18))
        + "</div></div>"
        + "".join(
            f'<div class="crow" id="sess-{k}"><span class="cname mono">{n}</span>'
            f'<span class="cwin mono">{w}</span>'
            f'<span class="cstate mono" id="state-{k}">—</span>'
            f'<span class="ceta mono" id="eta-{k}"></span></div>'
            for k, n, w in (("asia", "ASIA", "18:00–00:00"),
                            ("london", "LONDON", "02:00–05:00"),
                            ("ny", "NEW YORK", "09:30–16:00")))
        + _buckets_html(summary))

    body = (f'<div class="structgrid">'
            f'<div class="ladcol"><div class="ladder">{ladder}</div>'
            f'<div class="lnote mono">{note}</div></div>'
            f'<div class="clkcol">{clock}</div></div>')
    return section("02", title, body, age, anchor="structure")


# ------------------------------------------------------------------ stats

_BUCKET_TITLES = {"asia_high": "LONDON SWEEPS ASIA HIGH",
                  "asia_low": "LONDON SWEEPS ASIA LOW",
                  "none": "NO LONDON SWEEP",
                  "both": "LONDON SWEEPS BOTH SIDES"}


def _first_touch_pane(summary: dict) -> str:
    rows = ""
    for k, label in _LEVEL_LABELS.items():
        st = (summary.get("first_touch") or {}).get(k)
        if not isinstance(st, dict):
            continue
        pct = float(st.get("fakeout_pct") or 0)
        rows += (
            f'<tr><td class="mono lvl">{label}</td>'
            f'<td class="barcell"><span class="bar"><span class="fill" '
            f'style="width:{pct:.1f}%"></span></span></td>'
            f'<td class="mono amber">{pct:.0f}%</td>'
            f'<td class="mono">{int(st.get("touched") or 0)}</td>'
            f'<td class="mono up">+{float(st.get("fakeout_median_mfe60_pts") or 0):.0f} pts</td>'
            f'<td class="mono dn">{float(st.get("fakeout_median_mae120_pts") or 0):.0f} pts</td></tr>')
    return ('<table class="sheet"><thead><tr><th>LEVEL</th><th></th>'
            "<th>FAKEOUT</th><th>TOUCHES</th><th>MED. REVERSAL 60M</th>"
            f"<th>STOP-RUN RISK</th></tr></thead><tbody>{rows}</tbody></table>")


def _matrix_pane(summary: dict) -> str:
    rows = ""
    for k, title in _BUCKET_TITLES.items():
        b = (summary.get("london_manipulation") or {}).get(k)
        if not isinstance(b, dict):
            continue
        days = int(b.get("days") or 0)
        up = float(b.get("ny_up_pct") or 0)
        med = float(b.get("median_ny_change_pts") or 0)
        low_n = ' <span class="lown">low sample</span>' if days < 5 else ""
        rows += (
            f'<tr><td class="mono lvl">{title}{low_n}</td>'
            f'<td class="mono">{days}</td>'
            f'<td class="mono {"up" if up >= 50 else "dn"}">{up:.0f}%</td>'
            f'<td class="mono {chg_cls(med)}">{med:+.1f} pts</td></tr>')
    return ('<table class="sheet"><thead><tr><th>OVERNIGHT PATTERN</th>'
            "<th>DAYS</th><th>NY CLOSED UP</th><th>MEDIAN NY MOVE</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>")


_REGIME_LABELS = {"chop": "CONSOLIDATION (CHOP)", "mixed": "MIXED",
                  "trend_up": "TRENDING UP", "trend_down": "TRENDING DOWN"}


def _conditions_pane(summary: dict) -> str:
    """Cross-market confirmation + regime + event days — the conditioning
    layers that turn level stats into a playbook."""
    cm = summary.get("cross_market") or {}
    rows = ""
    for k, label in (("confirmed", "ES SWEPT ITS LEVEL TOO (CONFIRMED)"),
                     ("diverged", "NQ SWEPT ALONE (DIVERGED)")):
        v = cm.get(k) or {}
        if not v.get("touches"):
            continue
        rows += (f'<tr><td class="mono lvl">{label}</td>'
                 f'<td class="mono">{int(v["touches"])}</td>'
                 f'<td class="mono amber">{float(v["fakeout_pct"] or 0):.0f}%</td>'
                 f'<td class="mono">≈{float(v["fakeout_median_overshoot_pts"] or 0):.0f} pts</td>'
                 f'<td class="mono up">+{float(v["fakeout_median_mfe60_pts"] or 0):.0f} pts</td></tr>')
    cross = ""
    if rows:
        untagged = int(cm.get("untagged_touches") or 0)
        cross = (
            '<div class="colhead mono">CROSS-MARKET CONFIRMATION — WAS ES BEHIND THE SWEEP?</div>'
            '<table class="sheet"><thead><tr><th>AT NQ\'S FIRST TOUCH</th>'
            '<th>TOUCHES</th><th>FAKEOUT</th><th>STOPS RUN</th><th>MED. BOUNCE 60M</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            + (f'<div class="lnote mono">a sweep without broad index participation '
               f'is arbitrage-suspect — it reverses more and runs shallower'
               f'{f" · {untagged} early touches untagged (predate ES data)" if untagged else ""}'
               f'</div>'))

    rrows = ""
    for k, v in (summary.get("regime") or {}).items():
        if not v.get("days"):
            continue
        low_n = ' <span class="lown">low sample</span>' if v["days"] < 5 else ""
        rrows += (f'<tr><td class="mono lvl">{_REGIME_LABELS.get(k, k.upper())}{low_n}</td>'
                  f'<td class="mono">{int(v["days"])}</td>'
                  f'<td class="mono">{float(v["ny_up_pct"] or 0):.0f}%</td>'
                  f'<td class="mono amber">{float(v["fakeout_pct"] or 0):.0f}%</td>'
                  f'<td class="mono up">+{float(v["fakeout_median_mfe60_pts"] or 0):.0f} pts</td></tr>')
    regime = ""
    if rrows:
        regime = (
            '<div class="colhead mono" style="margin-top:22px">DAILY REGIME — WHICH PLAYBOOK APPLIES</div>'
            '<table class="sheet"><thead><tr><th>TAPE (20-SESSION EFFICIENCY RATIO)</th>'
            '<th>DAYS</th><th>NY UP</th><th>FAKEOUT</th><th>MED. BOUNCE 60M</th>'
            f'</tr></thead><tbody>{rrows}</tbody></table>'
            '<div class="lnote mono">fades pay biggest in consolidation — '
            'trend days favor the break side</div>')

    erows = ""
    for k, v in (summary.get("event_days") or {}).items():
        if not v.get("days"):
            continue
        label = {"none": "ORDINARY DAYS"}.get(k, _EVENT_LABELS.get(k, k).upper() + " DAYS")
        low_n = ' <span class="lown">low sample</span>' if v["days"] < 5 else ""
        med = float(v["median_ny_change_pts"] or 0)
        erows += (f'<tr><td class="mono lvl">{label}{low_n}</td>'
                  f'<td class="mono">{int(v["days"])}</td>'
                  f'<td class="mono">{float(v["ny_up_pct"] or 0):.0f}%</td>'
                  f'<td class="mono {chg_cls(med)}">{med:+.1f} pts</td>'
                  f'<td class="mono">{float(v["median_ny_range_pts"] or 0):.0f} pts</td></tr>')
    events = ""
    if erows:
        events = (
            '<div class="colhead mono" style="margin-top:22px">FLOW-EVENT DAYS — WHEN STRUCTURE TAKES A BACK SEAT</div>'
            '<table class="sheet"><thead><tr><th>DAY TYPE</th><th>DAYS</th>'
            '<th>NY UP</th><th>MEDIAN NY MOVE</th><th>MEDIAN NY RANGE</th>'
            f'</tr></thead><tbody>{erows}</tbody></table>')

    return cross + regime + events


def build_stats(summary: dict | None, age: float | None) -> str:
    if summary is None:
        return section("03", "The Numbers", "", anchor="stats", nofeed=True)
    cond = _conditions_pane(summary)
    cond_tab = cond_pane = ""
    if cond:
        cond_tab = ('<span class="tabsep">/</span>'
                    '<button class="tab" data-pane="pane-cond">CONDITIONS</button>')
        cond_pane = (f'<div class="tabpane" id="pane-cond">'
                     f'<div class="pane-title mono">CONDITIONS</div>{cond}</div>')
    body = (
        '<div class="tabbar mono">'
        '<button class="tab active" data-pane="pane-ft">FIRST-TOUCH BOARD</button>'
        '<span class="tabsep">/</span>'
        '<button class="tab" data-pane="pane-lm">LONDON MATRIX</button>'
        f'{cond_tab}'
        "</div>"
        f'<div class="tabpane active" id="pane-ft">'
        f'<div class="pane-title mono">FIRST-TOUCH BOARD</div>{_first_touch_pane(summary)}</div>'
        f'<div class="tabpane" id="pane-lm">'
        f'<div class="pane-title mono">LONDON MATRIX</div>{_matrix_pane(summary)}</div>'
        f'{cond_pane}')
    return section("03", "The Numbers", body, age, anchor="stats")


# ----------------------------------------------------------------- record

def build_record(sb: dict | None) -> str:
    """The Record — the call grades itself (roadmap feature 5, USP #1).

    Walk-forward: every row was computed with only the data available
    before that date, so every grade is out-of-sample. Verdicts are
    published either way — including "no edge".
    """
    title = "The Record — every call graded, walk-forward"
    if not sb or not (sb.get("record") or {}).get("calls"):
        return section("04", title, "", anchor="record", nofeed=True)
    r = sb["record"]
    fo = sb["fakeout_oos"]
    lt = sb["latest"]

    # latest graded call, stamped (show the call-side probability, not up_pct)
    said = float(lt["said_up_pct"])
    if lt["call"] != "up":
        said = 100 - said
    act = float(lt["actual_change_pts"])
    stamp = ('<span class="stamp hit">HIT</span>' if lt["hit"]
             else '<span class="stamp miss">MISS</span>')
    ev_chips = "".join(
        f' <span class="stamp open">{esc(_EVENT_LABELS.get(k, k).upper())} DAY</span>'
        for k in (lt.get("events") or []))
    left = (
        f'<div class="kicker mono">LATEST GRADED CALL · {esc(lt["date"])} '
        f'({esc(lt["weekday"]).upper()})</div>'
        f'<h1 class="verdict vsm">{esc(_PATTERN_TITLES[lt["pattern"]])}</h1>'
        f'<p class="dek">The script said NY closes '
        f'<span class="{"up" if lt["call"] == "up" else "dn"}">{esc(lt["call"])} '
        f'— {said:.0f}% of {int(lt["prior_days"])} prior days</span>. '
        f'New York closed <span class="mono {chg_cls(act)}">{act:+.1f} pts</span>. '
        f'{stamp}{ev_chips}</p>'
        f'<div class="l10 mono">'
        + "".join(f'<b class="{"up" if h else "dn"}">{"&check;" if h else "&cross;"}</b>'
                  for h in r["last10"])
        + '<span class="l10cap">LAST 10 CALLS</span></div>')

    right = (
        f'<div class="bigstat"><span class="bignum">{float(r["hit_pct"]):.0f}<i>%</i></span>'
        f'<span class="bigcap mono">DIRECTIONAL HIT RATE — {int(r["hits"])}/{int(r["calls"])} '
        f'calls graded out-of-sample<br>{esc(r["first_call"])} → {esc(r["last_call"])} · '
        f'{int(r["no_call_days"])} no-call days (low sample / coin flip)</span></div>')

    # the claims sheet: what we test, how it's running, verdict attached
    dir_pct = float(r["hit_pct"])
    dir_verdict = ('<span class="stamp hit">EDGE</span>' if dir_pct >= 60 else
                   '<span class="stamp miss">NO EDGE YET</span>' if dir_pct <= 55 else
                   '<span class="stamp open">UNPROVEN</span>')
    fo_pct = float(fo["oos_pct"] or 0)
    fo_claim = float(fo["avg_claim_pct"] or 0)
    fo_verdict = ('<span class="stamp hit">HOLDS</span>'
                  if abs(fo_pct - fo_claim) <= 10 and fo_pct > 50
                  else '<span class="stamp open">UNPROVEN</span>')
    rows = (
        f'<tr><td class="mono lvl">LONDON PATTERN CALLS NY DIRECTION</td>'
        f'<td class="mono">{int(r["calls"])} calls</td>'
        f'<td class="mono amber">{dir_pct:.0f}% hit</td>'
        f'<td>{dir_verdict}</td></tr>'
        f'<tr><td class="mono lvl">FIRST NY TOUCH IS USUALLY A FAKEOUT</td>'
        f'<td class="mono">{int(fo["touches"])} touches</td>'
        f'<td class="mono amber">claimed ~{fo_claim:.0f}% · ran {fo_pct:.0f}%</td>'
        f'<td>{fo_verdict}</td></tr>')
    for k, ttl in _BUCKET_TITLES.items():
        v = (sb.get("by_pattern") or {}).get(k) or {}
        if not v.get("calls"):
            continue
        rows += (f'<tr class="subrow"><td class="mono lvl sub">— {ttl}</td>'
                 f'<td class="mono">{int(v["calls"])} calls</td>'
                 f'<td class="mono">{float(v["hit_pct"]):.0f}% hit</td><td></td></tr>')
    table = ('<table class="sheet"><thead><tr><th>CLAIM UNDER TEST</th>'
             '<th>OUT-OF-SAMPLE</th><th>RESULT</th><th>VERDICT</th>'
             f'</tr></thead><tbody>{rows}</tbody></table>')

    note = ('<div class="lnote mono">walk-forward: every call is computed from '
            'the sessions before its date only — nothing here is backfit. '
            'negative verdicts are published, not buried.</div>')
    body = f'<div class="herogrid rec"><div>{left}</div>{right}</div>{table}{note}'
    return section("04", title, body,
                   right=f'<span class="st">THROUGH {esc(r["last_call"])}</span>',
                   anchor="record")


# ----------------------------------------------------------------- ledger

_OUTCOME_CHIPS = {"blown": ("BLOWN", "miss"), "passed": ("PASSED", "hit"),
                  "funded": ("FUNDED", "hit"), "active": ("ACTIVE", "open")}


def _days_between(a: str | None, b: str | None) -> int | None:
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days
    except (TypeError, ValueError):
        return None


def build_ledger(ledger: dict | None) -> str:
    """The Ledger — what the accounts cost vs what payouts came back.

    Personal edition ONLY: this is account P&L, which the public/product
    page explicitly excludes. Fees are recorded facts — a missing
    cost_usd renders as unknown, never as an estimate.
    """
    title = "The Ledger — account economics"
    if not ledger:
        return section("05", title, "", anchor="ledger", nofeed=True)
    accts = [a for a in ledger["accounts"] if isinstance(a, dict)]

    costs = [a["cost_usd"] for a in accts if isinstance(a.get("cost_usd"), (int, float))]
    unknown_fees = len(accts) - len(costs)
    spent = sum(costs)
    payouts = [p for a in accts for p in (a.get("payouts") or [])
               if isinstance(p, dict) and isinstance(p.get("amount_usd"), (int, float))]
    paid_out = sum(p["amount_usd"] for p in payouts)

    evals = [a for a in accts if a.get("type") == "eval"]
    resolved = [a for a in evals if a.get("outcome") in ("passed", "blown")]
    passed = sum(1 for a in resolved if a["outcome"] == "passed")
    lifespans = [d for a in accts if a.get("outcome") == "blown"
                 if (d := _days_between(a.get("bought"), a.get("outcome_date"))) is not None]
    med_life = sorted(lifespans)[len(lifespans) // 2] if lifespans else None

    if unknown_fees:
        spent_txt = (f"${spent:,.0f} recorded · {unknown_fees} fee"
                     f"{'s' if unknown_fees > 1 else ''} unrecorded"
                     if costs else f"unknown — {unknown_fees} fees unrecorded")
        net_txt = "net unknowable until fees are filled in"
    else:
        spent_txt = f"${spent:,.0f} across {len(accts)} accounts"
        net = paid_out - spent
        net_txt = f'net {"+" if net >= 0 else "−"}${abs(net):,.0f}'

    left = (
        f'<div class="kicker mono">SINCE THE FIRST TRACKED ACCOUNT · '
        f'{len(accts)} ACCOUNTS ACROSS '
        f'{len({a.get("firm") for a in accts})} FIRMS</div>'
        f'<h1 class="verdict vsm">{len(accts)} accounts. '
        f'{len(payouts) or "Zero"} payout{"" if len(payouts) == 1 else "s"}.</h1>'
        f'<p class="dek">Evals passed <span class="mono">{passed}/{len(resolved)}</span> '
        f'of those resolved'
        + (f' · blown accounts lasted a median <span class="mono">{med_life} '
           f'day{"s" if med_life != 1 else ""}</span>' if med_life is not None else "")
        + f'. Spent: <span class="mono amber">{esc(spent_txt)}</span> · '
        f'{esc(net_txt)}.</p>')
    right = (
        f'<div class="bigstat"><span class="bignum">${paid_out:,.0f}</span>'
        f'<span class="bigcap mono">TOTAL PAYOUTS RECEIVED — the only number '
        f'that ever comes back<br>fees go out on every account; this is the '
        f'return side of the ledger</span></div>')

    rows = ""
    for a in accts:
        chip_txt, chip_cls = _OUTCOME_CHIPS.get(str(a.get("outcome")), ("?", "open"))
        cost = a.get("cost_usd")
        life = _days_between(a.get("bought"), a.get("outcome_date"))
        a_pay = sum(p.get("amount_usd") or 0 for p in (a.get("payouts") or []))
        rows += (
            f'<tr><td class="mono lvl">{esc(a.get("firm") or "?").upper()} '
            f'{esc(a.get("label") or "")}</td>'
            f'<td class="mono">{esc(a.get("bought") or "—")}</td>'
            f'<td class="mono amber">{f"${cost:,.0f}" if isinstance(cost, (int, float)) else "—"}</td>'
            f'<td class="mono">{f"{life}d" if life is not None else "—"}</td>'
            f'<td><span class="stamp {chip_cls}">{chip_txt}</span></td>'
            f'<td class="mono">{f"${a_pay:,.0f}" if a_pay else "—"}</td></tr>')
    table = ('<table class="sheet"><thead><tr><th>ACCOUNT</th><th>BOUGHT</th>'
             '<th>COST</th><th>LIFESPAN</th><th>OUTCOME</th><th>PAYOUTS</th>'
             f'</tr></thead><tbody>{rows}</tbody></table>')

    note = ('<div class="lnote mono">fees are recorded facts, never estimates — '
            'fill cost_usd in data/prop_ledger.json to complete the net. '
            'personal edition only; never rendered on the public page.</div>')
    body = f'<div class="herogrid rec"><div>{left}</div>{right}</div>{table}{note}'
    return section("05", title, body,
                   right=f'<span class="st">UPDATED {esc(ledger.get("updated") or "?")}</span>',
                   anchor="ledger")


# ------------------------------------------------------------------- wire

def build_wire(brief: dict | None, age: float | None, num: str = "05") -> str:
    if brief is None:
        return section(num, "Wire", "", anchor="wire", nofeed=True)
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
        toggle = (f'<button class="linkbtn mono" id="wire-more">SHOW ALL '
                  f'({len(news)}) +</button>')
    cal = ""
    for ev in brief.get("calendar") or []:
        if not isinstance(ev, dict):
            continue
        cal += (f'<span class="calitem mono">{esc(ev.get("time") or "")} '
                f'{esc(ev.get("title") or "")} · {esc(ev.get("impact") or "")}'
                f'{" · f:" + esc(ev["forecast"]) if ev.get("forecast") else ""}</span>')
    if cal:
        cal = (f'<div class="colhead mono" style="margin-top:18px">TODAY\'S CALENDAR</div>'
               f'<div class="cal">{cal}</div>')
    return section(num, "Wire", items + toggle + cal, age, anchor="wire")


# ------------------------------------------------------------------ page

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:#0a0d10;color:#c9d4de;
  font:14px/1.55 "Segoe UI",system-ui,sans-serif;
  background-image:radial-gradient(900px 420px at 70% -140px,rgba(77,208,225,.045),transparent)}
::selection{background:#ffb454;color:#0a0d10}
::-webkit-scrollbar{width:10px}
::-webkit-scrollbar-thumb{background:#1e2833}
::-webkit-scrollbar-track{background:#0a0d10}
a{color:#c9d4de;text-decoration:none}
button{font:inherit;color:inherit;background:none;border:none;cursor:pointer}
.mono{font-family:"Cascadia Mono","Consolas",monospace}
.disp{font-family:"Bahnschrift SemiBold Condensed","Bahnschrift","Arial Narrow",
  "Segoe UI",sans-serif}
.up{color:#3ddc84}.dn{color:#ff5f56}.flat{color:#5c6a78}.amber{color:#ffb454}
.lown{color:#ff5f56;font-size:10px;letter-spacing:.1em;text-transform:uppercase}
.page{max-width:1200px;margin:0 auto;padding:0 32px}
.stale-dim{opacity:.55}
.nofeed{color:#39485a;font:15px "Cascadia Mono","Consolas",monospace;
  letter-spacing:.4em;padding:34px 0 8px}
/* ---------- masthead ---------- */
.mast{display:flex;align-items:baseline;justify-content:space-between;
  padding:26px 0 14px}
.brand{font-family:"Bahnschrift SemiBold Condensed","Bahnschrift","Arial Narrow",sans-serif;
  font-size:27px;letter-spacing:.04em;color:#e8eef4;white-space:nowrap}
.brand em{font-style:normal;color:#ffb454}
.brand .bsub{display:block;font:10px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.34em;margin-top:1px}
.mnav{display:flex;gap:4px;font:11px "Cascadia Mono","Consolas",monospace;
  letter-spacing:.1em;color:#5c6a78}
.mnav a{color:#5c6a78;padding:2px 8px}
.mnav a:hover{color:#ffb454}
.mnav .nsep{color:#2b3844}
.mmeta{text-align:right;font:10.5px "Cascadia Mono","Consolas",monospace;
  color:#5c6a78;letter-spacing:.06em;line-height:1.7}
.mmeta .n{color:#ffb454}
.mmeta .clk{color:#c9d4de;font-size:12px}
.rule2{border-top:3px solid #2b3844;border-bottom:1px solid #1e2833;height:6px}
/* ---------- ticker ---------- */
.ticker{overflow:hidden;border-bottom:1px solid #1e2833;padding:7px 0;
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 3%,#000 97%,transparent)}
.tickrun{display:inline-flex;white-space:nowrap;animation:tick 60s linear infinite}
.ticker:hover .tickrun{animation-play-state:paused}
.tickseg{display:inline-flex}
.ti{font:12px "Cascadia Mono","Consolas",monospace;color:#8b9aa8;padding:0 14px}
.ti b{font-weight:normal;color:#c9d4de}
.sep{color:#2b3844;font:12px "Cascadia Mono","Consolas",monospace}
@keyframes tick{from{transform:translateX(0)}to{transform:translateX(-50%)}}
/* ---------- sections ---------- */
.sec{padding:34px 0 6px}
.runhead{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:1px solid #1e2833;padding-bottom:8px;margin-bottom:18px;
  cursor:pointer;user-select:none}
.rh-l{font:11px "Cascadia Mono","Consolas",monospace;letter-spacing:.22em;
  text-transform:uppercase;color:#8b9aa8}
.rh-l i{font-style:normal;color:#ffb454;margin-right:14px}
.rh-l .fold{font-weight:normal;color:#2b3844;margin-left:14px}
.runhead:hover .fold{color:#ffb454}
.sec.collapsed .fold{color:#ffb454}
.sec.collapsed .secbody{display:none}
.rh-r{display:flex;gap:14px;font:10px "Cascadia Mono","Consolas",monospace;
  letter-spacing:.1em;color:#5c6a78}
.st{color:#5c6a78}.st.warn{color:#ffb454}.st.bad{color:#ff5f56}
.colhead{font-size:10px;letter-spacing:.2em;color:#39485a;margin:2px 0 10px}
.linkbtn{font-size:10.5px;letter-spacing:.14em;color:#ffb454;padding:10px 0 2px;display:block}
.linkbtn:hover{text-decoration:underline;text-underline-offset:3px}
/* ---------- hero ---------- */
.herogrid{display:grid;grid-template-columns:1.35fr .9fr;gap:44px;
  align-items:end;padding:6px 0 26px}
.kicker{font-size:10.5px;letter-spacing:.24em;color:#5c6a78;margin-bottom:14px}
h1.verdict{font-family:"Bahnschrift SemiBold Condensed","Bahnschrift","Arial Narrow",sans-serif;
  font-size:64px;line-height:.98;letter-spacing:.005em;color:#e8eef4;
  text-wrap:balance;margin:0 0 16px}
.dek{font-size:16px;line-height:1.6;color:#8b9aa8;max-width:56ch}
.dek .mono{font-size:14.5px}
.tags{margin-top:16px;display:flex;gap:12px;font-size:11px;letter-spacing:.08em}
.tag.warn{color:#ffb454}
.dek.flownote{margin-top:10px;font-size:13.5px;color:#ffb454;max-width:62ch}
.bigstat{text-align:right;padding-bottom:4px}
.bignum{font-family:"Bahnschrift SemiBold Condensed","Bahnschrift","Arial Narrow",sans-serif;
  font-size:150px;line-height:.82;color:#ffb454;display:block}
.bignum i{font-style:normal;font-size:64px}
.bigcap{display:block;margin-top:12px;font-size:10px;line-height:1.7;
  letter-spacing:.1em;color:#5c6a78;text-transform:uppercase}
/* ---------- record ---------- */
h1.verdict.vsm{font-size:42px}
.herogrid.rec{align-items:start;padding-bottom:18px}
.herogrid.rec .bignum{font-size:110px}
.stamp{display:inline-block;font:10.5px "Cascadia Mono","Consolas",monospace;
  letter-spacing:.2em;padding:2px 10px 1px;border:1px solid;margin-left:6px;
  vertical-align:2px}
.stamp.hit{color:#3ddc84;border-color:rgba(61,220,132,.55)}
.stamp.miss{color:#ff5f56;border-color:rgba(255,95,86,.55)}
.stamp.open{color:#ffb454;border-color:rgba(255,180,84,.55)}
.l10{display:flex;gap:5px;align-items:center;margin-top:18px}
.l10 b{font-weight:normal;font-size:10px;width:19px;height:19px;display:flex;
  align-items:center;justify-content:center;border:1px solid #1e2833}
.l10 b.up{border-color:rgba(61,220,132,.4)}
.l10 b.dn{border-color:rgba(255,95,86,.4)}
.l10cap{font-size:9px;letter-spacing:.18em;color:#39485a;margin-left:10px}
.sheet .lvl.sub{color:#5c6a78;padding-left:16px}
tr.subrow td{font-size:12px;color:#8b9aa8}
/* ---------- rate-sheet tables ---------- */
table.sheet{width:100%;border-collapse:collapse}
.sheet th{font:9.5px "Cascadia Mono","Consolas",monospace;letter-spacing:.16em;
  color:#39485a;text-align:left;font-weight:normal;padding:0 14px 7px 0;
  border-bottom:1px solid #2b3844}
.sheet td{padding:9px 14px 9px 0;border-bottom:1px solid #161d26;font-size:13px}
.sheet tr:hover td{background:rgba(255,255,255,.022)}
.sheet .lvl{color:#4dd0e1;font-size:11px;letter-spacing:.08em}
.sheet .barcell{width:26%}
.bar{display:block;height:5px;background:rgba(255,180,84,.12);position:relative}
.fill{position:absolute;inset:0 auto 0 0;background:#ffb454}
/* ---------- structure ---------- */
.structgrid{display:grid;grid-template-columns:1.5fr 1fr;gap:0}
.ladcol{padding-right:36px;border-right:1px solid #1e2833}
.clkcol{padding-left:36px}
.ladder{display:flex;flex-direction:column}
.lrow{display:flex;align-items:baseline;gap:9px;padding:3.5px 4px;min-height:22px}
.lrow[data-d]{cursor:pointer}
.lrow[data-d]:hover{background:rgba(255,255,255,.03)}
.lrow[data-d]:hover .lname{color:#c9d4de}
.rchev{color:#2b3844;font:10px "Cascadia Mono","Consolas",monospace;flex:none;width:8px}
.lrow.open .rchev{color:#ffb454}
.lrow .lname{color:#5c6a78;font-size:11.5px;letter-spacing:.05em}
.lrow .lpx{margin-left:auto;font-size:12.5px;color:#8b9aa8}
.lrow.gamma .lname{color:#8b9aa8}
.lrow.cwall .lname,.lrow.pwall .lname,.lrow.flip .lname{color:#c9d4de;font-weight:600}
.lrow.cwall .lname::after{content:" ▲";color:#3ddc84;font-size:9px}
.lrow.pwall .lname::after{content:" ▼";color:#ff5f56;font-size:9px}
.lrow.flip .lname::after{content:" ◆";color:#ffb454;font-size:9px}
.lrow.flip .lpx,.lrow.cwall .lpx,.lrow.pwall .lpx{color:#c9d4de}
.lrow.amd .lname{color:#4dd0e1;font-weight:600}
.lrow.amd .lpx{color:#4dd0e1}
.lrow.last{background:#ffb454;margin:3px 0;padding:5px 8px}
.lrow.last .lname,.lrow.last .lpx{color:#0a0d10;font-weight:700}
.lrow.last .lname{letter-spacing:.2em;font-size:10.5px}
.lrow.gap .lname{color:#232e3a;letter-spacing:4px}
.lrow.roomy{padding:10px 6px}
.lrow.roomy .lname{font-size:12.5px}.lrow.roomy .lpx{font-size:14.5px}
.lrow.roomy.last{padding:10px 8px}
.lodds{margin-left:16px;font-size:10.5px;color:#5c6a78}
.lnote{margin-top:12px;font-size:9.5px;letter-spacing:.14em;color:#39485a;
  text-transform:uppercase}
.dots{color:#ffb454;font-size:10px;letter-spacing:1px}
.drawer{border-left:1px solid #ffb454;margin:3px 0 8px 18px;padding:9px 0 9px 16px;
  font-size:12px;color:#8b9aa8;line-height:1.6}
.dgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px 18px}
.dcell{display:flex;flex-direction:column}
.dcell span{font:9px "Cascadia Mono","Consolas",monospace;color:#39485a;letter-spacing:.12em}
.dcell b{font-weight:normal;font-size:12.5px;color:#c9d4de}
/* clock column */
.clockbig{font-size:38px;color:#e8eef4;letter-spacing:.02em}
.clocksub{font-size:9.5px;letter-spacing:.34em;color:#39485a;margin:0 0 18px}
.tlwrap{margin-bottom:16px}
.tl{position:relative;height:10px;background:#10151b;overflow:hidden}
.tlblock{position:absolute;top:0;bottom:0;opacity:.5}
.tl-asia{background:#4dd0e1}.tl-london{background:#ffb454}.tl-ny{background:#3ddc84}
.tlneedle{position:absolute;top:-2px;bottom:-2px;width:2px;background:#e8eef4;left:0}
.tlticks{position:relative;height:13px;margin-top:3px}
.tltick{position:absolute;font-size:9px;color:#39485a}
.crow{display:flex;align-items:baseline;gap:10px;padding:9px 0;border-top:1px solid #161d26}
.cname{font-size:11px;letter-spacing:.18em;color:#c9d4de;width:86px}
.cwin{color:#39485a;font-size:10px}
.cstate{margin-left:auto;font-size:10.5px;letter-spacing:.12em}
.cstate.open{color:#3ddc84}.cstate.closed{color:#39485a}
.ceta{color:#5c6a78;font-size:10px;width:86px;text-align:right}
.buckets{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#161d26;
  border:1px solid #161d26}
.bucket{background:#0a0d10;padding:9px 12px;display:flex;flex-direction:column}
.bucket.active{box-shadow:inset 2px 0 0 #ffb454}
.bucket.active .brange{color:#ffb454}
.brange{font-size:9.5px;letter-spacing:.12em;color:#39485a}
.bfake{font-size:21px;color:#c9d4de;margin:1px 0}
.bsub{font-size:9.5px;color:#39485a}
.buckets+.colhead{margin-top:16px}
.colhead{margin-top:18px}
/* ---------- tabs ---------- */
.tabbar{display:flex;gap:10px;align-items:baseline;font-size:11px;
  letter-spacing:.16em;margin-bottom:16px}
.tab{color:#39485a;padding:0}
.tab:hover{color:#8b9aa8}
.tab.active{color:#ffb454}
.tabsep{color:#232e3a}
.tabpane{display:none}
.tabpane.active{display:block}
.pane-title{display:none;font-size:10px;letter-spacing:.2em;color:#39485a;margin:18px 0 8px}
/* ---------- wire ---------- */
.wrow{display:flex;gap:18px;padding:8px 0;border-bottom:1px solid #161d26;
  font-size:14px;align-items:baseline}
.wrow a:hover{color:#ffb454;text-decoration:underline;text-underline-offset:3px}
.wrow.wmore{display:none}
.wire-open .wrow.wmore{display:flex}
.wtime{color:#ffb454;font-size:11px;flex:none}
.cal{display:flex;gap:20px;flex-wrap:wrap}
.calitem{font-size:11px;color:#4dd0e1}
/* ---------- footer ---------- */
.foot{margin-top:44px;border-top:3px solid #2b3844;padding:16px 0 30px;
  display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;
  font:10px "Cascadia Mono","Consolas",monospace;color:#39485a;
  letter-spacing:.08em;line-height:1.8}
.foot .fh{color:#5c6a78;letter-spacing:.22em;display:block;margin-bottom:2px}
.foot .amber{color:#ffb454}
.foot .fc{text-align:center}.foot .fr{text-align:right}
/* ---------- expand-all (#png screenshot mode) ---------- */
body.expand-all .tabbar{display:none}
body.expand-all .tabpane{display:block}
body.expand-all .pane-title{display:block}
body.expand-all .wrow.wmore{display:flex}
body.expand-all #wire-more{display:none}
body.expand-all .lnote{display:none}
body.expand-all .tickrun{animation:none}
"""

JS = """
function etParts(){
  var p=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour12:false,
    hour:'2-digit',minute:'2-digit',second:'2-digit'}).formatToParts(new Date());
  var o={};p.forEach(function(x){o[x.type]=x.value});
  if(o.hour==='24')o.hour='00';return o;}
function utcStr(){
  return new Intl.DateTimeFormat('en-GB',{timeZone:'UTC',hour12:false,
    hour:'2-digit',minute:'2-digit'}).format(new Date());}
function fmtEta(mins){var h=Math.floor(mins/60),m=mins%60;
  return (h?h+'H ':'')+m+'M';}
var SESS=[['asia',1080,1440],['london',120,300],['ny',570,960]];
function tick(){
  var o=etParts(),et=o.hour+':'+o.minute+':'+o.second;
  var b=document.getElementById('et-big');if(b)b.textContent=et;
  var h=document.getElementById('mast-ny');if(h)h.textContent=o.hour+':'+o.minute+' NY';
  var u=document.getElementById('mast-utc');if(u)u.textContent=utcStr()+' UTC';
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

/* collapsible sections */
document.querySelectorAll('.runhead').forEach(function(h){
  h.addEventListener('click',function(e){
    if(e.target.closest('a,button'))return;
    var sec=h.parentElement;
    sec.classList.toggle('collapsed');
    h.querySelector('.fold').textContent=sec.classList.contains('collapsed')?'[+]':'[\\u2013]';
  });
});
/* ladder drawers */
document.querySelectorAll('.lrow[data-d]').forEach(function(row){
  row.addEventListener('click',function(){
    var d=document.getElementById(row.getAttribute('data-d'));
    if(!d)return;
    d.hidden=!d.hidden;
    row.classList.toggle('open',!d.hidden);
    var c=row.querySelector('.rchev');if(c)c.textContent=d.hidden?'+':'\\u2013';
  });
});
/* tabs */
document.querySelectorAll('.tab').forEach(function(t){
  t.addEventListener('click',function(){
    var wrap=t.closest('.secbody');
    wrap.querySelectorAll('.tab').forEach(function(x){x.classList.remove('active')});
    wrap.querySelectorAll('.tabpane').forEach(function(x){x.classList.remove('active')});
    t.classList.add('active');
    var p=document.getElementById(t.getAttribute('data-pane'));
    if(p)p.classList.add('active');
  });
});
/* wire show-more */
var wm=document.getElementById('wire-more');
if(wm){var wmOrig=wm.textContent;
  wm.addEventListener('click',function(){
    var p=wm.closest('.secbody');p.classList.toggle('wire-open');
    wm.textContent=p.classList.contains('wire-open')?'SHOW LESS \\u2013':wmOrig;
  });}
/* screenshot mode */
if(location.hash==='#png')document.body.classList.add('expand-all');
"""


def build_page(brief, brief_age, gex, gex_age, summary, summary_age,
               scoreboard: dict | None = None, ledger: dict | None = None,
               public: bool = False) -> str:
    n_sessions = "—"
    if summary:
        n_sessions = str((summary.get("sample") or {}).get("sessions", "—"))
    built = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    edition = ("PUBLIC EDITION · PRICE DATA ONLY<br>" if public else "")

    mast = (
        '<div class="mast">'
        '<div class="brand">NQ <em>SITUATION ROOM</em>'
        '<span class="bsub">BY SWEEPSTATS</span></div>'
        '<nav class="mnav">'
        '<a href="#script">SCRIPT</a><span class="nsep">/</span>'
        '<a href="#structure">STRUCTURE</a><span class="nsep">/</span>'
        '<a href="#stats">NUMBERS</a><span class="nsep">/</span>'
        '<a href="#record">RECORD</a><span class="nsep">/</span>'
        + ('' if public else '<a href="#ledger">LEDGER</a><span class="nsep">/</span>')
        + '<a href="#wire">WIRE</a></nav>'
        f'<div class="mmeta"><span class="clk" id="mast-ny">--:-- NY</span> · '
        f'<span id="mast-utc">--:-- UTC</span><br>'
        f'{edition}<span class="n">n={esc(n_sessions)} SESSIONS</span> · BUILD {esc(built)}</div>'
        '</div><div class="rule2"></div>')

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
        '<div class="foot">'
        f'<div><span class="fh">SWEEPSTATS</span>NQ session-level statistics — '
        f'a compounding dataset, updated daily.<br>{ages}</div>'
        '<div class="fc"><span class="fh">METHOD</span>5-minute closes decide. '
        'Fakeout = reclaim within 2 bars of the first NY breach of an '
        'Asia/London extreme.</div>'
        '<div class="fr"><span class="fh">DISCLAIMER</span>Past frequency ≠ future '
        'probability.<br>Not financial advice.</div>'
        "</div>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>NQ Situation Room</title>"
        f"<style>{CSS}</style></head><body><div class='page'>"
        + mast
        + build_ticker(brief)
        + build_hero(gex, summary)
        + build_structure(gex, brief, summary, gex_age, public)
        + build_stats(summary, summary_age)
        + build_record(scoreboard)
        + ("" if public else build_ledger(ledger))
        + build_wire(brief, brief_age, num="05" if public else "06")
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
    fully shown, ticker frozen) for the shareable screenshot.
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
    scoreboard = load_scoreboard()
    ledger = None if args.public else load_ledger()

    suffix = "_public" if args.public else ""
    out_html = RESULTS / f"situation_room{suffix}.html"
    out_png = RESULTS / f"situation_room{suffix}.png"

    out_html.write_text(
        build_page(brief, brief_age, gex, gex_age, summary, summary_age,
                   scoreboard, ledger, public=args.public),
        encoding="utf-8")
    print(f"HTML -> {out_html}")

    if args.png and render_png(out_html, out_png,
                               height=3350 if args.public else 4300):
        print(f"PNG  -> {out_png}")
    if args.open:
        webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()
