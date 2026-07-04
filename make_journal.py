"""
make_journal.py — Generate a draft trade journal entry for a day.

Assembles journal/YYYY-MM-DD.md from three sources:
  - my_trades.csv rows for that date (points, gross P&L, R:R computed)
  - data/tradeify_state.json guardrail snapshot, if present
  - results/morning_brief.json + results/gex_levels.json market context,
    included only when their capture date matches the journal date

Qualitative sections (execution, exit rationale, review) are emitted as
fill-in placeholders — the numbers are automated, the narrative is yours.

Usage:
    python make_journal.py              # today's entry
    python make_journal.py 2026-06-24   # specific date
    python make_journal.py --force      # overwrite an existing entry
"""

import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parent
TRADES_CSV = ROOT / "my_trades.csv"
STATE_FILE = ROOT / "data" / "tradeify_state.json"
BRIEF_JSON = ROOT / "results" / "morning_brief.json"
GEX_JSON = ROOT / "results" / "gex_levels.json"
JOURNAL_DIR = ROOT / "journal"

NQ_POINT_VALUE = 20.0  # $ per point per contract (E-mini)


def load_trades(day: str) -> list[dict]:
    """Rows from my_trades.csv matching the date, comments skipped."""
    if not TRADES_CSV.exists():
        return []
    trades = []
    with open(TRADES_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in f if not r.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        if (row.get("date") or "").strip() == day:
            trades.append({k: (v or "").strip() for k, v in row.items() if k})
    return trades


def _fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was misdecoded as cp1252 (e.g. 'Î“' -> 'Γ')."""
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def trade_metrics(t: dict) -> dict:
    """Points captured, gross P&L, and R multiple where computable."""
    entry, exit_, stop = _num(t.get("entry", "")), _num(t.get("exit_price", "")), _num(t.get("stop", ""))
    contracts = _num(t.get("contracts", "")) or 1
    sign = -1 if t.get("direction", "").lower() == "short" else 1
    m = {"contracts": int(contracts)}
    if entry is not None and exit_ is not None:
        m["points"] = sign * (exit_ - entry)
        m["pnl"] = m["points"] * NQ_POINT_VALUE * contracts
    if entry is not None and stop is not None and abs(entry - stop) > 0:
        m["risk_pts"] = abs(entry - stop)
        if "points" in m:
            m["r_multiple"] = m["points"] / m["risk_pts"]
    return m


def guardrail_snapshot() -> dict | None:
    if not STATE_FILE.exists():
        return None
    sys.path.insert(0, str(ROOT))
    from data.tradeify_account import check_guardrail

    state = json.loads(STATE_FILE.read_text())
    return check_guardrail(state["last_seen_balance"])


def market_context(day: str) -> dict:
    """Morning brief + GEX context, only if captured on the journal date."""
    ctx = {}
    if BRIEF_JSON.exists():
        brief = json.loads(BRIEF_JSON.read_text(encoding="utf-8"))
        if (brief.get("generated") or "")[:10] == day:
            ctx["prices"] = brief.get("ctx", {})
            ctx["sectors"] = brief.get("sectors", {})
    # levels and session_amd can come from different capture runs, so each
    # block is gated on its own date rather than the file as a whole
    if GEX_JSON.exists():
        gex = json.loads(GEX_JSON.read_text(encoding="utf-8"))
        if (gex.get("timestamp") or "")[:10] == day:
            for lv in gex.get("levels", []):
                lv["name"] = _fix_mojibake(lv["name"])
            ctx["gex_levels"] = gex
        amd = gex.get("session_amd", {})
        if amd.get("date") == day:
            ctx["gex_amd"] = amd
    return ctx


def fmt_trade(i: int, t: dict, m: dict) -> str:
    d = t.get("direction", "?").capitalize()
    rows = [
        f"## Trade {i} — {d} {t.get('asset', 'NQ')}1!",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Instrument | {t.get('asset', 'NQ')}1! ({m['contracts']} E-mini, ${NQ_POINT_VALUE:,.0f}/pt) |",
        f"| Direction | {d} |",
        f"| Entry | {t.get('entry') or '—'} |",
        f"| Stop | {t.get('stop') or '—'} |",
        f"| Exit | {t.get('exit_price') or '— (open?)'} |",
    ]
    if "points" in m:
        rows.append(f"| Points captured | {m['points']:+.2f} |")
        rows.append(f"| Gross P&L | {'+' if m['pnl'] >= 0 else ''}${m['pnl']:,.2f} |")
    if "r_multiple" in m:
        rows.append(f"| R:R achieved | {m['r_multiple']:.1f}:1 (risk {m['risk_pts']:.2f} pts) |")
    if t.get("outcome"):
        rows.append(f"| Outcome | {t['outcome']} |")

    gex_bits = [
        t.get("gex_level_type", ""),
        f"[{t['level_strength']}]" if t.get("level_strength") else "",
        "SMT confluence" if t.get("smt_confluence", "").lower() == "yes" else "",
        f"{t['gamma_regime']} gamma" if t.get("gamma_regime") else "",
    ]
    gex_line = " · ".join(b for b in gex_bits if b)

    rows += [
        "",
        "### Setup",
        t.get("setup_notes") or "_(fill in: what you saw)_",
        f"\n**GEX context:** {gex_line}" if gex_line else "",
        "",
        "### Execution",
        "_(fill in: entry quality, drawdown against stop, management)_",
        "",
        "### Exit rationale",
        "_(fill in: technical exit or other reason)_",
        "",
        "### What went right / what to review",
        "_(fill in)_",
    ]
    return "\n".join(r for r in rows if r is not None)


def fmt_context(day: str, ctx: dict, guard: dict | None) -> str:
    pretty = datetime.strptime(day, "%Y-%m-%d").strftime("%a %d %b %Y").replace(" 0", " ")
    out = [f"## Market context ({pretty})", ""]

    prices = ctx.get("prices")
    if prices:
        line = " | ".join(
            f"{sym} {v.get('last', '?')} ({v.get('chg_pct', 0):+.2f}%)" for sym, v in prices.items()
        )
        out.append(f"- Prices at brief: {line}")
    sectors = ctx.get("sectors")
    if sectors:
        moved = sorted(sectors.items(), key=lambda kv: abs(kv[1].get("chg_pct", 0)), reverse=True)[:3]
        out.append(
            "- Biggest sector moves: "
            + ", ".join(f"{v['name']} {v.get('chg_pct', 0):+.2f}%" for _, v in moved)
        )

    gex = ctx.get("gex_levels")
    if gex:
        px = gex.get("current_price")
        levels = gex.get("levels", [])
        flip = next((lv for lv in levels if lv["name"].lower().startswith("gamma flip") and "0dte" not in lv["name"].lower()), None)
        if flip and px:
            regime = "negative" if px < flip["price"] else "positive"
            out.append(f"- Gamma regime: **{regime}** (price {px:,.0f} vs Gamma Flip {flip['price']:,.0f})")
        if px and levels:
            above = min((lv for lv in levels if lv["price"] > px), key=lambda lv: lv["price"], default=None)
            below = max((lv for lv in levels if lv["price"] < px), key=lambda lv: lv["price"], default=None)
            if above:
                out.append(f"- Nearest level above: {above['name']} {above['price']:,.2f}")
            if below:
                out.append(f"- Nearest level below: {below['name']} {below['price']:,.2f}")

    amd = ctx.get("gex_amd")
    if amd:
        out.append(
            f"- AMD: Asia {amd.get('asia_low', '?')}–{amd.get('asia_high', '?')}, "
            f"London {amd.get('london_low', '?')}–{amd.get('london_high', '?')}, "
            f"sweep: {amd.get('london_sweep', 'none')}, bias: {amd.get('day_bias', '?')}"
        )

    if not ctx:
        out.append("_(no same-day brief/GEX capture found — fill in manually)_")

    if guard:
        out += [
            "",
            "### Account (Tradeify)",
            f"- Balance at last reading: ${guard['current_balance']:,.2f}",
            f"- Cushion to trailing floor: ${guard['distance_to_trailing_floor']:,.2f} | to daily floor: ${guard['distance_to_daily_floor']:,.2f}",
            f"- Payout available: ${guard['payout_available']:,.2f}" if guard["payout_eligible"] else "- Payout: not yet eligible",
        ]
    return "\n".join(out)


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    day = args[0] if args else date.today().isoformat()
    datetime.strptime(day, "%Y-%m-%d")  # validate early

    out_path = JOURNAL_DIR / f"{day}.md"
    if out_path.exists() and not force:
        print(f"{out_path} already exists - rerun with --force to overwrite.")
        raise SystemExit(1)

    trades = load_trades(day)
    ctx = market_context(day)
    guard = guardrail_snapshot()

    pretty = datetime.strptime(day, "%Y-%m-%d").strftime("%a %d %b %Y").replace(" 0", " ")
    parts = [f"# Trade Journal — {pretty}", ""]
    if trades:
        for i, t in enumerate(trades, 1):
            parts += [fmt_trade(i, t, trade_metrics(t)), "", "---", ""]
    else:
        parts += ["_No trades logged in my_trades.csv for this date._", "", "---", ""]
    parts.append(fmt_context(day, ctx, guard))

    JOURNAL_DIR.mkdir(exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Draft journal written: {out_path}")
    print(f"  Trades included : {len(trades)}")
    print(f"  Market context  : {'same-day data' if ctx else 'none found (stale/missing brief)'}")
    print(f"  Guardrail state : {'included' if guard else 'no tradeify_state.json yet'}")
    if trades:
        print("Fill in the qualitative sections (execution, exit rationale, review).")


if __name__ == "__main__":
    main()
