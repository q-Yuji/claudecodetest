"""
scoreboard_card.py — Render the evening scoreboard card from the walk-forward
grades (roadmap feature 5's "second daily content unit").

Reads the grades from backtest.scoreboard (computed live from the dataset)
and writes results/scoreboard_card.html: the graded verdict on the latest
session's call, the running out-of-sample record, and the claims-under-test
sheet with verdicts attached — including negative ones. The morning stats
card says what the data claims; this card shows how those claims are
actually running. Shares stats_card's CSS so the two posts read as one brand.

Run:
  python -m backtest.scoreboard_card
"""

from __future__ import annotations

from pathlib import Path

from backtest.scoreboard import compute_from_file
from backtest.stats_card import _CSS, SWEEP_LABELS

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS_DIR / "scoreboard_card.html"

_EXTRA_CSS = """
.stamp{display:inline-block; font-size:12px; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; padding:3px 12px; border-radius:5px; border:1.5px solid;
  vertical-align:middle}
.stamp.hit{color:var(--up); border-color:var(--up)}
.stamp.miss{color:var(--down); border-color:var(--down)}
.stamp.open{color:var(--ink-2); border-color:var(--baseline)}
.hero-stamp{font-size:20px; padding:6px 18px}
.l10{display:flex; gap:6px; align-items:center; margin-top:10px}
.l10 b{width:24px; height:24px; display:flex; align-items:center; justify-content:center;
  font-size:12px; font-weight:700; border-radius:5px; border:1.5px solid var(--grid)}
.l10 b.h{color:var(--up); border-color:var(--up)}
.l10 b.m{color:var(--down); border-color:var(--down)}
.l10 span{font-size:11px; color:var(--muted); margin-left:8px}
.recgrid{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:6px}
.recbox .big{font-size:34px; font-weight:700; letter-spacing:-.01em}
.recbox .cap{font-size:11.5px; color:var(--muted); line-height:1.5; margin-top:2px}
"""


def _l10(last10: list[bool]) -> str:
    boxes = "".join(f'<b class="{"h" if h else "m"}">{"✓" if h else "✗"}</b>'
                    for h in last10)
    return f'<div class="l10">{boxes}<span>LAST {len(last10)} CALLS</span></div>'


def build_card(sb: dict) -> str:
    r = sb["record"]
    fo = sb["fakeout_oos"]
    lt = sb["latest"]

    act = float(lt["actual_change_pts"])
    act_html = (f'<span class="{"up" if act > 0 else "down"}">{act:+.0f} pts</span>')
    stamp = (f'<span class="stamp hero-stamp {"hit" if lt["hit"] else "miss"}">'
             f'{"HIT" if lt["hit"] else "MISS"}</span>')

    dir_pct = float(r["hit_pct"])
    dir_verdict = ('<span class="stamp hit">Edge</span>' if dir_pct >= 60 else
                   '<span class="stamp miss">No edge yet</span>' if dir_pct <= 55
                   else '<span class="stamp open">Unproven</span>')
    fo_pct = float(fo["oos_pct"] or 0)
    fo_claim = float(fo["avg_claim_pct"] or 0)
    fo_verdict = ('<span class="stamp hit">Holds</span>'
                  if abs(fo_pct - fo_claim) <= 10 and fo_pct > 50
                  else '<span class="stamp open">Unproven</span>')

    pattern_rows = "".join(
        f'<tr><td>{SWEEP_LABELS[k]}</td>'
        f'<td class="num">{v["calls"]}</td>'
        f'<td class="num">{v["hits"]}</td>'
        f'<td class="num">{v["hit_pct"]:.0f}%</td></tr>'
        for k, v in sb["by_pattern"].items() if v["calls"])

    return f"""<title>SweepStats — The Record</title>
<style>{_CSS}{_EXTRA_CSS}</style>
<div class="viz-root">
<div class="card">
  <div class="brand">SweepStats · The Record</div>
  <div class="sub">NQ futures · every morning call graded against what New York
    actually did · walk-forward, out-of-sample by construction</div>

  <div class="hero-val">{lt['date']} {stamp}</div>
  <div class="hero-label">{SWEEP_LABELS[lt['pattern']]} — the data said NY closes
    <b>{lt['call']}</b> ({(float(lt['said_up_pct']) if lt['call'] == 'up'
    else 100 - float(lt['said_up_pct'])):.0f}% of {lt['prior_days']}
    prior days with this pattern). New York closed {act_html}.</div>
  {_l10(r['last10'])}

  <div class="sec">
    <h2>The running record</h2>
    <div class="recgrid">
      <div class="recbox"><div class="big">{r['hits']}/{r['calls']}</div>
        <div class="cap">directional calls hit ({dir_pct:.0f}%) ·
        {r['first_call']} → {r['last_call']} · {r['no_call_days']} no-call days
        (low sample / coin flip)</div></div>
      <div class="recbox"><div class="big">{fo_pct:.0f}%</div>
        <div class="cap">out-of-sample fakeout rate vs ~{fo_claim:.0f}% claimed
        at touch time · {fo['fakeouts']} of {fo['touches']} first touches</div></div>
    </div>
  </div>

  <div class="sec">
    <h2>Claims under test</h2>
    <table>
      <tr><th>Claim</th><th class="num">Out-of-sample</th>
          <th class="num">Result</th><th class="num">Verdict</th></tr>
      <tr><td>London's overnight pattern predicts NY direction</td>
          <td class="num">{r['calls']} calls</td>
          <td class="num">{dir_pct:.0f}% hit</td>
          <td class="num">{dir_verdict}</td></tr>
      <tr><td>The first NY touch of an overnight level is usually a fakeout</td>
          <td class="num">{fo['touches']} touches</td>
          <td class="num">ran {fo_pct:.0f}%</td>
          <td class="num">{fo_verdict}</td></tr>
    </table>
  </div>

  <div class="sec">
    <h2>Directional record by pattern</h2>
    <table>
      <tr><th>Overnight pattern</th><th class="num">Calls</th>
          <th class="num">Hits</th><th class="num">Hit rate</th></tr>
      {pattern_rows}
    </table>
  </div>

  <div class="foot">
    <b>Method:</b> every call is reconstructed using only the sessions before its
    date (min {sb['method']['min_days']} prior days per pattern, else no call) and
    graded against the NY close — nothing is backfit, and the append-only dataset
    makes the whole record reproducible. Negative verdicts are published, not
    buried: a claim that fails out-of-sample says so on the card.<br>
    Generated {sb['generated'][:10]}. Past frequency ≠ future probability.
    Not financial advice.
  </div>
</div>
</div>
"""


def main() -> None:
    sb = compute_from_file()
    if not sb["record"]["calls"]:
        print("scoreboard card: no graded calls yet — nothing to render")
        return
    OUT_FILE.write_text(build_card(sb), encoding="utf-8")
    print(f"Card -> {OUT_FILE}")


if __name__ == "__main__":
    main()
