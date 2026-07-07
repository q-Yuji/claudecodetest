"""
stats_card.py — Render the SweepStats shareable card from the stats summary.

Reads results/session_stats_summary.json (produced by backtest.session_stats)
and writes results/stats_card.html: a self-contained, theme-aware card with
the headline fakeout stat, per-level first-touch outcomes, London-manipulation
bias table, and time-of-day breakdown. This is the artifact that doubles as
MVP, first post, and pitch demo — regenerate it whenever the dataset grows.

Run:
  python -m backtest.stats_card
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
SUMMARY_FILE = RESULTS_DIR / "session_stats_summary.json"
OUT_FILE = RESULTS_DIR / "stats_card.html"

LEVEL_LABELS = {
    "asia_high": "Asia High",
    "asia_low": "Asia Low",
    "london_high": "London High",
    "london_low": "London Low",
}
SWEEP_LABELS = {
    "asia_high": "London sweeps Asia High",
    "asia_low": "London sweeps Asia Low",
    "none": "London stays inside Asia",
    "both": "London sweeps both sides",
}

_CSS = """
.viz-root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7;
  --border:rgba(11,11,11,.10); --series:#2a78d6; --series-soft:#cde2fb;
  --up:#006300; --down:#d03b3b;
}
@media (prefers-color-scheme: dark){
  .viz-root{
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --baseline:#383835;
    --border:rgba(255,255,255,.10); --series:#3987e5; --series-soft:#184f95;
    --up:#0ca30c; --down:#e66767;
  }
}
:root[data-theme="dark"] .viz-root{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#383835;
  --border:rgba(255,255,255,.10); --series:#3987e5; --series-soft:#184f95;
  --up:#0ca30c; --down:#e66767;
}
:root[data-theme="light"] .viz-root{
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7;
  --border:rgba(11,11,11,.10); --series:#2a78d6; --series-soft:#cde2fb;
  --up:#006300; --down:#d03b3b;
}
.viz-root{
  background:var(--page); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  min-height:100vh; padding:32px 16px; margin:0;
}
.card{
  max-width:680px; margin:0 auto; background:var(--surface);
  border:1px solid var(--border); border-radius:14px; padding:28px 30px 22px;
}
.brand{font-size:13px; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:var(--series); margin-bottom:2px}
.sub{font-size:12.5px; color:var(--ink-2); margin-bottom:22px}
.hero-val{font-size:56px; font-weight:700; line-height:1.05; letter-spacing:-.01em}
.hero-label{font-size:15px; color:var(--ink-2); margin-top:4px; max-width:34em}
.hero-note{font-size:12px; color:var(--muted); margin-top:6px}
.sec{margin-top:30px}
.sec h2{font-size:12px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ink-2); margin:0 0 12px}
.row{display:grid; grid-template-columns:96px 1fr 128px; gap:10px; align-items:center;
  padding:7px 0}
.row + .row{border-top:1px solid var(--grid)}
.lvl{font-size:13px; color:var(--ink)}
.track{position:relative; height:16px; background:transparent}
.track::before{content:""; position:absolute; left:0; top:0; bottom:0; width:100%;
  border-left:2px solid var(--baseline)}
.bar{position:absolute; left:0; top:0; bottom:0; background:var(--series);
  border-radius:0 4px 4px 0; min-width:2px}
.bar-val{position:absolute; top:50%; transform:translateY(-50%); font-size:12px;
  font-weight:700; color:var(--ink); white-space:nowrap}
.row-meta{font-size:11.5px; color:var(--muted); text-align:right; line-height:1.5}
.row:hover{background:color-mix(in srgb, var(--series) 6%, transparent)}
table{width:100%; border-collapse:collapse; font-size:12.5px}
th{text-align:left; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding:6px 8px 6px 0; border-bottom:1px solid var(--baseline)}
th.num,td.num{text-align:right; font-variant-numeric:tabular-nums; padding-right:0; padding-left:8px}
td{padding:7px 8px 7px 0; border-bottom:1px solid var(--grid); color:var(--ink)}
tr:last-child td{border-bottom:none}
.up{color:var(--up); font-weight:600}
.down{color:var(--down); font-weight:600}
.dim{color:var(--muted)}
.foot{margin-top:28px; padding-top:14px; border-top:1px solid var(--grid);
  font-size:11px; color:var(--muted); line-height:1.65}
.foot b{color:var(--ink-2); font-weight:600}
@media (max-width:520px){ .row{grid-template-columns:84px 1fr}
  .row-meta{grid-column:1/-1; text-align:left} .hero-val{font-size:44px} }
"""


def _bar_row(label: str, pct: float, meta: str, vmax: float = 100.0) -> str:
    w = max(2.0, pct / vmax * 100.0)
    inside = w > 30
    val_style = (f"right:calc({100 - w}% + 8px)" if inside
                 else f"left:calc({w}% + 8px)")
    val_color = "color:#fff" if inside else ""
    return f"""
  <div class="row">
    <div class="lvl">{label}</div>
    <div class="track"><div class="bar" style="width:{w:.1f}%"></div>
      <span class="bar-val" style="{val_style};{val_color}">{pct:.0f}%</span></div>
    <div class="row-meta">{meta}</div>
  </div>"""


def build_card(s: dict) -> str:
    ft = s["first_touch"]
    lm = s["london_manipulation"]
    tb = s["time_buckets"]
    sample = s["sample"]

    tot_touch = sum(v["touched"] for v in ft.values())
    tot_fake = sum(v["fakeout"] for v in ft.values())
    hero_pct = 100.0 * tot_fake / tot_touch if tot_touch else 0.0

    level_rows = "".join(
        _bar_row(
            LEVEL_LABELS[k],
            v["fakeout_pct"],
            f"n={v['touched']} touches · median reversal "
            f"{v['fakeout_median_mfe60_pts']:.0f} pts/60m",
        )
        for k, v in ft.items() if v["touched"]
    )

    def _chg(x):
        if x is None:
            return '<span class="dim">—</span>'
        cls = "up" if x > 0 else "down"
        return f'<span class="{cls}">{x:+.0f} pts</span>'

    lm_rows = "".join(
        f"<tr><td>{SWEEP_LABELS[k]}"
        + (' <span class="dim">(low sample)</span>' if v["days"] < 5 else "")
        + f'</td><td class="num">{v["days"]}</td>'
        f'<td class="num">{v["ny_up_pct"]:.0f}%</td>'
        f'<td class="num">{_chg(v["median_ny_change_pts"])}</td></tr>'
        for k, v in lm.items() if v["days"]
    )

    tb_rows = "".join(
        f"<tr><td>{k} ET"
        + (' <span class="dim">(low sample)</span>' if v["touches"] < 10 else "")
        + f'</td><td class="num">{v["touches"]}</td>'
        f'<td class="num">{v["fakeout_pct"]:.0f}%</td>'
        f'<td class="num">{f"{v['fakeout_median_mfe60_pts']:.0f}" if v["fakeout_median_mfe60_pts"] is not None else "—"}</td></tr>'
        for k, v in tb.items() if v["touches"]
    )

    return f"""<title>SweepStats — NQ Session Level Report</title>
<style>{_CSS}</style>
<div class="viz-root">
<div class="card">
  <div class="brand">SweepStats</div>
  <div class="sub">NQ futures · first New-York touch of overnight session levels ·
    {sample['sessions']} sessions, {sample['from']} → {sample['to']}</div>

  <div class="hero-val">{hero_pct:.0f}%</div>
  <div class="hero-label">of first NY touches of an Asia or London high/low were
    <b>fakeouts</b> — price broke the level but 5-minute closes reclaimed it
    within two bars.</div>
  <div class="hero-note">{tot_fake} fakeouts in {tot_touch} first-touch episodes</div>

  <div class="sec">
    <h2>Fakeout rate by level — first touch only</h2>
    {level_rows}
  </div>

  <div class="sec">
    <h2>London manipulation → NY direction</h2>
    <table>
      <tr><th>Overnight pattern</th><th class="num">Days</th>
          <th class="num">NY closed up</th><th class="num">Median NY move</th></tr>
      {lm_rows}
    </table>
  </div>

  <div class="sec">
    <h2>Fakeout rate by time of first touch</h2>
    <table>
      <tr><th>First touch window</th><th class="num">Touches</th>
          <th class="num">Fakeout</th><th class="num">Median rev. 60m (pts)</th></tr>
      {tb_rows}
    </table>
  </div>

  <div class="foot">
    <b>Definitions (v{s['dataset_version']}):</b> sessions in ET — Asia 6pm–midnight,
    London 2–5am, NY 9:30am–4pm. A <b>fakeout</b> = the first NY 5m bar to breach a
    session high/low, where that bar or either of the next two <b>closes</b> back on
    the original side (wicks don't count — closes decide). Otherwise it's a
    <b>break</b>. Reversal = max favorable move within 60 minutes of the reclaim
    close. Median NY move = close minus 9:30 open.<br>
    Data: CME NQ continuous front month, 5-minute bars. Generated {s['generated'][:10]}.
    Past frequency ≠ future probability. Not financial advice.
  </div>
</div>
</div>
"""


def main():
    s = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    OUT_FILE.write_text(build_card(s), encoding="utf-8")
    print(f"Card -> {OUT_FILE}")


if __name__ == "__main__":
    main()
