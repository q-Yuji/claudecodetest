"""
analyse_my_trades.py — Load your real trade log and compare to algorithm detections.

Usage:
    python3 analyse_my_trades.py

Fill in my_trades.csv first. The script will show:
  - Your real win rate, R:R, expectancy
  - Which of your trades the algorithm also detected (and which it missed)
  - What the algorithm detected that you didn't take (noise filter insight)
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "my_trades.csv"
POINT_VALUE = 20.0   # NQ: $20/point


def load_my_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_FILE, comment="#")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["date", "direction", "entry", "stop", "exit_price", "outcome"])
    df["entry"]      = pd.to_numeric(df["entry"],      errors="coerce")
    df["stop"]       = pd.to_numeric(df["stop"],       errors="coerce")
    df["target"]     = pd.to_numeric(df["target"],     errors="coerce")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df = df.dropna(subset=["entry", "stop", "exit_price"])

    # Infer contracts (2 standard, assume all standard for now)
    df["contracts"] = 2

    # Calculate P&L
    def calc_pnl(row):
        if row["direction"] == "long":
            return (row["exit_price"] - row["entry"]) * row["contracts"] * POINT_VALUE
        else:
            return (row["entry"] - row["exit_price"]) * row["contracts"] * POINT_VALUE

    df["pnl"] = df.apply(calc_pnl, axis=1)

    # Calculate R:R achieved
    def calc_rr(row):
        risk = abs(row["entry"] - row["stop"])
        if risk == 0:
            return 0
        reward = abs(row["exit_price"] - row["entry"])
        return round(reward / risk, 2)

    df["rr_achieved"] = df.apply(calc_rr, axis=1)

    # R:R planned
    def calc_planned_rr(row):
        risk = abs(row["entry"] - row["stop"])
        if risk == 0 or pd.isna(row["target"]):
            return np.nan
        reward = abs(row["target"] - row["entry"])
        return round(reward / risk, 2)

    df["rr_planned"] = df.apply(calc_planned_rr, axis=1)

    return df


def print_stats(df: pd.DataFrame):
    total  = len(df)
    wins   = df[df["outcome"] == "win"]
    losses = df[df["outcome"] == "loss"]
    bes    = df[df["outcome"] == "be"]

    win_rate     = len(wins) / total if total else 0
    gross_profit = wins["pnl"].sum()   if len(wins)   else 0
    gross_loss   = abs(losses["pnl"].sum()) if len(losses) else 0
    net_pnl      = df["pnl"].sum()
    avg_win      = gross_profit / len(wins)   if len(wins)   else 0
    avg_loss     = gross_loss   / len(losses) if len(losses) else 0
    pf           = gross_profit / gross_loss  if gross_loss  else float("inf")
    expectancy   = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    print(f"\n{'='*52}")
    print(f"  Your Real Trade Stats  ({total} trades)")
    print(f"{'='*52}")
    print(f"  Win rate      : {win_rate*100:.1f}%  ({len(wins)}W / {len(losses)}L / {len(bes)}BE)")
    print(f"  Net P&L       : ${net_pnl:,.2f}")
    print(f"  Avg win       : ${avg_win:,.2f}")
    print(f"  Avg loss      : ${avg_loss:,.2f}")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Expectancy    : ${expectancy:,.2f} / trade")
    print(f"  Avg R achieved: {df['rr_achieved'].mean():.2f}R")
    if df["rr_planned"].notna().any():
        print(f"  Avg R planned : {df['rr_planned'].mean():.2f}R")

    print(f"\n  By session:")
    for sess in df["session"].dropna().unique():
        s = df[df["session"] == sess]
        wr = (s["outcome"] == "win").mean() * 100
        print(f"    {sess:6s} : {len(s)} trades  {wr:.0f}% WR  ${s['pnl'].sum():,.0f} net")

    print(f"\n  By direction:")
    for d in ["long", "short"]:
        s = df[df["direction"] == d]
        if len(s):
            wr = (s["outcome"] == "win").mean() * 100
            print(f"    {d:5s} : {len(s)} trades  {wr:.0f}% WR  ${s['pnl'].sum():,.0f} net")

    print(f"\n  Stop sizes:")
    df["stop_pts"] = abs(df["entry"] - df["stop"])
    print(f"    Min  : {df['stop_pts'].min():.1f} pts")
    print(f"    Max  : {df['stop_pts'].max():.1f} pts")
    print(f"    Avg  : {df['stop_pts'].mean():.1f} pts")

    print()


def main():
    if not TRADES_FILE.exists():
        print(f"Trade file not found: {TRADES_FILE}")
        return

    df = load_my_trades()
    if df.empty:
        print("No trades found in my_trades.csv — fill in your trades first.")
        print("See the comment header in the file for the format.")
        return

    print_stats(df)

    # Save a summary JSON for the dashboard (optional)
    summary = {
        "total": len(df),
        "wins":  int((df["outcome"] == "win").sum()),
        "losses": int((df["outcome"] == "loss").sum()),
        "win_rate": round((df["outcome"] == "win").mean() * 100, 1),
        "net_pnl": round(df["pnl"].sum(), 2),
        "avg_rr_achieved": round(df["rr_achieved"].mean(), 2),
        "trades": df.to_dict("records"),
    }
    out = Path(__file__).parent / "results" / "my_trades_summary.json"
    import json
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved → {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    main()
