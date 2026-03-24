"""
Prop-firm backtesting engine — MyFutureFunded / Topstep $50k rules.

Eval stage:
  - Profit target:       $3,000  (pass eval at this point)
  - Max daily loss:      $1,500  (breach = fail day; account locked for session)
  - Max trailing DD:     $2,000  (trails from highest equity ever reached)
  - Position limit:      10 contracts NQ

Funded stage (simulated after eval pass):
  - Max daily loss:      $1,500
  - Trailing drawdown:   $2,000  (stops trailing once floor hits starting balance)

NQ specifics:
  - 1 point = $20
  - Min tick = 0.25 ($5)
  - Default risk per trade: ~$250–$500 (0.5–1% of account)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Literal
import pandas as pd
import numpy as np

POINT_VALUE  = {"NQ": 20.0}
MIN_TICK     = {"NQ": 0.25}
SLIPPAGE_PTS = {"NQ": 0.5}   # 2 ticks slippage on fill


@dataclass
class Trade:
    id: int
    asset: str
    direction: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    contracts: int
    risk_usd: float
    reward_usd: float
    rr: float
    div_confirmed:   bool = False
    entry_type:      str  = "IDM_POI"   # "IDM_POI" or "SMT"
    high_conviction: bool = False
    stop_pts:        float = 0.0

    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl: float | None = None
    outcome: Literal["win", "loss", "be"] | None = None


@dataclass
class PropFirmStatus:
    phase: Literal["eval", "funded", "failed", "passed"] = "eval"
    fail_reason: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    prop_status: PropFirmStatus = field(default_factory=PropFirmStatus)

    starting_balance: float = 50_000.0
    ending_balance:   float = 50_000.0

    # summary stats
    total_trades:     int   = 0
    wins:             int   = 0
    losses:           int   = 0
    breakevens:       int   = 0
    win_rate:         float = 0.0
    avg_rr:           float = 0.0
    profit_factor:    float = 0.0
    net_pnl:          float = 0.0
    max_drawdown:     float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_usd:      float = 0.0
    avg_loss_usd:     float = 0.0
    expectancy:       float = 0.0
    sharpe_ratio:     float = 0.0   # annualised, daily P&L basis
    sortino_ratio:    float = 0.0   # annualised, downside deviation only
    calmar_ratio:     float = 0.0   # annualised return / max drawdown

    def summarise(self):
        closed = [t for t in self.trades if t.pnl is not None]
        self.total_trades = len(closed)
        wins   = [t for t in closed if t.outcome == "win"]
        losses = [t for t in closed if t.outcome == "loss"]

        self.wins       = len(wins)
        self.losses     = len(losses)
        self.breakevens = len([t for t in closed if t.outcome == "be"])
        self.win_rate   = self.wins / self.total_trades if self.total_trades else 0
        self.avg_rr     = float(np.mean([t.rr for t in closed])) if closed else 0

        gross_profit = sum(t.pnl for t in wins)            if wins   else 0
        gross_loss   = abs(sum(t.pnl for t in losses))     if losses else 0

        self.profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
        self.net_pnl       = sum(t.pnl for t in closed)
        self.ending_balance = self.starting_balance + self.net_pnl
        self.avg_win_usd   = gross_profit / self.wins   if self.wins   else 0
        self.avg_loss_usd  = gross_loss   / self.losses if self.losses else 0
        self.expectancy    = (self.win_rate * self.avg_win_usd) - \
                             ((1 - self.win_rate) * self.avg_loss_usd)

        if self.equity_curve:
            eq   = pd.Series([e["equity"] for e in self.equity_curve])
            peak = eq.cummax()
            dd   = eq - peak
            self.max_drawdown     = float(abs(dd.min()))
            self.max_drawdown_pct = float(abs((dd / peak).min()) * 100)

        # ── risk-adjusted metrics ─────────────────────────────────────────────
        if closed:
            # Group P&L by trading day
            daily: dict[date, float] = {}
            for t in closed:
                d = t.exit_time.date()
                daily[d] = daily.get(d, 0.0) + t.pnl

            daily_pnl = pd.Series(list(daily.values()))
            mean_d    = daily_pnl.mean()
            std_d     = daily_pnl.std()
            neg_d     = daily_pnl[daily_pnl < 0]

            TRADING_DAYS = 252.0

            # Sharpe — annualised (risk-free rate ≈ 0 for futures)
            if std_d and std_d > 0:
                self.sharpe_ratio = round(float((mean_d / std_d) * np.sqrt(TRADING_DAYS)), 2)

            # Sortino — uses downside deviation only
            down_std = neg_d.std() if len(neg_d) > 1 else std_d
            if down_std and down_std > 0:
                self.sortino_ratio = round(float((mean_d / down_std) * np.sqrt(TRADING_DAYS)), 2)

            # Calmar — annualised return / max drawdown
            if self.max_drawdown > 0:
                trading_days_in_sample = len(daily)
                annualised_return = self.net_pnl * (TRADING_DAYS / max(trading_days_in_sample, 1))
                self.calmar_ratio = round(float(annualised_return / self.max_drawdown), 2)

        return self

    def to_dict(self) -> dict:
        self.summarise()
        return {
            "starting_balance": self.starting_balance,
            "ending_balance":   round(self.ending_balance, 2),
            "total_trades":     self.total_trades,
            "wins":             self.wins,
            "losses":           self.losses,
            "breakevens":       self.breakevens,
            "win_rate":         round(self.win_rate * 100, 1),
            "avg_rr":           round(self.avg_rr, 2),
            "profit_factor":    round(self.profit_factor, 2) if self.profit_factor != float("inf") else 999,
            "net_pnl":          round(self.net_pnl, 2),
            "max_drawdown":     round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "avg_win_usd":      round(self.avg_win_usd, 2),
            "avg_loss_usd":     round(self.avg_loss_usd, 2),
            "expectancy":       round(self.expectancy, 2),
            "sharpe_ratio":     self.sharpe_ratio,
            "sortino_ratio":    self.sortino_ratio,
            "calmar_ratio":     self.calmar_ratio,
            "prop_phase":       self.prop_status.phase,
            "prop_fail_reason": self.prop_status.fail_reason,
            "equity_curve":     self.equity_curve,
            "trades": [
                {
                    "id":             t.id,
                    "asset":          t.asset,
                    "direction":      t.direction,
                    "entry_time":     str(t.entry_time),
                    "exit_time":      str(t.exit_time),
                    "entry_price":    t.entry_price,
                    "exit_price":     t.exit_price,
                    "pnl":            round(t.pnl, 2),
                    "outcome":        t.outcome,
                    "rr":             round(t.rr, 2),
                    "contracts":      t.contracts,
                    "div_confirmed":  t.div_confirmed,
                    "entry_type":     t.entry_type,
                    "high_conviction": t.high_conviction,
                    "stop_pts":       round(t.stop_pts, 2),
                }
                for t in self.trades if t.pnl is not None
            ],
        }


class PropFirmEngine:
    """
    MyFutureFunded / Topstep $50k rules.

    Eval rules:
      - Profit target $3,000  → phase switches to "funded"
      - Max daily loss $1,500 → session locked for that day
      - Trailing drawdown $2,000 from equity peak → account failed
    Funded rules (after eval pass):
      - Max daily loss $1,500
      - Trailing drawdown $2,000, but floor stops at starting balance
    """

    START_BAL         = 50_000.0
    EVAL_TARGET       = 3_000.0
    MAX_DAILY_LOSS    = 1_500.0
    TRAILING_DD       = 2_000.0
    DEFAULT_CONTRACTS = 2         # standard position size
    MAX_CONTRACTS     = 3         # high-conviction only
    MAX_STOP_PTS      = 10.0      # never risk more than 10 NQ points on a stop
    MIN_RR            = 2.0
    MAX_RR            = 50.0     # distribution legs can be 10-30R+ at CE entries

    def __init__(self):
        self.balance       = self.START_BAL
        self.equity_peak   = self.START_BAL
        self.dd_floor      = self.START_BAL - self.TRAILING_DD   # = $48,000
        self.result        = BacktestResult(starting_balance=self.START_BAL)
        self._trade_id     = 0
        self._daily_pnl: dict[date, float] = {}
        self._day_locked:  set[date]       = set()
        self._open_trade: Trade | None     = None
        self._phase        = "eval"        # "eval" | "funded" | "failed"

    @property
    def active(self) -> bool:
        return self._phase in ("eval", "funded")

    # ── drawdown floor (trailing) ─────────────────────────────────────────────
    def _update_floor(self):
        """Recalculate the trailing DD floor each time equity changes."""
        self.equity_peak = max(self.equity_peak, self.balance)
        new_floor = self.equity_peak - self.TRAILING_DD

        if self._phase == "funded":
            # In funded: floor never goes below starting balance
            new_floor = max(new_floor, self.START_BAL)

        self.dd_floor = new_floor

    # ── daily loss check ──────────────────────────────────────────────────────
    def _daily_ok(self, today: date) -> bool:
        if today in self._day_locked:
            return False
        return self._daily_pnl.get(today, 0.0) > -self.MAX_DAILY_LOSS

    # ── position sizing ───────────────────────────────────────────────────────
    def _size(self, asset: str, entry: float, stop: float,
              high_conviction: bool = False) -> tuple[int, float]:
        """
        2 contracts standard, 3 on high conviction.
        Rejects the trade if the stop is wider than MAX_STOP_PTS.
        """
        stop_pts = abs(entry - stop)
        if stop_pts == 0 or stop_pts > self.MAX_STOP_PTS:
            return 0, 0.0          # stop too wide — don't take the trade
        pv           = POINT_VALUE[asset]
        contracts    = self.MAX_CONTRACTS if high_conviction else self.DEFAULT_CONTRACTS
        actual_risk  = contracts * stop_pts * pv
        return contracts, actual_risk

    # ── open a trade ─────────────────────────────────────────────────────────
    def open_trade(self, asset: str, direction: str, entry: float,
                   stop: float, target: float, entry_time: pd.Timestamp,
                   div_confirmed: bool = False,
                   high_conviction: bool = False,
                   entry_type: str = "IDM_POI") -> Trade | None:

        if not self.active or self._open_trade is not None:
            return None

        today = entry_time.date()
        if not self._daily_ok(today):
            return None

        if self.balance <= self.dd_floor:
            self._phase = "failed"
            self.result.prop_status.phase = "failed"
            self.result.prop_status.fail_reason = "Trailing drawdown breached"
            return None

        rr = abs(target - entry) / abs(stop - entry) if abs(stop - entry) > 0 else 0
        if rr < self.MIN_RR or rr > self.MAX_RR:
            return None

        # Apply slippage on entry
        slip = SLIPPAGE_PTS[asset]
        entry_filled = (entry + slip) if direction == "long" else (entry - slip)

        contracts, risk_usd = self._size(asset, entry_filled, stop, high_conviction)
        if contracts == 0:
            return None

        pv         = POINT_VALUE[asset]
        reward_usd = contracts * abs(target - entry_filled) * pv
        stop_pts   = abs(entry_filled - stop)

        self._trade_id += 1
        trade = Trade(
            id=self._trade_id,
            asset=asset,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_filled,
            stop_price=stop,
            target_price=target,
            contracts=contracts,
            risk_usd=risk_usd,
            reward_usd=reward_usd,
            rr=round(abs(target - entry_filled) / abs(stop - entry_filled), 2),
            div_confirmed=div_confirmed,
            high_conviction=high_conviction,
            entry_type=entry_type,
            stop_pts=stop_pts,
        )
        self._open_trade = trade
        self.result.trades.append(trade)
        return trade

    # ── close a trade ─────────────────────────────────────────────────────────
    def close_trade(self, exit_price: float, exit_time: pd.Timestamp):
        t = self._open_trade
        if t is None:
            return

        pv = POINT_VALUE[t.asset]
        if t.direction == "long":
            pnl = (exit_price - t.entry_price) * t.contracts * pv
        else:
            pnl = (t.entry_price - exit_price) * t.contracts * pv

        t.exit_time  = exit_time
        t.exit_price = exit_price
        t.pnl        = round(pnl, 2)
        t.outcome    = "win" if pnl > 50 else ("loss" if pnl < -50 else "be")

        self.balance += pnl
        self._update_floor()

        today = exit_time.date()
        self._daily_pnl[today] = self._daily_pnl.get(today, 0.0) + pnl

        # Lock day if daily loss limit hit
        if self._daily_pnl[today] <= -self.MAX_DAILY_LOSS:
            self._day_locked.add(today)

        # Check trailing drawdown breach
        if self.balance <= self.dd_floor:
            self._phase = "failed"
            self.result.prop_status.phase = "failed"
            self.result.prop_status.fail_reason = (
                f"Trailing drawdown breached at {exit_time} "
                f"(balance ${self.balance:,.0f} ≤ floor ${self.dd_floor:,.0f})"
            )

        # Check eval pass
        if self._phase == "eval" and (self.balance - self.START_BAL) >= self.EVAL_TARGET:
            self._phase = "funded"
            self.result.prop_status.phase = "funded"
            print(f"  *** EVAL PASSED at {exit_time} — balance ${self.balance:,.2f} ***")

        self.result.equity_curve.append({
            "time":     str(exit_time),
            "equity":   round(self.balance, 2),
            "dd_floor": round(self.dd_floor, 2),
            "trade_id": t.id,
            "outcome":  t.outcome,
            "phase":    self._phase,
        })

        self._open_trade = None

    # ── check stop/target on a candle ────────────────────────────────────────
    def check_exit(self, candle: pd.Series, exit_time: pd.Timestamp):
        t = self._open_trade
        if t is None:
            return

        if t.direction == "long":
            if candle["low"] <= t.stop_price:
                self.close_trade(t.stop_price, exit_time)
            elif candle["high"] >= t.target_price:
                self.close_trade(t.target_price, exit_time)
        else:
            if candle["high"] >= t.stop_price:
                self.close_trade(t.stop_price, exit_time)
            elif candle["low"] <= t.target_price:
                self.close_trade(t.target_price, exit_time)
