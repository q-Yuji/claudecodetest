"""
Morning Brief -- pre-session market analysis.

Run before market open:
  python morning_brief.py

Pulls overnight prices, sector ETFs, economic calendar, news, GEX screenshot,
IBKR positions, and 2-week AMD backtest review -- renders a tabbed HTML
dashboard (Morning Brief | Backtesting) and opens it in the browser.
"""

import base64
import json
import socket
import subprocess
import sys
import warnings
import webbrowser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path

import yfinance as yf
import requests

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


# -- Overnight prices ----------------------------------------------------------

def get_overnight_context() -> dict:
    symbols = {"NQ": "NQ=F", "ES": "ES=F", "SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX"}
    ctx = {}
    for name, ticker in symbols.items():
        try:
            data = yf.download(ticker, period="2d", interval="1d",
                               auto_adjust=True, progress=False)
            if len(data) >= 2:
                close      = data["Close"].squeeze()
                prev_close = float(close.iloc[-2])
                last       = float(close.iloc[-1])
                chg_pct    = (last - prev_close) / prev_close * 100
                ctx[name]  = {
                    "prev_close": round(prev_close, 2),
                    "last":       round(last, 2),
                    "chg_pct":    round(chg_pct, 2),
                }
        except Exception as e:
            ctx[name] = {"error": str(e)}
    return ctx


# -- Sector / market ETFs ------------------------------------------------------

def get_sectors() -> dict:
    tickers = {
        "XLK":      "Tech",
        "XLF":      "Financials",
        "XLE":      "Energy",
        "XLV":      "Health",
        "IWM":      "Russell",
        "GLD":      "Gold",
        "TLT":      "Bonds",
        "^TNX":     "10Y Yld",
        "DX-Y.NYB": "Dollar",
    }
    result = {}
    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, period="2d", interval="1d",
                               auto_adjust=True, progress=False)
            if len(data) >= 2:
                close  = data["Close"].squeeze()
                prev   = float(close.iloc[-2])
                last   = float(close.iloc[-1])
                chg    = (last - prev) / prev * 100
                result[ticker] = {
                    "name":    name,
                    "last":    round(last, 2),
                    "chg_pct": round(chg, 2),
                }
        except Exception as e:
            result[ticker] = {"name": name, "error": str(e)}
    return result


# -- Economic calendar ---------------------------------------------------------

def get_econ_calendar() -> list:
    try:
        resp = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=5,
            headers={**_HEADERS, "Referer": "https://www.forexfactory.com/"},
        )
        try:
            raw = resp.json()
        except Exception:
            return [{"error": "Calendar feed blocked (try again during market hours)"}]
        today  = date.today().isoformat()
        events = []
        for e in raw:
            if e.get("country") != "USD":
                continue
            if e.get("impact") not in ("High", "Medium"):
                continue
            if e.get("date", "")[:10] != today:
                continue
            events.append({
                "time":     e["date"][11:16],
                "title":    e.get("title", ""),
                "impact":   e.get("impact", ""),
                "forecast": e.get("forecast", "") or "-",
                "previous": e.get("previous", "") or "-",
            })
        return sorted(events, key=lambda x: x["time"])
    except Exception as e:
        return [{"error": str(e)}]


# -- News headlines ------------------------------------------------------------

def get_news_headlines(n: int = 10) -> list:
    try:
        resp = requests.get(
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US",
            timeout=5, headers=_HEADERS,
        )
        root  = ET.fromstring(resp.text)
        items = root.findall(".//item")
        headlines = []
        for item in items[:n]:
            title = item.findtext("title", "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            url = item.findtext("link", "").strip()
            if title:
                headlines.append({"title": title, "date": pubdate, "url": url})
        return headlines
    except Exception as e:
        return [{"error": str(e)}]


# -- IBKR positions ------------------------------------------------------------

def get_ibkr_data() -> dict:
    try:
        from data.ibkr import get_accounts, get_positions, get_account_summary
        accounts = get_accounts()
        if not accounts:
            return {"error": "No accounts -- log in at https://localhost:5000"}
        acc_id    = accounts[0]["accountId"]
        positions = get_positions(acc_id)
        summary   = get_account_summary(acc_id)
        nl   = (summary.get("netliquidation") or {}).get("amount", 0)
        dpnl = (summary.get("dailypnl")       or {}).get("amount", 0)
        bp   = (summary.get("buyingpower")    or {}).get("amount", 0)
        return {
            "account":    acc_id,
            "net_liq":    nl,
            "daily_pnl":  dpnl,
            "buying_pow": bp,
            "positions":  positions,
        }
    except requests.exceptions.ConnectionError:
        return {"error": "IBKR Gateway not running"}
    except Exception as e:
        return {"error": str(e)}


# -- GEX screenshot ------------------------------------------------------------

def capture_gex() -> dict:
    try:
        from data.gex import screenshot_gex
        path = screenshot_gex("SPY")
        return {"SPY": str(path) if path else None}
    except Exception as e:
        return {"error": str(e)}


# -- Backtest (AMD 2-week review) ----------------------------------------------

def _get_backtest_days() -> list:
    try:
        from backtest.amd_session_review import (
            fetch_data, fetch_calendar_week, analyse_day
        )
        data = fetch_data()
        cal  = fetch_calendar_week()
        df_ref = data["NQ"]["15m"]
        if df_ref.empty:
            return []
        all_dates    = sorted(set(df_ref.index.date))
        trading_days = [d for d in all_dates if d.weekday() < 5][-10:]
        return [analyse_day(td, data, cal) for td in trading_days]
    except Exception as e:
        print(f"  [backtest] {e}")
        return []


# -- HTML dashboard ------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Morning Brief</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#07090f;--surface:#0d111a;--card:#0f1420;--border:#1a2035;
  --accent:#4f8ef7;--accent-dim:rgba(79,142,247,0.08);
  --green:#22c55e;--green-dim:rgba(34,197,94,0.08);
  --red:#ef4444;--red-dim:rgba(239,68,68,0.08);
  --yellow:#f59e0b;--yellow-dim:rgba(245,158,11,0.1);
  --text:#e2e8f0;--muted:#4a5568;--subtle:#1e293b;
}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',Consolas,monospace;font-size:13px;min-height:100vh;padding:20px 24px 40px}

/* HEADER */
.header{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:0;padding-bottom:16px;border-bottom:1px solid var(--border)}
.brand{font-size:10px;font-weight:700;letter-spacing:.25em;color:var(--accent);text-transform:uppercase;margin-bottom:4px}
.header-date{font-size:18px;font-weight:700;color:var(--text)}
.live{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--green);letter-spacing:.1em}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* TABS */
.tab-nav{display:flex;gap:2px;margin-bottom:20px;border-bottom:1px solid var(--border);padding-top:0}
.tab-btn{background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);font-family:inherit;font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;padding:12px 18px;cursor:pointer;margin-bottom:-1px;transition:color .15s,border-color .15s}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-count{font-size:9px;background:var(--accent-dim);color:var(--accent);padding:1px 6px;border-radius:10px;margin-left:6px;font-weight:700}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* PRICE CARDS */
.price-row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:10px}
.price-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;transition:border-color .2s}
.price-card:hover{border-color:var(--accent)}
.price-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.price-card.up::after{background:linear-gradient(90deg,var(--green),transparent)}
.price-card.down::after{background:linear-gradient(90deg,var(--red),transparent)}
.price-card.flat::after{background:linear-gradient(90deg,var(--muted),transparent)}
.p-ticker{font-size:10px;font-weight:700;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin-bottom:10px}
.p-value{font-size:20px;font-weight:700;color:var(--text);margin-bottom:2px;letter-spacing:-.02em}
.p-change{font-size:14px;font-weight:700;margin-bottom:4px}
.p-change.up{color:var(--green)}.p-change.down{color:var(--red)}.p-change.flat{color:var(--muted)}
.p-prev{font-size:10px;color:var(--muted)}

/* SECTORS */
.sector-row{display:grid;grid-template-columns:repeat(9,1fr);gap:8px;margin-bottom:18px}
.sector-chip{background:var(--card);border:1px solid var(--border);border-radius:7px;padding:8px 10px;display:flex;flex-direction:column;align-items:center;gap:3px}
.s-name{font-size:9px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.s-chg{font-size:12px;font-weight:700}
.s-chg.up{color:var(--green)}.s-chg.down{color:var(--red)}.s-chg.flat{color:var(--muted)}

/* MAIN GRID */
.main-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}

/* CARDS */
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.card-head{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;background:rgba(255,255,255,.01)}
.card-head-icon{width:5px;height:5px;border-radius:50%;background:var(--accent);flex-shrink:0}
.card-head-title{font-size:9px;font-weight:700;letter-spacing:.18em;color:var(--muted);text-transform:uppercase}
.card-body{padding:14px 16px}

/* CALENDAR */
.cal-item{display:flex;align-items:flex-start;gap:10px;padding:9px 0;border-bottom:1px solid var(--subtle)}
.cal-item:last-child{border-bottom:none}
.cal-time{font-size:11px;color:var(--muted);width:38px;flex-shrink:0;padding-top:1px}
.cal-badge{font-size:8px;font-weight:700;letter-spacing:.05em;padding:2px 5px;border-radius:3px;flex-shrink:0;margin-top:1px}
.cal-badge.high{background:var(--red-dim);color:var(--red)}
.cal-badge.med{background:var(--yellow-dim);color:var(--yellow)}
.cal-name{flex:1;font-size:12px;color:var(--text);line-height:1.4}
.cal-vals{font-size:10px;color:var(--muted);text-align:right;flex-shrink:0;line-height:1.7}
.empty-msg{color:var(--muted);font-size:11px;padding:10px 0;text-align:center;font-style:italic}

/* NEWS */
.news-item{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid var(--subtle);align-items:flex-start}
.news-item:last-child{border-bottom:none}
.news-bar{width:2px;min-height:30px;background:var(--accent);border-radius:2px;flex-shrink:0;opacity:.4}
.news-title{font-size:12px;color:var(--text);line-height:1.5;margin-bottom:2px}
.news-title a{color:var(--text);text-decoration:none}
.news-title a:hover{color:var(--accent);text-decoration:underline}
.news-date{font-size:10px;color:var(--muted)}

/* IBKR */
.ibkr-stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px}
.istat{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:7px;padding:10px 12px}
.istat-label{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px}
.istat-val{font-size:15px;font-weight:700;color:var(--text)}
.istat-val.up{color:var(--green)}.istat-val.down{color:var(--red)}
.pos-table{width:100%;border-collapse:collapse}
.pos-table th{font-size:9px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;padding:4px 0;border-bottom:1px solid var(--border);text-align:left;font-weight:600}
.pos-table th:not(:first-child){text-align:right}
.pos-table td{font-size:11px;padding:7px 0;border-bottom:1px solid var(--subtle);vertical-align:middle}
.pos-table td:not(:first-child){text-align:right}
.pos-table tr:last-child td{border-bottom:none}
.pos-sym{color:var(--text);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pos-qty{color:var(--muted)}
.pnl.pos{color:var(--green)}.pnl.neg{color:var(--red)}.pnl.flat{color:var(--muted)}

/* BACKTEST */
.bt-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.bt-stat{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 14px}
.bt-stat-label{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
.bt-stat-val{font-size:20px;font-weight:700;color:var(--text)}
.bt-day{background:var(--card);border:1px solid var(--border);border-radius:10px;margin-bottom:12px;overflow:hidden}
.bt-day-head{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;background:rgba(255,255,255,.01)}
.bt-date{font-size:13px;font-weight:700;color:var(--text)}
.bias-chip{font-size:9px;font-weight:700;letter-spacing:.1em;padding:2px 7px;border-radius:4px;text-transform:uppercase}
.bias-chip.bullish{background:var(--green-dim);color:var(--green)}
.bias-chip.bearish{background:var(--red-dim);color:var(--red)}
.bias-chip.unknown{background:rgba(74,85,104,.2);color:var(--muted)}
.bt-body{display:grid;grid-template-columns:200px 1fr}
.bt-sessions{padding:12px 16px;border-right:1px solid var(--border)}
.bt-sess-label{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;margin-top:10px}
.bt-sess-label:first-child{margin-top:0}
.bt-hl-row{display:flex;justify-content:space-between;margin-bottom:2px}
.hl-k{color:var(--muted);font-size:11px}.hl-v{font-size:11px;color:var(--text)}
.bt-right{padding:12px 16px}
.bt-section{margin-bottom:10px}
.bt-sec-title{font-size:9px;font-weight:700;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin-bottom:5px}
.bt-amd-note{font-size:11px;color:var(--text);padding:3px 0;border-bottom:1px solid var(--subtle)}
.bt-amd-note:last-child{border-bottom:none}
.bt-amd-note.muted{color:var(--muted);border-bottom:none}
.bt-gex{font-size:11px;padding:3px 0}
.bt-gex.muted{color:var(--muted);font-style:italic}
.bt-tag{display:inline-block;font-size:9px;font-weight:700;padding:2px 6px;border-radius:3px;margin:1px 2px 1px 0;letter-spacing:.04em}
.bt-tag.sweep-long{background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.2)}
.bt-tag.sweep-short{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.2)}
.bt-tag.fbos{background:rgba(245,158,11,.1);color:var(--yellow);border:1px solid rgba(245,158,11,.2)}
.bt-tag.rbos{background:rgba(79,142,247,.1);color:var(--accent);border:1px solid rgba(79,142,247,.2)}
.bt-tag.smt{background:rgba(168,85,247,.1);color:#a855f7;border:1px solid rgba(168,85,247,.2)}
.bt-tag.div{background:rgba(20,184,166,.1);color:#14b8a6;border:1px solid rgba(20,184,166,.2)}
.bt-tag.news-high{background:var(--red-dim);color:var(--red);border:1px solid rgba(239,68,68,.25)}
.bt-tag.news-med{background:var(--yellow-dim);color:var(--yellow);border:1px solid rgba(245,158,11,.2)}
.bt-tag.big-c{background:rgba(255,255,255,.04);color:var(--text);border:1px solid var(--border)}
.bt-empty{color:var(--muted);font-size:12px;padding:40px 0;text-align:center;font-style:italic}

/* AI CHAT */
.chat-wrap{display:flex;flex-direction:column;height:calc(100vh - 180px);min-height:420px}
.chat-msgs{flex:1;overflow-y:auto;padding:0 2px 16px;display:flex;flex-direction:column;gap:10px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.chat-msg{display:flex;flex-direction:column;gap:3px}
.chat-msg.user{align-self:flex-end;align-items:flex-end;max-width:75%}
.chat-msg.ai{align-self:flex-start;align-items:flex-start;max-width:88%}
.chat-label{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
.chat-bubble{padding:10px 14px;border-radius:10px;font-size:12px;line-height:1.65;white-space:pre-wrap;word-break:break-word}
.chat-msg.user .chat-bubble{background:var(--accent-dim);border:1px solid rgba(79,142,247,.22);color:var(--text)}
.chat-msg.ai   .chat-bubble{background:var(--card);border:1px solid var(--border);color:var(--text)}
.chat-footer{display:flex;gap:8px;padding-top:12px;border-top:1px solid var(--border);align-items:flex-end}
.chat-input{flex:1;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:inherit;font-size:12px;padding:10px 14px;resize:none;outline:none;line-height:1.5;min-height:42px;max-height:120px}
.chat-input:focus{border-color:rgba(79,142,247,.4)}
.chat-send{background:var(--accent);border:none;border-radius:8px;color:#fff;cursor:pointer;font-family:inherit;font-size:11px;font-weight:700;letter-spacing:.06em;padding:10px 20px;white-space:nowrap;align-self:stretch}
.chat-send:hover{opacity:.85}.chat-send:disabled{opacity:.35;cursor:default}
.chat-key-btn{background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--muted);cursor:pointer;font-size:15px;padding:0 12px;align-self:stretch}
.chat-key-btn:hover{border-color:rgba(79,142,247,.4);color:var(--text)}
.chat-warn{background:var(--yellow-dim);border:1px solid rgba(245,158,11,.25);border-radius:8px;color:var(--yellow);font-size:11px;padding:10px 14px;margin-bottom:12px;line-height:1.6;display:none}
.chat-warn code{background:rgba(245,158,11,.15);padding:1px 5px;border-radius:3px;font-family:inherit}
.chat-key-form{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:14px;display:none}
.chat-key-title{font-size:9px;font-weight:700;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;margin-bottom:8px}
.chat-key-row{display:flex;gap:8px;margin-top:8px}
.chat-key-input{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:7px;color:var(--text);font-family:inherit;font-size:12px;padding:8px 12px;outline:none}
.chat-key-input:focus{border-color:rgba(79,142,247,.4)}
.chat-key-save{background:var(--accent);border:none;border-radius:7px;color:#fff;cursor:pointer;font-family:inherit;font-size:11px;font-weight:700;padding:8px 16px}
.typing-dot{display:inline-block;width:4px;height:4px;border-radius:50%;background:var(--muted);margin:0 1px;animation:tdot 1.2s ease-in-out infinite}
.typing-dot:nth-child(2){animation-delay:.2s}.typing-dot:nth-child(3){animation-delay:.4s}
@keyframes tdot{0%,80%,100%{opacity:.2}40%{opacity:1}}
</style>
</head>
<body>
<script>
const D  = __DATA__;
const BT = __BACKTEST_DATA__;

const fmt  = (n,d=2) => n==null?'--':Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct  = n => n==null?'--':(n>=0?'+':'')+fmt(n)+'%';
const money= n => n==null?'--':(n>=0?'+$':'-$')+fmt(Math.abs(n),0);
const cls  = n => n>0?'up':n<0?'down':'flat';

// Tab switching
function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('date').textContent = D.generated;

  // PRICES
  const priceGrid = document.getElementById('price-row');
  ['NQ','ES','SPY','QQQ','VIX'].forEach(sym => {
    const d = D.ctx[sym]; if(!d||d.error) return;
    const c = cls(d.chg_pct);
    priceGrid.innerHTML += `
    <div class="price-card ${c}">
      <div class="p-ticker">${sym}</div>
      <div class="p-value">${Number(d.last).toLocaleString('en-US')}</div>
      <div class="p-change ${c}">${pct(d.chg_pct)}</div>
      <div class="p-prev">prev ${fmt(d.prev_close)}</div>
    </div>`;
  });

  // SECTORS
  const srow = document.getElementById('sector-row');
  Object.entries(D.sectors||{}).forEach(([,s]) => {
    if(s.error) return;
    const c = cls(s.chg_pct);
    srow.innerHTML += `<div class="sector-chip">
      <div class="s-name">${s.name}</div>
      <div class="s-chg ${c}">${pct(s.chg_pct)}</div>
    </div>`;
  });

  // CALENDAR
  const cal = document.getElementById('cal-body');
  const evts = (D.calendar||[]).filter(e => !e.error);
  if(!evts.length) {
    cal.innerHTML = '<div class="empty-msg">No high-impact USD events today</div>';
  } else {
    evts.forEach(e => {
      cal.innerHTML += `<div class="cal-item">
        <div class="cal-time">${e.time}</div>
        <div class="cal-badge ${e.impact==='High'?'high':'med'}">${e.impact.toUpperCase()}</div>
        <div class="cal-name">${e.title}</div>
        <div class="cal-vals">f: ${e.forecast}<br>p: ${e.previous}</div>
      </div>`;
    });
  }

  // NEWS
  const news = document.getElementById('news-body');
  (D.news||[]).forEach(n => {
    if(n.error) return;
    const d = n.date ? n.date.replace(/\s*\+\d+\s*$/,'').trim() : '';
    news.innerHTML += `<div class="news-item">
      <div class="news-bar"></div>
      <div class="news-content">
        <div class="news-title">${n.url ? `<a href="${n.url}" target="_blank" rel="noopener">${n.title}</a>` : n.title}</div>
        ${d ? `<div class="news-date">${d}</div>` : ''}
      </div>
    </div>`;
  });

  // IBKR
  const ibkr = D.ibkr||{};
  const ibkrBody = document.getElementById('ibkr-body');
  if(ibkr.error) {
    ibkrBody.innerHTML = `<div class="empty-msg">${ibkr.error}</div>`;
  } else {
    const pnlC = cls(ibkr.daily_pnl||0);
    document.getElementById('ibkr-stats').innerHTML = `
      <div class="istat"><div class="istat-label">Net Liq</div><div class="istat-val">$${fmt(ibkr.net_liq,0)}</div></div>
      <div class="istat"><div class="istat-label">Daily P&L</div><div class="istat-val ${pnlC}">${money(ibkr.daily_pnl)}</div></div>
      <div class="istat"><div class="istat-label">Buying Power</div><div class="istat-val">$${fmt(ibkr.buying_pow,0)}</div></div>
    `;
    const pos = D.positions||[];
    const posEl = document.getElementById('ibkr-pos');
    if(!pos.length) {
      posEl.innerHTML = '<div class="empty-msg">No open positions</div>';
    } else {
      posEl.innerHTML = `<table class="pos-table">
        <thead><tr><th>Contract</th><th>Qty</th><th>P&L</th></tr></thead>
        <tbody>${pos.map(p => {
          const sym = (p.contractDesc||p.ticker||'?').slice(0,48);
          const qty = p.position||0;
          const pnl = p.unrealizedPnl||0;
          const pc  = cls(pnl);
          return `<tr>
            <td class="pos-sym">${sym}</td>
            <td class="pos-qty">${qty>0?'+':''}${qty}</td>
            <td class="pnl ${pc}">${money(pnl)}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>`;
    }
  }

  // BACKTEST
  const container = document.getElementById('bt-days');
  if(!BT || !BT.length) {
    container.innerHTML = '<div class="bt-empty">Backtest data unavailable — check console for errors.</div>';
  } else {
    const totalSweeps = BT.reduce((s,d) => s + d.sweeps.length, 0);
    const totalFbos   = BT.reduce((s,d) => s + d.fbos_bull + d.fbos_bear, 0);
    const totalSmt    = BT.reduce((s,d) => s + d.smt_bull + d.smt_bear, 0);
    document.getElementById('bt-summary').innerHTML = `
      <div class="bt-stat"><div class="bt-stat-label">Days Reviewed</div><div class="bt-stat-val">${BT.length}</div></div>
      <div class="bt-stat"><div class="bt-stat-label">Session Sweeps</div><div class="bt-stat-val">${totalSweeps}</div></div>
      <div class="bt-stat"><div class="bt-stat-label">FBOS (15m)</div><div class="bt-stat-val">${totalFbos}</div></div>
      <div class="bt-stat"><div class="bt-stat-label">SMT Signals</div><div class="bt-stat-val">${totalSmt}</div></div>
    `;
    document.getElementById('bt-days-count').textContent = BT.length + 'd';

    const tag = (label, cls) => `<span class="bt-tag ${cls}">${label}</span>`;

    BT.forEach(d => {
      const dt      = new Date(d.date + 'T12:00:00');
      const dateStr = dt.toLocaleDateString('en-US', {weekday:'long', day:'numeric', month:'long', year:'numeric'});

      const amdHtml = d.amd_notes && d.amd_notes.length
        ? d.amd_notes.map(n => `<div class="bt-amd-note">${n}</div>`).join('')
        : '<div class="bt-amd-note muted">No session sweeps detected</div>';

      const sweepTags = (d.sweeps||[]).map(sw =>
        tag(`${sw.type} ${sw.time}`, sw.direction==='long'?'sweep-long':'sweep-short')
      ).join('');

      const sigTags = [
        d.fbos_bull ? tag(`FBOS bull x${d.fbos_bull}`, 'fbos') : '',
        d.fbos_bear ? tag(`FBOS bear x${d.fbos_bear}`, 'fbos') : '',
        d.rbos_bull ? tag(`RBOS bull x${d.rbos_bull}`, 'rbos') : '',
        d.rbos_bear ? tag(`RBOS bear x${d.rbos_bear}`, 'rbos') : '',
        d.smt_bull  ? tag(`SMT bull x${d.smt_bull}`,   'smt')  : '',
        d.smt_bear  ? tag(`SMT bear x${d.smt_bear}`,   'smt')  : '',
        d.div_bull  ? tag(`DIV bull x${d.div_bull}`,   'div')  : '',
        d.div_bear  ? tag(`DIV bear x${d.div_bear}`,   'div')  : '',
      ].join('');

      const newsTags = (d.news||[]).map(n =>
        tag(`${n.time} ${n.title}`, n.impact==='High'?'news-high':'news-med')
      ).join('');

      const bcTags = (d.big_candles||[]).map(bc =>
        tag(`${bc.time} ${bc.direction==='bull'?'up':'dn'} ${bc.range_pts}pts (${bc.atr_mult}x ATR)`, 'big-c')
      ).join('');

      const gexHtml = d.gex_notes
        ? `<div class="bt-gex">${d.gex_notes}</div>`
        : '<div class="bt-gex muted">GEX levels not yet populated — run TV replay</div>';

      container.innerHTML += `
      <div class="bt-day">
        <div class="bt-day-head">
          <div class="bt-date">${dateStr}</div>
          <div class="bias-chip ${d.day_bias}">${d.day_bias}</div>
        </div>
        <div class="bt-body">
          <div class="bt-sessions">
            <div class="bt-sess-label">Asia (6pm–midnight ET)</div>
            <div class="bt-hl-row"><span class="hl-k">High</span><span class="hl-v">${d.asia_high??'--'}</span></div>
            <div class="bt-hl-row"><span class="hl-k">Low</span><span class="hl-v">${d.asia_low??'--'}</span></div>
            <div class="bt-sess-label">London (2am–5am ET)</div>
            <div class="bt-hl-row"><span class="hl-k">High</span><span class="hl-v">${d.london_high??'--'}</span></div>
            <div class="bt-hl-row"><span class="hl-k">Low</span><span class="hl-v">${d.london_low??'--'}</span></div>
            <div class="bt-sess-label">NY Session</div>
            <div class="bt-hl-row"><span class="hl-k">Open</span><span class="hl-v">${d.ny_open??'--'}</span></div>
            <div class="bt-hl-row"><span class="hl-k">High</span><span class="hl-v">${d.ny_high??'--'}</span></div>
            <div class="bt-hl-row"><span class="hl-k">Low</span><span class="hl-v">${d.ny_low??'--'}</span></div>
            <div class="bt-hl-row"><span class="hl-k">Close</span><span class="hl-v">${d.ny_close??'--'}</span></div>
          </div>
          <div class="bt-right">
            <div class="bt-section">
              <div class="bt-sec-title">AMD / Session Sweeps</div>
              ${amdHtml}
            </div>
            ${sweepTags||sigTags ? `<div class="bt-section"><div class="bt-sec-title">Signals — 15m FBOS/RBOS · 3m SMT · 5m DIV</div>${sweepTags}${sigTags}</div>` : ''}
            ${newsTags ? `<div class="bt-section"><div class="bt-sec-title">News Events</div>${newsTags}</div>` : ''}
            ${bcTags ? `<div class="bt-section"><div class="bt-sec-title">Big Candles</div>${bcTags}</div>` : ''}
            <div class="bt-section">
              <div class="bt-sec-title">GEX Suite Levels</div>
              ${gexHtml}
            </div>
          </div>
        </div>
      </div>`;
    });
  }
});
</script>

<div class="header">
  <div class="header-left">
    <div class="brand">Morning Brief</div>
    <div class="header-date" id="date"></div>
  </div>
  <div class="live"><div class="live-dot"></div>LIVE</div>
</div>

<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('brief',this)">Morning Brief</button>
  <button class="tab-btn" onclick="switchTab('backtest',this)">Backtesting <span class="tab-count" id="bt-days-count"></span></button>
  <button class="tab-btn" onclick="switchTab('ai',this);aiCheck()">AI Assistant</button>
</div>

<div id="tab-brief" class="tab-panel active">
  <div class="price-row" id="price-row"></div>
  <div class="sector-row" id="sector-row"></div>
  <div class="main-grid">
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="card">
        <div class="card-head"><div class="card-head-icon"></div><div class="card-head-title">Economic Calendar</div></div>
        <div class="card-body" id="cal-body"></div>
      </div>
      <div class="card">
        <div class="card-head"><div class="card-head-icon"></div><div class="card-head-title">IBKR Positions</div></div>
        <div class="card-body" id="ibkr-body">
          <div id="ibkr-stats" class="ibkr-stats"></div>
          <div id="ibkr-pos"></div>
        </div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="card">
        <div class="card-head"><div class="card-head-icon"></div><div class="card-head-title">Market News</div></div>
        <div class="card-body" id="news-body"></div>
      </div>
    </div>
  </div>
</div>

<div id="tab-backtest" class="tab-panel">
  <div class="bt-summary" id="bt-summary"></div>
  <div id="bt-days"></div>
</div>

<div id="tab-ai" class="tab-panel">
  <div class="chat-warn" id="ai-warn">
    Chat server not running &mdash; start it with: <code>python chat_server.py</code>
  </div>
  <div class="chat-key-form" id="ai-key-form">
    <div class="chat-key-title">Anthropic API Key</div>
    <div style="font-size:11px;color:var(--text)">Enter once &mdash; saved to chat_server config.</div>
    <div class="chat-key-row">
      <input class="chat-key-input" id="ai-key-input" type="password" placeholder="sk-ant-api03-...">
      <button class="chat-key-save" onclick="aiSaveKey()">Save</button>
    </div>
  </div>
  <div class="chat-wrap">
    <div class="chat-msgs" id="ai-msgs"></div>
    <div class="chat-footer">
      <textarea class="chat-input" id="ai-input" rows="2" placeholder="Ask about NQ bias, GEX levels, trade setups, session structure... (Ctrl+Enter to send)"></textarea>
      <button class="chat-send" id="ai-send" onclick="aiSend()">Send</button>
      <button class="chat-key-btn" onclick="aiToggleKey()" title="Set API Key">&#x1F511;</button>
    </div>
  </div>
</div>

<script src="chat.js"></script>
</body>
</html>"""


def save_session_context(days: list):
    """Merge today's AMD session data into gex_levels.json without clobbering TV GEX levels."""
    if not days:
        return
    gex_path = RESULTS_DIR / "gex_levels.json"
    data: dict = {}
    if gex_path.exists():
        try:
            data = json.loads(gex_path.read_text())
        except Exception:
            pass

    today = days[-1]
    ah, al = today.get("asia_high", 0), today.get("asia_low", 0)
    lh, ll = today.get("london_high", 0), today.get("london_low", 0)
    if lh and ah and lh > ah:
        sweep = "Asia High"
    elif ll and al and ll < al:
        sweep = "Asia Low"
    else:
        sweep = "none"

    data["session_amd"] = {
        "date":          str(today.get("date")),
        "asia_high":     ah,
        "asia_low":      al,
        "london_high":   lh,
        "london_low":    ll,
        "london_sweep":  sweep,
        "amd_notes":     today.get("amd_notes", []),
        "day_bias":      today.get("day_bias", "unknown"),
    }

    recent = []
    for r in reversed(days[-6:-1]):
        rh, rl = r.get("london_high", 0), r.get("london_low", 0)
        ra, ra2 = r.get("asia_high", 0), r.get("asia_low", 0)
        rs = "Asia High" if (rh and ra and rh > ra) else "Asia Low" if (rl and ra2 and rl < ra2) else "none"
        recent.append({
            "date":          str(r.get("date")),
            "london_sweep":  rs,
            "day_bias":      r.get("day_bias", "unknown"),
            "notes":         "; ".join(r.get("amd_notes", [])) or "No session sweeps",
        })
    data["recent_amd"] = recent

    gex_path.write_text(json.dumps(data, indent=2, default=str))


def _start_chat_server():
    """Start chat_server.py in a new console if not already running."""
    try:
        with socket.create_connection(("localhost", 8765), timeout=0.5):
            print("  AI Chat server already running on http://localhost:8765")
            return
    except OSError:
        pass
    chat = Path(__file__).parent / "chat_server.py"
    if not chat.exists():
        return
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen([sys.executable, str(chat)], creationflags=flags)
    print("  AI Chat server started -> http://localhost:8765")


def generate_dashboard(ctx, sectors, calendar, news, ibkr, gex, days) -> Path:
    gex_b64  = ""
    gex_path = gex.get("SPY")
    if gex_path and Path(gex_path).exists():
        gex_b64 = base64.b64encode(Path(gex_path).read_bytes()).decode()

    data = {
        "generated": datetime.now().strftime("%A, %d %B %Y  %H:%M"),
        "ctx":       ctx,
        "sectors":   sectors,
        "calendar":  calendar,
        "news":      news,
        "ibkr":      {k: v for k, v in ibkr.items() if k != "positions"},
        "positions": ibkr.get("positions", []),
        "gex_img":   gex_b64,
    }

    html = _HTML.replace("__DATA__", json.dumps(data, default=str))
    html = html.replace("__BACKTEST_DATA__", json.dumps(days, default=str))
    out  = RESULTS_DIR / "morning_brief.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    print("  Fetching data (morning brief + backtest in parallel)...")

    def _brief():
        ctx      = get_overnight_context()
        sectors  = get_sectors()
        calendar = get_econ_calendar()
        news     = get_news_headlines(10)
        ibkr     = get_ibkr_data()
        gex      = capture_gex()
        return ctx, sectors, calendar, news, ibkr, gex

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_brief = ex.submit(_brief)
        f_bt    = ex.submit(_get_backtest_days)
        ctx, sectors, calendar, news, ibkr, gex = f_brief.result()
        days = f_bt.result()

    print(f"  Morning brief ready. Backtest: {len(days)} days.")
    print("  Building dashboard...")
    html_path = generate_dashboard(ctx, sectors, calendar, news, ibkr, gex, days)
    print(f"  Saved -> {html_path.name}")

    webbrowser.open(html_path.as_uri())
    print("  Opened in browser.")

    out = {
        "generated": datetime.now().isoformat(),
        "ctx": ctx, "sectors": sectors,
        "calendar": calendar, "news": news,
        "ibkr": {k: v for k, v in ibkr.items() if k != "positions"},
        "positions": ibkr.get("positions", []),
    }
    (RESULTS_DIR / "morning_brief.json").write_text(
        json.dumps(out, indent=2, default=str)
    )

    save_session_context(days)
    _start_chat_server()


if __name__ == "__main__":
    main()
