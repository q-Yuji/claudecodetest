"""
Morning Brief -- pre-session market analysis.

Run before market open:
  python morning_brief.py

Pulls overnight prices, sector ETFs, economic calendar, news, GEX screenshot,
and IBKR positions -- then renders a dashboard HTML and opens it in the browser.
"""

import base64
import json
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime, date
from pathlib import Path

import yfinance as yf
import requests

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
            if title:
                headlines.append({"title": title, "date": pubdate})
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
.header{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.header-left{}
.brand{font-size:10px;font-weight:700;letter-spacing:.25em;color:var(--accent);text-transform:uppercase;margin-bottom:4px}
.header-date{font-size:18px;font-weight:700;color:var(--text)}
.live{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--green);letter-spacing:.1em}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

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
.news-item{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid var(--subtle);align-items:flex-start;cursor:default}
.news-item:last-child{border-bottom:none}
.news-bar{width:2px;min-height:30px;background:var(--accent);border-radius:2px;flex-shrink:0;opacity:.4}
.news-content{}
.news-title{font-size:12px;color:var(--text);line-height:1.5;margin-bottom:2px}
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

/* GEX */
.gex-img{width:100%;border-radius:6px;display:block}

/* DIVIDER */
.section-gap{height:12px}
</style>
</head>
<body>
<script>
const D = __DATA__;

const fmt  = (n,d=2) => n==null?'--':Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct  = n => n==null?'--':(n>=0?'+':'')+fmt(n)+'%';
const money= n => n==null?'--':(n>=0?'+$':'-$')+fmt(Math.abs(n),0);
const cls  = n => n>0?'up':n<0?'down':'flat';

window.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('date').textContent = D.generated;

  // PRICES
  const priceGrid = document.getElementById('price-row');
  ['NQ','ES','SPY','QQQ','VIX'].forEach(sym=>{
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
  Object.entries(D.sectors||{}).forEach(([,s])=>{
    if(s.error) return;
    const c = cls(s.chg_pct);
    srow.innerHTML += `<div class="sector-chip">
      <div class="s-name">${s.name}</div>
      <div class="s-chg ${c}">${pct(s.chg_pct)}</div>
    </div>`;
  });

  // CALENDAR
  const cal = document.getElementById('cal-body');
  const evts = (D.calendar||[]).filter(e=>!e.error);
  if(!evts.length){
    cal.innerHTML='<div class="empty-msg">No high-impact USD events today</div>';
  } else {
    evts.forEach(e=>{
      cal.innerHTML+=`<div class="cal-item">
        <div class="cal-time">${e.time}</div>
        <div class="cal-badge ${e.impact==='High'?'high':'med'}">${e.impact.toUpperCase()}</div>
        <div class="cal-name">${e.title}</div>
        <div class="cal-vals">f: ${e.forecast}<br>p: ${e.previous}</div>
      </div>`;
    });
  }

  // NEWS
  const news = document.getElementById('news-body');
  (D.news||[]).forEach(n=>{
    if(n.error) return;
    const d = n.date ? n.date.replace(/\s*\+\d+\s*$/,'').trim() : '';
    news.innerHTML+=`<div class="news-item">
      <div class="news-bar"></div>
      <div class="news-content">
        <div class="news-title">${n.title}</div>
        ${d?`<div class="news-date">${d}</div>`:''}
      </div>
    </div>`;
  });

  // IBKR
  const ibkr = D.ibkr||{};
  const ibkrBody = document.getElementById('ibkr-body');
  if(ibkr.error){
    ibkrBody.innerHTML=`<div class="empty-msg">${ibkr.error}</div>`;
  } else {
    const pnlC = cls(ibkr.daily_pnl||0);
    document.getElementById('ibkr-stats').innerHTML=`
      <div class="istat"><div class="istat-label">Net Liq</div><div class="istat-val">$${fmt(ibkr.net_liq,0)}</div></div>
      <div class="istat"><div class="istat-label">Daily P&L</div><div class="istat-val ${pnlC}">${money(ibkr.daily_pnl)}</div></div>
      <div class="istat"><div class="istat-label">Buying Power</div><div class="istat-val">$${fmt(ibkr.buying_pow,0)}</div></div>
    `;
    const pos = D.positions||[];
    const posEl = document.getElementById('ibkr-pos');
    if(!pos.length){
      posEl.innerHTML='<div class="empty-msg">No open positions</div>';
    } else {
      let rows = pos.map(p=>{
        const sym  = (p.contractDesc||p.ticker||'?').slice(0,48);
        const qty  = p.position||0;
        const pnl  = p.unrealizedPnl||0;
        const pc   = cls(pnl);
        return `<tr>
          <td class="pos-sym">${sym}</td>
          <td class="pos-qty">${qty>0?'+':''}${qty}</td>
          <td class="pnl ${pc}">${money(pnl)}</td>
        </tr>`;
      }).join('');
      posEl.innerHTML=`<table class="pos-table">
        <thead><tr><th>Contract</th><th>Qty</th><th>P&L</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    }
  }

  // GEX
  if(D.gex_img){
    document.getElementById('gex-img').src='data:image/png;base64,'+D.gex_img;
    document.getElementById('gex-img').style.display='block';
    document.getElementById('gex-empty').style.display='none';
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
    <div class="card">
      <div class="card-head"><div class="card-head-icon"></div><div class="card-head-title">GEX Suite -- SPY Heatmap</div></div>
      <div class="card-body" style="padding:10px">
        <img id="gex-img" class="gex-img" style="display:none" alt="GEX heatmap"/>
        <div id="gex-empty" class="empty-msg">Launch Chrome + GEX Suite first</div>
      </div>
    </div>
  </div>
</div>

</body>
</html>"""


def generate_dashboard(ctx, sectors, calendar, news, ibkr, gex) -> Path:
    gex_b64 = ""
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
    out  = RESULTS_DIR / "morning_brief.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    print("  Fetching market data...")
    ctx      = get_overnight_context()
    sectors  = get_sectors()
    calendar = get_econ_calendar()
    news     = get_news_headlines(10)
    ibkr     = get_ibkr_data()
    gex      = capture_gex()

    print("  Building dashboard...")
    html_path = generate_dashboard(ctx, sectors, calendar, news, ibkr, gex)
    print(f"  Saved -> {html_path.name}")

    webbrowser.open(html_path.as_uri())
    print("  Opened in browser.")

    # Also save JSON for reference
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


if __name__ == "__main__":
    main()
