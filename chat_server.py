"""
Local AI chat proxy for the trading dashboard.

Start : python chat_server.py
URL   : http://localhost:8765

Context sources (read on every request):
  results/morning_brief.json  -- prices, sectors, news, IBKR
  results/gex_levels.json     -- GEX levels + session AMD
"""

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import requests as _req

RESULTS      = Path(__file__).parent / "results"
PORT         = 8765
MODEL        = "claude-sonnet-4-6"
API_URL      = "https://api.anthropic.com/v1/messages"
CONFIG_FILE  = RESULTS / ".chat_config.json"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_key() -> str:
    if k := os.environ.get("ANTHROPIC_API_KEY"):
        return k
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text()).get("api_key", "")
        except Exception:
            pass
    return ""


def _save_key(key: str):
    RESULTS.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"api_key": key}))


# ---------------------------------------------------------------------------
# Context + system prompt
# ---------------------------------------------------------------------------

def _load_ctx() -> dict:
    ctx = {}
    for name, fname in [("brief", "morning_brief.json"), ("gex", "gex_levels.json")]:
        p = RESULTS / fname
        if p.exists():
            try:
                ctx[name] = json.loads(p.read_text())
            except Exception:
                pass
    return ctx


def _build_system(ctx: dict) -> str:
    now = datetime.now().strftime("%A %d %B %Y  %H:%M ET")
    lines = [
        f"You are Jarvis, a sharp NQ futures trading assistant. Time: {now}.",
        "The user trades NQ futures and options on IBKR using ICT/SMC methodology and GEX Suite confluence.",
        "Be direct, specific, concise. Use trading terminology. No generic risk disclaimers.",
        "When discussing levels: always state what a hold or break means for directional bias.",
        "",
    ]

    brief = ctx.get("brief", {})
    if brief:
        lines.append("=== MARKET SNAPSHOT ===")
        for sym in ["NQ", "ES", "SPY", "QQQ", "VIX"]:
            d = brief.get("ctx", {}).get(sym, {})
            if d and not d.get("error"):
                s = "+" if d.get("chg_pct", 0) >= 0 else ""
                lines.append(f"  {sym}: {d['last']}  ({s}{d['chg_pct']}%  prev {d['prev_close']})")

        secs = brief.get("sectors", {})
        if secs:
            lines.append("\n=== SECTORS ===")
            for _, s in secs.items():
                if not s.get("error"):
                    sg = "+" if s.get("chg_pct", 0) >= 0 else ""
                    lines.append(f"  {s['name']}: {sg}{s['chg_pct']}%")

        news = [n for n in brief.get("news", []) if not n.get("error")]
        if news:
            lines.append("\n=== NEWS HEADLINES ===")
            for n in news[:6]:
                lines.append(f"  - {n['title']}")

        cal = [e for e in brief.get("calendar", []) if not e.get("error")]
        if cal:
            lines.append("\n=== ECONOMIC CALENDAR ===")
            for e in cal:
                lines.append(f"  {e['time']} [{e['impact'].upper()}] {e['title']}  F:{e['forecast']}  P:{e['previous']}")

        pos = brief.get("positions", [])
        ibkr = brief.get("ibkr", {})
        if ibkr.get("error"):
            lines.append(f"\nIBKR: {ibkr['error']}")
        elif pos:
            lines.append("\n=== IBKR POSITIONS ===")
            for p in pos:
                sym = (p.get("contractDesc") or p.get("ticker") or "?")[:40]
                lines.append(f"  {sym}: qty={p.get('position')}  PnL=${p.get('unrealizedPnl', 0):.0f}")
        else:
            lines.append("\nIBKR: No open positions")

    gex = ctx.get("gex", {})
    if gex:
        lines.append("\n=== GEX SUITE LEVELS (NQ) ===")
        cur    = gex.get("current_price")
        levels = gex.get("levels", [])
        ts     = gex.get("timestamp", "unknown")

        gf = next((l for l in levels if l["name"] == "Gamma Flip"), None)
        if gf and cur:
            regime = ("POSITIVE — choppy, mean-reverting; fade/scalp mode"
                      if cur > gf["price"] else
                      "NEGATIVE — trending; momentum accelerates; do NOT fade hard moves")
            lines.append(f"  Gamma Regime: {regime}")
        if cur:
            lines.append(f"  NQ price when fetched: {cur}")
        lines.append(f"  (fetched {ts})\n  Levels sorted high→low:")

        for lv in sorted(levels, key=lambda x: x["price"], reverse=True):
            dist = f"  ({'+' if lv['price'] >= (cur or 0) else ''}{lv['price'] - (cur or 0):.0f} pts)" if cur else ""
            flag = "  <<< NEAREST" if cur and abs(lv["price"] - cur) < 100 else ""
            lines.append(f"    {lv['name']:<35} {lv['price']}{dist}{flag}")

        amd = gex.get("session_amd", {})
        if amd:
            lines.append("\n=== TODAY'S SESSION (AMD) ===")
            lines.append(f"  Asia:   H={amd.get('asia_high')}  L={amd.get('asia_low')}")
            lines.append(f"  London: H={amd.get('london_high')}  L={amd.get('london_low')}")
            sweep = amd.get("london_sweep", "none")
            if sweep == "none":
                lines.append("  London sweep: NONE — London inside Asia range.")
                lines.append("  → Manipulation phase has NOT happened yet. Watch for NY open to take a side.")
            else:
                lines.append(f"  London sweep: {sweep}")
                lines.append(f"  → Classic AMD: expect NY distribution opposite to the sweep direction.")
            for note in amd.get("amd_notes", []):
                lines.append(f"  {note}")
            lines.append(f"  Algo day bias: {amd.get('day_bias', 'unknown')}")

        recent = gex.get("recent_amd", [])
        if recent:
            lines.append("\n=== RECENT AMD PATTERN (last 5 sessions) ===")
            for r in recent:
                lines.append(f"  {r['date']}  sweep={r.get('london_sweep','—')}  bias={r['day_bias']}  |  {r['notes']}")

    lines += [
        "",
        "=== TRADING FRAMEWORK ===",
        "GEX: Gamma Flip is the regime line. Above=positive gamma (chop/fade). Below=negative gamma (trend/momentum).",
        "GEX: 0DTE Call Wall / Put Wall are same-day intraday magnets. Break=momentum. Hold=reversal.",
        "GEX: Gamma Levels 1–10: Level 1 strongest. Price steps between them like stair steps.",
        "AMD: Accumulation (Asia quiet range) → Manipulation (London sweeps one side) → Distribution (NY delivers).",
        "ICT: London sweep of Asia High/Low = manipulation; expect NY reversal from that extreme (FBO setup).",
        "Confluence: GEX level + AMD sweep level = highest probability reaction zone.",
        "Strength modifiers: +++ on a GEX level = maximum conviction; prioritise over plain levels regardless of type.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "key_set": bool(_load_key())}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _sse(self, text: str):
        self.wfile.write(f'data: {json.dumps({"text": text})}\n\n'.encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def do_POST(self):
        if self.path == "/set-key":
            body = self._body()
            _save_key(body.get("key", ""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())
            return

        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return

        body     = self._body()
        messages = body.get("messages", [])
        api_key  = _load_key()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()

        if not api_key:
            self._sse("No API key set. Click the \U0001f511 icon to add your Anthropic API key.")
            return

        ctx    = _load_ctx()
        system = _build_system(ctx)

        try:
            resp = _req.post(
                API_URL,
                headers={
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      MODEL,
                    "max_tokens": 1024,
                    "system":     system,
                    "messages":   messages,
                    "stream":     True,
                },
                stream=True,
                timeout=30,
            )

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    break
                try:
                    ev   = json.loads(data)
                    text = ev.get("delta", {}).get("text", "") if ev.get("type") == "content_block_delta" else ""
                    if text:
                        self.wfile.write(f'data: {json.dumps({"text": text})}\n\n'.encode())
                        self.wfile.flush()
                except Exception:
                    pass

        except Exception as e:
            self._sse(f"Error calling Claude API: {e}")


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    srv = _Server(("localhost", PORT), _Handler)
    key_status = "SET" if _load_key() else "not set — add via dashboard 🔑"
    print(f"  AI Chat server  http://localhost:{PORT}  |  API key: {key_status}")
    print("  Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("  Stopped.")


if __name__ == "__main__":
    main()
