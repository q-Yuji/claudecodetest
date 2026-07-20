"""build_site.py — assemble the complete deployable Situation Room site.

Everything a professional go-live needs, built and testable locally long
before any of it is public (the go-live steps live in GO-LIVE.md):

  publish/
    index.html          landing page — pitch, live proof numbers, pricing
    login/index.html    magic-code login (calls /api/login + /api/verify)
    sample/index.html   FREE tier: yesterday's full page, banner on top
    room/index.html     MEMBERS: today's page (gated by functions/_middleware)
    terms/, privacy/    boilerplate the professional look requires
    functions/          Cloudflare Pages Functions (auth, paywall, webhook)
    _routes.json        which paths run through the worker
    og.png              social-share card (daily stats card when available)

Design decisions:
  - The room/sample pages ARE the --public edition of situation_room.py —
    one renderer, one aesthetic. The landing page reuses its CSS.
  - Free tier = yesterday's page (roadmap decision): each build archives
    the raw public edition to results/room_archive/YYYY-MM-DD.html and
    the sample page is the newest archive older than today.
  - The redaction gate runs on EVERY page written (landing included) —
    copy that trips the gate gets rewritten, the gate does not get loosened.
  - Shared function helpers are inlined at the {{LIB}} marker so the
    deployed functions/ tree contains only route files.

Run:
  python -m tools.build_site          # build into publish/
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
SITE = ROOT / "site"
STAGE = ROOT / "publish"
ARCHIVE = RESULTS / "room_archive"

ET = ZoneInfo("America/New_York")

# Markers that must NEVER appear on any public page (vocabulary, not just
# secrets — a leaked panel brings its words with it). Fail-safe: copy that
# trips the gate gets rewritten; the list only changes in a reviewed commit.
FORBIDDEN = [
    "gex", "gamma", "call wall", "put wall",          # licensed-feed data
    "mffu", "tradeify", "mffuevbldr", "u24694898",    # firms + account ids
    "my funded futures", "apex",                      # firms
    "ledger", "payout", "cost_usd", "prop_ledger",    # account economics
    "trailing floor", "buffer", "roll_day",           # guardrail language
    "tradovate", "ibkr", "interactive brokers",       # brokers
    "tradeify_state", "drawdown",                     # state + risk language
]


def redaction_gate(html: str) -> list[str]:
    low = html.lower()
    return [t for t in FORBIDDEN if t in low]


def load_site_config() -> dict:
    cfg = {}
    try:
        cfg = json.loads((ROOT / "publish_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    site = cfg.get("site") or {}
    return {
        "brand": site.get("brand") or "NQ Situation Room",
        "by": site.get("by") or "SweepStats",
        "tagline": site.get("tagline") or "No gurus. No narratives. Distributions.",
        "price_label": site.get("price_label") or "$29/mo",
        "checkout_url": site.get("checkout_url"),  # None until Whop exists
        "site_url": site.get("site_url"),          # None until domain exists
        "contact_email": site.get("contact_email") or "contact@example.com",
    }


# ------------------------------------------------------------------ pages

def _room_html() -> str:
    """Today's public edition, rendered fresh through situation_room.py."""
    import situation_room as sr
    brief, brief_age = sr.load("morning_brief.json", "generated")
    gex, gex_age = sr.load("gex_levels.json", "timestamp")
    summary, summary_age = sr.load("session_stats_summary.json", "generated")
    return sr.build_page(brief, brief_age, gex, gex_age, summary, summary_age,
                         sr.load_scoreboard(), None, None, public=True)


def _inject_after_body(html: str, fragment: str) -> str:
    marker = "<body><div class='page'>"
    if marker in html:
        return html.replace(marker, f"<body>{fragment}<div class='page'>", 1)
    return html.replace("<body>", f"<body>{fragment}", 1)


_BAR_CSS = (
    "font:10px 'Cascadia Mono','Consolas',monospace;letter-spacing:.14em;"
    "padding:8px 32px;display:flex;justify-content:space-between;"
    "border-bottom:1px solid #1e2833;color:#5c6a78")


def _member_bar() -> str:
    return (
        f'<div style="{_BAR_CSS}"><span>MEMBER EDITION · TODAY\'S PAGE</span>'
        '<span><a href="/sample/" style="color:#5c6a78">SAMPLE</a>'
        '<span style="color:#2b3844"> / </span>'
        '<a href="#" style="color:#ffb454" '
        'onclick="fetch(\'/api/logout\',{method:\'POST\'})'
        '.then(function(){location=\'/\'});return false">LOG OUT</a></span></div>')


def _sample_banner(cfg: dict) -> str:
    join = (f'<a href="{cfg["checkout_url"]}" style="color:#0a0d10">'
            if cfg["checkout_url"] else '<a href="/#pricing" style="color:#0a0d10">')
    return (
        '<div style="background:#ffb454;color:#0a0d10;padding:10px 32px;'
        "font:11px 'Cascadia Mono','Consolas',monospace;letter-spacing:.14em\">"
        "FREE SAMPLE — THIS IS YESTERDAY'S PAGE, IN FULL. MEMBERS GET TODAY'S "
        f'EVERY MORNING. {join}<b>JOIN →</b></a></div>')


def _rotate_sample(room_html: str, today: str) -> str:
    """Archive today's raw page; the sample is the newest archive from a
    prior day (first build ever: today's page is the sample too)."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / f"{today}.html").write_text(room_html, encoding="utf-8")
    for f in sorted(ARCHIVE.glob("*.html"))[:-14]:
        f.unlink()  # keep two weeks
    older = [f for f in sorted(ARCHIVE.glob("*.html")) if f.stem < today]
    return (older[-1].read_text(encoding="utf-8") if older else room_html)


def _noindex(html: str) -> str:
    return html.replace("<head>", '<head><meta name="robots" content="noindex">', 1)


# ---------------------------------------------------- landing + login pages

def _mast(cfg: dict, active: str = "") -> str:
    return (
        '<div class="mast">'
        f'<div class="brand">NQ <em>SITUATION ROOM</em>'
        f'<span class="bsub">BY {cfg["by"].upper()}</span></div>'
        '<nav class="mnav">'
        '<a href="/sample/">SAMPLE</a><span class="nsep">/</span>'
        '<a href="/#pricing">PRICING</a><span class="nsep">/</span>'
        '<a href="/login/">LOG IN</a><span class="nsep">/</span>'
        '<a href="/room/">MEMBER AREA</a></nav>'
        f'<div class="mmeta">{cfg["tagline"].upper()}</div>'
        "</div><div class='rule2'></div>")


def _page_shell(cfg: dict, title: str, body: str, extra_css: str = "",
                extra_js: str = "") -> str:
    import situation_room as sr
    desc = ("NQ session statistics that grade themselves — first-touch "
            "fakeout odds, stop-run depths, and a public walk-forward record.")
    og = ""
    if cfg["site_url"]:
        og = (f'<meta property="og:image" content="{cfg["site_url"]}/og.png">'
              f'<meta property="og:url" content="{cfg["site_url"]}">')
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        f"<meta name='description' content='{desc}'>"
        f"<meta property='og:title' content='{title}'>"
        f"<meta property='og:description' content='{desc}'>{og}"
        f"<style>{sr.CSS}{extra_css}</style></head>"
        f"<body><div class='page'>{body}</div>"
        f"<script>{extra_js}</script></body></html>")


LANDING_CSS = """
.hero-l{padding:60px 0 40px}
.cta{display:inline-block;background:#ffb454;color:#0a0d10;font:600 13px
  "Cascadia Mono","Consolas",monospace;letter-spacing:.14em;padding:13px 30px;
  margin-top:22px}
.cta:hover{background:#e8eef4}
.cta.ghost{background:none;border:1px solid #2b3844;color:#c9d4de;margin-left:12px}
.cta.ghost:hover{border-color:#ffb454;color:#ffb454}
.plist{list-style:none;margin:8px 0}
.plist li{padding:11px 0;border-bottom:1px solid #161d26;font-size:15px;
  color:#8b9aa8;display:flex;gap:14px;align-items:baseline}
.plist li b{color:#e8eef4;font-weight:600;white-space:nowrap}
.plist li i{font-style:normal;color:#ffb454;
  font:11px "Cascadia Mono","Consolas",monospace}
.pricebox{border:1px solid #2b3844;padding:34px 38px;max-width:430px}
.pricebig{font-family:"Bahnschrift SemiBold Condensed","Bahnschrift",
  "Arial Narrow",sans-serif;font-size:64px;color:#e8eef4;line-height:1}
.pricebig i{font-style:normal;font-size:22px;color:#5c6a78}
.finep{font-size:12px;color:#5c6a78;margin-top:14px;line-height:1.7}
"""


def build_landing(cfg: dict, summary: dict | None, sb: dict | None) -> str:
    n = fake_pct = touches = 0
    frm = to = "?"
    if summary:
        import situation_room as sr
        fake_pct, _, touches = sr._hero_numbers(summary)
        s = summary.get("sample") or {}
        n, frm, to = s.get("sessions", 0), s.get("from", "?"), s.get("to", "?")

    rec_html = ""
    if sb and (sb.get("record") or {}).get("calls"):
        r, fo = sb["record"], sb["fakeout_oos"]
        rec_html = (
            f'<p class="dek">Our directional call is hitting '
            f'<span class="mono amber">{float(r["hit_pct"]):.0f}%</span> over '
            f'{int(r["calls"])} graded calls — <b>below chance, and we publish '
            f'that</b>. The fakeout claim? Claimed ~{float(fo["avg_claim_pct"] or 0):.0f}%, '
            f'running <span class="mono amber">{float(fo["oos_pct"] or 0):.0f}%</span> '
            f'out-of-sample across {int(fo["touches"])} touches. Every number on '
            f'the page is graded walk-forward, in public, every day. Sellers of '
            f'certainty can\'t do this. Distributions can.</p>')

    join = (f'<a class="cta" href="{cfg["checkout_url"]}">JOIN — {cfg["price_label"]}</a>'
            if cfg["checkout_url"] else
            '<span class="cta" style="opacity:.55;cursor:default">OPENING SOON</span>')

    body = (
        _mast(cfg)
        + '<div class="hero-l"><div class="herogrid"><div>'
          f'<div class="kicker mono">NQ SESSION STATISTICS · UPDATED EVERY TRADING DAY</div>'
          f'<h1 class="verdict">{cfg["tagline"]}</h1>'
          '<p class="dek">Every overnight level the market watches — with the '
          'measured odds of what happens at the first touch: how often it\'s a '
          'trap, how far stops get run past it, what the bounce pays. One page, '
          'one instrument, rebuilt before every New York open from a dataset '
          'that compounds daily.</p>'
          f'{join}<a class="cta ghost" href="/sample/">SEE YESTERDAY\'S PAGE FREE</a>'
          "</div>"
          f'<div class="bigstat"><span class="bignum">{fake_pct}<i>%</i></span>'
          f'<span class="bigcap mono">OF FIRST NY TOUCHES OF OVERNIGHT LEVELS '
          f'ARE FAKEOUTS<br>n={touches} touches · {n} sessions · {frm} → {to}'
          "</span></div></div></div>"
        + sr_section("01", "What the page tells you, every morning", (
            '<ul class="plist">'
            "<li><i>SCRIPT</i><span><b>Today's Script.</b> Which overnight pattern "
            "printed and what New York historically did with it — with sample "
            "sizes, never vibes.</span></li>"
            "<li><i>MAP</i><span><b>The liquidity map.</b> Asia/London extremes, "
            "prior-day and prior-week highs and lows — each with its first-touch "
            "odds attached.</span></li>"
            "<li><i>STOPS</i><span><b>Where your stop dies.</b> Median run PAST "
            "each level before the reversal — placement geometry nobody else "
            "publishes.</span></li>"
            "<li><i>CLOCK</i><span><b>Time-of-day odds.</b> A 9:45 touch and a "
            "13:30 touch are different trades. The page knows the difference."
            "</span></li>"
            "<li><i>GAP</i><span><b>Weekend gap mechanics.</b> How often the "
            "Friday-to-Sunday gap fills before the open — measured, not "
            "asserted.</span></li>"
            "<li><i>RECORD</i><span><b>The Record.</b> Every morning call graded "
            "next day, walk-forward, misses included. The scoreboard is the "
            "product.</span></li></ul>"))
        + sr_section("02", "The record grades itself — even when it loses",
                     rec_html or '<p class="dek">The graded record publishes '
                                 'here as the dataset grows.</p>')
        + sr_section("03", "Pricing", (
            '<div class="pricebox" id="pricing">'
            f'<div class="pricebig">{cfg["price_label"].split("/")[0]}'
            f'<i>/{cfg["price_label"].split("/")[-1]}</i></div>'
            '<p class="dek" style="margin-top:10px">Full page every trading '
            'morning · all odds tables · the graded record · cancel anytime.</p>'
            f'{join}'
            '<p class="finep">Billed through Whop. The free sample is '
            "yesterday's page, in full — judge the product on it. Past "
            "frequency ≠ future probability; nothing here is financial "
            "advice.</p></div>"))
        + sr_section("04", "Method", (
            '<p class="dek">5-minute closes decide everything. A fakeout is a '
            'reclaim within 2 bars of the first New York breach of an overnight '
            'extreme. Definitions are pinned and versioned; when a claim stops '
            'holding out-of-sample, the page says so. '
            f'Questions: <a href="mailto:{cfg["contact_email"]}" '
            f'style="color:#ffb454">{cfg["contact_email"]}</a></p>'))
        + _foot(cfg))
    return _page_shell(cfg, f"{cfg['brand']} — NQ session odds, graded daily",
                       body, LANDING_CSS)


def sr_section(num: str, title: str, inner: str) -> str:
    import situation_room as sr
    return sr.section(num, title, inner)


def _foot(cfg: dict) -> str:
    return ('<div class="foot">'
            f'<div><span class="fh">{cfg["by"].upper()}</span>NQ session-level '
            'statistics — a compounding dataset, updated daily.</div>'
            '<div class="fc"><span class="fh">LEGAL</span>'
            '<a href="/terms/">Terms</a> · <a href="/privacy/">Privacy</a></div>'
            '<div class="fr"><span class="fh">DISCLAIMER</span>Past frequency ≠ '
            'future probability.<br>Not financial advice.</div></div>')


LOGIN_CSS = """
.loginbox{max-width:430px;margin:70px auto 120px}
.loginbox h1{font-family:"Bahnschrift SemiBold Condensed","Bahnschrift",
  "Arial Narrow",sans-serif;font-size:40px;color:#e8eef4;margin-bottom:8px}
.loginbox .dek{margin-bottom:26px}
.fld{width:100%;background:#10151b;border:1px solid #2b3844;color:#e8eef4;
  font:15px "Cascadia Mono","Consolas",monospace;padding:12px 14px;margin:6px 0 14px}
.fld:focus{outline:none;border-color:#ffb454}
.btn{width:100%;background:#ffb454;border:none;color:#0a0d10;cursor:pointer;
  font:600 13px "Cascadia Mono","Consolas",monospace;letter-spacing:.14em;
  padding:13px}
.btn:hover{background:#e8eef4}
.msg{font:12px "Cascadia Mono","Consolas",monospace;margin-top:14px;
  line-height:1.7;color:#8b9aa8;min-height:18px}
.msg.err{color:#ff5f56}.msg.ok{color:#3ddc84}
#step2{display:none}
"""

LOGIN_JS = """
var nextUrl=new URLSearchParams(location.search).get('next')||'/room/';
var email='';
function show(id,on){document.getElementById(id).style.display=on?'block':'none'}
function msg(t,cls){var m=document.getElementById('msg');
  m.textContent=t;m.className='msg '+(cls||'')}
if(new URLSearchParams(location.search).get('expired'))
  msg('Your membership looks inactive — log in again or rejoin.','err');
document.getElementById('f1').addEventListener('submit',function(e){
  e.preventDefault();
  email=document.getElementById('email').value.trim();
  msg('Sending code\\u2026');
  fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email:email})})
  .then(function(r){return r.json().then(function(d){return{s:r.status,d:d}})})
  .then(function(x){
    if(x.s===200){show('step1',false);show('step2',true);
      msg(x.d.dev_code?('DEV CODE: '+x.d.dev_code):'Code sent \\u2014 check your email.','ok');}
    else if(x.s===403&&x.d.checkout_url){
      msg('No active membership for that email.','err');
      var a=document.getElementById('joinlink');a.href=x.d.checkout_url;
      a.style.display='inline';}
    else msg(x.d.error||'Something went wrong.','err');
  }).catch(function(){msg('Network error.','err')});
});
document.getElementById('f2').addEventListener('submit',function(e){
  e.preventDefault();
  fetch('/api/verify',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email:email,code:document.getElementById('code').value.trim()})})
  .then(function(r){return r.json().then(function(d){return{s:r.status,d:d}})})
  .then(function(x){
    if(x.s===200){msg('Welcome back.','ok');location=nextUrl;}
    else msg(x.d.error||'Wrong code.','err');
  }).catch(function(){msg('Network error.','err')});
});
"""


def build_login(cfg: dict) -> str:
    body = (
        _mast(cfg)
        + '<div class="loginbox">'
          '<h1>Member login.</h1>'
          '<p class="dek">Enter the email you joined with — we\'ll send a '
          'six-digit code. No passwords, ever.</p>'
          '<div id="step1"><form id="f1">'
          '<input class="fld" id="email" type="email" placeholder="you@example.com" '
          'autocomplete="email" required>'
          '<button class="btn" type="submit">SEND CODE</button></form></div>'
          '<div id="step2"><form id="f2">'
          '<input class="fld" id="code" inputmode="numeric" pattern="\\d{6}" '
          'maxlength="6" placeholder="6-digit code" required>'
          '<button class="btn" type="submit">LOG IN</button></form></div>'
          '<div class="msg" id="msg"></div>'
          '<a id="joinlink" href="#" style="display:none;color:#ffb454" '
          'class="mono">JOIN FIRST →</a>'
          "</div>"
        + _foot(cfg))
    return _page_shell(cfg, f"Log in — {cfg['brand']}", body, LOGIN_CSS, LOGIN_JS)


def build_legal(cfg: dict, kind: str) -> str:
    if kind == "terms":
        title, text = "Terms of Service", (
            "<p class='dek'>This service publishes historical statistics about "
            "futures market behavior. It is informational content, not "
            "investment advice, portfolio management, or a recommendation to "
            "trade any instrument. Past frequency does not predict future "
            "probability. Trading futures involves substantial risk of loss.</p>"
            "<p class='dek'>Subscriptions are billed and can be cancelled "
            "through Whop at any time; access ends when a subscription "
            "lapses. Content is for the subscriber's personal use and may not "
            "be redistributed or resold. The service is provided as-is, "
            "without warranty; totals and statistics may be revised when data "
            "errors are found — corrections are published, not hidden.</p>")
    else:
        title, text = "Privacy", (
            "<p class='dek'>We store the minimum needed to run a membership: "
            "your email address, membership status, and a login session "
            "cookie. Login is by emailed one-time code — no passwords are "
            "stored. Payments are processed by Whop; we never see card "
            "details. No analytics trackers, no data sales, no third-party "
            "ad tech.</p>"
            f"<p class='dek'>Data questions or deletion requests: "
            f"<a href='mailto:{cfg['contact_email']}' style='color:#ffb454'>"
            f"{cfg['contact_email']}</a>.</p>")
    body = (_mast(cfg)
            + f'<div style="max-width:640px;padding:50px 0 90px">'
              f'<h1 class="verdict vsm">{title}.</h1>{text}</div>'
            + _foot(cfg))
    return _page_shell(cfg, f"{title} — {cfg['brand']}", body)


# ------------------------------------------------------------------ build

def _write_gated(path: Path, html: str, page: str) -> None:
    hits = redaction_gate(html)
    if hits:
        raise SystemExit(f"REDACTION GATE FAILED on {page}: {hits} — "
                         "not writing the site. Fix the copy/leak.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _inline_functions() -> int:
    """Copy site/functions_src -> publish/functions, inlining _shared.js
    at each {{LIB}} marker."""
    lib = (SITE / "functions_src" / "_shared.js").read_text(encoding="utf-8")
    dst_root = STAGE / "functions"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    n = 0
    for src in (SITE / "functions_src").rglob("*.js"):
        if src.name == "_shared.js":
            continue
        rel = src.relative_to(SITE / "functions_src")
        out = src.read_text(encoding="utf-8").replace("{{LIB}}", lib)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(out, encoding="utf-8")
        n += 1
    return n


def build() -> None:
    cfg = load_site_config()
    today = datetime.now(ET).date().isoformat()
    STAGE.mkdir(exist_ok=True)

    try:
        summary = json.loads((RESULTS / "session_stats_summary.json")
                             .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = None
    try:
        from backtest.scoreboard import compute_from_file
        sb = compute_from_file()
    except Exception:
        sb = None

    room_raw = _room_html()
    sample_raw = _rotate_sample(room_raw, today)

    _write_gated(STAGE / "room" / "index.html",
                 _noindex(_inject_after_body(room_raw, _member_bar())), "room")
    _write_gated(STAGE / "sample" / "index.html",
                 _inject_after_body(sample_raw, _sample_banner(cfg)), "sample")
    _write_gated(STAGE / "index.html", build_landing(cfg, summary, sb), "landing")
    _write_gated(STAGE / "login" / "index.html", build_login(cfg), "login")
    _write_gated(STAGE / "terms" / "index.html", build_legal(cfg, "terms"), "terms")
    _write_gated(STAGE / "privacy" / "index.html", build_legal(cfg, "privacy"),
                 "privacy")

    shutil.copy(SITE / "static" / "_routes.json", STAGE / "_routes.json")
    (STAGE / ".nojekyll").write_text("", encoding="utf-8")
    n_fn = _inline_functions()

    card = RESULTS / "stats_card.png"
    if card.exists():
        shutil.copy(card, STAGE / "og.png")

    print(f"  site -> {STAGE}  (room/sample/landing/login/legal, "
          f"{n_fn} edge functions, gate clean on every page)")


if __name__ == "__main__":
    build()
