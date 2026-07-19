# Situation Room — product roadmap (what makes it worth building & sellable)

Written by Fable 5, 2026-07-09, planning session before user is away
(back Sunday 2026-07-12 evening). This is NOT a build spec — blueprint #6
(`situation-room.md`) is the v1 build spec and stands unchanged. This doc
answers the open question from 2026-07-08: **which capabilities justify a
subscription**, ranked through the user's filter (proven demand · specific
wedge · distribution).

## Core thesis

SitDeck went viral on looks; looks are the *distribution*, not the product.
A generic trading war room loses to TradingView on day one. The wedge is
that the page can answer, live, the question every NQ session trader asks
and nobody serves with real numbers:

> "Given what happened overnight, what are the odds of THIS trade, right now?"

Static aggregates ("61% of first touches are fakeouts") are content — great
for posting, not worth paying for. **Conditioning** — today's pattern
matched against the compounding dataset, per level, per time of day, with
follow-through and stop-run distances attached — is the utility. The
dataset that makes that possible compounds daily and is the moat.

## Demand evidence (2026-07-09 web scan) — who the average trader is and what he pays for

**The target customer is the prop-eval futures trader**, and he is measurably
desperate in exactly the places this dataset speaks to:

- Eval pass rates run **5–20%**; only **~7% of prop traders ever see a
  payout**; **~half of funded traders lose the account within 90 days**
  (quantvps.com / damnpropfirms.com / apextraderfunding.com, 2026 data).
- **~70% of eval failures are drawdown breaches** (max trailing or daily
  loss) — not bad analysis, bad risk placement/sizing. This crowd re-buys
  evals and resets over and over; their pain is denominated in repeated
  $50–300 fees, so "saves one blown eval" is a self-evident value prop.
- Stop-run pain is so universal that platforms write education about it
  (NinjaTrader's liquidity-trap guide: retail stops cluster at prior-day
  highs/lows and **overnight session extremes** — literally the levels
  SweepStats measures).
- What they already pay monthly: TradingView Premium ~$60, NinjaTrader
  $36–56, order-flow tools $69+, paid Discords **$99–299** for morning
  bias calls, and — the key comparable — **Edgeful at $49/mo**.

**Edgeful is the proof AND the competitor to beat.** 150+ statistical
reports (gap fill by gap-size bucket, opening-range breakout, initial
balance), TradingView integration, alerts, $49/mo ($33 annual). It proves
traders pay for session statistics at scale. The honest read of the gap:

1. **Language/concept gap** — Edgeful's reports are classic pit-session
   stats (gaps, ORB, IB). The ICT/SMC generation thinks in *sweeps,
   liquidity, manipulation, killzones* — SweepStats quantifies THAT
   vocabulary (fakeout %, London-manipulation bias, overshoot depth).
   The 2026-07-07 scan already found rigorous ICT-concept statistics
   don't exist; Edgeful doesn't change that.
2. **Answer vs library** — Edgeful is a report library you go query; the
   Situation Room is one pushed page: "here is today's script." Average
   traders don't want 150 reports, they want to be told what the data
   says about *today*.
3. **Prop-awareness gap** — nobody, Edgeful included, translates setup
   statistics into prop math ("median adverse excursion on this setup is
   $1,030/contract — that breaches your $1k daily floor on ONE lot").
   Given 70% of failures are drawdown breaches, this is the feature with
   the clearest "pays for itself" story.
4. **Receipts gap** — the self-grading public scoreboard (feature 5)
   differentiates from both Edgeful (no forward-graded daily call) and
   the $99–299 Discord gurus (no verifiable track record at all).

**Risk, stated plainly:** Edgeful could add sweep reports any quarter.
The durable moat is therefore NOT the stat definitions — it's the
compounding forward-tested record, the prop-risk fusion, the daily-script
UX, and distribution. Move accordingly: publish the graded record early.

**Pricing anchor:** Edgeful sets the category at $49/mo. Enter at
$29–39/mo (or free page + paid alerts) — undercut with a sharper wedge
rather than compete on report count.

## Feature ladder, ranked by (live-trader utility × uniqueness)

### 1. "Today's Script" — conditional bias panel  ★ the killer feature
At London close (07:00 ET-ish), classify today's overnight pattern
(which of the 4 london_manipulation buckets) and surface the matching
historical distribution:
- "London swept Asia High → NY up 62% of days, median +147 pts (n=21)"
- Fakeout odds for each still-untested level today.
All computable NOW from `session_stats_summary.json` + today's session
levels (the AMD engine already extracts them). This single panel converts
the page from "stats site" to "daily decision tool" and is v2's centerpiece.

### 2. Live level board with per-level playbook numbers
Asia H/L, London H/L rows, each showing live status (UNTESTED / TOUCHED →
FAKEOUT or BREAK) plus, on the untested ones, the playbook if touched:
- fakeout probability (58–69% depending on level),
- **median overshoot = stop-run depth** ("sweeps run 25–64 pts past this
  level before reversing — a stop inside that band is donated"),
- median reversal MFE60 (the target-setting number).
The overshoot band is the most directly actionable stat in the whole
dataset and nobody publishes it.

### 3. First-touch alerts — the paid moment of value
Discord/Telegram push the moment NQ first touches a tracked level, with
the odds attached ("Asia High touched 09:47 · historically 58% fakeout ·
median overshoot 25 pts · median bounce +121 pts/60m"). The screenshot is
free marketing; the alert AT the touch is what a subscription buys.
Event-driven, not polled — consistent with [[feedback-ondemand-over-polling]].

### 4. Time-of-day odds strip
The `time_buckets` data, live-highlighted: a 09:45 touch and a 13:30 touch
are different trades (57.6% vs 72.7% fakeout — but late-day median MFE60
is only 11 pts: "the afternoon fade wins and still pays nothing" is
exactly the kind of insight that makes the product feel smart).

### 5. Morning-after scoreboard — self-grading = trust
Auto-compare yesterday's Script against what happened, archive every card
with timestamp. A public running record of "we said 62%, out-of-sample
it's running X%" is the credibility asset that separates this from guru
anecdotes — and it's a SECOND daily auto-post (morning script + evening
scoreboard = 2 content units/day from the same pipeline).

**BUILT 2026-07-19** (`backtest/scoreboard.py` + The Record panel in both
editions) as a walk-forward re-grader rather than a card archive: every
call is reconstructed from only the sessions before its date, so the full
out-of-sample record exists retroactively and is reproducible from the
append-only dataset (whose git history is the tamper-proof archive).
First grading, n=53: **the directional call has NO edge** — 41% hit over
34 calls, and filtering to bigger buckets or stronger claimed edge makes
it WORSE (n≥10: 37%; n≥10 & edge≥10pts: 43%) — while **the fakeout
headline holds**: claimed ~60% at touch time, ran 64% over 128
out-of-sample touches. Product consequence: the fakeout/overshoot stats
(features 2–3, USP #2) are the validated sellable core; Today's Script
directional lean stays published-but-flagged (NO EDGE YET) as anti-guru
proof until vendor data (n≈250) settles whether any conditional slice
earns an EDGE stamp. The evening auto-post upgrade is still open.

### 6. Risk/sizing calculator — prop-aware wedge combo
Generic prop-compliance is OCCUPIED (TradesViz etc.), but nobody fuses
setup statistics with prop math: median MAE120 on an Asia-High fakeout is
51.5 pts ≈ **$1,030/contract on NQ — already past a $1k daily floor on one
lot**. "Given your remaining drawdown, max size for this setup's
historical adverse excursion" is a 20-line calculation with real teeth.
Seed logic exists in `data/tradeify_account.py`.

### 7. Regime slicing (needs bigger n)
Fakeout % conditioned on gamma regime / VIX band / day-of-week.
`backtest/gex_history.json` is being banked daily for exactly this.
**Constraint:** the public product cannot ship anything GEX-Suite-derived
(user rule, 2026-07-07) — regime slices stay personal-page-only until an
independent gamma source exists (own computation from CBOE/OCC options
data — real moat, big lift, parked as Phase 5 research).

### 8. Multi-instrument (ES first)
The engine is instrument-agnostic once data is bought. Each instrument
multiplies daily content output and addressable audience. ES is the
obvious second (same session structure, biggest futures crowd).

## Concept expansion map (added 2026-07-10 — user: "there is more to
trading than London highs / Asia lows")

Correct — Asia/London is concept #1 through what is really a
**concept-quantification engine**: pin a falsifiable definition → detect
per session → measure outcomes → compound → grade publicly. Expansion
ranked by computability from existing 5m OHLCV + where stops actually
rest:

- **Tier 1 (same framework, bigger audience):** prior day H/L (THE
  stop-cluster levels — build first), previous week H/L, midnight open /
  NY open anchors, equal highs/lows (liquidity pools, definable as two
  swings within X pts).
- **Tier 2 (session behavior):** killzone stats (10–11 silver bullet,
  lunch reversal, MOC), day-of-week conditioning (which weekday forms the
  weekly H/L — classic ICT claim, never tested), prior-day-context
  conditioning (after up day / inside day / big range), FVG fill rates,
  NWOG/NDOG respect.
- **Tier 3 (uniquely ours): SMT divergence** — NQ vs ES at sweeps; the
  user trades SMT confluence personally (column in my_trades.csv), data
  is already dual-instrument, and NOBODY publishes "sweep with SMT
  diverging reverses X% vs Y% without."
- **Endgame: confluence odds** — Today's Script goes multi-factor
  ("swept Asia High + above midnight open + Monday → NY up 78%, n=14").
  This is what a trader faces at 9:28 and no one serves. GATE: needs
  n≈250 (vendor purchase) — at n=49 stacked conditions are anecdotes.

Anti-trap: this is NOT license to build 150 shallow reports (Edgeful won
that game). Fewer concepts, deeper treatment: pinned public definition,
overshoot geometry, forward grading. Sequence: PDH/PDL → SMT → confluence.

### POSITIONING CORRECTION (2026-07-10, user veto — OVERRIDES the ICT
framing above and in the demand section)

The user does NOT want an ICT product: he considers most ICT outdated,
its crowd largely unprofitable, and he never bases trades on it. He uses
only OTE, order blocks, exhaustion candles, and NWOG — and only as minor
confluences, never main bias.

- **Brand language: neutral quant, never ICT dialect.** "Overnight range
  break" not killzone/PO3; "cross-market divergence" not SMT; "stop
  cluster" not liquidity pool. The phenomena (session extremes, stop
  runs, first-touch reversals) are microstructure, not ICT — keep them,
  drop the vocabulary.
- **Positioning: the anti-guru.** "No gurus. No narratives.
  Distributions." The engine TESTS claims — including publishing "we
  tested X: no edge." Negative results are content guru-sellers can
  never publish; perfectly aligned with the self-grading USP.
- **Test queue reordered to the user's own confluences:** NWOG
  respect/fill (precise by construction), OTE-zone return odds after
  sweep-reversals, order-block hold rates (pin: last opposing candle
  before displacement), exhaustion-candle reversal odds (definition
  needs pinning). Dogfooding question it answers: do HIS confluences
  measurably add edge on NQ?
- Marketing may still TEST popular ICT claims as myth-buster content
  (engagement without endorsement) — the brand never adopts the dialect.

## What NOT to build (settled by prior market scans — don't relitigate)
- Drag-and-drop widget platform / 180-feed aggregation (SitDeck's commodity
  half; losing fight vs TradingView).
- Generic trade journal, generic prop-compliance dashboard (OCCUPIED).
- Broker integration / order routing.
- Anything branded or fed by GEX Suite in the public product.

## Phases

**Phase 0 — prerequisites (now / this month)**
- BUY THE VENDOR DATA (~$10–30, importer already built and tested).
  n=48 → n≈250 is the single highest-leverage unlock: every conditional
  slice above is noise at n=48 (the "both" bucket has ONE day) and
  respectable at n=250. Nothing sellable before this.
- Pick the brand name (SweepStats is the working name; product brand must
  be separate from personal identity per sellability criteria).
- Daily pipeline keeps compounding automatically (already live, 18:30).

**Phase 1 — v1 wrapper (blueprint #6, as spec'd, unchanged)**
Static shareable artifact. Upgrades the posting loop, becomes the brand
face. Sonnet-buildable cold. Build any time.

**Phase 2 — utility panels (personal tool, local, no server)**
Features 1, 2, 4, 5 as new panels in situation_room.py, computed from the
dataset + today's session levels. This is where "worth building
trading-wise" gets settled — the user trades with it every session before
anyone is asked to pay. Dogfooding = retention signal from user #1.

**Phase 3 — public face (still no server)**
Pipeline pushes a REDACTED page (no account P&L, no GEX panel, no
positions) to static hosting (Cloudflare/GitHub Pages) after the morning
run and the 18:30 run. Free tier = yesterday's numbers + headline stats.
Email capture. Two auto-posts/day drive traffic to it.

**Phase 4 — paid**
Discord bot delivering feature 3 (touch alerts with odds) + full
conditional dashboards + feature 6 sizing + ES. Sell via Whop/Discord
subscription first — zero auth/billing to build, and futures traders
already live in Discord. Web accounts only after revenue proves demand.
Track retention from subscriber #1.

**Phase 5 — research options**
Independent gamma computation (unlocks regime slices publicly), more
instruments, API access for the dataset.

## USP (added 2026-07-10 — user: "need stuff that makes this truly unique")

A USP is the sentence a trader repeats to a friend, not a feature list.
Features are copyable in a quarter; the moats are data only we have and
proof only time can build. Ranked candidates:

1. **"The morning call that grades itself."** Daily directional read,
   publicly scored next morning, forever. A year of forward-graded calls
   cannot be backfilled by any competitor — the time moat. Turns being
   unknown into an asset: don't ask for trust, show the scoreboard.
2. **"We tell you where your stop dies."** The overshoot distribution
   (stops run ≈25–64 pts past the level before reversing) = placement
   geometry, not outcome probability. Emotionally resonant, unavailable
   anywhere, screenshot-native.
3. **"Odds your account can survive."** Stats × prop drawdown math —
   personal, sticky, aimed at what kills 70% of the target customers.

NOT the USP: the aesthetic (distribution, copyable), the war-room format,
generic session stats (Edgeful's turf), breadth of instruments/reports
(the incumbent's game).

How to find it (not from a chair): (a) dogfood two weeks — the number the
user's eyes go to at 9:28am IS the product; (b) the daily posting loop is
a USP detector — the stat that gets quoted/argued-with in replies wins.
Weekend questions for the user: which of the three would HE pay $39/mo
for; the one-sentence pitch without the word "dashboard"; the name.

## Tiering (added 2026-07-10, after the v1 draft was built)

The GEX constraint is solved by architecture, not by dropping features —
`situation_room.py` now renders two editions from one codebase:

- **Base product (`--public`)**: price-derived data only. The GEX ladder
  is replaced by the **Session Liquidity Ladder** — Asia/London H/L +
  prev close, each annotated with its SweepStats odds (fakeout %,
  stop-run depth, bounce). Nothing on the page requires a licensed feed
  beyond standard price data. This is the sellable core, and the ladder
  is arguably STRONGER here: universal levels every trader watches,
  wearing proprietary odds nobody else has.
- **Premium / BYO-GEX tier (user idea, 2026-07-10)**: subscribers who
  already pay for GEX Suite connect their own feed and the GEX ladder
  panel lights up for them privately. The product never redistributes
  GEX data — each user displays what they personally license (same model
  as the repo reading the user's own chart). CAUTION before shipping
  this tier: the user's standing rule is "cannot link the product to GEX
  Suite" — an official "works with GEX Suite" integration needs the
  mentor's blessing; an unofficial one must stay generic ("import your
  own levels JSON/CSV"), which also opens it to SpotGamma/MenthorQ
  subscribers and avoids naming anyone.

DECISION: the paid base tier must be complete without any BYO data —
premium integrations are additive, never load-bearing.

## Production data architecture (added 2026-07-10, user question: "how
would you get the data when I can't be the host?")

The chart screen-reading exists ONLY because GEX Suite and the Tradovate
panel have no API — the sellable product never touches a chart. Every
public-facing number is computed from raw 5m OHLCV bars by
`session_stats.py`, which already runs unattended daily. Hosted shape:

- **Data vendor:** Databento (CME Globex, NQ) — historical API to backfill
  the dataset (same purchase already planned) + live/intraday feed for the
  alert bot. Tens of $/mo at small scale, usage-based.
- **Pipeline server:** $5–10/mo VPS or scheduled cloud function running the
  exact current pipeline (pull bars → update dataset → recompute stats →
  render site → push to static hosting). Replaces the Windows 18:30 task.
- **Alert bot:** the only real-time consumer — watches live price vs
  today's levels, fires Discord pings on first touch (the paid tier).
- **User's machine:** personal cockpit only (GEX ladder, Tradovate reads);
  never in the product path.
- **Licensing note:** published statistics are *derived data* (yours,
  free/cheap) vs *redistributing market data* (live price display —
  exchange fees). Keep the public page stats-and-levels, no streaming
  tape, until revenue justifies display licensing.
- **IBKR's role (user asked 2026-07-10 — cheap API via existing login):**
  bridge, not production. Personal-use data license can't power a paid
  service (and voids non-pro status); CP Gateway needs daily interactive
  2FA login (unattendable); CP API serves no expired contracts (proven
  2026-07-07), so Databento backfill is needed regardless. USE IBKR for:
  personal cockpit, dataset cross-validation (HMDS matched yfinance 4/5
  sessions), and the free-content period. RULE: first paying alert
  subscriber = feed moves to Databento that day.

## DECISIONS made in this doc
- DECISION: v1 blueprint stays as-is; conditional features are Phase 2
  panels added to the same `situation_room.py`, not a rewrite.
- DECISION: monetize via Discord/Whop before building accounts.
- DECISION: vendor-data purchase is a hard gate before any public claim —
  headline stats must survive n≈250 before they're marketed.
- DECISION: GEX-derived anything stays off the public product until an
  independent gamma source exists.

## Open (user to settle, no rush)
- Brand name.
- Whether Phase 1+2 get built during the away weekend or after Sunday.
- Pricing shape (flat monthly vs alert-tier split) — decide after Phase 2
  dogfooding, with real usage feel.
