# GO-LIVE — taking the Situation Room public

Everything below the checklist is ALREADY BUILT and locally tested
(2026-07-20, full auth/paywall/webhook flow green under
`wrangler pages dev`). Going live is configuration, not code.

## What exists

- `tools/build_site.py` — builds the whole site into `publish/`:
  landing (live stats + pricing), `/login/`, FREE `/sample/`
  (yesterday's page), members-only `/room/` (today's page), terms,
  privacy, `og.png`, `_routes.json`, and the edge functions.
  A 22-marker redaction gate runs on EVERY page; a hit aborts the build.
- `site/functions_src/` — Cloudflare Pages Functions (source of truth;
  the build inlines `_shared.js` into each route):
  - `_middleware.js` gates `/room/*` (signed session cookie + live
    membership re-check)
  - `api/login` (6-digit emailed code), `api/verify`, `api/logout`,
    `api/me`
  - `api/whop-webhook` (HMAC-verified; `went_valid` grants,
    `went_invalid` revokes)
  - `api/admin` (Bearer ADMIN_TOKEN; add/remove/list members — comp
    the mentor here)
- `tools/publish_public.py` — build + gate + push to the page repo in
  `publish_config.json` (remote `null` = dry run). Runs at the end of
  the 18:30 pipeline already.

## Local dev / regression test (no accounts needed)

```
python -m tools.build_site
cp site/dev/.dev.vars publish/.dev.vars
cd publish && npx wrangler pages dev . --kv MEMBERS --port 8788
```
DEV_MODE=1 returns login codes in the API response instead of emailing.
Flow: POST /api/admin (add member) → /api/login → /api/verify →
GET /room/ 200. Without a cookie /room/ 302s to /login/.

## The switch-flip, in order (~1 evening)

1. **Name the brand** (blocks everything cosmetic): update the `site`
   block in `publish_config.json` (brand, tagline, contact_email) and
   the brand strings in `situation_room.py`'s masthead.
2. **Domain**: buy it at Cloudflare Registrar (at-cost). Neutral name,
   not tied to personal identity (sellability criterion).
3. **Cloudflare Pages project** (free tier is plenty):
   `npx wrangler login` → create project → create the KV namespace and
   bind it as `MEMBERS` → attach the custom domain.
4. **Secrets** (Pages → Settings → Environment variables; never in git):
   - `SESSION_SECRET` — long random string (`openssl rand -hex 32`)
   - `ADMIN_TOKEN` — long random string
   - `RESEND_API_KEY` + `EMAIL_FROM` — resend.com free tier (100
     emails/day), verify the sending domain
   - `WHOP_WEBHOOK_SECRET`, `WHOP_API_KEY` — from step 5
   - Do NOT set `DEV_MODE` in production.
5. **Whop**: create the product at the chosen price → checkout link
   into `publish_config.json` `site.checkout_url` (the landing page's
   OPENING SOON buttons become JOIN buttons automatically) → add a
   webhook pointed at `https://<domain>/api/whop-webhook` for
   membership valid/invalid events. **Verify against current Whop docs:**
   the exact signature header name/scheme (the function accepts
   `x-whop-signature` / `whop-signature`, hex HMAC-SHA256 of the raw
   body) and the v2/v5 memberships-by-email endpoint used by the lazy
   fallback (`WHOP_MEMBERS_ENDPOINT` env overrides the default).
6. **Deploy**: either direct (`cd publish && npx wrangler pages deploy .`)
   or set `publish_config.json` `remote` to a page repo wired to
   Cloudflare Pages git integration — then the existing 18:30 pipeline
   publishes automatically every evening. For a morning publish too,
   run `python -m tools.publish_public` at the end of the session-start
   routine.
7. **Smoke test in production**: buy a $1 test product (or comp
   yourself via `/api/admin`), log in with the emailed code, check
   `/room/`, cancel, confirm the webhook revokes access.
8. **Point of no return**: announce. The Record panel does the talking.

## Standing rules

- The personal edition (GEX ladder, The Floor, The Ledger) NEVER ships;
  only `build_site.py` output reaches `publish/`, and the redaction
  gate is the enforcement. Loosen `FORBIDDEN` only in a reviewed commit.
- Real secrets exist only in the Cloudflare dashboard. `.dev.vars`
  holds dev placeholders only.
- First paying subscriber = the data feed moves to Databento that day
  (standing decision, roadmap §data architecture).
