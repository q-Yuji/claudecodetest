/* _shared.js — common helpers for all Situation Room edge functions.
 *
 * NOT deployed as its own file: tools/build_site.py inlines this source
 * into each function at the `{{LIB}}` marker (build-time bundling), so
 * the functions/ directory contains only route files and there is no
 * ambiguity about shared modules becoming accidental routes.
 *
 * Env bindings/secrets (set in Cloudflare dashboard at go-live; dev
 * placeholders live in site/dev/.dev.vars):
 *   MEMBERS             KV namespace — members, login codes, rate limits
 *   SESSION_SECRET      HMAC key for session cookies (long random string)
 *   ADMIN_TOKEN         bearer token for /api/admin
 *   DEV_MODE            "1" = login codes returned in API response, no email
 *   RESEND_API_KEY      transactional email (go-live)
 *   EMAIL_FROM          e.g. "Situation Room <login@example.com>"
 *   WHOP_WEBHOOK_SECRET HMAC secret for /api/whop-webhook
 *   WHOP_API_KEY        lazy membership lookup fallback (go-live)
 *   WHOP_CHECKOUT_URL   where non-members are sent to buy
 */

const enc = new TextEncoder();

function b64url(bytes) {
  let s = btoa(String.fromCharCode(...new Uint8Array(bytes)));
  return s.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s) {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  return atob(s);
}

async function hmacHex(secret, msg) {
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" },
    false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(msg));
  return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function sha256Hex(msg) {
  const d = await crypto.subtle.digest("SHA-256", enc.encode(msg));
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, "0")).join("");
}

/* constant-time-ish compare (both sides are fixed-length hex we computed) */
function safeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length)
    return false;
  let out = 0;
  for (let i = 0; i < a.length; i++) out |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return out === 0;
}

function normEmail(raw) {
  const e = String(raw || "").trim().toLowerCase();
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e) && e.length <= 254 ? e : null;
}

/* ---- sessions: stateless signed cookie  v1.<email_b64url>.<exp>.<sig> ---- */

const SESSION_COOKIE = "sr_session";
const SESSION_DAYS = 30;

async function makeSession(env, email) {
  const exp = Math.floor(Date.now() / 1000) + SESSION_DAYS * 86400;
  const body = `${b64url(enc.encode(email))}.${exp}`;
  const sig = await hmacHex(env.SESSION_SECRET, body);
  return `v1.${body}.${sig}`;
}

async function readSession(env, request) {
  const cookie = request.headers.get("Cookie") || "";
  const m = cookie.match(new RegExp(`${SESSION_COOKIE}=([^;]+)`));
  if (!m) return null;
  const parts = m[1].split(".");
  if (parts.length !== 4 || parts[0] !== "v1") return null;
  const [, emailB64, expStr, sig] = parts;
  const expect = await hmacHex(env.SESSION_SECRET, `${emailB64}.${expStr}`);
  if (!safeEqual(sig, expect)) return null;
  if (parseInt(expStr, 10) < Date.now() / 1000) return null;
  try {
    return { email: b64urlDecode(emailB64) };
  } catch {
    return null;
  }
}

function sessionSetCookie(token) {
  return `${SESSION_COOKIE}=${token}; Max-Age=${SESSION_DAYS * 86400}; ` +
         "Path=/; HttpOnly; Secure; SameSite=Lax";
}

function sessionClearCookie() {
  return `${SESSION_COOKIE}=; Max-Age=0; Path=/; HttpOnly; Secure; SameSite=Lax`;
}

/* ---- membership store (KV) ---- */

async function getMember(env, email) {
  const raw = await env.MEMBERS.get(`member:${email}`);
  if (!raw) return null;
  try {
    const m = JSON.parse(raw);
    return m && m.status === "active" ? m : null;
  } catch {
    return null;
  }
}

async function putMember(env, email, fields) {
  const rec = { status: "active", since: new Date().toISOString(),
                source: "manual", ...fields };
  await env.MEMBERS.put(`member:${email}`, JSON.stringify(rec));
  return rec;
}

async function revokeMember(env, email, reason) {
  const raw = await env.MEMBERS.get(`member:${email}`);
  let rec = {};
  try { rec = raw ? JSON.parse(raw) : {}; } catch { /* rebuild */ }
  rec.status = "revoked";
  rec.revoked_at = new Date().toISOString();
  rec.revoked_reason = reason || "unspecified";
  await env.MEMBERS.put(`member:${email}`, JSON.stringify(rec));
}

/* Lazy fallback: if the KV has no record, ask Whop directly (covers a
 * missed webhook). Endpoint shape verified against Whop docs at go-live;
 * absent WHOP_API_KEY this simply reports "not a member". */
async function whopHasValidMembership(env, email) {
  if (!env.WHOP_API_KEY) return false;
  const base = env.WHOP_MEMBERS_ENDPOINT ||
    "https://api.whop.com/api/v2/memberships?valid=true&email=";
  try {
    const r = await fetch(base + encodeURIComponent(email), {
      headers: { Authorization: `Bearer ${env.WHOP_API_KEY}` },
    });
    if (!r.ok) return false;
    const data = await r.json();
    const list = data && (data.data || data.memberships || []);
    return Array.isArray(list) && list.length > 0;
  } catch {
    return false;
  }
}

async function isMember(env, email) {
  if (await getMember(env, email)) return true;
  if (await whopHasValidMembership(env, email)) {
    await putMember(env, email, { source: "whop-lazy" });
    return true;
  }
  return false;
}

/* ---- small helpers ---- */

function json(obj, status = 200, headers = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...headers },
  });
}

async function readJson(request) {
  try {
    return await request.json();
  } catch {
    return null;
  }
}

/* KV-backed counter with TTL — crude but effective rate limiting */
async function bumpCounter(env, key, ttlSeconds) {
  const raw = await env.MEMBERS.get(key);
  const n = (raw ? parseInt(raw, 10) : 0) + 1;
  await env.MEMBERS.put(key, String(n), { expirationTtl: ttlSeconds });
  return n;
}
