/* POST /api/whop-webhook — Whop membership lifecycle -> KV member store.
 *
 * membership.went_valid   -> member active
 * membership.went_invalid -> member revoked
 *
 * Signature: HMAC-SHA256 of the raw body with WHOP_WEBHOOK_SECRET,
 * compared against the signature header. Header name and exact scheme
 * MUST be re-verified against current Whop docs at go-live (see
 * GO-LIVE.md) — both plausible header names are checked here. Without
 * the secret configured, the hook only works in DEV_MODE; in production
 * an unsigned request is always rejected.
 *
 * Every accepted event is journaled to KV (whoplog:<ts>) for debugging.
 */

{{LIB}}

function findEmail(obj, depth = 0) {
  if (!obj || typeof obj !== "object" || depth > 4) return null;
  if (typeof obj.email === "string") return normEmail(obj.email);
  for (const v of Object.values(obj)) {
    const hit = findEmail(v, depth + 1);
    if (hit) return hit;
  }
  return null;
}

export async function onRequestPost({ request, env }) {
  const raw = await request.text();

  const sig = request.headers.get("x-whop-signature") ||
              request.headers.get("whop-signature") || "";
  if (env.WHOP_WEBHOOK_SECRET) {
    const expect = await hmacHex(env.WHOP_WEBHOOK_SECRET, raw);
    const given = sig.replace(/^sha256=/, "").toLowerCase();
    if (!safeEqual(given, expect)) {
      return json({ error: "bad signature" }, 401);
    }
  } else if (env.DEV_MODE !== "1") {
    return json({ error: "webhook secret not configured" }, 503);
  }

  let event;
  try {
    event = JSON.parse(raw);
  } catch {
    return json({ error: "invalid JSON" }, 400);
  }

  const action = String(event.action || event.event || event.type || "");
  const email = findEmail(event.data || event);
  if (!email) return json({ error: "no email in payload" }, 422);

  if (/went_valid|membership[._]created|payment[._]succeeded/i.test(action)) {
    await putMember(env, email, { source: "whop-webhook", whop_action: action });
  } else if (/went_invalid|membership[._](deleted|cancelled|canceled|expired)/i.test(action)) {
    await revokeMember(env, email, action);
  } else {
    return json({ ok: true, ignored: action });
  }

  await env.MEMBERS.put(`whoplog:${Date.now()}`,
                        JSON.stringify({ action, email }),
                        { expirationTtl: 30 * 86400 });
  return json({ ok: true, action, email });
}
