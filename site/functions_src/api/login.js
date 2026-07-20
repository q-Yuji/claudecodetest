/* POST /api/login  {email}
 * Members get a 6-digit code (10-minute TTL) by email; in DEV_MODE the
 * code is returned in the response instead of sent. Non-members get a
 * pointer at the checkout. Rate-limited per email and per IP. */

{{LIB}}

const CODE_TTL = 600; // seconds

async function sendCodeEmail(env, email, code) {
  if (!env.RESEND_API_KEY) return false;
  const r = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: env.EMAIL_FROM || "Situation Room <login@example.com>",
      to: [email],
      subject: `${code} is your Situation Room login code`,
      text: `Your login code is ${code}\n\nIt expires in 10 minutes. ` +
            "If you didn't request this, ignore this email.",
    }),
  });
  return r.ok;
}

export async function onRequestPost({ request, env }) {
  const body = await readJson(request);
  const email = normEmail(body && body.email);
  if (!email) return json({ error: "valid email required" }, 400);

  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  if (await bumpCounter(env, `rl:login:ip:${ip}`, 3600) > 20 ||
      await bumpCounter(env, `rl:login:em:${email}`, 3600) > 6) {
    return json({ error: "too many requests — try again later" }, 429);
  }

  if (!(await isMember(env, email))) {
    return json({
      error: "no active membership for this email",
      checkout_url: env.WHOP_CHECKOUT_URL || null,
    }, 403);
  }

  const code = String(Math.floor(100000 + Math.random() * 900000));
  await env.MEMBERS.put(`code:${email}`, await sha256Hex(code),
                        { expirationTtl: CODE_TTL });
  await env.MEMBERS.delete(`codetries:${email}`);

  if (env.DEV_MODE === "1") {
    return json({ ok: true, dev_code: code,
                  note: "DEV_MODE — code returned instead of emailed" });
  }
  if (!(await sendCodeEmail(env, email, code))) {
    return json({ error: "could not send the login email — try again" }, 502);
  }
  return json({ ok: true, sent: true });
}
