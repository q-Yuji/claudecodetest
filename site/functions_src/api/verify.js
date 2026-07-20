/* POST /api/verify  {email, code}
 * Exchanges a valid login code for a 30-day signed session cookie.
 * Five attempts per code, then it burns. */

{{LIB}}

export async function onRequestPost({ request, env }) {
  const body = await readJson(request);
  const email = normEmail(body && body.email);
  const code = String((body && body.code) || "").trim();
  if (!email || !/^\d{6}$/.test(code)) {
    return json({ error: "email and 6-digit code required" }, 400);
  }

  const tries = await bumpCounter(env, `codetries:${email}`, 600);
  if (tries > 5) {
    await env.MEMBERS.delete(`code:${email}`);
    return json({ error: "too many attempts — request a new code" }, 429);
  }

  const expected = await env.MEMBERS.get(`code:${email}`);
  if (!expected || !safeEqual(await sha256Hex(code), expected)) {
    return json({ error: "wrong or expired code" }, 401);
  }
  await env.MEMBERS.delete(`code:${email}`);
  await env.MEMBERS.delete(`codetries:${email}`);

  const token = await makeSession(env, email);
  return json({ ok: true, email },
              200, { "Set-Cookie": sessionSetCookie(token) });
}
