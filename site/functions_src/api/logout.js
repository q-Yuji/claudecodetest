/* POST /api/logout — clears the session cookie. */

{{LIB}}

export async function onRequestPost() {
  return json({ ok: true }, 200, { "Set-Cookie": sessionClearCookie() });
}
