/* GET /api/me — session + membership status for the front-end. */

{{LIB}}

export async function onRequestGet({ request, env }) {
  const session = await readSession(env, request);
  if (!session) return json({ logged_in: false });
  return json({
    logged_in: true,
    email: session.email,
    member: await isMember(env, session.email),
  });
}
