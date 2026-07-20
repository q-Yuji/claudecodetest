/* /api/admin — manual membership management (comp the mentor, remove a
 * refund, inspect state). Guarded by the ADMIN_TOKEN secret:
 *   Authorization: Bearer <ADMIN_TOKEN>
 *
 * GET             -> list members
 * POST {action:"add"|"remove", email}
 */

{{LIB}}

function authed(request, env) {
  const h = request.headers.get("Authorization") || "";
  return env.ADMIN_TOKEN && safeEqual(h, `Bearer ${env.ADMIN_TOKEN}`);
}

export async function onRequestGet({ request, env }) {
  if (!authed(request, env)) return json({ error: "unauthorized" }, 401);
  const out = [];
  let cursor;
  do {
    const page = await env.MEMBERS.list({ prefix: "member:", cursor });
    for (const k of page.keys) {
      const raw = await env.MEMBERS.get(k.name);
      try {
        out.push({ email: k.name.slice(7), ...JSON.parse(raw) });
      } catch {
        out.push({ email: k.name.slice(7), status: "corrupt" });
      }
    }
    cursor = page.list_complete ? null : page.cursor;
  } while (cursor);
  return json({ members: out });
}

export async function onRequestPost({ request, env }) {
  if (!authed(request, env)) return json({ error: "unauthorized" }, 401);
  const body = await readJson(request);
  const email = normEmail(body && body.email);
  const action = body && body.action;
  if (!email || !["add", "remove"].includes(action)) {
    return json({ error: "need {action: add|remove, email}" }, 400);
  }
  if (action === "add") {
    return json({ ok: true, member: await putMember(env, email, { source: "admin" }) });
  }
  await revokeMember(env, email, "admin");
  return json({ ok: true, removed: email });
}
