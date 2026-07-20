/* _middleware.js — the paywall. Runs on every request routed to the
 * worker (_routes.json includes /room/* and /api/*). Members-only paths
 * require a valid signed session cookie; everything else passes through
 * to static assets or the API routes. */

{{LIB}}

export async function onRequest(context) {
  const { request, env, next } = context;
  const url = new URL(request.url);

  if (url.pathname === "/room" || url.pathname.startsWith("/room/")) {
    const session = await readSession(env, request);
    if (!session) {
      return Response.redirect(
        `${url.origin}/login/?next=${encodeURIComponent(url.pathname)}`, 302);
    }
    // membership can lapse mid-session — re-check on each page view
    if (!(await isMember(env, session.email))) {
      return Response.redirect(`${url.origin}/login/?expired=1`, 302);
    }
  }
  return next();
}
