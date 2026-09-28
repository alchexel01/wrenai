"""
Wren Google Sign-In
====================
Verifies Google identity server-side and hands the app back a verified
email, using the same "app opens a browser, polls a status endpoint"
pattern already used for Paystack checkout in premium.py.

Flow:
    1. App generates a random session_id, opens the browser to
       GOOGLE_AUTH_URL with state=session_id and our redirect_uri.
    2. User signs in with Google. Google redirects the browser to
       GET /auth/google/callback?code=...&state=session_id on THIS
       backend.
    3. We exchange the code for tokens (server-side — the Client
       Secret never touches the app), verify the ID token's signature
       against Google's public keys, and pull out the verified email.
    4. We store {session_id: email} in memory with a timestamp and
       show the user a plain "you can return to the app" page.
    5. The app polls GET /auth/google/status/{session_id} until it
       sees done=true, then treats that email exactly like the email
       from the old manual-entry onboarding screen (premium.py already
       keys everything off email, so nothing downstream needs to
       change).

Storage: Postgres (DATABASE_URL env var), same pool pattern as
premium.py. This used to be an in-memory dict — fine in theory since
a login session only lives a few minutes — but an in-memory dict is
only shared within a single process. The moment Render runs more than
one worker/instance (or restarts between /start and the browser
callback), /start and /callback/status can land on different
processes that never heard of each other's sessions, so the app polls
forever and the sign-in silently never completes. Postgres fixes that
the same way it fixed premium persistence: one shared store every
process reads from, regardless of which process handled which
request. Expired/collected rows are just deleted, so the table never
grows unbounded.

Env vars required (Render dashboard, same place as the others):
    GOOGLE_CLIENT_ID       — from the Web application OAuth client
    GOOGLE_CLIENT_SECRET   — from the same client (Client secret)
    GOOGLE_REDIRECT_URI    — must exactly match an Authorized redirect
                             URI registered on that client, e.g.
                             https://wrenai-application.onrender.com/auth/google/callback
"""

import os
import time
import hmac
import hashlib
import logging
import secrets
import datetime as dt
from urllib.parse import urlencode
from typing import Optional

import httpx
import asyncpg
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

log = logging.getLogger("wren-backend")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

# ── Sign-in proof token ────────────────────────────────────────────────
# Typing an email proves nothing, and an email address is not a secret. So
# when Google verifies an email, we also hand the app a signed token for
# that exact email. Anything sensitive that is keyed by email (today:
# /premium/restore) must present it, which proves the caller really
# signed in with Google as that address. Stateless (HMAC), so no DB table,
# and it works across workers/restarts.
#
# Signed with AUTH_TOKEN_SECRET; falls back to GOOGLE_CLIENT_SECRET so it
# works with no new config. Set AUTH_TOKEN_SECRET (any long random string)
# on Render to decouple it - changing it signs everyone's token out, which
# only means they sign in with Google again before their next Restore.
AUTH_TOKEN_SECRET = (os.environ.get("AUTH_TOKEN_SECRET", "").strip()
                     or GOOGLE_CLIENT_SECRET)
TOKEN_TTL_SECONDS = 180 * 24 * 3600


def _sign(email: str, iat: int) -> str:
    return hmac.new(AUTH_TOKEN_SECRET.encode("utf-8"),
                    f"wren-auth-v1|{email}|{iat}".encode("utf-8"),
                    hashlib.sha256).hexdigest()


def make_token(email: str) -> Optional[str]:
    email = (email or "").strip().lower()
    if not AUTH_TOKEN_SECRET or not email:
        return None
    iat = int(time.time())
    return f"{iat}.{_sign(email, iat)}"


def verify_token(email: str, token: Optional[str]) -> bool:
    email = (email or "").strip().lower()
    if not AUTH_TOKEN_SECRET or not email or not token:
        return False
    try:
        iat_s, sig = token.split(".", 1)
        iat = int(iat_s)
    except (ValueError, AttributeError):
        return False
    age = time.time() - iat
    if age < -60 or age > TOKEN_TTL_SECONDS:
        return False
    return hmac.compare_digest(sig, _sign(email, iat))


# How long an unfinished login session is kept before we give up on it
# and let it be garbage-collected. Generous, since a user might sit on
# Google's account picker for a while.
SESSION_TTL_SECONDS = 10 * 60

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

_pool = None

_google_request = google_requests.Request()


async def init_db():
    global _pool
    if not DATABASE_URL:
        log.error("[auth] DATABASE_URL not set - google sign-in endpoints will fail")
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS auth_sessions ("
            "session_id TEXT PRIMARY KEY, "
            "email TEXT NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL"
            ")"
        )
    log.info("[auth] DB pool ready, table ensured")


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _require_pool(rid):
    if _pool is None:
        log.error(f"[{rid}] /auth/google: DB pool not initialized (DATABASE_URL missing?)")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing DATABASE_URL", "request_id": rid})


async def _purge_expired(conn):
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=SESSION_TTL_SECONDS)
    await conn.execute("DELETE FROM auth_sessions WHERE created_at < $1", cutoff)


def _require_config(rid):
    missing = [name for name, val in (
        ("GOOGLE_CLIENT_ID", GOOGLE_CLIENT_ID),
        ("GOOGLE_CLIENT_SECRET", GOOGLE_CLIENT_SECRET),
        ("GOOGLE_REDIRECT_URI", GOOGLE_REDIRECT_URI),
    ) if not val]
    if missing:
        log.error(f"[{rid}] /auth/google: missing env vars: {', '.join(missing)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"server missing {', '.join(missing)}", "request_id": rid},
        )


class AuthStartResponse(BaseModel):
    authorization_url: str
    session_id: str


class AuthStatusResponse(BaseModel):
    done: bool
    email: Optional[str] = None
    token: Optional[str] = None


async def auth_google_start(rid: str) -> AuthStartResponse:
    """Called by the app before opening the browser. Mints a fresh
    session_id and builds the Google consent-screen URL around it, so
    the Client ID/redirect URI live here on the backend rather than
    being hardcoded into the app (easier to rotate later)."""
    _require_config(rid)
    _require_pool(rid)
    async with _pool.acquire() as conn:
        await _purge_expired(conn)

    session_id = secrets.token_urlsafe(24)
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email",
        "state": session_id,
        # Forces the account chooser rather than silently reusing
        # whatever Google session is already active in the browser —
        # matters most on a shared/test device.
        "prompt": "select_account",
    }
    auth_url = f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}"

    log.info(f"[{rid}] /auth/google/start: issued session_id={session_id[:8]}...")
    return AuthStartResponse(authorization_url=auth_url, session_id=session_id)


async def auth_google_callback(code: str, state: str, rid: str) -> HTMLResponse:
    """Google redirects here after the user signs in. Exchanges the
    code for tokens, verifies the ID token signature, and stashes the
    verified email under the session_id (Google's 'state' param) for
    the app to pick up via polling."""
    _require_config(rid)
    _require_pool(rid)

    session_id = state
    if not session_id:
        log.warning(f"[{rid}] /auth/google/callback: missing state param")
        return _result_page("Something went wrong — missing session. "
                             "Please return to the app and try again.", ok=False)

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                GOOGLE_TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /auth/google/callback: token exchange request failed: {e!r}")
            return _result_page("Couldn't reach Google. Please return to the app and try again.", ok=False)

    body = resp.json()
    if resp.status_code != 200 or "id_token" not in body:
        log.error(f"[{rid}] /auth/google/callback: token exchange failed "
                  f"{resp.status_code}: {body}")
        return _result_page("Sign-in failed. Please return to the app and try again.", ok=False)

    try:
        claims = google_id_token.verify_oauth2_token(
            body["id_token"], _google_request, GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        # Signature invalid, expired, wrong audience, etc. — never
        # trust an ID token we haven't verified against Google's keys.
        log.error(f"[{rid}] /auth/google/callback: ID token verification failed: {e}")
        return _result_page("Sign-in couldn't be verified. Please try again.", ok=False)

    if not claims.get("email_verified", False):
        log.warning(f"[{rid}] /auth/google/callback: email not verified on Google's side "
                    f"(email={claims.get('email')})")
        return _result_page(
            "That Google account's email isn't verified. Please verify it with "
            "Google first, then try again.", ok=False,
        )

    email = (claims.get("email") or "").strip().lower()
    if not email:
        log.error(f"[{rid}] /auth/google/callback: verified token had no email claim")
        return _result_page("Sign-in failed. Please return to the app and try again.", ok=False)

    now = dt.datetime.now(dt.timezone.utc)
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO auth_sessions (session_id, email, created_at) "
            "VALUES ($1, $2, $3) "
            "ON CONFLICT (session_id) DO UPDATE SET "
            "email = EXCLUDED.email, created_at = EXCLUDED.created_at",
            session_id, email, now,
        )
    log.info(f"[{rid}] /auth/google/callback: session_id={session_id[:8]}... "
             f"verified email={email}")
    return _result_page("You're signed in. You can close this tab and return to Wren.", ok=True)


async def auth_google_status(session_id: str, rid: str) -> AuthStatusResponse:
    """Polled by the app after it opens the browser. Returns the
    verified email once the callback above has landed, then the
    session is consumed (deleted) so it can't be polled/reused again.
    Reading from Postgres here (instead of an in-process dict) is what
    makes this work no matter which worker/instance handled /start,
    /callback, or this poll — they all see the same row."""
    _require_pool(rid)
    async with _pool.acquire() as conn:
        await _purge_expired(conn)
        row = await conn.fetchrow(
            "DELETE FROM auth_sessions WHERE session_id = $1 "
            "RETURNING email", session_id)

    if not row:
        return AuthStatusResponse(done=False)

    email = row["email"]
    log.info(f"[{rid}] /auth/google/status: session_id={session_id[:8]}... "
             f"collected email={email}")
    return AuthStatusResponse(done=True, email=email, token=make_token(email))


def _result_page(message: str, ok: bool) -> HTMLResponse:
    color = "#16a34a" if ok else "#dc2626"
    html = f"""
    <html>
      <head>
        <title>Sign in to Wren AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="font-family: -apple-system, sans-serif; text-align: center; padding: 48px 24px;">
        <p style="font-size: 15px; color: #6b7280; margin-bottom: 4px;">Wren AI</p>
        <p style="color: {color}; font-size: 18px;">{message}</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200 if ok else 400)
