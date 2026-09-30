"""
Wren Google Sign-In
====================
Verifies Google identity server-side and hands the app back a verified
email plus a signed proof-of-sign-in token, using the same "app opens a
browser, polls a status endpoint" pattern used for Paystack checkout in
premium.py.

Flow:
    1. App calls GET /auth/google/start. We mint a random session_id AND
       a short numeric confirmation code, store both in Postgres, and
       return the Google URL (state=session_id) plus the code. The app
       opens the browser and shows the code on screen.
    2. User signs in with Google. Google redirects the browser to
       GET /auth/google/callback?code=...&state=session_id on THIS
       backend.
    3. We only accept a callback for a session that /start really
       created and that hasn't been used yet. We exchange the code for
       tokens (server-side - the Client Secret never touches the app),
       verify the ID token's signature against Google's public keys, and
       store the verified email on the session (NOT yet released).
    4. The browser page asks the user to type the confirmation code
       that is showing in the app. Only when it matches is the session
       marked confirmed. (This is what stops "sign-in link phishing":
       a victim who was tricked into completing someone else's sign-in
       has no code on their own screen to type. See "Phishing note".)
    5. The app polls GET /auth/google/status/{session_id}. Once the
       session is confirmed it gets {done, email, token} exactly once
       and the row is deleted.

Phishing note
-------------
The attack: attacker starts a session, sends the victim the Google link,
victim signs in, attacker's poll collects the victim's email + token.
Step 4 breaks that for anyone who isn't being actively social-engineered
into reading out a code. It cannot stop a victim who is told "type this
code" by the attacker - the only complete fix for that is returning the
result to the app itself with an Android App Link / custom-scheme
redirect instead of polling. That is an app-manifest change, so it is
left as a follow-up. The code step is OFF by default (it made sign-in feel
like too much); set AUTH_REQUIRE_CODE=1 to switch it on.

Sign-in proof token
-------------------
Typing an email proves nothing, and an email address is not a secret.
So when Google verifies an email we also hand the app a signed token for
that exact email. Every route that reads or changes something keyed by
an email (chats, premium purchase/restore/reset) must present it via
require_user(). Stateless (HMAC), so no DB table, and it works across
workers/restarts.

Signed with AUTH_TOKEN_SECRET; falls back to GOOGLE_CLIENT_SECRET so it
works with no new config. Set AUTH_TOKEN_SECRET (any long random string)
on Render to decouple it - changing it signs everyone out, which only
means they sign in with Google again.

Storage: Postgres (DATABASE_URL), shared by every worker/instance, so
/start, /callback, /confirm and /status can land on different processes.
Expired rows are deleted on every call, so the table never grows
unbounded.

Env vars (Render dashboard):
    GOOGLE_CLIENT_ID       - from the Web application OAuth client
    GOOGLE_CLIENT_SECRET   - from the same client
    GOOGLE_REDIRECT_URI    - must exactly match an Authorized redirect URI,
                             e.g. https://wrenai-application.onrender.com/auth/google/callback
    AUTH_TOKEN_SECRET      - (recommended) long random string for signing tokens
    AUTH_REQUIRE_TOKEN     - default on. "0" = stop enforcing tokens (escape hatch only)
    AUTH_REQUIRE_CODE      - default OFF. "1" = add the extra confirmation-code step
"""

import os
import time
import hmac
import html
import hashlib
import asyncio
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

# Default ON. "0" is a temporary escape hatch, same convention as
# RESTORE_REQUIRE_TOKEN in premium.py.
REQUIRE_TOKEN = os.environ.get("AUTH_REQUIRE_TOKEN", "1") != "0"
REQUIRE_CODE = os.environ.get("AUTH_REQUIRE_CODE", "0") == "1"

_EXPLICIT_TOKEN_SECRET = os.environ.get("AUTH_TOKEN_SECRET", "").strip()
AUTH_TOKEN_SECRET = _EXPLICIT_TOKEN_SECRET or GOOGLE_CLIENT_SECRET
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
        iat_s, sig = token.strip().split(".", 1)
        iat = int(iat_s)
    except (ValueError, AttributeError):
        return False
    age = time.time() - iat
    if age < -60 or age > TOKEN_TTL_SECONDS:
        return False
    # Compare as bytes: hmac.compare_digest raises TypeError on a str that
    # contains non-ASCII characters, and the token comes straight from an
    # HTTP header the caller controls - that would be a 500 instead of a 401.
    return hmac.compare_digest(sig.encode("utf-8", "ignore"),
                               _sign(email, iat).encode("utf-8"))


def require_user(email: str, token: Optional[str], rid: str = "-"):
    """Gate for every route keyed by an email address. Raises 401 unless
    `token` is a valid, unexpired sign-in token issued for exactly this
    email. Knowing someone's address is not enough to act as them."""
    if not REQUIRE_TOKEN:
        return
    if not verify_token(email, token):
        log.warning(f"[{rid}] auth: missing/invalid sign-in token for email="
                    f"{(email or '').strip().lower()[:3]}***")
        raise HTTPException(
            status_code=401,
            detail={"error": "please sign in again to continue", "request_id": rid},
        )


# How long an unfinished login session is kept. The app polls for about
# 3 minutes, so 5 leaves slack without leaving stale sessions lying around.
SESSION_TTL_SECONDS = 5 * 60
# Wrong confirmation-code tries before the session is thrown away.
CODE_MAX_ATTEMPTS = 5
# Hard cap on pending sessions so /start can't be used to fill the table.
MAX_PENDING_SESSIONS = 2000

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

_pool = None

_google_request = google_requests.Request()

_TABLE = "auth_login_sessions"


async def init_db():
    global _pool
    if not AUTH_TOKEN_SECRET:
        log.error("[auth] neither AUTH_TOKEN_SECRET nor GOOGLE_CLIENT_SECRET is set - "
                  "sign-in tokens cannot be issued or verified")
    elif not _EXPLICIT_TOKEN_SECRET:
        log.warning("[auth] AUTH_TOKEN_SECRET not set - signing tokens with "
                    "GOOGLE_CLIENT_SECRET. Set AUTH_TOKEN_SECRET on Render to a "
                    "separate long random string.")
    if not REQUIRE_TOKEN:
        log.warning("[auth] AUTH_REQUIRE_TOKEN=0 - email-keyed routes are NOT "
                    "checking sign-in tokens")
    if not DATABASE_URL:
        log.error("[auth] DATABASE_URL not set - google sign-in endpoints will fail")
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
            "session_id TEXT PRIMARY KEY, "
            "code TEXT NOT NULL, "
            "email TEXT, "
            "confirmed BOOLEAN NOT NULL DEFAULT FALSE, "
            "attempts INTEGER NOT NULL DEFAULT 0, "
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
    await conn.execute(f"DELETE FROM {_TABLE} WHERE created_at < $1", cutoff)


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
    # Shown in the app; the user must type it into the browser page.
    # None when AUTH_REQUIRE_CODE=0.
    user_code: Optional[str] = None


class AuthStatusResponse(BaseModel):
    done: bool
    email: Optional[str] = None
    token: Optional[str] = None


async def auth_google_start(rid: str) -> AuthStartResponse:
    """Called by the app before opening the browser. Registers a fresh
    session (so the callback can later be checked against it) and builds
    the Google consent-screen URL around it."""
    _require_config(rid)
    _require_pool(rid)

    session_id = secrets.token_urlsafe(24)
    code = f"{secrets.randbelow(10000):04d}" if REQUIRE_CODE else ""
    now = dt.datetime.now(dt.timezone.utc)

    async with _pool.acquire() as conn:
        await _purge_expired(conn)
        pending = await conn.fetchval(f"SELECT COUNT(*) FROM {_TABLE}")
        if pending is not None and pending >= MAX_PENDING_SESSIONS:
            log.warning(f"[{rid}] /auth/google/start: {pending} pending sessions - refusing")
            raise HTTPException(
                status_code=429,
                detail={"error": "too many sign-ins in progress, try again in a minute",
                        "request_id": rid})
        await conn.execute(
            f"INSERT INTO {_TABLE} (session_id, code, created_at) VALUES ($1, $2, $3)",
            session_id, code, now)

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email",
        "state": session_id,
        # Forces the account chooser rather than silently reusing
        # whatever Google session is already active in the browser.
        "prompt": "select_account",
    }
    auth_url = f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}"

    log.info(f"[{rid}] /auth/google/start: issued session_id={session_id[:8]}...")
    return AuthStartResponse(authorization_url=auth_url, session_id=session_id,
                             user_code=code or None)


async def auth_google_callback(code: str, state: str, rid: str,
                               error: str = "") -> HTMLResponse:
    """Google redirects here after the user signs in. Exchanges the code
    for tokens, verifies the ID token signature, and attaches the
    verified email to the session that /start created."""
    _require_config(rid)
    _require_pool(rid)

    session_id = (state or "").strip()
    if not session_id:
        log.warning(f"[{rid}] /auth/google/callback: missing state param")
        return _result_page("Something went wrong - missing session. "
                            "Please return to the app and try again.", ok=False)

    if error or not code:
        # User pressed Cancel / denied consent on Google's screen.
        log.info(f"[{rid}] /auth/google/callback: cancelled or denied (error={error!r})")
        async with _pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {_TABLE} WHERE session_id = $1", session_id)
        return _result_page("Sign-in was cancelled. You can close this tab and "
                            "try again from the app.", ok=False)

    # Only a session that /start really created, that is still fresh and
    # that hasn't already received an email, may complete. Without this any
    # state string was accepted and a callback could be replayed.
    async with _pool.acquire() as conn:
        await _purge_expired(conn)
        row = await conn.fetchrow(
            f"SELECT email FROM {_TABLE} WHERE session_id = $1", session_id)
    if not row or row["email"] is not None:
        log.warning(f"[{rid}] /auth/google/callback: unknown/expired/used "
                    f"session_id={session_id[:8]}...")
        return _result_page("This sign-in link has expired or was already used. "
                            "Please return to the app and start again.", ok=False)

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

    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200 or "id_token" not in body:
        log.error(f"[{rid}] /auth/google/callback: token exchange failed "
                  f"{resp.status_code}: {str(body)[:300]}")
        return _result_page("Sign-in failed. Please return to the app and try again.", ok=False)

    try:
        # verify_oauth2_token does blocking network I/O the first time (it
        # downloads Google's signing certs). Run it off the event loop so a
        # sign-in can never stall every other request on this worker.
        claims = await asyncio.to_thread(
            google_id_token.verify_oauth2_token,
            body["id_token"], _google_request, GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        # Signature invalid, expired, wrong audience, etc. - never
        # trust an ID token we haven't verified against Google's keys.
        log.error(f"[{rid}] /auth/google/callback: ID token verification failed: {e}")
        return _result_page("Sign-in couldn't be verified. Please try again.", ok=False)

    if not claims.get("email_verified", False):
        log.warning(f"[{rid}] /auth/google/callback: email not verified on Google's side")
        return _result_page(
            "That Google account's email isn't verified. Please verify it with "
            "Google first, then try again.", ok=False,
        )

    email = (claims.get("email") or "").strip().lower()
    if not email:
        log.error(f"[{rid}] /auth/google/callback: verified token had no email claim")
        return _result_page("Sign-in failed. Please return to the app and try again.", ok=False)

    async with _pool.acquire() as conn:
        # "AND email IS NULL" makes attaching the email atomic: two racing
        # callbacks for one session can't both win.
        result = await conn.execute(
            f"UPDATE {_TABLE} SET email = $2, confirmed = $3 "
            "WHERE session_id = $1 AND email IS NULL",
            session_id, email, not REQUIRE_CODE)
    if not str(result).endswith(" 1"):
        log.warning(f"[{rid}] /auth/google/callback: session_id={session_id[:8]}... "
                    f"lost the race / expired before email could be attached")
        return _result_page("This sign-in link has expired or was already used. "
                            "Please return to the app and start again.", ok=False)

    log.info(f"[{rid}] /auth/google/callback: session_id={session_id[:8]}... "
             f"google-verified, awaiting code={'yes' if REQUIRE_CODE else 'no'}")

    if not REQUIRE_CODE:
        return _result_page("You're signed in. You can close this tab and return to Wren.", ok=True)
    return _code_page(session_id, email)


async def auth_google_confirm(state: str, code: str, rid: str) -> HTMLResponse:
    """The user typed the code that the app is showing. If it matches, the
    session is released to whoever holds the session_id (the app)."""
    _require_pool(rid)
    session_id = (state or "").strip()
    typed = (code or "").strip()

    async with _pool.acquire() as conn:
        await _purge_expired(conn)
        row = await conn.fetchrow(
            f"SELECT code, email, confirmed, attempts FROM {_TABLE} WHERE session_id = $1",
            session_id)
        if not row or row["email"] is None:
            return _result_page("This sign-in link has expired or was already used. "
                                "Please return to the app and start again.", ok=False)
        if row["confirmed"]:
            return _result_page("You're signed in. You can close this tab and return to Wren.", ok=True)
        if row["attempts"] >= CODE_MAX_ATTEMPTS:
            await conn.execute(f"DELETE FROM {_TABLE} WHERE session_id = $1", session_id)
            return _result_page("Too many wrong codes. Please return to the app and start again.", ok=False)

        if row["code"] and hmac.compare_digest(typed.encode("utf-8", "ignore"),
                                                row["code"].encode("utf-8")):
            await conn.execute(
                f"UPDATE {_TABLE} SET confirmed = TRUE WHERE session_id = $1", session_id)
            log.info(f"[{rid}] /auth/google/confirm: session_id={session_id[:8]}... confirmed")
            return _result_page("You're signed in. You can close this tab and return to Wren.", ok=True)

        attempts = await conn.fetchval(
            f"UPDATE {_TABLE} SET attempts = attempts + 1 WHERE session_id = $1 "
            "RETURNING attempts", session_id)
        log.warning(f"[{rid}] /auth/google/confirm: wrong code for "
                    f"session_id={session_id[:8]}... (attempt {attempts})")
        if attempts is not None and attempts >= CODE_MAX_ATTEMPTS:
            await conn.execute(f"DELETE FROM {_TABLE} WHERE session_id = $1", session_id)
            return _result_page("Too many wrong codes. Please return to the app and start again.", ok=False)
        return _code_page(session_id, row["email"], error="That code doesn't match. Check the app and try again.")


async def auth_google_status(session_id: str, rid: str) -> AuthStatusResponse:
    """Polled by the app after it opens the browser. Returns the verified
    email + signed token once (and only once) the browser step has been
    confirmed; the session is then consumed (deleted) so it can't be
    polled or reused again."""
    _require_pool(rid)
    async with _pool.acquire() as conn:
        await _purge_expired(conn)
        row = await conn.fetchrow(
            f"DELETE FROM {_TABLE} WHERE session_id = $1 AND confirmed = TRUE "
            "RETURNING email", session_id)

    if not row or not row["email"]:
        return AuthStatusResponse(done=False)

    email = row["email"]
    log.info(f"[{rid}] /auth/google/status: session_id={session_id[:8]}... collected")
    return AuthStatusResponse(done=True, email=email, token=make_token(email))


_PAGE_STYLE = ("font-family: -apple-system, sans-serif; text-align: center; "
               "padding: 48px 24px;")


def _result_page(message: str, ok: bool) -> HTMLResponse:
    color = "#16a34a" if ok else "#dc2626"
    html_doc = f"""
    <html>
      <head>
        <title>Sign in to Wren AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="{_PAGE_STYLE}">
        <p style="font-size: 15px; color: #6b7280; margin-bottom: 4px;">Wren AI</p>
        <p style="color: {color}; font-size: 18px;">{html.escape(message)}</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html_doc, status_code=200 if ok else 400)


def _code_page(session_id: str, email: str, error: str = "") -> HTMLResponse:
    err_html = (f'<p style="color:#dc2626;font-size:15px;">{html.escape(error)}</p>'
                if error else "")
    html_doc = f"""
    <html>
      <head>
        <title>Sign in to Wren AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="{_PAGE_STYLE}">
        <p style="font-size: 15px; color: #6b7280; margin-bottom: 4px;">Wren AI</p>
        <p style="font-size: 18px;">One last step</p>
        <p style="font-size: 15px;">Signing in as <b>{html.escape(email)}</b>.<br>
           Enter the 4-digit code shown in the Wren AI app.</p>
        {err_html}
        <form method="post" action="/auth/google/confirm">
          <input type="hidden" name="state" value="{html.escape(session_id)}">
          <input name="code" type="text" inputmode="numeric" pattern="[0-9]*"
                 maxlength="4" autocomplete="off" autofocus
                 style="font-size: 28px; width: 6em; text-align: center; padding: 8px;">
          <br><br>
          <button type="submit" style="font-size: 16px; padding: 10px 28px;">Continue</button>
        </form>
        <p style="font-size: 13px; color: #6b7280; margin-top: 28px;">
          Didn't just tap "Continue with Google" in the Wren AI app on your own
          device? Close this page. Never enter a code someone else gave you.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html_doc, status_code=200)
