"""
Wren email sign-in with a 6-digit code (no passwords)
=====================================================
A second way to sign in next to Google. It ends the same way Google does:
the app receives {email, token}, where token is the signed proof from
auth.make_token(), so chats / premium / restore work unchanged.

Flow:
    1. POST /auth/email/request {email}
       We email a 6-digit code to that address.
    2. POST /auth/email/verify {email, code}
       Right code -> {email, token}. That is the proof the person can read
       that inbox, which is exactly what "signed in as this email" means.

There is no password and no account table: nothing to leak, reset or
brute-force offline. Typing someone else's address only makes THEM receive
an email; without their inbox you can't get past step 2.

Protections:
  * codes are stored as HMACs (never plaintext), single use, expire in 10 min
  * 5 wrong guesses kill the code (a fresh one must be requested)
  * one code email per address per 60 s, at most 5 per hour, plus a rough
    global hourly cap - so this can't be used to spam a victim's inbox or
    run up your mail bill
  * the reply to /request is identical no matter what, and a failed mail
    send removes the pending code so the user can just retry

Env vars (Render dashboard):
    RESEND_API_KEY   API key from resend.com (sends over HTTPS - Render's
                     free tier blocks SMTP ports)
    MAIL_FROM        e.g. "Wren AI <noreply@yourdomain.com>". The domain
                     must be verified with Resend, otherwise it can only
                     deliver to your own address.
    EMAIL_AUTH       default on. "0" turns email sign-in off entirely.
    EMAIL_MAX_PER_HOUR  rough global cap on code emails per hour (default 300)
"""

import os
import re
import hmac
import hashlib
import logging
import secrets
import datetime as dt

import httpx
from fastapi import HTTPException
from pydantic import BaseModel

import auth

log = logging.getLogger("wren-backend")

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
MAIL_FROM = os.environ.get("MAIL_FROM", "").strip()
EMAIL_AUTH_ENABLED = os.environ.get("EMAIL_AUTH", "1") != "0"
try:
    MAX_EMAILS_PER_HOUR = int(os.environ.get("EMAIL_MAX_PER_HOUR", "300"))
except ValueError:
    MAX_EMAILS_PER_HOUR = 300

CODE_TTL_SECONDS = 10 * 60
CODE_MAX_ATTEMPTS = 5
RESEND_COOLDOWN_SECONDS = 60
MAX_CODES_PER_EMAIL_PER_HOUR = 5

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")

_OK_MSG = "A 6-digit code is on its way to your email."


class EmailRequest(BaseModel):
    email: str


class EmailVerifyRequest(BaseModel):
    email: str
    code: str


class EmailOkResponse(BaseModel):
    ok: bool = True
    message: str = ""
    # Seconds the app should wait before offering "Resend code".
    retry_after: int = RESEND_COOLDOWN_SECONDS


class EmailTokenResponse(BaseModel):
    email: str
    token: str


def _err(status: int, msg: str, rid: str):
    return HTTPException(status_code=status, detail={"error": msg, "request_id": rid})


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _check_email(email, rid: str) -> str:
    e = (email or "").strip().lower() if isinstance(email, str) else ""
    if len(e) > 254 or not _EMAIL_RE.match(e):
        raise _err(400, "enter a valid email address", rid)
    return e


def _code_hash(email: str, code: str) -> str:
    return hmac.new(auth.AUTH_TOKEN_SECRET.encode("utf-8"),
                    f"wren-email-otp-v1|{email}|{code}".encode("utf-8"),
                    hashlib.sha256).hexdigest()


def _require_ready(rid: str):
    if not EMAIL_AUTH_ENABLED:
        raise _err(404, "email sign-in is turned off", rid)
    if auth._pool is None:
        log.error(f"[{rid}] /auth/email: DB pool not initialized")
        raise _err(500, "server missing DATABASE_URL", rid)
    if not auth.AUTH_TOKEN_SECRET:
        log.error(f"[{rid}] /auth/email: no token secret configured")
        raise _err(500, "server not configured for sign-in", rid)


def _require_mail(rid: str):
    if not RESEND_API_KEY or not MAIL_FROM:
        log.error(f"[{rid}] /auth/email: RESEND_API_KEY / MAIL_FROM not set - "
                  "cannot send sign-in codes")
        raise _err(503, "email sign-in isn't available right now - use Google instead", rid)


async def init_tables():
    if auth._pool is None:
        return
    async with auth._pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS auth_email_otps ("
            "email TEXT PRIMARY KEY, "
            "code_hash TEXT NOT NULL, "
            "attempts INTEGER NOT NULL DEFAULT 0, "
            "window_start TIMESTAMPTZ NOT NULL, "
            "sent_count INTEGER NOT NULL DEFAULT 1, "
            "last_sent_at TIMESTAMPTZ NOT NULL, "
            "expires_at TIMESTAMPTZ NOT NULL)")
    if not RESEND_API_KEY or not MAIL_FROM:
        log.warning("[auth-email] RESEND_API_KEY/MAIL_FROM not set - email "
                    "sign-in will answer 503 until they are")
    log.info("[auth-email] table ensured")


async def _send_mail(to: str, subject: str, text: str, rid: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={"from": MAIL_FROM, "to": [to], "subject": subject, "text": text})
    except httpx.RequestError as e:
        log.error(f"[{rid}] /auth/email: mail request failed: {e!r}")
        raise _err(502, "couldn't send the email - try again in a moment", rid)
    if resp.status_code >= 300:
        # Never log the message body: it contains the code.
        log.error(f"[{rid}] /auth/email: mail provider said {resp.status_code}: "
                  f"{resp.text[:200]}")
        raise _err(502, "couldn't send the email - try again in a moment", rid)


async def email_request(payload, rid: str) -> EmailOkResponse:
    """Step 1: email a fresh code. Inside the resend cooldown / hourly cap the
    call still answers ok (without sending again), so the reply never depends
    on what is or isn't pending for an address."""
    _require_ready(rid)
    email = _check_email(payload.email, rid)
    _require_mail(rid)

    now = _now()
    code = f"{secrets.randbelow(10 ** 6):06d}"
    async with auth._pool.acquire() as conn:
        await conn.execute("DELETE FROM auth_email_otps WHERE expires_at < $1", now)
        recent = await conn.fetchval(
            "SELECT COUNT(*) FROM auth_email_otps WHERE last_sent_at > $1",
            now - dt.timedelta(hours=1))
        if recent is not None and recent >= MAX_EMAILS_PER_HOUR:
            log.warning(f"[{rid}] /auth/email/request: global hourly email cap reached")
            raise _err(429, "too many sign-ins right now, try again later", rid)

        row = await conn.fetchrow(
            "SELECT window_start, sent_count, last_sent_at FROM auth_email_otps "
            "WHERE email = $1", email)
        window_start, sent_count = now, 1
        if row:
            if (now - row["last_sent_at"]).total_seconds() < RESEND_COOLDOWN_SECONDS:
                return EmailOkResponse(message=_OK_MSG)
            if (now - row["window_start"]) < dt.timedelta(hours=1):
                if row["sent_count"] >= MAX_CODES_PER_EMAIL_PER_HOUR:
                    log.info(f"[{rid}] /auth/email/request: hourly per-address cap hit")
                    return EmailOkResponse(message=_OK_MSG)
                window_start, sent_count = row["window_start"], row["sent_count"] + 1
        await conn.execute(
            "INSERT INTO auth_email_otps (email, code_hash, attempts, window_start, "
            "sent_count, last_sent_at, expires_at) VALUES ($1, $2, 0, $3, $4, $5, $6) "
            "ON CONFLICT (email) DO UPDATE SET "
            "code_hash = EXCLUDED.code_hash, attempts = 0, "
            "window_start = EXCLUDED.window_start, sent_count = EXCLUDED.sent_count, "
            "last_sent_at = EXCLUDED.last_sent_at, expires_at = EXCLUDED.expires_at",
            email, _code_hash(email, code), window_start, sent_count, now,
            now + dt.timedelta(seconds=CODE_TTL_SECONDS))

    try:
        await _send_mail(
            email, f"Your Wren AI sign-in code: {code}",
            f"Your Wren AI sign-in code is {code}\n\n"
            f"Enter it in the app to sign in. It expires in "
            f"{CODE_TTL_SECONDS // 60} minutes.\n\n"
            "If you didn't ask for this, ignore this email - nobody can sign in "
            "as you without the code.", rid)
    except HTTPException:
        # Nothing was delivered, so don't leave a pending row behind that
        # would silence retries for the whole resend cooldown.
        async with auth._pool.acquire() as conn:
            await conn.execute("DELETE FROM auth_email_otps WHERE email = $1", email)
        raise
    log.info(f"[{rid}] /auth/email/request: code sent")
    return EmailOkResponse(message=_OK_MSG)


async def email_verify(payload, rid: str) -> EmailTokenResponse:
    """Step 2: right code -> signed token for that email. Single use."""
    _require_ready(rid)
    email = _check_email(payload.email, rid)
    typed = (payload.code or "").strip() if isinstance(payload.code, str) else ""
    bad = _err(400, "that code is wrong or has expired", rid)
    now = _now()

    async with auth._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT code_hash, attempts, expires_at FROM auth_email_otps WHERE email = $1",
            email)
        if not row or row["expires_at"] < now:
            raise bad
        if row["attempts"] >= CODE_MAX_ATTEMPTS:
            await conn.execute("DELETE FROM auth_email_otps WHERE email = $1", email)
            raise bad
        if not hmac.compare_digest(_code_hash(email, typed).encode(),
                                   row["code_hash"].encode()):
            n = await conn.fetchval(
                "UPDATE auth_email_otps SET attempts = attempts + 1 "
                "WHERE email = $1 RETURNING attempts", email)
            log.warning(f"[{rid}] /auth/email/verify: wrong code (attempt {n})")
            if n is not None and n >= CODE_MAX_ATTEMPTS:
                await conn.execute("DELETE FROM auth_email_otps WHERE email = $1", email)
            raise bad
        # Right code. DELETE ... RETURNING makes "use it once" atomic: if two
        # requests race with the same correct code, only one gets the row.
        used = await conn.fetchrow(
            "DELETE FROM auth_email_otps WHERE email = $1 AND code_hash = $2 "
            "RETURNING email", email, row["code_hash"])
    if not used:
        raise bad

    tok = auth.make_token(email)
    if not tok:
        raise _err(500, "server not configured for sign-in", rid)
    log.info(f"[{rid}] /auth/email/verify: signed in")
    return EmailTokenResponse(email=email, token=tok)
