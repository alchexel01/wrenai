"""
Wren Google Play review login
=============================
Lets the Google Play reviewer sign in to ONE fixed account and get Pro
without paying, using a password that only lives on Render.

Flow:
    1. In the app the reviewer types REVIEW_EMAIL (pro_email@wren.ai) on the
       sign-in screen; the button turns into "Continue".
    2. Continue opens a password screen. The app calls
       POST /auth/review/login {email, password, device_id}.
    3. If the password matches REVIEW_PASSWORD we:
         - give that email a Pro row in premium_subscriptions (365 days,
           renewed on every successful login), and
         - return the same signed token Google / email sign-in return,
           so every other route (chats, premium status, usage) just works.

Safety:
    * Only REVIEW_EMAIL can ever use this route; it cannot grant Pro to any
      other account.
    * No REVIEW_PASSWORD on Render = the route is switched off (404).
    * Wrong guesses are limited: MAX_FAILS wrong passwords in LOCK_SECONDS
      locks the route for that window (answers 429), even for the right
      password, so it cannot be brute-forced.
    * The password is compared in constant time and is never logged.

Env vars (Render dashboard):
    REVIEW_PASSWORD   - the password you give Google (long and random).
                        Remove it to turn the review login off.
    REVIEW_EMAIL      - optional, defaults to pro_email@wren.ai
"""

import os
import time
import hmac
import logging
import datetime as dt
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

import auth
import premium

log = logging.getLogger("wren-backend")

REVIEW_EMAIL = os.environ.get("REVIEW_EMAIL", "pro_email@wren.ai").strip().lower()
REVIEW_PASSWORD = os.environ.get("REVIEW_PASSWORD", "").strip()

REVIEW_PRO_DAYS = 365
MAX_FAILS = 8            # wrong passwords allowed ...
LOCK_SECONDS = 15 * 60   # ... within this window before the route locks

_fails: list = []        # unix times of recent wrong passwords (per worker)


class ReviewLoginRequest(BaseModel):
    email: str
    password: str
    device_id: Optional[str] = ""


class ReviewLoginResponse(BaseModel):
    email: str
    token: str
    is_premium: bool = True
    expires_at: Optional[str] = None


def _recent_fails(now: float) -> int:
    cutoff = now - LOCK_SECONDS
    while _fails and _fails[0] < cutoff:
        _fails.pop(0)
    return len(_fails)


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest((a or "").encode("utf-8", "ignore"),
                               (b or "").encode("utf-8", "ignore"))


async def review_login(payload: ReviewLoginRequest, rid: str) -> ReviewLoginResponse:
    if not REVIEW_PASSWORD:
        # Feature is off. Look exactly like a route that does not exist.
        raise HTTPException(status_code=404,
                            detail={"error": "not found", "request_id": rid})

    now = time.time()
    if _recent_fails(now) >= MAX_FAILS:
        log.warning(f"[{rid}] /auth/review/login: locked after too many wrong passwords")
        raise HTTPException(
            status_code=429,
            detail={"error": "Too many wrong attempts. Please try again in 15 minutes.",
                    "request_id": rid})

    email = (payload.email or "").strip().lower()
    # Always run both comparisons so timing does not reveal which was wrong.
    email_ok = _same(email, REVIEW_EMAIL)
    pass_ok = _same((payload.password or "").strip(), REVIEW_PASSWORD)
    if not (email_ok and pass_ok):
        _fails.append(now)
        log.warning(f"[{rid}] /auth/review/login: wrong credentials "
                    f"({_recent_fails(now)}/{MAX_FAILS})")
        raise HTTPException(status_code=400,
                            detail={"error": "Incorrect password.", "request_id": rid})

    token = auth.make_token(REVIEW_EMAIL)
    if not token:
        log.error(f"[{rid}] /auth/review/login: cannot sign a token (no AUTH_TOKEN_SECRET)")
        raise HTTPException(status_code=500,
                            detail={"error": "sign-in is not configured", "request_id": rid})

    premium._require_pool(rid)
    started = dt.datetime.now(dt.timezone.utc)
    expires = started + dt.timedelta(days=REVIEW_PRO_DAYS)
    device_id = (payload.device_id or "").strip()[:64] or "review-account"
    async with premium._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO premium_subscriptions "
            "(email, device_id, reference, purchased_at, expires_at, status, restore_used) "
            "VALUES ($1, $2, 'review-account', $3, $4, 'active', FALSE) "
            "ON CONFLICT (email) DO UPDATE SET "
            "device_id = EXCLUDED.device_id, reference = EXCLUDED.reference, "
            "purchased_at = EXCLUDED.purchased_at, expires_at = EXCLUDED.expires_at, "
            "status = 'active'",
            REVIEW_EMAIL, device_id, started, expires)

    _fails.clear()
    log.info(f"[{rid}] /auth/review/login: review account signed in, Pro until "
             f"{expires.isoformat()}")
    return ReviewLoginResponse(email=REVIEW_EMAIL, token=token, is_premium=True,
                               expires_at=expires.isoformat())
