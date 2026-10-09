"""
Wren Account Deletion
=====================
Permanently deletes everything the backend holds for one email, in a single
database transaction (all-or-nothing, so a failure halfway can never leave an
account half-deleted):

    chat_histories         every synced conversation        (chat_history.py)
    premium_subscriptions  the premium purchase record      (premium.py)
    auth_login_sessions    any unfinished Google sign-in    (auth.py)

Endpoint (wired into main.py):
    DELETE /account/{email}

Security
--------
This is the most destructive route on the backend, so it is stricter than the
other email-keyed routes:

* It needs the X-App-Secret header like every route, AND the signed sign-in
  token for exactly this email (X-Auth-Token), verified with
  auth.verify_token() DIRECTLY - not auth.require_user(). require_user()
  silently allows everything when AUTH_REQUIRE_TOKEN=0 (the escape hatch), and
  that must never open up "delete anyone's account if you know their address".
  If the token secret isn't configured, verify_token() fails closed -> 401.
* It is idempotent: deleting an account that has no data still returns 200,
  so a retry after a dropped connection succeeds instead of erroring.

Premium note
------------
The premium row is deleted with the account. Signing in again with the same
email starts from a clean slate (free tier, and the email can buy again).
The Paystack reference is written to the server log before deletion so there
is still a trail for payment disputes.
"""

import logging
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

import auth
import chat_history
import premium

log = logging.getLogger("wren-backend")


class AccountDeleteResponse(BaseModel):
    ok: bool
    chats_deleted: int = 0
    premium_deleted: bool = False


def _normalize_email(email):
    return (email or "").strip().lower()


def _get_pool():
    # All three modules point at the same DATABASE_URL; use one pool so the
    # three deletes can share one transaction.
    return chat_history._pool or premium._pool or auth._pool


async def account_delete(email, token: Optional[str], rid) -> AccountDeleteResponse:
    email = _normalize_email(email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400,
                            detail={"error": "a valid email is required", "request_id": rid})

    if not auth.verify_token(email, token):
        log.warning(f"[{rid}] /account/delete: missing/invalid sign-in token for "
                    f"email={email[:3]}***")
        raise HTTPException(status_code=401,
                            detail={"error": "sign in with Google to continue",
                                    "request_id": rid})

    pool = _get_pool()
    if pool is None:
        log.error(f"[{rid}] /account/delete: DB pool not initialized (DATABASE_URL missing?)")
        raise HTTPException(status_code=500,
                            detail={"error": "server missing DATABASE_URL", "request_id": rid})

    async with pool.acquire() as conn:
        async with conn.transaction():
            prem = await conn.fetchrow(
                "SELECT reference, expires_at FROM premium_subscriptions WHERE email = $1",
                email)
            chats = await conn.execute(
                "DELETE FROM chat_histories WHERE email = $1", email)
            prem_res = await conn.execute(
                "DELETE FROM premium_subscriptions WHERE email = $1", email)
            await conn.execute(
                "DELETE FROM auth_login_sessions WHERE email = $1", email)

    # asyncpg returns command tags like "DELETE 3".
    def _count(tag):
        try:
            return int(str(tag).rsplit(" ", 1)[-1])
        except ValueError:
            return 0

    n_chats = _count(chats)
    had_premium = _count(prem_res) > 0
    log.warning(f"[{rid}] /account/delete: email={email} chats={n_chats} "
                f"premium={'yes' if had_premium else 'no'}"
                + (f" reference={prem['reference']} expires={prem['expires_at'].isoformat()}"
                   if prem else ""))
    return AccountDeleteResponse(ok=True, chats_deleted=n_chats, premium_deleted=had_premium)
