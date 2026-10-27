"""
Google Play Billing verification for Wren premium
=================================================
Sits next to premium.py (Paystack). Same product: one-time, 365 days of
premium per verified purchase, keyed by email + device_id, with the backend
(not the phone) as the only thing that ever grants it.

Route (registered in main.py):
    POST /premium/google/verify

The phone sends the purchase token Google gave it. We:
  1. check the product and the account tag the phone claims,
  2. ask the Google Play Developer API what that token REALLY is
     (purchases.productsv2.getproductpurchasev2),
  3. require PURCHASED state, the right product, and that Google's own
     obfuscatedExternalAccountId equals sha256(email) - this is what stops one
     Google purchase being redeemed under a different Wren account,
  4. record the token (unique) and grant premium in ONE database transaction.

Answers (these are exactly what the app's _google_verify_purchase expects):
    200  verified + premium active (same shape as /premium/status). Idempotent:
         the same token for the same account answers 200 again.
    202  Google says the purchase is still PENDING - app retries later
    400  invalid / cancelled / wrong product - retrying cannot help
    403  purchase carries no account tag (not made through our app flow)
    409  token already redeemed by, or tagged for, a different account
    401  (raised by auth.require_user in the route) sign-in token rejected
    503  Google/our credentials unavailable - app retries; NOT consumed

Environment variables (set on Render):
    GOOGLE_PLAY_PACKAGE_NAME            org.wrenai.wrenaiofficial
    GOOGLE_PLAY_SERVICE_ACCOUNT_JSON    the whole service-account key JSON, as one value
    GOOGLE_PREMIUM_PRODUCT_ID           wren_premium_365   (optional; this is the default)
Add `google-auth` and `requests` to requirements.txt (google-auth's token
refresh uses requests; it runs in a worker thread so it never blocks the loop).
"""

import os
import json
import time
import hashlib
import datetime as dt
import asyncio
import logging
from typing import Optional
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import premium

log = logging.getLogger("wren-backend")

PACKAGE_NAME = os.environ.get("GOOGLE_PLAY_PACKAGE_NAME", "").strip()
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_PLAY_SERVICE_ACCOUNT_JSON", "").strip()
PRODUCT_ID = os.environ.get("GOOGLE_PREMIUM_PRODUCT_ID", "wren_premium_365").strip()
PREMIUM_DAYS = 365

_SCOPE = "https://www.googleapis.com/auth/androidpublisher"
_API = "https://androidpublisher.googleapis.com/androidpublisher/v3"


class GoogleVerifyRequest(BaseModel):
    device_id: str = ""
    email: str
    token: str = ""            # the app also sends its sign-in token here; the
                               # route authenticates with the X-Auth-Token header
    product_id: str
    purchase_token: str
    order_id: str = ""
    account_id: str = ""       # sha256(email.strip().lower()) hex


def account_tag(email: str) -> str:
    """Must match _google_account_id() in the app (main.py)."""
    return hashlib.sha256((email or "").strip().lower().encode("utf-8")).hexdigest()


# ── Calling Google ───────────────────────────────────────────────────────

class GoogleUnavailable(Exception):
    """Google (or our credentials for it) can't answer right now -> 503."""


class GoogleRejected(Exception):
    """Google says this token is not a valid purchase for this app -> 400."""


_creds = None


def _access_token_sync() -> str:
    """Blocking: builds/refreshes the service-account credentials. Called via
    asyncio.to_thread so the event loop is never held."""
    global _creds
    if not SERVICE_ACCOUNT_JSON or not PACKAGE_NAME:
        raise GoogleUnavailable("GOOGLE_PLAY_* env vars not set")
    from google.oauth2 import service_account
    import google.auth.transport.requests as gtr
    if _creds is None:
        _creds = service_account.Credentials.from_service_account_info(
            json.loads(SERVICE_ACCOUNT_JSON), scopes=[_SCOPE])
    if not _creds.valid:
        _creds.refresh(gtr.Request())
    return _creds.token


async def _fetch_purchase(purchase_token: str, rid: str) -> dict:
    """GET the ProductPurchaseV2 for a token. Raises GoogleRejected or
    GoogleUnavailable; returns the parsed JSON on success."""
    try:
        access = await asyncio.to_thread(_access_token_sync)
    except GoogleUnavailable:
        raise
    except Exception as e:
        raise GoogleUnavailable(f"credentials: {type(e).__name__}: {e}")

    url = (f"{_API}/applications/{quote(PACKAGE_NAME, safe='')}"
           f"/purchases/productsv2/tokens/{quote(purchase_token, safe='')}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=20)) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {access}"})
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise GoogleUnavailable(f"network: {type(e).__name__}")

    if r.status_code == 200:
        return r.json()
    if r.status_code in (400, 404, 410):
        # Malformed token, unknown token, or purchase too old/removed.
        raise GoogleRejected(f"google answered {r.status_code}")
    if r.status_code in (401, 403):
        # OUR side is misconfigured (service account not granted in Play Console,
        # API not enabled...). The user did nothing wrong: surface as 503.
        log.error(f"[{rid}] Google Play API refused our credentials "
                  f"({r.status_code}): {r.text[:300]}")
        raise GoogleUnavailable(f"google auth {r.status_code}")
    raise GoogleUnavailable(f"google answered {r.status_code}")


# ── Judging what Google said (pure function: easy to test) ───────────────

def evaluate_purchase(data: dict, expected_product: str, expected_account: str) -> tuple:
    """Returns (verdict, message, order_id).
    verdict: 'ok' | 'pending' | 'invalid' | 'forbidden' | 'mismatch'"""
    order_id = (data.get("orderId") or "").strip()

    products = [(i or {}).get("productId") for i in (data.get("productLineItem") or [])]
    if expected_product not in products:
        return "invalid", "This purchase is for a different product.", order_id

    state = ((data.get("purchaseStateContext") or {}).get("purchaseState") or "")
    if state == "PENDING":
        return "pending", "Payment is still pending.", order_id
    if state != "PURCHASED":
        return "invalid", f"Purchase is not completed (state={state or 'unknown'}).", order_id

    tagged = (data.get("obfuscatedExternalAccountId") or "").strip()
    if not tagged:
        return "forbidden", "Purchase has no account tag.", order_id
    if tagged != expected_account:
        return "mismatch", "Purchase belongs to a different account.", order_id

    return "ok", "", order_id


# ── Storage: THE ONE PART THAT NEEDS premium.py ──────────────────────────

# The table (created in premium.init_db()):
#   google_purchases(purchase_token PK, order_id, product_id, email,
#                    device_id, granted_at)
# One Google purchase token can grant premium exactly once, to one email.
#
# Claim + grant happen in ONE transaction, so a failure between them can never
# leave a claimed token that produced no premium. Grant rules deliberately
# mirror premium.premium_verify (Paystack): one premium_subscriptions row per
# email, a new purchase re-links the row to the buying device and resets
# restore_used. The one difference: if the email still has ACTIVE premium, the
# new 365 days are added on top of the remaining time (the user has really
# paid, so they must not lose what they already had).

async def _store_claim_and_grant(*, purchase_token: str, order_id: str,
                                 product_id: str, email: str, device_id: str,
                                 days: int) -> str:
    """Returns "new" (granted now), "same" (this token was already granted to
    this email - idempotent success) or "other" (token belongs to another
    email)."""
    now = dt.datetime.now(dt.timezone.utc)
    reference = "gp:" + (order_id or hashlib.sha256(purchase_token.encode()).hexdigest()[:32])

    async with premium._pool.acquire() as conn:
        async with conn.transaction():
            inserted = await conn.fetchrow(
                "INSERT INTO google_purchases "
                "(purchase_token, order_id, product_id, email, device_id, granted_at) "
                "VALUES ($1, $2, $3, $4, $5, $6) "
                "ON CONFLICT (purchase_token) DO NOTHING "
                "RETURNING purchase_token",
                purchase_token, order_id or "", product_id, email, device_id, now)

            if inserted is None:
                owner = await conn.fetchrow(
                    "SELECT email FROM google_purchases WHERE purchase_token = $1",
                    purchase_token)
                if owner is None:
                    raise RuntimeError("purchase row vanished mid-transaction")
                if owner["email"] != email:
                    return "other"
                # Already granted to this email. If it is still linked to a
                # different device (e.g. reinstall before the consume went
                # through), the Google purchase is proof enough to re-link it,
                # but only using the account's single restore, like /restore.
                row = await conn.fetchrow(
                    "SELECT device_id, restore_used FROM premium_subscriptions "
                    "WHERE email = $1", email)
                if row and row["device_id"] != device_id and not row["restore_used"]:
                    await conn.execute(
                        "UPDATE premium_subscriptions "
                        "SET device_id = $2, restore_used = TRUE WHERE email = $1",
                        email, device_id)
                return "same"

            existing = await conn.fetchrow(
                "SELECT expires_at FROM premium_subscriptions WHERE email = $1 FOR UPDATE",
                email)
            base = now
            if existing and existing["expires_at"] > now:
                base = existing["expires_at"]
            expires = base + dt.timedelta(days=days)

            await conn.execute(
                "INSERT INTO premium_subscriptions "
                "(email, device_id, reference, purchased_at, expires_at, status, restore_used) "
                "VALUES ($1, $2, $3, $4, $5, 'active', FALSE) "
                "ON CONFLICT (email) DO UPDATE SET "
                "device_id = EXCLUDED.device_id, reference = EXCLUDED.reference, "
                "purchased_at = EXCLUDED.purchased_at, expires_at = EXCLUDED.expires_at, "
                "status = 'active', restore_used = FALSE",
                email, device_id, reference, now, expires)
            return "new"


# ── The route's logic ────────────────────────────────────────────────────

def _err(status: int, msg: str, rid: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": msg, "request_id": rid})


async def google_verify(payload: GoogleVerifyRequest, rid: str):
    premium._require_pool(rid)
    email = premium._normalize_email(payload.email)
    token = (payload.purchase_token or "").strip()
    device_id = (payload.device_id or "").strip()
    if not email or "@" not in email or not token:
        raise _err(400, "a valid email and purchase_token are required", rid)
    if not premium._device_id_ok(device_id):
        # Same rule as Paystack's initialize: premium is tied to (email, device),
        # so an unidentifiable device can never hold it. Nothing is consumed on
        # this answer, so Google refunds the purchase automatically.
        raise _err(400, "this device could not be identified - update the app and try again", rid)
    if payload.product_id != PRODUCT_ID:
        raise _err(400, "unknown product", rid)

    expected_account = account_tag(email)
    if payload.account_id and payload.account_id != expected_account:
        # The phone's own claim disagrees with the signed-in email.
        raise _err(409, "purchase belongs to a different account", rid)

    try:
        data = await _fetch_purchase(token, rid)
    except GoogleRejected as e:
        log.warning(f"[{rid}] google verify rejected token: {e}")
        raise _err(400, "invalid purchase", rid)
    except GoogleUnavailable as e:
        log.warning(f"[{rid}] google verify unavailable: {e}")
        raise _err(503, "purchase verification temporarily unavailable", rid)

    verdict, message, order_id = evaluate_purchase(data, PRODUCT_ID, expected_account)
    if verdict == "pending":
        return JSONResponse(status_code=202,
                            content={"is_premium": False, "state": "pending"})
    if verdict == "invalid":
        raise _err(400, message, rid)
    if verdict == "forbidden":
        raise _err(403, message, rid)
    if verdict == "mismatch":
        raise _err(409, message, rid)

    if data.get("testPurchaseContext"):
        log.info(f"[{rid}] google test purchase accepted for {email}")

    try:
        outcome = await _store_claim_and_grant(
            purchase_token=token, order_id=order_id or payload.order_id,
            product_id=PRODUCT_ID, email=email,
            device_id=device_id, days=PREMIUM_DAYS)
    except Exception as e:
        # Database trouble: the app retries and the purchase stays unconsumed.
        log.error(f"[{rid}] google grant failed: {type(e).__name__}: {e}")
        raise _err(503, "could not record purchase, try again", rid)

    if outcome == "other":
        raise _err(409, "this purchase is already linked to another account", rid)

    log.info(f"[{rid}] google purchase {outcome}: {email} order={order_id}")
    # Same response shape as every other premium route.
    return await premium.premium_status(device_id, rid, email=email)
