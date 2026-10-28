"""
Google Play subscriptions for Wren premium
==========================================
Sits next to premium.py. Premium is now a SUBSCRIPTION: one Play subscription
(GOOGLE_PREMIUM_SUB_ID, default "wren_premium") with two auto-renewing base
plans, "monthly" (N1,000) and "annual" (N10,000). Premium stays a property of
(email, device) in premium_subscriptions exactly as before; this module keeps
that row's expires_at in step with what Google says.

Routes (registered in main.py):
    POST /premium/google/verify   the app sends the purchase token after buying
and a hook, refresh_if_due(), that main.py calls from /premium/status.

How entitlement works
---------------------
Google's own rule: lineItems[].expiryTime decides access. ACTIVE and CANCELED
(auto-renew switched off, but paid up to expiryTime) are entitled until that
time; IN_GRACE_PERIOD (renewal payment failing) stays entitled while Google
retries; ON_HOLD / PAUSED / EXPIRED are not. A renewal simply moves
expiryTime forward, so every renewal is picked up by re-reading the purchase.

There is no Pub/Sub plumbing. Instead, refresh_if_due() re-reads a user's
subscription from Google when it is "due" (about every 6h, every 10 min in the
last day before expiry, every minute while it still needs acknowledging) and
is triggered by /premium/status, which the app already calls on startup and
resume. Between Google calls the stored expiry is used, so most status calls
cost nothing extra. Refunds / revocations show up as EXPIRED on the next
refresh.

/premium/google/verify answers (what the app expects)
    200  verified; body is the normal /premium/status response. Idempotent.
    202  Google says the purchase is still PENDING - the app retries later
    400  not a valid / active subscription for this app - retrying can't help
    403  purchase carries no account tag (not bought through our app flow)
    409  token belongs to a different account
    401  (from auth.require_user in the route) sign-in token rejected
    503  Google or our database can't answer - the app retries

Environment variables (Render)
    GOOGLE_PLAY_PACKAGE_NAME            org.wrenai.wrenaiofficial
    GOOGLE_PLAY_SERVICE_ACCOUNT_JSON    the whole service-account key JSON
    GOOGLE_PREMIUM_SUB_ID               wren_premium         (default)
    GOOGLE_PREMIUM_BASE_PLANS           monthly,annual       (default)
requirements.txt: google-auth, requests (token refresh runs in a worker thread).
"""

import os
import re
import json
import hashlib
import asyncio
import logging
import datetime as dt
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
SUB_PRODUCT_ID = os.environ.get("GOOGLE_PREMIUM_SUB_ID", "wren_premium").strip()
BASE_PLANS = {p.strip() for p in
              os.environ.get("GOOGLE_PREMIUM_BASE_PLANS", "monthly,annual").split(",")
              if p.strip()}

_SCOPE = "https://www.googleapis.com/auth/androidpublisher"
_API = "https://androidpublisher.googleapis.com/androidpublisher/v3"
GRACE_ACCESS = dt.timedelta(days=1)     # re-checked daily while Google retries payment


class GoogleVerifyRequest(BaseModel):
    device_id: str = ""
    email: str
    token: str = ""            # the app also sends its sign-in token here; the
                               # route authenticates with the X-Auth-Token header
    product_id: str
    purchase_token: str
    order_id: str = ""
    account_id: str = ""       # sha256(email.strip().lower()) hex
    base_plan_id: str = ""     # informational; Google's own data is what we trust


def account_tag(email: str) -> str:
    """Must match _google_account_id() in the app (main.py)."""
    return hashlib.sha256((email or "").strip().lower().encode("utf-8")).hexdigest()


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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


async def _google_request(method: str, url: str, rid: str, json_body=None):
    try:
        access = await asyncio.to_thread(_access_token_sync)
    except GoogleUnavailable:
        raise
    except Exception as e:
        raise GoogleUnavailable(f"credentials: {type(e).__name__}: {e}")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=20)) as client:
            if method == "GET":
                r = await client.get(url, headers={"Authorization": f"Bearer {access}"})
            else:
                r = await client.post(url, headers={"Authorization": f"Bearer {access}"},
                                      json=json_body if json_body is not None else {})
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise GoogleUnavailable(f"network: {type(e).__name__}")
    return r


async def _fetch_subscription(purchase_token: str, rid: str) -> dict:
    """GET the SubscriptionPurchaseV2 for a token."""
    url = (f"{_API}/applications/{quote(PACKAGE_NAME, safe='')}"
           f"/purchases/subscriptionsv2/tokens/{quote(purchase_token, safe='')}")
    r = await _google_request("GET", url, rid)
    if r.status_code == 200:
        return r.json()
    if r.status_code in (400, 404, 410):
        raise GoogleRejected(f"google answered {r.status_code}")
    if r.status_code in (401, 403):
        # OUR side is misconfigured (service account not invited in Play
        # Console, API not enabled...). The user did nothing wrong: 503.
        log.error(f"[{rid}] Google Play API refused our credentials "
                  f"({r.status_code}): {r.text[:300]}")
        raise GoogleUnavailable(f"google auth {r.status_code}")
    raise GoogleUnavailable(f"google answered {r.status_code}")


async def _acknowledge(product_id: str, purchase_token: str, rid: str) -> bool:
    """Play refunds a subscription that is not acknowledged within 3 days."""
    url = (f"{_API}/applications/{quote(PACKAGE_NAME, safe='')}"
           f"/purchases/subscriptions/{quote(product_id, safe='')}"
           f"/tokens/{quote(purchase_token, safe='')}:acknowledge")
    try:
        r = await _google_request("POST", url, rid, json_body={})
    except GoogleUnavailable as e:
        log.warning(f"[{rid}] acknowledge unavailable: {e}")
        return False
    if r.status_code in (200, 204):
        return True
    log.warning(f"[{rid}] acknowledge failed {r.status_code}: {r.text[:200]}")
    return False


async def _cancel_remote(product_id: str, purchase_token: str, rid: str) -> bool:
    """Turns auto-renew off (user keeps access until expiryTime). Outcome is
    judged afterwards by re-reading the subscription, not by this call."""
    url = (f"{_API}/applications/{quote(PACKAGE_NAME, safe='')}"
           f"/purchases/subscriptions/{quote(product_id, safe='')}"
           f"/tokens/{quote(purchase_token, safe='')}:cancel")
    try:
        r = await _google_request("POST", url, rid, json_body={})
    except GoogleUnavailable as e:
        log.warning(f"[{rid}] cancel unavailable: {e}")
        return False
    if r.status_code in (200, 204):
        return True
    log.warning(f"[{rid}] cancel failed {r.status_code}: {r.text[:200]}")
    return False


# ── Reading what Google said (pure functions: easy to test) ──────────────

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


def _parse_ts(s) -> Optional[dt.datetime]:
    if not s:
        return None
    m = _TS.match(str(s).strip())
    if not m:
        return None
    base, frac, tz = m.groups()
    frac = (frac or ".0")[:7]              # Python parses at most microseconds
    tz = "+00:00" if tz == "Z" else tz
    try:
        return dt.datetime.fromisoformat(f"{base}{frac}{tz}")
    except ValueError:
        return None


def entitlement_end(state: str, expiry: Optional[dt.datetime], now: dt.datetime):
    """When does paid access end, or None if the user has no access now?
    state is Google's SUBSCRIPTION_STATE_* without the prefix."""
    if state in ("ACTIVE", "CANCELED"):
        return expiry if expiry and expiry > now else None
    if state == "IN_GRACE_PERIOD":
        return max(expiry or now, now + GRACE_ACCESS)
    return None                             # ON_HOLD, PAUSED, EXPIRED, PENDING...


def parse_subscription(data: dict, expected_product: str, now: dt.datetime) -> Optional[dict]:
    """Flattens a SubscriptionPurchaseV2. None if it isn't for our product."""
    item = next((i for i in (data.get("lineItems") or [])
                 if (i or {}).get("productId") == expected_product), None)
    if item is None:
        return None
    state = (data.get("subscriptionState") or "").replace("SUBSCRIPTION_STATE_", "")
    expiry = _parse_ts(item.get("expiryTime"))
    until = entitlement_end(state, expiry, now)
    return {
        "state": (state or "UNSPECIFIED").lower(),
        "entitled_until": until,
        "expires_at": until or expiry,
        "base_plan_id": (item.get("offerDetails") or {}).get("basePlanId") or "",
        "auto_renewing": bool((item.get("autoRenewingPlan") or {}).get("autoRenewEnabled")),
        "order_id": item.get("latestSuccessfulOrderId") or data.get("latestOrderId") or "",
        "acknowledged": data.get("acknowledgementState") == "ACKNOWLEDGEMENT_STATE_ACKNOWLEDGED",
        "account_id": ((data.get("externalAccountIdentifiers") or {})
                       .get("obfuscatedExternalAccountId") or "").strip(),
        "linked_token": (data.get("linkedPurchaseToken") or "").strip(),
        "test": bool(data.get("testPurchase")),
    }


def evaluate_new_purchase(sub: Optional[dict], expected_account: str) -> tuple:
    """Verdict for a token the app has just submitted.
    ('ok' | 'pending' | 'invalid' | 'forbidden' | 'mismatch', message)"""
    if sub is None:
        return "invalid", "This purchase is for a different product."
    if BASE_PLANS and sub["base_plan_id"] and sub["base_plan_id"] not in BASE_PLANS:
        return "invalid", "This purchase is for an unknown plan."
    if sub["state"] == "pending":
        return "pending", "Payment is still pending."
    if not sub["account_id"]:
        return "forbidden", "Purchase has no account tag."
    if sub["account_id"] != expected_account:
        return "mismatch", "Purchase belongs to a different account."
    if sub["entitled_until"] is None:
        return "invalid", f"Subscription is not active (state={sub['state']})."
    return "ok", ""


# ── Storage (asyncpg / Postgres) ─────────────────────────────────────────
# Tables are created in premium.init_db():
#   google_subscriptions: one row per purchase token (see the CREATE there).
#   premium_subscriptions (existing): one row per email; its expires_at is what
#   /premium/status reads. _reconcile() keeps it in step with Google.

async def _upsert_sub(conn, token: str, email: str, device_id: str, sub: dict,
                      now: dt.datetime) -> str:
    """'new' | 'same' (already this email's) | 'other' (another email's)."""
    entitled = sub["entitled_until"] is not None
    args = (token, email, device_id, SUB_PRODUCT_ID, sub["base_plan_id"], sub["state"],
            entitled, sub["expires_at"], sub["auto_renewing"], sub["order_id"],
            sub["acknowledged"], now, now)
    inserted = await conn.fetchrow(
        "INSERT INTO google_subscriptions (purchase_token, email, device_id, product_id, "
        "base_plan_id, state, entitled, expires_at, auto_renewing, latest_order_id, "
        "acknowledged, granted_at, last_checked_at) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) "
        "ON CONFLICT (purchase_token) DO NOTHING RETURNING purchase_token", *args)
    if inserted is not None:
        return "new"
    owner = await conn.fetchrow(
        "SELECT email FROM google_subscriptions WHERE purchase_token = $1", token)
    if owner is None:
        raise RuntimeError("subscription row vanished mid-transaction")
    if owner["email"] != email:
        return "other"
    await conn.execute(
        "UPDATE google_subscriptions SET state=$2, entitled=$3, expires_at=$4, "
        "auto_renewing=$5, latest_order_id=$6, acknowledged=$7, base_plan_id=$8, "
        "last_checked_at=$9 WHERE purchase_token=$1",
        token, sub["state"], entitled, sub["expires_at"], sub["auto_renewing"],
        sub["order_id"], sub["acknowledged"] or False, sub["base_plan_id"], now)
    return "same"


async def _reconcile(conn, email: str, now: dt.datetime, link_device: str = ""):
    """Makes premium_subscriptions agree with the stored Google state:
      * the entitled subscription that lasts longest sets expires_at;
      * a legacy (Paystack) row that outlasts it is left alone;
      * when no subscription is entitled any more, a Google-granted row ends now.
    link_device: set only for a brand-new purchase (links the row to the buying
    device and resets the one-time restore, as a Paystack purchase did)."""
    best = await conn.fetchrow(
        "SELECT purchase_token, latest_order_id, expires_at, device_id "
        "FROM google_subscriptions "
        "WHERE email = $1 AND NOT replaced AND entitled AND expires_at > $2 "
        "ORDER BY expires_at DESC LIMIT 1", email, now)
    row = await conn.fetchrow(
        "SELECT device_id, reference, expires_at FROM premium_subscriptions "
        "WHERE email = $1 FOR UPDATE", email)

    if best is None:
        if row and row["reference"].startswith("gs:") and row["expires_at"] > now:
            await conn.execute(
                "UPDATE premium_subscriptions SET expires_at = $2 WHERE email = $1",
                email, now)
        return

    g_exp = best["expires_at"]
    ref = "gs:" + (best["latest_order_id"] or best["purchase_token"][:32])
    if row is None:
        await conn.execute(
            "INSERT INTO premium_subscriptions "
            "(email, device_id, reference, purchased_at, expires_at, status, restore_used) "
            "VALUES ($1, $2, $3, $4, $5, 'active', FALSE)",
            email, link_device or best["device_id"], ref, now, g_exp)
        return
    if row["reference"].startswith("gs:") or row["expires_at"] <= g_exp:
        await conn.execute(
            "UPDATE premium_subscriptions SET expires_at = $2, reference = $3, "
            "status = 'active' WHERE email = $1", email, g_exp, ref)
    if link_device:
        await conn.execute(
            "UPDATE premium_subscriptions SET device_id = $2, restore_used = FALSE "
            "WHERE email = $1", email, link_device)


# ── /premium/google/verify ───────────────────────────────────────────────

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
        # Premium is tied to (email, device) - same rule as Paystack's initialize.
        # Nothing is acknowledged on this answer, so Google refunds the purchase.
        raise _err(400, "this device could not be identified - update the app and try again", rid)
    if payload.product_id != SUB_PRODUCT_ID:
        raise _err(400, "unknown product", rid)

    expected_account = account_tag(email)
    if payload.account_id and payload.account_id != expected_account:
        raise _err(409, "purchase belongs to a different account", rid)

    now = _utcnow()
    try:
        data = await _fetch_subscription(token, rid)
    except GoogleRejected as e:
        log.warning(f"[{rid}] google verify rejected token: {e}")
        raise _err(400, "invalid purchase", rid)
    except GoogleUnavailable as e:
        log.warning(f"[{rid}] google verify unavailable: {e}")
        raise _err(503, "purchase verification temporarily unavailable", rid)

    sub = parse_subscription(data, SUB_PRODUCT_ID, now)
    verdict, message = evaluate_new_purchase(sub, expected_account)
    if verdict == "pending":
        return JSONResponse(status_code=202, content={"is_premium": False, "state": "pending"})
    if verdict == "invalid":
        raise _err(400, message, rid)
    if verdict == "forbidden":
        raise _err(403, message, rid)
    if verdict == "mismatch":
        raise _err(409, message, rid)
    if sub["test"]:
        log.info(f"[{rid}] google TEST subscription accepted for {email}")

    try:
        async with premium._pool.acquire() as conn:
            async with conn.transaction():
                outcome = await _upsert_sub(conn, token, email, device_id, sub, now)
                if outcome == "other":
                    raise _err(409, "this purchase is already linked to another account", rid)
                if sub["linked_token"]:
                    # Plan change / resubscribe: the old token is superseded.
                    await conn.execute(
                        "UPDATE google_subscriptions SET replaced = TRUE "
                        "WHERE purchase_token = $1 AND email = $2",
                        sub["linked_token"], email)
                if outcome == "same":
                    # Re-verification of a token already granted to this email. If
                    # the row is linked to another device (reinstall), the Google
                    # purchase is proof enough to re-link, once, using the account's
                    # single restore - like /premium/restore.
                    row = await conn.fetchrow(
                        "SELECT device_id, restore_used FROM premium_subscriptions "
                        "WHERE email = $1", email)
                    if row and row["device_id"] != device_id and not row["restore_used"]:
                        await conn.execute(
                            "UPDATE premium_subscriptions SET device_id = $2, "
                            "restore_used = TRUE WHERE email = $1", email, device_id)
                    await _reconcile(conn, email, now)
                else:
                    await _reconcile(conn, email, now, link_device=device_id)
    except HTTPException:
        raise
    except Exception as e:
        # Database trouble: the app retries; nothing was acknowledged.
        log.error(f"[{rid}] google grant failed: {type(e).__name__}: {e}")
        raise _err(503, "could not record purchase, try again", rid)

    # Acknowledge AFTER the grant is committed. A failure here is not fatal:
    # refresh_if_due() retries it every minute until it sticks.
    if not sub["acknowledged"]:
        if await _acknowledge(SUB_PRODUCT_ID, token, rid):
            async with premium._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE google_subscriptions SET acknowledged = TRUE "
                    "WHERE purchase_token = $1", token)

    log.info(f"[{rid}] google subscription {outcome}: {email} plan={sub['base_plan_id']} "
             f"state={sub['state']} until={sub['entitled_until']}")
    return await premium.premium_status(device_id, rid, email=email)


# ── /premium/google/cancel ───────────────────────────────────────────────

class GoogleCancelRequest(BaseModel):
    email: str
    token: str = ""            # route authenticates with the X-Auth-Token header


async def auto_renewing_for(email: str) -> Optional[bool]:
    """True/False when this email has an entitled Google subscription
    (is it set to renew?), None when it has none (e.g. legacy Paystack)."""
    email = premium._normalize_email(email)
    if not email or getattr(premium, "_pool", None) is None:
        return None
    try:
        async with premium._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT bool_or(auto_renewing) AS ar FROM google_subscriptions "
                "WHERE email = $1 AND NOT replaced AND entitled AND expires_at > $2",
                email, _utcnow())
        return None if row is None else row["ar"]
    except Exception as e:
        log.warning(f"auto_renewing_for failed: {type(e).__name__}: {e}")
        return None


async def google_cancel(payload: GoogleCancelRequest, rid: str):
    """Stops renewal of the caller's subscription. Access continues until the
    already-paid period ends, exactly like cancelling inside Google Play."""
    premium._require_pool(rid)
    email = premium._normalize_email(payload.email)
    if not email or "@" not in email:
        raise _err(400, "a valid email is required", rid)
    now = _utcnow()
    async with premium._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT purchase_token, product_id FROM google_subscriptions "
            "WHERE email = $1 AND NOT replaced AND entitled AND expires_at > $2",
            email, now)
    if not rows:
        raise _err(404, "no active recurring subscription to cancel", rid)

    latest_exp = None
    still_renewing = False
    for r in rows:
        await _cancel_remote(r["product_id"], r["purchase_token"], rid)
        try:
            data = await _fetch_subscription(r["purchase_token"], rid)
        except GoogleRejected:
            continue
        except GoogleUnavailable as e:
            log.warning(f"[{rid}] cancel re-read unavailable: {e}")
            raise _err(503, "could not confirm cancellation, try again", rid)
        sub = parse_subscription(data, r["product_id"], now)
        if sub is None:
            continue
        still_renewing = still_renewing or sub["auto_renewing"]
        if sub["expires_at"] and (latest_exp is None or sub["expires_at"] > latest_exp):
            latest_exp = sub["expires_at"]
        async with premium._pool.acquire() as conn:
            await conn.execute(
                "UPDATE google_subscriptions SET state=$2, entitled=$3, expires_at=$4, "
                "auto_renewing=$5, last_checked_at=$6 WHERE purchase_token=$1",
                r["purchase_token"], sub["state"], sub["entitled_until"] is not None,
                sub["expires_at"], sub["auto_renewing"], now)
    if still_renewing:
        raise _err(502, "Google did not accept the cancellation", rid)
    try:
        async with premium._pool.acquire() as conn:
            async with conn.transaction():
                await _reconcile(conn, email, _utcnow())
    except Exception as e:
        log.warning(f"[{rid}] reconcile after cancel failed: {type(e).__name__}: {e}")
    log.info(f"[{rid}] google subscription cancelled: {email} until {latest_exp}")
    return {"ok": True, "auto_renewing": False,
            "expires_at": latest_exp.isoformat() if latest_exp else None}


# ── Keeping entitlement fresh (called from /premium/status) ──────────────

def _is_due(r, now: dt.datetime) -> bool:
    age = now - r["last_checked_at"]
    if r["entitled"]:
        if not r["acknowledged"]:
            return age > dt.timedelta(seconds=60)
        exp = r["expires_at"]
        if exp is not None and exp - now < dt.timedelta(days=1):
            return age > dt.timedelta(minutes=10)
        return age > dt.timedelta(hours=6)
    return age > dt.timedelta(hours=1)      # on_hold / paused / pending / lapsed


async def _refresh_one(r, rid: str, now: dt.datetime):
    token = r["purchase_token"]
    async with premium._pool.acquire() as conn:
        # Claim this check so concurrent status calls don't all hit Google.
        claimed = await conn.fetchrow(
            "UPDATE google_subscriptions SET last_checked_at = $2 "
            "WHERE purchase_token = $1 AND last_checked_at = $3 RETURNING purchase_token",
            token, now, r["last_checked_at"])
    if claimed is None:
        return
    try:
        data = await _fetch_subscription(token, rid)
    except GoogleRejected:
        # Google no longer knows the token (very old / revoked): not entitled.
        async with premium._pool.acquire() as conn:
            await conn.execute(
                "UPDATE google_subscriptions SET state = 'expired', entitled = FALSE "
                "WHERE purchase_token = $1", token)
        return
    except GoogleUnavailable as e:
        log.warning(f"[{rid}] subscription refresh unavailable: {e}")
        return
    sub = parse_subscription(data, r["product_id"], now)
    if sub is None:
        return
    async with premium._pool.acquire() as conn:
        await conn.execute(
            "UPDATE google_subscriptions SET state=$2, entitled=$3, expires_at=$4, "
            "auto_renewing=$5, latest_order_id=$6, acknowledged=$7, base_plan_id=$8 "
            "WHERE purchase_token=$1",
            token, sub["state"], sub["entitled_until"] is not None, sub["expires_at"],
            sub["auto_renewing"], sub["order_id"], sub["acknowledged"], sub["base_plan_id"])
    if sub["entitled_until"] is not None and not sub["acknowledged"]:
        if await _acknowledge(r["product_id"], token, rid):
            async with premium._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE google_subscriptions SET acknowledged = TRUE "
                    "WHERE purchase_token = $1", token)


async def refresh_if_due(email: str, rid: str):
    """Call before reading premium status. Re-reads this user's Google
    subscriptions that are due, then reconciles premium_subscriptions from the
    stored state. Never raises: a failure just leaves the previous state."""
    email = premium._normalize_email(email)
    if not email or getattr(premium, "_pool", None) is None:
        return
    try:
        now = _utcnow()
        async with premium._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT purchase_token, product_id, entitled, expires_at, acknowledged, "
                "last_checked_at FROM google_subscriptions "
                "WHERE email = $1 AND NOT replaced AND state <> 'expired'", email)
        for r in [r for r in rows if _is_due(r, now)][:3]:
            await _refresh_one(r, rid, now)
        if rows:
            async with premium._pool.acquire() as conn:
                async with conn.transaction():
                    await _reconcile(conn, email, _utcnow())
    except Exception as e:
        log.warning(f"[{rid}] subscription refresh failed: {type(e).__name__}: {e}")
