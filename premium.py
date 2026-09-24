"""
Wren Premium — Paystack integration
====================================
One-time (non-recurring) premium purchase: N3,000 for 365 days of access.
Not a subscription — Paystack never auto-charges again. When the 365
days are up, the app simply goes back to free until the user pays again.

Identity model: EMAIL is the durable identity, not device_id.
---------------------------------------------------------------
Device fingerprinting (Android ID, or a random UUID as a fallback when
Android ID is unavailable) is what /premium/status checks day-to-day
because it's instant and needs no user input. But it's not durable
across an uninstall: on many devices the UUID fallback regenerates
from scratch on reinstall, silently orphaning a paying user's premium
with no way back in. Email survives that, because the user can type
it in again.

So: one premium row per email, ever. A given email can complete a real
purchase exactly once (a second attempt is rejected as "email already
used" rather than granting a second 365-day period or moving the
subscription). If that email's original device is later lost — most
commonly a reinstall that regenerated the device UUID — the user gets
exactly ONE lifetime "Restore Purchase": re-enter the same email, and
premium re-links to whatever device asks. A second restore attempt is
rejected too, same reasoning as the purchase limit: there's no
reliable way to tell "my phone again" from "someone else has my
email," so the honest, simple rule is one shot each, with manual
support (via /premium/reset, testing-only today) as the fallback for
real edge cases.

Storage: Postgres (DATABASE_URL env var, set automatically by Render
when you attach a Postgres instance to this service in the same
project). Deliberately NOT SQLite/a local file — this service's own
disk is wiped on every redeploy, which would silently un-premium every
paying user the next time you ship a code change. Postgres survives
that because it's a separate managed service.

Endpoints (wired into main.py):
    POST   /premium/initialize        — start a Paystack transaction, get checkout URL
    GET    /premium/verify/{reference}— confirm a transaction, activate premium on success
    GET    /premium/status/{device_id}— is this device currently premium, until when
    POST   /premium/restore           — one-time: re-link an email's premium to a new device_id
    DELETE /premium/reset/{email}     — TESTING ONLY: wipe an email's premium record entirely

All endpoints still require the existing X-App-Secret header, same as
every other route on this backend.
"""

import os
import logging
import datetime as dt
from typing import Optional

import json

import httpx
import asyncpg
from fastapi import HTTPException
from pydantic import BaseModel

log = logging.getLogger("wren-backend")

PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY", "").strip()
PAYSTACK_BASE_URL = "https://api.paystack.co"

PREMIUM_PRICE_KOBO = 300_000
PREMIUM_DURATION_DAYS = 365

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

_pool = None


async def init_db():
    global _pool
    if not DATABASE_URL:
        log.error("[premium] DATABASE_URL not set - premium endpoints will fail")
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS premium_subscriptions ("
            "email TEXT PRIMARY KEY, "
            "device_id TEXT NOT NULL, "
            "reference TEXT NOT NULL, "
            "purchased_at TIMESTAMPTZ NOT NULL, "
            "expires_at TIMESTAMPTZ NOT NULL, "
            "status TEXT NOT NULL DEFAULT 'active', "
            "restore_used BOOLEAN NOT NULL DEFAULT FALSE"
            ")"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_premium_device_id "
            "ON premium_subscriptions (device_id)"
        )
    log.info("[premium] DB pool ready, table ensured")


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _require_pool(rid):
    if _pool is None:
        log.error(f"[{rid}] /premium: DB pool not initialized (DATABASE_URL missing?)")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing DATABASE_URL", "request_id": rid})


def _require_paystack_key(rid):
    if not PAYSTACK_SECRET_KEY:
        log.error(f"[{rid}] /premium: server missing PAYSTACK_SECRET_KEY")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing PAYSTACK_SECRET_KEY", "request_id": rid})


def _normalize_email(email):
    return (email or "").strip().lower()


def _parse_paystack_json(resp, rid, endpoint):
    """Paystack normally returns JSON, but under transient load (gateway
    timeouts, Cloudflare hiccups) a non-2xx response sometimes comes back
    as an HTML error page instead. resp.json() raises on that, which
    turned an occasional upstream hiccup into an unhandled 500 instead of
    the clean 502 this module intends - that's most of the "verification
    fails sometimes" reports. Parse defensively so those cases fail the
    same clean way as a normal Paystack error response."""
    try:
        return resp.json()
    except ValueError:
        log.error(f"[{rid}] {endpoint}: Paystack returned non-JSON "
                  f"(status={resp.status_code}): {resp.text[:200]!r}")
        raise HTTPException(status_code=502,
                             detail={"error": "paystack returned an invalid response",
                                     "source": "paystack", "request_id": rid})


class InitializeRequest(BaseModel):
    device_id: str
    email: str


class InitializeResponse(BaseModel):
    authorization_url: str
    reference: str


class StatusResponse(BaseModel):
    is_premium: bool
    expires_at: Optional[str] = None


class RestoreRequest(BaseModel):
    device_id: str
    email: str


async def premium_initialize(payload, rid):
    _require_pool(rid)
    _require_paystack_key(rid)

    device_id = (payload.device_id or "").strip()
    email = _normalize_email(payload.email)
    if not device_id:
        raise HTTPException(status_code=400, detail={"error": "device_id is required", "request_id": rid})
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail={"error": "a valid email is required", "request_id": rid})

    async with _pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT expires_at FROM premium_subscriptions WHERE email = $1", email)
    if existing:
        now = dt.datetime.now(dt.timezone.utc)
        if existing["expires_at"] > now:
            log.info(f"[{rid}] /premium/initialize: email={email} already has active premium - rejecting")
            raise HTTPException(status_code=409, detail={"error": "email already used", "request_id": rid})

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                f"{PAYSTACK_BASE_URL}/transaction/initialize",
                headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"},
                json={
                    "email": email,
                    "amount": PREMIUM_PRICE_KOBO,
                    "currency": "NGN",
                    "metadata": {"device_id": device_id, "email": email},
                },
            )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /premium/initialize: Paystack request failed: {e!r}")
            raise HTTPException(status_code=502,
                                 detail={"error": f"upstream request failed: {e}",
                                         "source": "paystack", "request_id": rid})

    body = _parse_paystack_json(resp, rid, "/premium/initialize")
    if resp.status_code != 200 or not body.get("status"):
        log.error(f"[{rid}] /premium/initialize: Paystack error {resp.status_code}: {body}")
        raise HTTPException(status_code=502,
                             detail={"error": body.get("message", "paystack initialize failed"),
                                     "source": "paystack", "request_id": rid})

    data = body["data"]
    log.info(f"[{rid}] /premium/initialize: email={email} device={device_id} reference={data['reference']}")
    return InitializeResponse(authorization_url=data["authorization_url"], reference=data["reference"])


async def premium_verify(reference, rid):
    _require_pool(rid)
    _require_paystack_key(rid)

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                f"{PAYSTACK_BASE_URL}/transaction/verify/{reference}",
                headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"},
            )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /premium/verify: Paystack request failed: {e!r}")
            raise HTTPException(status_code=502,
                                 detail={"error": f"upstream request failed: {e}",
                                         "source": "paystack", "request_id": rid})

    body = _parse_paystack_json(resp, rid, "/premium/verify")
    if resp.status_code != 200 or not body.get("status"):
        log.error(f"[{rid}] /premium/verify: Paystack error {resp.status_code}: {body}")
        raise HTTPException(status_code=502,
                             detail={"error": body.get("message", "paystack verify failed"),
                                     "source": "paystack", "request_id": rid})

    data = body["data"]
    paystack_status = data.get("status")
    meta = data.get("metadata") or {}
    if isinstance(meta, str):
        # Paystack occasionally round-trips metadata as a JSON-encoded
        # string instead of the object we sent - a known quirk that
        # depends on the payment channel/dashboard settings, not
        # something under our control. meta.get(...) below would raise
        # AttributeError on a str, which is the other big source of
        # intermittent verify failures. Decode it back to a dict.
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            log.error(f"[{rid}] /premium/verify: reference={reference} "
                      f"metadata was a string and not valid JSON: {meta!r}")
            meta = {}
    device_id = meta.get("device_id")
    email = _normalize_email(meta.get("email") or (data.get("customer") or {}).get("email"))

    if paystack_status != "success":
        log.info(f"[{rid}] /premium/verify: reference={reference} status={paystack_status} (not activating)")
        return StatusResponse(is_premium=False)

    if not device_id or not email:
        log.error(f"[{rid}] /premium/verify: reference={reference} succeeded but missing device_id/email in metadata")
        raise HTTPException(status_code=500,
                             detail={"error": "payment verified but device_id/email missing on record",
                                     "request_id": rid})

    now = dt.datetime.now(dt.timezone.utc)
    expires = now + dt.timedelta(days=PREMIUM_DURATION_DAYS)

    async with _pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT reference, expires_at FROM premium_subscriptions WHERE email = $1", email)
        if existing and existing["reference"] == reference:
            expires = existing["expires_at"]
        else:
            await conn.execute(
                "INSERT INTO premium_subscriptions "
                "(email, device_id, reference, purchased_at, expires_at, status, restore_used) "
                "VALUES ($1, $2, $3, $4, $5, 'active', FALSE) "
                "ON CONFLICT (email) DO UPDATE SET "
                "device_id = EXCLUDED.device_id, reference = EXCLUDED.reference, "
                "purchased_at = EXCLUDED.purchased_at, expires_at = EXCLUDED.expires_at, "
                "status = 'active', restore_used = FALSE",
                email, device_id, reference, now, expires,
            )

    log.info(f"[{rid}] /premium/verify: email={email} device={device_id} reference={reference} "
             f"ACTIVATED until {expires.isoformat()}")
    return StatusResponse(is_premium=True, expires_at=expires.isoformat())


async def premium_status(device_id, rid):
    _require_pool(rid)
    device_id = (device_id or "").strip()
    if not device_id:
        raise HTTPException(status_code=400, detail={"error": "device_id is required", "request_id": rid})

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at FROM premium_subscriptions WHERE device_id = $1", device_id)

    if not row:
        return StatusResponse(is_premium=False)

    now = dt.datetime.now(dt.timezone.utc)
    expires_at = row["expires_at"]
    if expires_at <= now:
        return StatusResponse(is_premium=False, expires_at=expires_at.isoformat())

    return StatusResponse(is_premium=True, expires_at=expires_at.isoformat())


async def premium_restore(payload, rid):
    _require_pool(rid)
    device_id = (payload.device_id or "").strip()
    email = _normalize_email(payload.email)
    if not device_id:
        raise HTTPException(status_code=400, detail={"error": "device_id is required", "request_id": rid})
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail={"error": "a valid email is required", "request_id": rid})

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT expires_at, restore_used FROM premium_subscriptions WHERE email = $1", email)
        if not row:
            log.info(f"[{rid}] /premium/restore: email={email} - no premium record found")
            raise HTTPException(status_code=404,
                                 detail={"error": "no premium purchase found for this email", "request_id": rid})

        now = dt.datetime.now(dt.timezone.utc)
        if row["expires_at"] <= now:
            log.info(f"[{rid}] /premium/restore: email={email} - premium already expired")
            raise HTTPException(status_code=410,
                                 detail={"error": "premium for this email has already expired", "request_id": rid})

        if row["restore_used"]:
            log.info(f"[{rid}] /premium/restore: email={email} - restore already used")
            raise HTTPException(status_code=409,
                                 detail={"error": "restore already used - contact support", "request_id": rid})

        await conn.execute(
            "UPDATE premium_subscriptions SET device_id = $2, restore_used = TRUE WHERE email = $1",
            email, device_id,
        )
        expires_at = row["expires_at"]

    log.info(f"[{rid}] /premium/restore: email={email} re-linked to device={device_id} until {expires_at.isoformat()}")
    return StatusResponse(is_premium=True, expires_at=expires_at.isoformat())


async def premium_reset(email, rid):
    _require_pool(rid)
    email = _normalize_email(email)
    if not email:
        raise HTTPException(status_code=400, detail={"error": "email is required", "request_id": rid})

    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM premium_subscriptions WHERE email = $1", email)

    log.warning(f"[{rid}] /premium/reset: email={email} - {result}")
    return StatusResponse(is_premium=False)
