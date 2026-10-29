"""
usage.py - server-authoritative token budgets for Wren
======================================================

The REAL token counts live here, on the backend. The app only ever shows a
SCALED copy ("display tokens"), so editing the app's local file, memory or
APK changes nothing that matters.

Everything is controlled by Render environment variables (all optional):

    WREN_FREE_TOKENS_REAL            real free chat tokens / window      (20000)
    WREN_STUDY_FREE_TOKENS_REAL      real free Study Mode tokens / window (10000)
    WREN_STUDY_PREMIUM_TOKENS_REAL   real premium Study Mode tokens       (30000)

    WREN_FREE_TOKENS_SHOWN           what the app displays as the limit   (20000)
    WREN_STUDY_FREE_TOKENS_SHOWN     "                                     (10000)
    WREN_STUDY_PREMIUM_TOKENS_SHOWN  "                                     (30000)

    WREN_USAGE_WINDOW_SECS           window length, seconds               (86400)
    WREN_REQUIRE_SIGNIN_FOR_CHAT     1 = /chat needs a valid sign-in      (1)
    WREN_USAGE_DB_URL                Postgres URL (falls back to DATABASE_URL);
                                     if neither is set, a local SQLite file is
                                     used (wiped on every Render redeploy
                                     unless you attach a persistent disk).

Calibration
-----------
    shown_used = real_used / real_limit * shown_limit

So with REAL=30000 and SHOWN=20000 the app's bar reaches 0 exactly when the
backend's real budget is spent, and the user is never cut off early. Change
REAL up or down on Render and the displayed numbers re-scale automatically;
no app update needed.
"""

import os
import re
import json
import time
import asyncio
import sqlite3
import logging
import threading
from typing import Optional, Tuple

log = logging.getLogger("wren-backend")


def _env_int(name: str, default: int) -> int:
    try:
        v = int(str(os.environ.get(name, "")).strip() or default)
        return v if v > 0 else default
    except Exception:
        return default


def _cfg() -> dict:
    """Read env on every call so a Render env edit + restart is all it takes."""
    return {
        "window": _env_int("WREN_USAGE_WINDOW_SECS", 86400),
        "chat_free":     (_env_int("WREN_FREE_TOKENS_REAL", 20000),
                          _env_int("WREN_FREE_TOKENS_SHOWN", 20000)),
        "study_free":    (_env_int("WREN_STUDY_FREE_TOKENS_REAL", 10000),
                          _env_int("WREN_STUDY_FREE_TOKENS_SHOWN", 10000)),
        "study_premium": (_env_int("WREN_STUDY_PREMIUM_TOKENS_REAL", 30000),
                          _env_int("WREN_STUDY_PREMIUM_TOKENS_SHOWN", 30000)),
    }


def require_signin() -> bool:
    return os.environ.get("WREN_REQUIRE_SIGNIN_FOR_CHAT", "1").strip() != "0"


# ── Pool resolution ──────────────────────────────────────────────────────
def resolve_pool(requested: str, is_premium: bool) -> Tuple[Optional[str], int, int]:
    """Map (what the app asked for, server-verified premium) to
    (pool_name, real_limit, shown_limit). pool_name None = unmetered
    (premium normal chat)."""
    c = _cfg()
    if (requested or "").strip().lower() == "study":
        key = "study_premium" if is_premium else "study_free"
        real, shown = c[key]
        return key, real, shown
    if is_premium:
        return None, 0, 0
    real, shown = c["chat_free"]
    return "chat_free", real, shown


def to_shown(real_used: int, real_limit: int, shown_limit: int) -> int:
    """Scale real usage onto the displayed limit. The bar only reads 'full'
    when the real budget is truly spent, never earlier."""
    if real_limit <= 0:
        return 0
    if real_used >= real_limit:
        return shown_limit
    s = int(real_used * shown_limit / real_limit)
    return max(0, min(shown_limit - 1, s))


# ── Storage (Postgres if available, else SQLite) ─────────────────────────
_DDL = ("CREATE TABLE IF NOT EXISTS wren_usage ("
        "user_key TEXT NOT NULL, pool TEXT NOT NULL, "
        "used BIGINT NOT NULL DEFAULT 0, window_start DOUBLE PRECISION NOT NULL, "
        "PRIMARY KEY (user_key, pool))")

_pg = None            # asyncpg pool, if in use
_sqlite = None        # sqlite3 connection, if in use
_sqlite_lock = threading.Lock()


async def init_db():
    global _pg, _sqlite
    url = (os.environ.get("WREN_USAGE_DB_URL") or os.environ.get("DATABASE_URL") or "").strip()
    if url.startswith("postgres"):
        try:
            import asyncpg  # type: ignore
            _pg = await asyncpg.create_pool(url, min_size=1, max_size=3)
            async with _pg.acquire() as c:
                await c.execute(_DDL)
            log.info("[usage] using Postgres")
            return
        except Exception as e:
            log.error(f"[usage] Postgres unavailable ({e!r}); falling back to SQLite")
            _pg = None
    path = os.environ.get("WREN_USAGE_SQLITE", "wren_usage.sqlite3")
    _sqlite = sqlite3.connect(path, check_same_thread=False)
    _sqlite.execute(_DDL.replace("BIGINT", "INTEGER").replace("DOUBLE PRECISION", "REAL"))
    _sqlite.commit()
    log.warning(f"[usage] using SQLite at {path} (not persistent on Render without a disk)")


async def close_db():
    global _pg, _sqlite
    if _pg is not None:
        await _pg.close()
        _pg = None
    if _sqlite is not None:
        _sqlite.close()
        _sqlite = None


def _sqlite_op(fn):
    with _sqlite_lock:
        return fn(_sqlite)


async def _read(user_key: str, pool: str, now: float, window: int) -> Tuple[int, float]:
    """(used, window_start), rolling the window over if it has expired."""
    if _pg is not None:
        async with _pg.acquire() as c:
            row = await c.fetchrow(
                "SELECT used, window_start FROM wren_usage WHERE user_key=$1 AND pool=$2",
                user_key, pool)
            if row is None or now - row["window_start"] >= window:
                await c.execute(
                    "INSERT INTO wren_usage(user_key,pool,used,window_start) VALUES($1,$2,0,$3) "
                    "ON CONFLICT (user_key,pool) DO UPDATE SET used=0, window_start=$3",
                    user_key, pool, now)
                return 0, now
            return int(row["used"]), float(row["window_start"])

    def _do(db):
        r = db.execute("SELECT used, window_start FROM wren_usage WHERE user_key=? AND pool=?",
                       (user_key, pool)).fetchone()
        if r is None or now - r[1] >= window:
            db.execute("INSERT OR REPLACE INTO wren_usage(user_key,pool,used,window_start) "
                       "VALUES(?,?,0,?)", (user_key, pool, now))
            db.commit()
            return 0, now
        return int(r[0]), float(r[1])
    return await asyncio.to_thread(_sqlite_op, _do)


async def _add(user_key: str, pool: str, delta: int, now: float, window: int):
    await _read(user_key, pool, now, window)          # make sure the row/window is current
    if _pg is not None:
        async with _pg.acquire() as c:
            await c.execute("UPDATE wren_usage SET used = GREATEST(0, used + $3) "
                            "WHERE user_key=$1 AND pool=$2", user_key, pool, int(delta))
        return

    def _do(db):
        db.execute("UPDATE wren_usage SET used = MAX(0, used + ?) WHERE user_key=? AND pool=?",
                   (int(delta), user_key, pool))
        db.commit()
    await asyncio.to_thread(_sqlite_op, _do)


# ── Public API used by main.py ───────────────────────────────────────────
async def snapshot(user_key: str, requested_pool: str, is_premium: bool) -> dict:
    """Current state for one user+pool, in the shape the app needs."""
    pool, real_limit, shown_limit = resolve_pool(requested_pool, is_premium)
    if pool is None:
        return {"pool": "chat_premium", "unlimited": True, "exhausted": False,
                "shown_used": 0, "shown_limit": 0, "shown_left": 0,
                "reset_in": 0, "window_secs": _cfg()["window"]}
    now, window = time.time(), _cfg()["window"]
    used, start = await _read(user_key, pool, now, window)
    shown_used = to_shown(used, real_limit, shown_limit)
    return {
        "pool": pool, "unlimited": False, "exhausted": used >= real_limit,
        "shown_used": shown_used, "shown_limit": shown_limit,
        "shown_left": shown_limit - shown_used,
        "reset_in": max(0, int(window - (now - start))), "window_secs": window,
        # real numbers stay server-side; only logged, never returned
        "_real_used": used, "_real_limit": real_limit,
    }


async def charge(user_key: str, requested_pool: str, is_premium: bool, real_tokens: int):
    pool, _, _ = resolve_pool(requested_pool, is_premium)
    if pool is None or real_tokens <= 0:
        return
    await _add(user_key, pool, int(real_tokens), time.time(), _cfg()["window"])


def public(snap: dict) -> dict:
    """Strip server-only fields before sending to the app."""
    return {k: v for k, v in snap.items() if not k.startswith("_")}


# ── Real usage from Groq's response ──────────────────────────────────────
class UsageTap:
    """Feed it every raw byte chunk of the upstream reply (SSE or plain
    JSON). At the end, .total() gives Groq's real total_tokens when present,
    otherwise a conservative estimate from what actually streamed."""

    _TOTAL_RE = re.compile(rb'"total_tokens"\s*:\s*(\d+)')
    _CONTENT_RE = re.compile(rb'"content"\s*:\s*"((?:[^"\\]|\\.)*)"')

    def __init__(self, request_body: dict):
        self._tail = b""
        self._real: Optional[int] = None
        self._content_chars = 0
        try:
            self._req_chars = len(json.dumps(request_body.get("messages", []), ensure_ascii=False))
        except Exception:
            self._req_chars = 0

    def feed(self, chunk: bytes):
        buf = self._tail + chunk
        # Groq may nest usage (x_groq.usage, per-model breakdowns for compound);
        # the largest total_tokens seen is the request's overall total.
        for m in self._TOTAL_RE.finditer(buf):
            v = int(m.group(1))
            if self._real is None or v > self._real:
                self._real = v
        for m in self._CONTENT_RE.finditer(chunk):
            self._content_chars += len(m.group(1))
        self._tail = buf[-600:]       # a usage object can straddle two chunks

    def total(self) -> int:
        if self._real is not None:
            return self._real
        return max(1, (self._req_chars + self._content_chars) // 4)
