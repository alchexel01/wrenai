"""
Wren feedback
=============
Thumbs up / thumbs down (and, later, a "Report" reason) on AI replies.

Stored in the same Postgres database as premium / chats (it re-uses the
connection pool premium.init_db() already opens, exactly like
premium_google.py does), so there is nothing new to configure on Render.

Routes (wired up in main.py):
    POST /feedback        - app sends one rating for one AI reply
    GET  /admin/feedback  - you read what came in (needs X-Admin-Key)
"""

import hashlib
import logging
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

import premium

log = logging.getLogger("wren-backend")

_RATINGS = {"up", "down", "none"}   # 'none' = the user took their thumb back
_REASONS = {"", "offensive", "harmful", "inaccurate", "other"}


class FeedbackRequest(BaseModel):
    chat_id: str = ""
    msg_index: int = -1
    rating: str                 # 'up' | 'down' | 'none'
    reason: str = ""            # for the thumbs-down / report sheet
    comment: str = ""
    ai_text: str = ""           # the reply being rated (capped below)
    user_text: str = ""         # the prompt that produced it (capped below)
    model: str = ""
    app_version: str = ""


class FeedbackOk(BaseModel):
    ok: bool = True


def _pool():
    pool = getattr(premium, "_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database not ready")
    return pool


async def init_db():
    """Create the table. Call after premium.init_db() (needs its pool)."""
    async with _pool().acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS message_feedback (
                id          BIGSERIAL PRIMARY KEY,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                user_key    TEXT NOT NULL,
                email       TEXT NOT NULL DEFAULT '',
                chat_id     TEXT NOT NULL DEFAULT '',
                msg_key     TEXT NOT NULL,
                rating      TEXT NOT NULL,
                reason      TEXT NOT NULL DEFAULT '',
                comment     TEXT NOT NULL DEFAULT '',
                ai_text     TEXT NOT NULL DEFAULT '',
                user_text   TEXT NOT NULL DEFAULT '',
                model       TEXT NOT NULL DEFAULT '',
                app_version TEXT NOT NULL DEFAULT '',
                UNIQUE (user_key, chat_id, msg_key)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS message_feedback_rating_idx
            ON message_feedback (rating, created_at DESC)
        """)
    log.info("[startup] feedback table ready")


def _msg_key(p: FeedbackRequest) -> str:
    # Same reply -> same key, so tapping up, then down, then up again on one
    # reply updates a single row instead of piling up duplicates.
    basis = (p.ai_text or "")[:2000] or f"{p.chat_id}:{p.msg_index}"
    return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:16]


async def feedback_save(p: FeedbackRequest, user_key: str, email: str,
                        rid: str) -> FeedbackOk:
    rating = (p.rating or "").strip().lower()
    reason = (p.reason or "").strip().lower()
    if rating not in _RATINGS:
        raise HTTPException(status_code=400,
                            detail={"error": "rating must be up, down or none",
                                    "request_id": rid})
    if reason not in _REASONS:
        reason = "other"

    chat_id = (p.chat_id or "")[:80]
    mk = _msg_key(p)

    async with _pool().acquire() as conn:
        if rating == "none":
            await conn.execute(
                "DELETE FROM message_feedback "
                "WHERE user_key=$1 AND chat_id=$2 AND msg_key=$3",
                user_key, chat_id, mk)
            log.info(f"[{rid}] feedback cleared ({user_key})")
            return FeedbackOk()

        await conn.execute(
            """
            INSERT INTO message_feedback
                (user_key, email, chat_id, msg_key, rating, reason, comment,
                 ai_text, user_text, model, app_version)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (user_key, chat_id, msg_key) DO UPDATE SET
                rating      = EXCLUDED.rating,
                reason      = EXCLUDED.reason,
                comment     = EXCLUDED.comment,
                model       = EXCLUDED.model,
                app_version = EXCLUDED.app_version,
                updated_at  = now()
            """,
            user_key, email, chat_id, mk, rating, reason,
            (p.comment or "")[:1000],
            (p.ai_text or "")[:6000],
            (p.user_text or "")[:2000],
            (p.model or "")[:80],
            (p.app_version or "")[:40])
    log.info(f"[{rid}] feedback saved: {rating} {reason or '-'} ({user_key})")
    return FeedbackOk()


async def feedback_list(rating: str = "", reason: str = "",
                        limit: int = 50) -> dict:
    limit = max(1, min(int(limit or 50), 200))
    where, args = [], []
    if rating in ("up", "down"):
        args.append(rating)
        where.append(f"rating = ${len(args)}")
    if reason:
        args.append(reason)
        where.append(f"reason = ${len(args)}")
    sql = ("SELECT id, created_at, updated_at, email, user_key, chat_id, rating, "
           "reason, comment, model, app_version, user_text, ai_text "
           "FROM message_feedback")
    if where:
        sql += " WHERE " + " AND ".join(where)
    args.append(limit)
    sql += f" ORDER BY updated_at DESC LIMIT ${len(args)}"

    async with _pool().acquire() as conn:
        rows = await conn.fetch(sql, *args)
        counts = await conn.fetch(
            "SELECT rating, count(*) AS n FROM message_feedback GROUP BY rating")
    return {
        "counts": {r["rating"]: r["n"] for r in counts},
        "items": [
            {**{k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in dict(r).items() if k not in ("user_text", "ai_text")},
             "user_text": (r["user_text"] or "")[:400],
             "ai_text": (r["ai_text"] or "")[:800]}
            for r in rows
        ],
    }
