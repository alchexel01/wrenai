"""
Wren Syllabus Backend
=======================
Small FastAPI service that serves JAMB syllabus grounding over HTTP,
so subject content can be updated/added without shipping a new APK.

Run locally:
    pip install fastapi uvicorn
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                     — liveness check
    GET  /subjects                   — list loaded subjects
    GET  /exam-bodies                — list exam bodies with data loaded, each with its subjects
    POST /rag/context                — get grounding text for a query
    POST /admin/reload               — re-scan syllabus_data/ (needs API key)
    POST /premium/initialize         — start a Paystack transaction, get checkout URL
    GET  /premium/verify/{reference} — confirm payment, activate premium for 365 days
    GET  /premium/status/{device_id} — is this email premium (on any device), until when
    POST /premium/restore            — re-link an email's premium to a new device_id (no limit)
    POST /premium/google/verify      — verify a Google Play subscription token, activate premium
    DELETE /premium/reset/{email}    — TESTING ONLY: wipe an email's premium record
    GET  /auth/google/start          — get a Google sign-in URL + session_id
    GET  /auth/google/callback       — Google redirects here after sign-in
    POST /auth/google/confirm        — browser page submits the code shown in the app
    POST /auth/email/request         — email a 6-digit sign-in code
    POST /auth/email/verify          — enter the code: returns email + token
    GET  /auth/google/status/{id}    — poll: has this session's sign-in completed?
    POST /chats/save                 — upsert one chat for an email
    GET  /chats/{email}              — fetch every chat saved for an email
    DELETE /chats/{email}/{chat_id}  — delete one chat
    DELETE /chats/{email}            — delete every chat for an email
    POST /feedback                   — thumbs up/down on an AI reply
    GET  /admin/feedback             — read feedback (needs admin key)
    POST /chat                       — proxies to Groq chat completions
    POST /transcribe                 — proxies to Groq Whisper transcription
    POST /image                      — safe image generation (filters the prompt, proxies to Pollinations)
    GET  /model/{key}                — streams an offline model file from HF
    GET  /scholarships               — scholarships from the repo's scholarships/ folder
    GET  /scholarships/{id}          — one scholarship
    GET  /news                       — news and updates from the repo's news/ folder
    GET  /news/{id}                  — one news item
    GET  /past-questions/papers      — which subjects/years have past questions loaded
    GET  /past-questions             — questions for one paper (?subject=&year=&limit=&offset=&shuffle=)
    GET  /past-questions/images/{subject}/{file} — one diagram picture (PNG)
    POST /past-questions/flag        — the app reports a question that looks wrong
    GET  /past-questions/flags       — (admin) flagged questions, most-reported first, with fix templates
    DELETE /past-questions/flags/{id} — (admin) clear a question's reports after fixing it

See README.md for deployment and how AI.py should call this.
"""

import os
import sys
import time
import uuid
import asyncio
import logging
import traceback
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel
from typing import Optional
import httpx

import rag_engine
import premium
import premium_google
import auth
import auth_email
import auth_review
import chat_history
import textbooks_api
import scholarships
import news
import past_questions
import usage
import feedback

app = FastAPI(title="Wren Syllabus Backend", version="1.0")


@app.on_event("startup")
async def _startup():
    await premium.init_db()
    await feedback.init_db()          # re-uses premium's DB pool
    await chat_history.init_db()
    await auth.init_db()
    await auth_email.init_tables()
    await usage.init_db()


@app.on_event("shutdown")
async def _shutdown():
    await premium.close_db()
    await chat_history.close_db()
    await auth.close_db()
    await usage.close_db()

# ── Error / crash logging ────────────────────────────────────────────────
# Every request gets a short request_id. It's included in:
#   - every log line for that request (so you can grep one request's
#     whole story out of Render's log stream),
#   - every HTTPException/error JSON body sent back to the app, and
#   - AI.py's record_error() calls already tag entries with a 'source'
#     string (e.g. 'Chat/http-error') — pair that with the request_id
#     printed here and you can match a client-side log line to the
#     exact backend request that produced it.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,  # Render captures stdout as the service log
)
log = logging.getLogger("wren-backend")


def _rid() -> str:
    return uuid.uuid4().hex[:8]


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """Wraps every request: logs entry/exit/timing, and — critically —
    catches any exception that escapes a route handler. Without this,
    an unhandled exception in a route (bug, upstream surprise, etc.)
    just becomes an opaque 500 with an empty body and nothing in the
    logs pointing at where it happened. This guarantees every failure
    is logged with a traceback, a request_id, and which route it came
    from, before FastAPI has any chance to swallow it."""
    rid = _rid()
    request.state.rid = rid
    start = time.time()
    log.info(f"[{rid}] --> {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception:
        dur_ms = int((time.time() - start) * 1000)
        tb = traceback.format_exc()
        log.error(
            f"[{rid}] !! UNHANDLED EXCEPTION in {request.method} "
            f"{request.url.path} after {dur_ms}ms\n{tb}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "internal server error",
                "request_id": rid,
                "route": request.url.path,
            },
        )
    dur_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = rid
    log.info(
        f"[{rid}] <-- {request.method} {request.url.path} "
        f"{response.status_code} ({dur_ms}ms)"
    )
    return response

# Simple shared-secret admin key for the reload endpoint, so random
# internet traffic can't trigger reloads. Set this as an environment
# variable on whatever host you deploy to — never hardcode a real
# secret in the file itself.
ADMIN_KEY = os.environ.get("WREN_ADMIN_KEY", "")

# Shared password between the phone app and this backend. NOT a real
# provider key — just stops random internet traffic from riding on
# your Groq/HF credentials for free. Rotate this any time by changing
# the env var on Render; the app needs the same value set on its side
# (see WREN_APP_SECRET in AI.py).
#
# .strip() guards against the single most common cause of "the values
# look identical but auth still fails": Render's dashboard textbox (or
# a copy-paste) silently including a trailing space or newline in the
# saved env var. Without stripping, "secret" and "secret\n" compare as
# different strings even though they render identically on screen.
APP_SECRET = os.environ.get("WREN_APP_SECRET", "").strip()

# Real provider credentials. These never leave the server.
#
# Preferred: set GROQ_API_KEYS on Render to a comma-separated list of
# every key you have — "key_one,key_two,key_three". To add an 11th
# key later, just add it to this list and restart the Render service;
# no code change needed.
#
# Also supported for backwards compatibility: individually numbered
# GROQ_API_KEY_1, GROQ_API_KEY_2, ... vars (gaps are fine). Any keys
# found this way are merged in with GROQ_API_KEYS, de-duplicated.
GROQ_API_KEYS_ENV_NAME = "GROQ_API_KEYS"
GROQ_API_KEY_NUMBERED_ENV_NAMES = [f"GROQ_API_KEY_{i}" for i in range(1, 11)]
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN", "")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class GroqKeyPool:
    """Sticky round-robin pool over any number of Groq API keys.

    "Sticky" means we keep using the same key across requests until it
    actually fails — we don't shuffle keys on every request for no
    reason. When a key comes back 429 (rate limited) or 401/403 (bad/
    revoked key), that key is put on a cooldown timer and we instantly
    advance to the next live key, so a single exhausted key never
    stalls the app — it just quietly keeps going on the next one.
    """

    # How long to skip a key that just got rate-limited, if Groq didn't
    # tell us via Retry-After. Long enough to matter, short enough that
    # a key that recovers isn't left idle for the rest of the day.
    DEFAULT_COOLDOWN_SECS = 60.0
    # 401/403 usually means the key itself is bad (revoked/typo'd), not
    # a transient limit — cool it down much longer so we don't keep
    # hammering a dead key every rotation.
    AUTH_FAILURE_COOLDOWN_SECS = 3600.0

    def __init__(self, list_env_name: str, numbered_env_names: list[str]):
        self.keys: list[str] = []
        seen: set[str] = set()

        # Primary source: one comma-separated env var. Adding a new
        # key later is just "edit this env var, restart" — no touching
        # this file.
        raw = os.environ.get(list_env_name, "")
        for part in raw.split(","):
            val = part.strip()
            if val and val not in seen:
                self.keys.append(val)
                seen.add(val)

        # Back-compat source: individually numbered vars, merged in on
        # top of anything already found above.
        for name in numbered_env_names:
            val = os.environ.get(name, "").strip()
            if val and val not in seen:
                self.keys.append(val)
                seen.add(val)

        self._idx = 0
        self._cooldown_until: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def __len__(self):
        return len(self.keys)

    async def current(self) -> Optional[str]:
        """Return a currently-usable key without advancing the pool."""
        async with self._lock:
            return self._pick_locked()

    def _pick_locked(self) -> Optional[str]:
        if not self.keys:
            return None
        now = time.time()
        n = len(self.keys)
        # Prefer the key at self._idx if it's live; otherwise scan
        # forward for the next one that's off cooldown.
        for offset in range(n):
            idx = (self._idx + offset) % n
            key = self.keys[idx]
            if self._cooldown_until.get(key, 0.0) <= now:
                self._idx = idx
                return key
        # Every key is cooling down (e.g. all 10 rate-limited at once)
        # — fall back to whichever one recovers soonest rather than
        # failing outright.
        soonest = min(self.keys, key=lambda k: self._cooldown_until.get(k, 0.0))
        self._idx = self.keys.index(soonest)
        return soonest

    async def advance(self, bad_key: Optional[str] = None, cooldown_secs: Optional[float] = None):
        """Mark `bad_key` as cooling down and move on to the next key."""
        async with self._lock:
            if bad_key is not None and cooldown_secs is not None:
                self._cooldown_until[bad_key] = time.time() + cooldown_secs
            if self.keys:
                self._idx = (self._idx + 1) % len(self.keys)


groq_pool = GroqKeyPool(GROQ_API_KEYS_ENV_NAME, GROQ_API_KEY_NUMBERED_ENV_NAMES)
log.info(f"[startup] Groq key pool: {len(groq_pool)} key(s) loaded")

# Registry of offline model files the app can download. Mirrors
# OFFLINE_MODELS in AI.py — add entries here if you add them there.
OFFLINE_MODELS = {
    "gemma3_1b": {
        "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/"
               "resolve/main/gemma3-1b-it-int4.litertlm",
    },
}


import hashlib


def _fingerprint(s: str) -> str:
    """Never logs the actual secret — just enough to compare two
    values without ever printing either one: length + a short hash
    prefix. Two equal secrets always produce an identical fingerprint;
    two secrets that merely *look* the same (e.g. one has a trailing
    space, or was truncated on paste) will show either a different
    length or a different hash, which is the tell."""
    if not s:
        return "EMPTY"
    return f"len={len(s)} sha256={hashlib.sha256(s.encode()).hexdigest()[:10]}"


log.info(f"[startup] WREN_APP_SECRET fingerprint: {_fingerprint(APP_SECRET)}")


def _check_app_secret(x_app_secret: str, rid: str = "-"):
    """Every proxy route requires the shared app secret. Without this,
    anyone who finds this URL could spend your Groq/HF quota for free.
    Logged explicitly so a rotated/mismatched WREN_APP_SECRET shows up
    as a clear 401 in the logs instead of looking like a generic
    network failure on the client."""
    incoming = (x_app_secret or "").strip()
    if not APP_SECRET or incoming != APP_SECRET:
        log.warning(
            f"[{rid}] auth failed: X-App-Secret mismatch. "
            f"server={_fingerprint(APP_SECRET)} "
            f"received={_fingerprint(incoming)}"
        )
        raise HTTPException(status_code=401,
                             detail={"error": "invalid or missing app secret", "request_id": rid})


class ContextRequest(BaseModel):
    query: str
    subject: Optional[str] = None   # e.g. "Biology" — omit to search all subjects


class ContextResponse(BaseModel):
    context: str
    matched: bool


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(rag_engine.syllabus_rag.chunks)}


@app.get("/subjects")
def subjects():
    return {"subjects": rag_engine.list_subjects()}


@app.get("/exam-bodies")
def exam_bodies():
    """One entry per exam body that actually has syllabus data loaded
    (e.g. JAMB), each with the subjects available for it. An exam body
    with no chunks loaded simply doesn't appear — the client renders
    whatever comes back as-is, no "coming soon" placeholder needed."""
    return {"exam_bodies": rag_engine.list_exam_bodies()}


@app.post("/rag/context", response_model=ContextResponse)
def rag_context(req: ContextRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    context = rag_engine.get_context_for(req.query, subject=req.subject)
    matched = "NO MATCH" not in context
    return ContextResponse(context=context, matched=matched)


@app.post("/admin/reload")
def admin_reload(x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing admin key")
    n = rag_engine.reload_all()
    return {"reloaded": True, "chunks_loaded": n}


# ── Premium (Paystack) ───────────────────────────────────────────────────
# One-time N3,000 / 365-day purchase. Not a subscription — see
# premium.py for the full design notes and why Postgres (not this
# service's own disk) is the store of record.

@app.post("/premium/initialize", response_model=premium.InitializeResponse)
async def premium_initialize_route(
    payload: premium.InitializeRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    # The purchase gets attached to this email, so the caller must be signed
    # in as it - otherwise anyone could start (and lock in) a purchase for
    # somebody else's address.
    auth.require_user(payload.email, x_auth_token, rid)
    return await premium.premium_initialize(payload, rid)


@app.get("/premium/verify/{reference}", response_model=premium.StatusResponse)
async def premium_verify_route(
    reference: str,
    req: Request,
    email: str = "",
    device_id: str = "",
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_verify(reference, rid, caller_email=email,
                                        caller_device_id=device_id)


@app.get("/premium/status/{device_id}", response_model=premium.StatusResponse)
async def premium_status_route(
    device_id: str,
    req: Request,
    email: str = "",
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    # Google Play subscriptions renew, lapse and get refunded without the app
    # telling us: re-read this user's subscription from Google when it is due
    # (cheap no-op otherwise), then answer from the up-to-date row.
    if email:
        await premium_google.refresh_if_due(email, rid)
    res = await premium.premium_status(device_id, rid, email=email)
    if email and res.is_premium:
        res.auto_renewing = await premium_google.auto_renewing_for(email)
    return res


@app.post("/premium/restore", response_model=premium.StatusResponse)
async def premium_restore_route(
    payload: premium.RestoreRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    """Re-links an email's existing premium purchase to whatever device_id
    is asking, with no limit. Premium follows the signed-in email, so this
    is no longer needed to unlock a new device; see premium.py."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_restore(payload, rid)


@app.post("/premium/google/verify")
async def premium_google_verify_route(
    payload: premium_google.GoogleVerifyRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    """Google Play Billing: the app sends the purchase token Google gave it;
    we verify it with the Google Play Developer API and activate premium.
    See premium_google.py for the full contract (200/202/400/409/503)."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    # Premium is granted to this email, so the caller must be signed in as it.
    auth.require_user(payload.email, x_auth_token, rid)
    res = await premium_google.google_verify(payload, rid)
    if isinstance(res, premium.StatusResponse) and res.is_premium:
        res.auto_renewing = await premium_google.auto_renewing_for(payload.email)
    return res


@app.post("/premium/google/cancel")
async def premium_google_cancel_route(
    payload: premium_google.GoogleCancelRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    """In-app cancel: stops auto-renewal via the Google Play Developer API.
    Premium stays active until the paid period ends. Needs the service
    account to have 'Manage orders and subscriptions' in Play Console."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    auth.require_user(payload.email, x_auth_token, rid)
    return await premium_google.google_cancel(payload, rid)


@app.delete("/premium/reset/{email}", response_model=premium.StatusResponse)
async def premium_reset_route(
    email: str,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    """TESTING ONLY — wipes any premium record for an email so the
    purchase flow can be re-run from scratch. Same X-App-Secret gate as
    every other route here; nothing extra-locked-down about it, so
    don't rely on this being hidden from anyone who has the app
    secret — it's meant for you during development, not as a
    production admin feature."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    auth.require_user(email, x_auth_token, rid)   # only your own record, even when enabled
    return await premium.premium_reset(email, rid)


# ── Google Sign-In ────────────────────────────────────────────────────
# App opens a browser to /auth/google/start's authorization_url, Google
# redirects back to /auth/google/callback on this backend, and the app
# polls /auth/google/status/{session_id} until the verified email shows
# up — same shape as the Paystack initialize/verify polling above. The
# user also types the code shown in the app into the browser page
# (/auth/google/confirm) before the session is released.
# See auth.py for the full design notes.

@app.get("/auth/google/start", response_model=auth.AuthStartResponse)
async def auth_google_start_route(req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await auth.auth_google_start(rid)


@app.get("/auth/google/callback")
async def auth_google_callback_route(req: Request, code: str = "", state: str = "",
                                     error: str = ""):
    # No X-App-Secret here on purpose — Google itself calls this URL
    # via browser redirect, not the app, so it can't attach that
    # header. Security instead comes from (a) the session having been
    # created by /start, (b) verifying the ID token's signature
    # server-side in auth.py, and (c) the confirmation code step.
    # `code`/`error` are optional because Google sends ?error=access_denied
    # (no code) when the user taps Cancel.
    rid = req.state.rid
    return await auth.auth_google_callback(code, state, rid, error=error)


@app.post("/auth/google/confirm")
async def auth_google_confirm_route(req: Request, state: str = Form(default=""),
                                    code: str = Form(default="")):
    # Submitted by the HTML page the callback returns, from the user's
    # browser — so also no X-App-Secret. The session_id (state) is an
    # unguessable 32-char token and wrong codes burn the session.
    rid = req.state.rid
    return await auth.auth_google_confirm(state, code, rid)


@app.get("/auth/google/status/{session_id}", response_model=auth.AuthStatusResponse)
async def auth_google_status_route(session_id: str, req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await auth.auth_google_status(session_id, rid)


# ── Email sign-in with a 6-digit code ────────────────────────────────
# Alternative to Google. /request emails a code, /verify swaps the right code
# for the same signed token Google sign-in returns. No passwords. See
# auth_email.py.

@app.post("/auth/email/request", response_model=auth_email.EmailOkResponse)
async def auth_email_request_route(payload: auth_email.EmailRequest, req: Request,
                                   x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await auth_email.email_request(payload, rid)


@app.post("/auth/email/verify", response_model=auth_email.EmailTokenResponse)
async def auth_email_verify_route(payload: auth_email.EmailVerifyRequest, req: Request,
                                  x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await auth_email.email_verify(payload, rid)


# ── Google Play review login (fixed account + password kept on Render) ───
# See auth_review.py. Off unless REVIEW_PASSWORD is set on Render.

@app.post("/auth/review/login", response_model=auth_review.ReviewLoginResponse)
async def auth_review_login_route(payload: auth_review.ReviewLoginRequest, req: Request,
                                  x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    res = await auth_review.review_login(payload, rid)
    _PREMIUM_CACHE.pop(res.email, None)     # usage gate must see Pro at once
    return res


# ── Chat history (email-scoped sync) ─────────────────────────────────────
# Ties every saved conversation to the signed-in email, same identity
# model as premium above. See chat_history.py for the full design
# notes and why Postgres (not this service's own disk) is the store
# of record.

@app.post("/chats/save", response_model=chat_history.OkResponse)
async def chats_save_route(
    payload: chat_history.ChatSaveRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    # X-App-Secret ships inside every APK, so it only proves "this is the
    # Wren app". The sign-in token proves WHICH USER, and must match the
    # email whose chats are being written.
    auth.require_user(getattr(payload, "email", ""), x_auth_token, rid)
    return await chat_history.chats_save(payload, rid)


@app.get("/chats/{email}", response_model=chat_history.ChatListResponse)
async def chats_list_route(
    email: str,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    auth.require_user(email, x_auth_token, rid)
    return await chat_history.chats_list(email, rid)


@app.delete("/chats/{email}/{chat_id}", response_model=chat_history.OkResponse)
async def chat_delete_route(
    email: str,
    chat_id: str,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    auth.require_user(email, x_auth_token, rid)
    return await chat_history.chat_delete(email, chat_id, rid)


@app.delete("/chats/{email}", response_model=chat_history.OkResponse)
async def chats_delete_all_route(
    email: str,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    auth.require_user(email, x_auth_token, rid)
    return await chat_history.chats_delete_all(email, rid)


# ── Feedback on AI replies (thumbs up / down) ────────────────────────────
@app.post("/feedback", response_model=feedback.FeedbackOk)
async def feedback_route(
    payload: feedback.FeedbackRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
    x_user_email: str = Header(default=""),
    x_device_id: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    email = (x_user_email or "").strip().lower()
    if email and x_auth_token:
        auth.require_user(email, x_auth_token, rid)   # 401s if token != email
        user_key = f"u:{email}"
    else:
        email = ""
        user_key = f"d:{(x_device_id or '').strip()[:64] or 'unknown'}"
    return await feedback.feedback_save(payload, user_key, email, rid)


@app.get("/admin/feedback")
async def admin_feedback(
    rating: str = "",
    reason: str = "",
    limit: int = 50,
    x_admin_key: str = Header(default=""),
):
    """Read what users sent. e.g. /admin/feedback?rating=down&limit=20"""
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing admin key")
    return await feedback.feedback_list(rating, reason, limit)


# ── Server-side token budgets (see usage.py) ─────────────────────────────
# The app's own counter is only a display copy. The real balance lives in
# usage.py, is charged from Groq's real usage numbers as each reply streams
# through /chat, and is enforced here before anything is forwarded.
_PREMIUM_CACHE: dict = {}          # email -> (expires_at, is_premium)
_PREMIUM_CACHE_TTL = 60.0


async def _is_premium_cached(email: str, device_id: str, rid: str) -> bool:
    now = time.time()
    hit = _PREMIUM_CACHE.get(email)
    if hit and hit[0] > now:
        return hit[1]
    try:
        res = await premium.premium_status(device_id, rid, email=email)
        val = bool(res.is_premium)
    except Exception as e:
        # Never cut off a paying customer because a status lookup hiccuped:
        # reuse the last known answer, otherwise let the request through.
        log.error(f"[{rid}] premium lookup failed for usage gate: {e!r}")
        return bool(hit[1]) if hit else True
    _PREMIUM_CACHE[email] = (now + _PREMIUM_CACHE_TTL, val)
    return val


async def _usage_identity(email: str, token: str, device_id: str, rid: str):
    """-> (user_key, email_or_empty, is_premium). Signed-in users are keyed by
    their verified email, so reinstalling or clearing app data can't reset
    their allowance."""
    email = (email or "").strip().lower()
    if email and token:
        auth.require_user(email, token, rid)          # 401s if token doesn't match
        return f"u:{email}", email, await _is_premium_cached(email, device_id, rid)
    if usage.require_signin():
        raise HTTPException(status_code=401,
                            detail={"error": "sign in required", "request_id": rid})
    dev = (device_id or "").strip()[:64] or "unknown"
    return f"d:{dev}", "", False


def _usage_headers(snap: dict) -> dict:
    if snap.get("unlimited"):
        return {"X-Usage-Unlimited": "1"}
    return {
        "X-Usage-Pool": snap["pool"],
        "X-Usage-Shown-Used": str(snap["shown_used"]),
        "X-Usage-Shown-Limit": str(snap["shown_limit"]),
        "X-Usage-Reset-In": str(snap["reset_in"]),
    }


@app.get("/usage/me")
async def usage_me(
    req: Request,
    pool: str = "chat",
    device_id: str = "",
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
    x_user_email: str = Header(default=""),
):
    """The app calls this after every reply and on start-up to refresh its
    display. Returns only SCALED numbers; real limits never leave the server."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    user_key, _email, is_prem = await _usage_identity(x_user_email, x_auth_token, device_id, rid)
    snap = await usage.snapshot(user_key, pool, is_prem)
    return {**usage.public(snap), "is_premium": is_prem}



@app.post("/chat")
async def chat(
    request: dict,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
    x_user_email: str = Header(default=""),
    x_device_id: str = Header(default=""),
    x_usage_pool: str = Header(default="chat"),
):
    """Forwards the app's chat-completion body to Groq, attaching the
    real Groq key server-side.

    IMPORTANT: the app (AI.py) always sends {"stream": True} and reads
    the response with rsp.iter_lines(), parsing Server-Sent-Events
    ('data: {...}' lines ending in 'data: [DONE]') as they arrive.
    This route MUST therefore open a real streaming connection to Groq
    and forward each chunk to the client as it arrives — buffering the
    whole reply into resp.content and wrapping it as a single chunk
    (the previous bug here) produces a byte blob the client's SSE
    parser never recognizes as valid, the loop exits with no [DONE],
    and the app treats that as a dead backend and falls back to
    Offline Intelligence / a "No internet connection" message even
    though nothing was actually wrong with connectivity.
    """
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    if not len(groq_pool):
        log.error(f"[{rid}] /chat: server has no Groq API keys configured")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing Groq API keys", "request_id": rid})

    is_streaming_req = bool(request.get("stream"))
    model = request.get("model", "?")
    log.info(f"[{rid}] /chat: model={model} stream={is_streaming_req}")

    # ── Server-side allowance gate (real numbers, never trusts the app) ──
    user_key, _email, is_prem = await _usage_identity(x_user_email, x_auth_token, x_device_id, rid)
    pool_req = "study" if (x_usage_pool or "").strip().lower() == "study" else "chat"
    snap = await usage.snapshot(user_key, pool_req, is_prem)
    if snap.get("exhausted"):
        log.info(f"[{rid}] /chat: {user_key} pool={snap['pool']} exhausted "
                 f"(real {snap['_real_used']}/{snap['_real_limit']})")
        raise HTTPException(
            status_code=402,
            detail={"error": "usage_limit_reached", "pool": snap["pool"],
                    "reset_in": snap["reset_in"], "request_id": rid},
            headers=_usage_headers(snap),
        )


    # Try every key in the pool before giving up. A key that's rate
    # limited or rejected gets cooled down and we instantly move to the
    # next one — the caller never sees the individual key failures,
    # only the final success or (if every key is down) the final error.
    attempts = len(groq_pool)
    last_status = 502
    last_body_text = "all Groq API keys exhausted"

    for attempt in range(1, attempts + 1):
        key = await groq_pool.current()
        client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=60))
        try:
            upstream_req = client.build_request(
                "POST", GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=request,
            )
            upstream = await client.send(upstream_req, stream=True)
        except httpx.RequestError as e:
            await client.aclose()
            log.error(f"[{rid}] /chat: key #{attempt}/{attempts} network error: {e!r}")
            last_status, last_body_text = 502, f"upstream request failed: {e}"
            await groq_pool.advance()
            continue

        if upstream.status_code == 429 or upstream.status_code in (401, 403):
            body = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            body_text = body.decode("utf-8", errors="replace")[:1000]
            cooldown = GroqKeyPool.DEFAULT_COOLDOWN_SECS
            if upstream.status_code == 429:
                retry_after = upstream.headers.get("retry-after")
                if retry_after and retry_after.strip().isdigit():
                    cooldown = float(retry_after)
            else:
                cooldown = GroqKeyPool.AUTH_FAILURE_COOLDOWN_SECS
            log.warning(
                f"[{rid}] /chat: key #{attempt}/{attempts} hit HTTP "
                f"{upstream.status_code} — cooling it down {cooldown:.0f}s and "
                f"rotating to next key: {body_text}"
            )
            last_status, last_body_text = upstream.status_code, body_text
            await groq_pool.advance(bad_key=key, cooldown_secs=cooldown)
            continue

        if upstream.status_code != 200:
            # Not a key/limit issue (bad model name, invalid param,
            # etc.) — another key won't fix this, so fail immediately
            # instead of burning through the whole pool.
            body = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            body_text = body.decode("utf-8", errors="replace")[:1000]
            log.error(
                f"[{rid}] /chat: Groq returned HTTP {upstream.status_code}: {body_text}"
            )
            raise HTTPException(
                status_code=upstream.status_code,
                detail={"error": body_text, "source": "groq", "request_id": rid},
            )

        # Success on this key.
        if attempt > 1:
            log.info(f"[{rid}] /chat: succeeded on key #{attempt}/{attempts} after failover")

        tap = usage.UsageTap(request)

        async def _stream(client=client, upstream=upstream, tap=tap):
            chunk_count = 0
            byte_count = 0
            try:
                async for chunk in upstream.aiter_bytes():
                    chunk_count += 1
                    byte_count += len(chunk)
                    tap.feed(chunk)
                    yield chunk
            except Exception as e:
                # A failure mid-stream (Groq connection dropped, read
                # timeout, etc.) after the 200 status and headers have
                # already been sent to the client — we can no longer
                # change the HTTP status or switch keys at this point,
                # so log it loudly server-side; the client sees this as
                # a stream that ended without a [DONE] sentinel (AI.py
                # already detects and logs that case itself as
                # 'Chat/stream').
                log.error(
                    f"[{rid}] /chat: stream broke after {chunk_count} chunks "
                    f"({byte_count} bytes): {type(e).__name__}: {e}"
                )
            finally:
                log.info(f"[{rid}] /chat: stream finished — {chunk_count} chunks, {byte_count} bytes")
                # Charge the REAL tokens Groq reported (or a conservative
                # estimate). Done as its own task so it still lands if the
                # phone disconnects mid-reply.
                if chunk_count:
                    asyncio.create_task(
                        usage.charge(user_key, pool_req, is_prem, tap.total()))
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            _stream(),
            status_code=200,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"X-Request-ID": rid, **_usage_headers(snap)},
        )

    # Every key in the pool failed.
    log.error(f"[{rid}] /chat: exhausted all {attempts} Groq keys — last error: {last_body_text}")
    raise HTTPException(
        status_code=last_status if last_status not in (429, 401, 403) else 429,
        detail={"error": last_body_text, "source": "groq", "request_id": rid},
    )


@app.post("/transcribe")
async def transcribe(
    req: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3-turbo"),
    response_format: str = Form("text"),
    x_app_secret: str = Header(default=""),
):
    """Forwards the recorded audio clip to Groq's Whisper endpoint,
    attaching the real Groq key server-side. Returns plain text, since
    the app reads resp.text.strip() directly."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    if not len(groq_pool):
        log.error(f"[{rid}] /transcribe: server has no Groq API keys configured")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing Groq API keys", "request_id": rid})

    audio_bytes = await file.read()
    log.info(f"[{rid}] /transcribe: {len(audio_bytes)} bytes, model={model}")

    attempts = len(groq_pool)
    last_status = 502
    last_body_text = "all Groq API keys exhausted"

    for attempt in range(1, attempts + 1):
        key = await groq_pool.current()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    GROQ_TRANSCRIBE_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (file.filename, audio_bytes, file.content_type)},
                    data={"model": model, "response_format": response_format},
                )
        except httpx.RequestError as e:
            log.error(f"[{rid}] /transcribe: key #{attempt}/{attempts} network error: {e!r}")
            last_status, last_body_text = 502, f"upstream request failed: {e}"
            await groq_pool.advance()
            continue

        if resp.status_code == 429 or resp.status_code in (401, 403):
            body_text = resp.text[:500]
            cooldown = GroqKeyPool.DEFAULT_COOLDOWN_SECS
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.strip().isdigit():
                    cooldown = float(retry_after)
            else:
                cooldown = GroqKeyPool.AUTH_FAILURE_COOLDOWN_SECS
            log.warning(
                f"[{rid}] /transcribe: key #{attempt}/{attempts} hit HTTP "
                f"{resp.status_code} — cooling it down {cooldown:.0f}s and "
                f"rotating to next key: {body_text}"
            )
            last_status, last_body_text = resp.status_code, body_text
            await groq_pool.advance(bad_key=key, cooldown_secs=cooldown)
            continue

        if resp.status_code != 200:
            body_text = resp.text[:500]
            log.error(f"[{rid}] /transcribe: Groq returned HTTP {resp.status_code}: {body_text}")
            raise HTTPException(
                status_code=resp.status_code,
                detail={"error": body_text, "source": "groq", "request_id": rid},
            )

        if attempt > 1:
            log.info(f"[{rid}] /transcribe: succeeded on key #{attempt}/{attempts} after failover")
        return PlainTextResponse(resp.text)

    log.error(f"[{rid}] /transcribe: exhausted all {attempts} Groq keys — last error: {last_body_text}")
    raise HTTPException(
        status_code=last_status if last_status not in (429, 401, 403) else 429,
        detail={"error": last_body_text, "source": "groq", "request_id": rid},
    )



# ═══════════════════════════════════════════════════════════════════════
#  POST /image — safe image generation (Pollinations, gen.pollinations.ai)
# ═══════════════════════════════════════════════════════════════════════
# The app no longer talks to Pollinations directly. It sends the prompt
# here, and this route:
#   1. checks the app secret,
#   2. refuses sexual / graphic prompts with a keyword filter (same rules
#      as the app's, but this copy can't be bypassed by editing the APK),
#   3. rate-limits per user so nobody can burn your Pollinations credits,
#   4. asks a Groq safety model (Llama Guard) to review the prompt,
#   5. generates the image with the Pollinations key kept server-side and
#      Pollinations' own safety filter switched on, and
#   6. streams the image bytes back.
#
# Render environment variables:
#   POLLINATIONS_API_KEY   (required)  secret key from enter.pollinations.ai
#   POLLINATIONS_SAFE      (optional)  value for the `safe` parameter. Default
#                          "nsfw". Pollinations documents `safe` as a
#                          comma-separated list of: privacy, secrets, sexual,
#                          violence, shield, true, nsfw.
#   POLLINATIONS_IMAGE_MODEL (optional) default "black-forest-labs/flux.1-schnell"
#                          (FLUX Schnell: cheap and fast). Needs to be a model
#                          your key is allowed to use.
#   IMAGE_GUARD_ENABLED    (optional)  "0" turns the Llama Guard check off
#   IMAGE_GUARD_MODEL      (optional)  default "meta-llama/llama-guard-4-12b"
#   IMAGE_RATE_MAX / IMAGE_RATE_WINDOW_SECS (optional) default 12 per 600s

POLLINATIONS_IMAGE_URL = "https://gen.pollinations.ai/image/"
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "").strip()
POLLINATIONS_SAFE = os.environ.get("POLLINATIONS_SAFE", "nsfw").strip() or "nsfw"
POLLINATIONS_IMAGE_MODEL = (os.environ.get("POLLINATIONS_IMAGE_MODEL",
                                           "black-forest-labs/flux.1-schnell").strip()
                            or "black-forest-labs/flux.1-schnell")
IMAGE_GUARD_ENABLED = os.environ.get("IMAGE_GUARD_ENABLED", "1").strip() != "0"
IMAGE_GUARD_MODEL = (os.environ.get("IMAGE_GUARD_MODEL", "meta-llama/llama-guard-4-12b").strip()
                     or "meta-llama/llama-guard-4-12b")
IMAGE_MAX_PROMPT_CHARS = 500


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


IMAGE_RATE_MAX = _env_int("IMAGE_RATE_MAX", 12)
IMAGE_RATE_WINDOW_SECS = _env_int("IMAGE_RATE_WINDOW_SECS", 600)

# The Pollinations key is sent in an Authorization header (never in the URL),
# and httpx's request logging is kept quiet as a second safeguard.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log.info(f"[startup] /image: pollinations key {'set' if POLLINATIONS_API_KEY else 'MISSING'}, "
         f"safe={POLLINATIONS_SAFE!r}, guard={'on' if IMAGE_GUARD_ENABLED else 'off'} ({IMAGE_GUARD_MODEL})")

import re
import unicodedata
import urllib.parse as _urlparse

_IMG_BLOCK_ALWAYS = re.compile(
    r"\b(?:porn\w*|nsfw|xxx|hentai|erotic\w*|nude[sd]?|nudity|naked|"
    r"topless|bottomless|undress\w*|strip(?:ping|per|tease)?|lingerie|"
    r"sexy|sexual\w*|sex\s*(?:scene|act|position|toy)s?|having\s+sex|"
    r"intercourse|orgasm\w*|masturbat\w*|fetish\w*|bdsm|"
    r"onlyfans|lewd|smut|explicit|boudoir|"
    r"loli\w*|shota|underage|child\s*(?:porn|abuse)|"
    r"gore|gory|bloodbath|dismember\w*|decapitat\w*|behead\w*|"
    r"mutilat\w*|disembowel\w*|eviscerat\w*|torture[sd]?|torturing|"
    r"snuff|massacre|slaughter(?:ed|ing)?\s+(?:people|humans?|children|kids)|"
    r"rape[sd]?|raping|molest\w*|self[\s-]*harm|"
    r"suicide|hanging\s+(?:himself|herself|themself|body)|"
    r"slit(?:ting)?\s+(?:my|his|her|their)?\s*wrists?|"
    r"(?:dead|mangled|bloody|burnt|burned)\s+(?:bod(?:y|ies)|corpses?|babies|baby|children|child)|"
    r"graphic\s+(?:violence|injur\w*|death)|bleeding\s+out|"
    r"dead|deceased|death|corpses?|lifeless|graveyards?|cemeter(?:y|ies)|"
    r"graves?|gravestones?|tombstones?|coffins?|morgue|murder\w*)\b")
_IMG_BLOCK_ANATOMY = re.compile(
    r"\b(?:penis|vagina|vulva|genitals?|genitalia|breasts?|boobs?|"
    r"nipples?|butt|buttocks|anus|testicles?|scrotum|clitoris|pussy|"
    r"cock|dick|ass)\b")
_IMG_EDU_CONTEXT = re.compile(
    r"\b(?:anatomy|anatomical|diagram|labell?ed|labelling|cross[\s-]*section|"
    r"histolog\w*|physiolog\w*|biolog\w*|medical|textbook|pharmac\w*|"
    r"reproductive\s+system|urinary\s+system|mammary|cancer|"
    r"examination|lactation|breastfeeding|tissue|cell|organ)\b")
_IMG_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                           "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})


def _image_prompt_blocked(msg: str) -> bool:
    """True if an image request is sexual / graphic. Normalises common
    evasions (leetspeak, spaced-out letters, stretched letters) first."""
    try:
        t = (msg or "").lower()
        t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
        t = t.translate(_IMG_LEET)
        t = re.sub(r"\b(?:[a-z][\s._\-*]){2,}[a-z]\b",
                   lambda m: re.sub(r"[\s._\-*]", "", m.group(0)), t)
        t = re.sub(r"([a-z])\1{2,}", r"\1", t)
        if _IMG_BLOCK_ALWAYS.search(t):
            return True
        if _IMG_BLOCK_ANATOMY.search(t) and not _IMG_EDU_CONTEXT.search(t):
            return True
    except Exception:
        pass
    return False


_IMG_RATE: dict = {}   # user_key -> list of recent request timestamps


def _image_rate_check(user_key: str):
    """Returns 0 if allowed (and records the request), else seconds to wait."""
    now = time.time()
    hits = [t for t in _IMG_RATE.get(user_key, []) if now - t < IMAGE_RATE_WINDOW_SECS]
    if len(hits) >= IMAGE_RATE_MAX:
        _IMG_RATE[user_key] = hits
        return max(1, int(IMAGE_RATE_WINDOW_SECS - (now - hits[0])))
    hits.append(now)
    _IMG_RATE[user_key] = hits
    if len(_IMG_RATE) > 5000:          # keep memory bounded
        for k in [k for k, v in _IMG_RATE.items() if not v or now - v[-1] > IMAGE_RATE_WINDOW_SECS]:
            _IMG_RATE.pop(k, None)
    return 0


async def _image_guard_unsafe(prompt: str, rid: str) -> bool:
    """Second opinion from a Groq safety model. Returns True only on a clear
    'unsafe' verdict. If the model is unavailable or errors, this returns
    False (the keyword filter above has already run, and Pollinations
    applies its own filter) and the problem is logged."""
    if not IMAGE_GUARD_ENABLED or not len(groq_pool):
        return False
    try:
        key = await groq_pool.current()
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": IMAGE_GUARD_MODEL,
                      "messages": [{"role": "user",
                                    "content": f"Image generation request: {prompt}"}],
                      "max_tokens": 20, "temperature": 0},
            )
        if resp.status_code != 200:
            log.warning(f"[{rid}] /image: guard model HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        verdict = (resp.json()["choices"][0]["message"]["content"] or "").strip().lower()
        return verdict.startswith("unsafe")
    except Exception as e:
        log.warning(f"[{rid}] /image: guard check failed ({e!r}); continuing without it")
        return False


class ImageRequest(BaseModel):
    prompt: str


def _policy_error(rid: str) -> HTTPException:
    # 451, not 422: FastAPI already uses 422 for malformed request bodies.
    return HTTPException(status_code=451,
                         detail={"error": "content_policy", "request_id": rid})


@app.post("/image")
async def image(
    body: ImageRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
    x_auth_token: str = Header(default=""),
    x_user_email: str = Header(default=""),
    x_device_id: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "empty_prompt", "request_id": rid})
    if len(prompt) > IMAGE_MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail={"error": "prompt_too_long", "request_id": rid})

    # 1. Keyword filter (free, instant).
    if _image_prompt_blocked(prompt):
        log.info(f"[{rid}] /image: refused by keyword filter (len={len(prompt)})")
        raise _policy_error(rid)

    if not POLLINATIONS_API_KEY:
        log.error(f"[{rid}] /image: POLLINATIONS_API_KEY is not set")
        raise HTTPException(status_code=503,
                            detail={"error": "image_service_not_configured", "request_id": rid})

    # 2. Per-user rate limit (protects your Pollinations credits).
    user_key, _email, _is_prem = await _usage_identity(x_user_email, x_auth_token, x_device_id, rid)
    wait = _image_rate_check(user_key)
    if wait:
        log.info(f"[{rid}] /image: {user_key} rate limited, retry in {wait}s")
        raise HTTPException(status_code=429,
                            detail={"error": "image_rate_limited", "retry_after": wait,
                                    "request_id": rid},
                            headers={"Retry-After": str(wait)})

    # 3. Llama Guard second opinion.
    if await _image_guard_unsafe(prompt, rid):
        log.info(f"[{rid}] /image: refused by guard model (len={len(prompt)})")
        raise _policy_error(rid)

    # 4. Generate with Pollinations (safety filter on).
    # A short, neutral suffix nudges the model toward school-appropriate output
    # without forcing every picture into a textbook-diagram style.
    gen_prompt = prompt + ". Safe for school, family-friendly, fully clothed if people are shown."
    url = POLLINATIONS_IMAGE_URL + _urlparse.quote(gen_prompt, safe="")
    params = {"model": POLLINATIONS_IMAGE_MODEL, "width": 512, "height": 512,
              "nologo": "true", "safe": POLLINATIONS_SAFE}
    try:
        async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
            resp = await client.get(
                url, params=params,
                headers={"Authorization": f"Bearer {POLLINATIONS_API_KEY}"})
    except httpx.RequestError as e:
        log.error(f"[{rid}] /image: Pollinations unreachable: {type(e).__name__}")
        raise HTTPException(status_code=502,
                            detail={"error": "image_upstream_unreachable", "request_id": rid})

    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if resp.status_code == 200 and ctype.startswith("image/") and len(resp.content) > 1000:
        return Response(content=resp.content, media_type=ctype,
                        headers={"Cache-Control": "no-store"})

    body_text = (resp.text or "")[:300] if not ctype.startswith("image/") else ""
    log.warning(f"[{rid}] /image: Pollinations HTTP {resp.status_code} ctype={ctype!r} "
                f"body={body_text!r}")
    low = body_text.lower()
    if resp.status_code in (400, 403, 422) and any(
            w in low for w in ("safe", "safety", "policy", "moderat", "nsfw", "content")):
        raise _policy_error(rid)
    if resp.status_code == 402:
        # Out of Pollen credits / key budget used up. Users just see "busy";
        # this line is for you: top up at enter.pollinations.ai.
        log.error(f"[{rid}] /image: Pollinations returned 402 - the API key is out of "
                  f"credits or over budget. Top up at enter.pollinations.ai")
        raise HTTPException(status_code=429,
                            detail={"error": "image_busy", "request_id": rid})
    if resp.status_code == 429:
        raise HTTPException(status_code=429,
                            detail={"error": "image_busy", "request_id": rid})
    # 401/402/other: key or credits problem on our side. Never leak details.
    raise HTTPException(status_code=502,
                        detail={"error": "image_service_unavailable", "request_id": rid})


@app.get("/model/{key}")
async def download_model(key: str, req: Request, x_app_secret: str = Header(default="")):
    """Streams an offline model file from Hugging Face, attaching the
    real HF token server-side. Streamed rather than buffered in memory
    since these files run several hundred MB."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    meta = OFFLINE_MODELS.get(key)
    if not meta:
        log.warning(f"[{rid}] /model/{key}: unknown model key")
        raise HTTPException(status_code=404,
                             detail={"error": "unknown model key", "request_id": rid})
    if not HF_ACCESS_TOKEN:
        log.error(f"[{rid}] /model/{key}: server missing HF_ACCESS_TOKEN")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing HF_ACCESS_TOKEN", "request_id": rid})

    # follow_redirects is essential: huggingface.co/.../resolve/main/<file> answers
    # with a 302 to a CDN host. httpx does NOT follow redirects by default, so
    # without this the 302 was treated as an error and passed to the app as
    # "Server returned status 302". httpx drops the Authorization header on
    # the cross-host hop, so the HF token is never sent to the CDN.
    client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=120),
                               follow_redirects=True)
    try:
        upstream_req = client.build_request(
            "GET", meta["url"],
            headers={"Authorization": f"Bearer {HF_ACCESS_TOKEN}"},
        )
        upstream = await client.send(upstream_req, stream=True)
    except httpx.RequestError as e:
        await client.aclose()
        log.error(f"[{rid}] /model/{key}: upstream (HF) request failed: {e!r}")
        raise HTTPException(
            status_code=502,
            detail={"error": f"upstream request failed: {e}",
                    "source": "huggingface", "request_id": rid},
        )

    if upstream.status_code in (401, 403):
        await upstream.aclose()
        await client.aclose()
        log.error(f"[{rid}] /model/{key}: Hugging Face auth failed ({upstream.status_code})")
        raise HTTPException(
            status_code=upstream.status_code,
            detail={"error": "Hugging Face auth failed", "source": "huggingface", "request_id": rid},
        )
    if upstream.status_code != 200:
        await upstream.aclose()
        await client.aclose()
        log.error(f"[{rid}] /model/{key}: HF returned HTTP {upstream.status_code}")
        raise HTTPException(
            status_code=upstream.status_code,
            detail={"error": "upstream error", "source": "huggingface", "request_id": rid},
        )

    async def _stream():
        byte_count = 0
        try:
            async for chunk in upstream.aiter_bytes():
                byte_count += len(chunk)
                yield chunk
        except Exception as e:
            log.error(f"[{rid}] /model/{key}: stream broke after {byte_count} bytes: {e!r}")
        finally:
            log.info(f"[{rid}] /model/{key}: stream finished — {byte_count} bytes")
            await upstream.aclose()
            await client.aclose()

    headers = {"X-Request-ID": rid}
    if "content-length" in upstream.headers:
        headers["content-length"] = upstream.headers["content-length"]

    return StreamingResponse(_stream(), status_code=200, headers=headers,
                              media_type="application/octet-stream")


# ── Textbook Library ─────────────────────────────────────────────────────
# Backing logic lives in textbooks_api.py (plain module, same split as
# rag_engine.py) — add PDFs under textbooks/<Subject>/<Title>.pdf in
# the repo, no upload endpoint needed. See textbooks_api.py's module
# docstring for the full folder convention and Render persistence notes.
@app.get("/textbooks")
def get_textbooks(req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return {"textbooks": textbooks_api.list_textbooks()}


@app.get("/textbooks/{book_id}/page/{page}")
def get_textbook_page(book_id: str, page: int, req: Request,
                       bg: int = 0, sid: str = "",
                       x_app_secret: str = Header(default="")):
    # bg=1: the app's background downloader — only ever returns a page that
    #       is already rendered (404 otherwise), never triggers a render.
    # sid : short random id of the app session, so a student jumping to a far
    #       page cancels only THEIR OWN stale queued requests.
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    try:
        path = textbooks_api.get_page_path(book_id, page, track=(bg == 0),
                                           student=sid[:32])
    except textbooks_api.TextbookNotFound:
        log.warning(f"[{rid}] /textbooks/{book_id}/page/{page}: unknown textbook")
        raise HTTPException(status_code=404,
                             detail={"error": "textbook not found", "request_id": rid})
    except textbooks_api.PageOutOfRange:
        log.warning(f"[{rid}] /textbooks/{book_id}/page/{page}: page out of range")
        raise HTTPException(status_code=404,
                             detail={"error": "page out of range", "request_id": rid})
    except textbooks_api.PageRenderError as e:
        log.error(f"[{rid}] /textbooks/{book_id}/page/{page}: render failed — {e}")
        raise HTTPException(status_code=500,
                             detail={"error": "page render failed",
                                     "request_id": rid,
                                     "detail": str(e)})
    return FileResponse(path, media_type="image/jpeg", headers={"X-Request-ID": rid})


@app.get("/textbooks/{book_id}/ready")
def get_textbook_ready(book_id: str, req: Request,
                        x_app_secret: str = Header(default="")):
    """Page numbers already rendered on the server (cheap folder listing,
    never renders). The app polls this and downloads only those pages."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    try:
        return {"pages": textbooks_api.ready_pages(book_id)}
    except textbooks_api.TextbookNotFound:
        log.warning(f"[{rid}] /textbooks/{book_id}/ready: unknown textbook")
        raise HTTPException(status_code=404,
                             detail={"error": "textbook not found", "request_id": rid})


# ── Scholarships ─────────────────────────────────────────────────────────
# Backing logic lives in scholarships.py (plain module, same split as
# textbooks_api.py) — add a .txt file under scholarships/ in the repo, no
# upload endpoint or admin page needed. See scholarships.py's module
# docstring for the file format.
@app.get("/scholarships", response_model=scholarships.ScholarshipListResponse)
def get_scholarships(req: Request, limit: int = 50,
                     x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return scholarships.list_scholarships(limit)


@app.get("/scholarships/{item_id}", response_model=scholarships.ScholarshipItem)
def get_scholarship(item_id: str, req: Request,
                    x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return scholarships.get_scholarship(item_id, rid)


# ── News and Updates ─────────────────────────────────────────────────────
# Same design as scholarships above: backing logic lives in news.py, add a
# .txt file under news/ in the repo (same Title:/Link:/Date: header format).
@app.get("/news", response_model=news.NewsListResponse)
def get_news(req: Request, limit: int = 50,
             x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return news.list_news(limit)


@app.get("/news/{item_id}", response_model=news.NewsItem)
def get_news_item(item_id: str, req: Request,
                  x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return news.get_news_item(item_id, rid)


# ── Past Questions ───────────────────────────────────────────────────────
# Same design as scholarships/news: backing logic lives in past_questions.py,
# data lives in past_questions_data/<subject>_<year>.json in the repo (made
# offline by tools/parse_questions.py). Add a JSON file, push, Render
# redeploys, the paper appears. See past_questions.py's docstring.
@app.get("/past-questions/papers", response_model=past_questions.PapersResponse)
def get_past_question_papers(req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return past_questions.list_papers()


@app.get("/past-questions", response_model=past_questions.QuestionsResponse)
def get_past_questions(req: Request, subject: str, year: int,
                       limit: int = 60, offset: int = 0, shuffle: bool = False,
                       x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return past_questions.get_questions(subject, year, limit, offset, shuffle, rid)


@app.get("/past-questions/images/{subject}/{filename}")
def get_past_question_image(subject: str, filename: str, req: Request,
                            x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    path = past_questions.image_path(subject, filename)
    return FileResponse(path, media_type="image/png",
                        headers={"X-Request-ID": rid, "Cache-Control": "public, max-age=86400"})


# ── Flagged past questions ───────────────────────────────────────────────
# The app posts a question here when a student flags it. You read them with
# GET /past-questions/flags (X-Admin-Key: <WREN_ADMIN_KEY>), fix them with a
# past_questions_data/fixes/<subject>_<year>.json file, then DELETE the entry.
# Storage and the "disk is wiped on deploy" note: see past_questions.py.
@app.post("/past-questions/flag")
def flag_past_question(body: past_questions.FlagIn, req: Request,
                       x_app_secret: str = Header(default=""),
                       x_device_id: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return past_questions.save_flag(body, x_device_id, rid)


@app.get("/past-questions/flags")
def list_flagged_past_questions(limit: int = 200, x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing admin key")
    return past_questions.list_flags(limit)


@app.delete("/past-questions/flags/{qid}")
def clear_flagged_past_question(qid: str, x_admin_key: str = Header(default="")):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing admin key")
    return past_questions.clear_flags(qid)
