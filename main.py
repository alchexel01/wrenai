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
    GET  /premium/status/{device_id} — is this device currently premium, until when
    POST /premium/restore            — one-time: re-link an email's premium to a new device_id
    DELETE /premium/reset/{email}    — TESTING ONLY: wipe an email's premium record
    GET  /auth/google/start          — get a Google sign-in URL + session_id
    GET  /auth/google/callback       — Google redirects here after sign-in
    GET  /auth/google/status/{id}    — poll: has this session's sign-in completed?
    POST /chats/save                 — upsert one chat for an email
    GET  /chats/{email}              — fetch every chat saved for an email
    DELETE /chats/{email}/{chat_id}  — delete one chat
    DELETE /chats/{email}            — delete every chat for an email
    POST /chat                       — proxies to Groq chat completions
    POST /transcribe                 — proxies to Groq Whisper transcription
    GET  /model/{key}                — streams an offline model file from HF

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
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import httpx

import rag_engine
import premium
import auth
import chat_history
import textbooks_api

app = FastAPI(title="Wren Syllabus Backend", version="1.0")


@app.on_event("startup")
async def _startup():
    await premium.init_db()
    await chat_history.init_db()
    await auth.init_db()


@app.on_event("shutdown")
async def _shutdown():
    await premium.close_db()
    await chat_history.close_db()
    await auth.close_db()

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
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_initialize(payload, rid)


@app.get("/premium/verify/{reference}", response_model=premium.StatusResponse)
async def premium_verify_route(
    reference: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_verify(reference, rid)


@app.get("/premium/status/{device_id}", response_model=premium.StatusResponse)
async def premium_status_route(
    device_id: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_status(device_id, rid)


@app.post("/premium/restore", response_model=premium.StatusResponse)
async def premium_restore_route(
    payload: premium.RestoreRequest,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    """One-time-per-email restore: re-links an email's existing premium
    purchase to whatever device_id is asking. Meant for the case where
    a reinstall regenerated the device's fallback UUID and orphaned a
    paying user's premium — see premium.py for the full identity model."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_restore(payload, rid)


@app.delete("/premium/reset/{email}", response_model=premium.StatusResponse)
async def premium_reset_route(
    email: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    """TESTING ONLY — wipes any premium record for an email so the
    purchase flow can be re-run from scratch. Same X-App-Secret gate as
    every other route here; nothing extra-locked-down about it, so
    don't rely on this being hidden from anyone who has the app
    secret — it's meant for you during development, not as a
    production admin feature."""
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await premium.premium_reset(email, rid)


# ── Google Sign-In ────────────────────────────────────────────────────
# App opens a browser to /auth/google/start's authorization_url, Google
# redirects back to /auth/google/callback on this backend, and the app
# polls /auth/google/status/{session_id} until the verified email shows
# up — same shape as the Paystack initialize/verify polling above.
# See auth.py for the full design notes.

@app.get("/auth/google/start", response_model=auth.AuthStartResponse)
async def auth_google_start_route(req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await auth.auth_google_start(rid)


@app.get("/auth/google/callback")
async def auth_google_callback_route(code: str, state: str, req: Request):
    # No X-App-Secret here on purpose — Google itself calls this URL
    # via browser redirect, not the app, so it can't attach that
    # header. Security instead comes from verifying the ID token's
    # signature server-side in auth.py.
    rid = req.state.rid
    return await auth.auth_google_callback(code, state, rid)


@app.get("/auth/google/status/{session_id}", response_model=auth.AuthStatusResponse)
async def auth_google_status_route(session_id: str, req: Request, x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await auth.auth_google_status(session_id, rid)


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
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await chat_history.chats_save(payload, rid)


@app.get("/chats/{email}", response_model=chat_history.ChatListResponse)
async def chats_list_route(
    email: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await chat_history.chats_list(email, rid)


@app.delete("/chats/{email}/{chat_id}", response_model=chat_history.OkResponse)
async def chat_delete_route(
    email: str,
    chat_id: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await chat_history.chat_delete(email, chat_id, rid)


@app.delete("/chats/{email}", response_model=chat_history.OkResponse)
async def chats_delete_all_route(
    email: str,
    req: Request,
    x_app_secret: str = Header(default=""),
):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    return await chat_history.chats_delete_all(email, rid)


@app.post("/chat")
async def chat(request: dict, req: Request, x_app_secret: str = Header(default="")):
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

        async def _stream(client=client, upstream=upstream):
            chunk_count = 0
            byte_count = 0
            try:
                async for chunk in upstream.aiter_bytes():
                    chunk_count += 1
                    byte_count += len(chunk)
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
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            _stream(),
            status_code=200,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"X-Request-ID": rid},
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

    client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=120))
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
                       x_app_secret: str = Header(default="")):
    rid = req.state.rid
    _check_app_secret(x_app_secret, rid)
    try:
        path = textbooks_api.get_page_path(book_id, page)
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
                             detail={"error": "page render failed", "request_id": rid})
    return FileResponse(path, media_type="image/jpeg", headers={"X-Request-ID": rid})
