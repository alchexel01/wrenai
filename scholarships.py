"""
Wren Scholarships — paste on an admin page, every user sees it
===============================================================
You open  https://<your-service>.onrender.com/admin/scholarships  (a page
served by this backend), enter WREN_ADMIN_KEY once, paste the scholarship
text, press "Publish". It is saved in Postgres and instantly available to
every app install through GET /scholarships.

How users get told
------------------
1. In-app feed (always on): the app calls GET /scholarships?after_id=<last
   id it has seen>. Anything newer than that is "new" - the app turns it
   into a notification and a badge. Works for every user, no Google setup.
2. Real phone push (optional): if FIREBASE_SERVICE_ACCOUNT_JSON is set, a
   single Firebase Cloud Messaging message is also sent to the topic
   "scholarships" at publish time, so phones that subscribed get it even
   when the app is closed. Not set = skipped, nothing breaks.

Storage: Postgres (DATABASE_URL), same as premium.py / chat_history.py -
this service's own disk is wiped on every redeploy.

Endpoints (wired into main.py):
    GET    /admin/scholarships             - the paste page (HTML, holds no secrets)
    POST   /admin/scholarships/api         - publish one (X-Admin-Key)
    GET    /admin/scholarships/api/list    - recent ones for the page (X-Admin-Key)
    DELETE /admin/scholarships/api/{id}    - remove one for everyone (X-Admin-Key)
    GET    /scholarships                   - feed for the app (X-App-Secret)
    GET    /scholarships/{id}              - one item (X-App-Secret)

Environment variables:
    DATABASE_URL                   - already set (shared Postgres)
    WREN_ADMIN_KEY                 - already set (same key as /admin/reload)
    FIREBASE_SERVICE_ACCOUNT_JSON  - OPTIONAL: whole service-account JSON, one value
"""

import os
import re
import json
import hmac
import asyncio
import logging
from typing import Optional, List

import httpx
import asyncpg
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

log = logging.getLogger("wren-backend")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
ADMIN_KEY = os.environ.get("WREN_ADMIN_KEY", "")

FCM_TOPIC = "scholarships"
FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
ANDROID_CHANNEL_ID = "scholarships"

MAX_TEXT_CHARS = 20_000
MIN_TEXT_CHARS = 20

_pool = None
_fcm_creds = None
_fcm_project_id = ""


# ── DB lifecycle ─────────────────────────────────────────────────────────
async def init_db():
    global _pool
    if not DATABASE_URL:
        log.error("[scholarships] DATABASE_URL not set - scholarships endpoints will fail")
        return
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS scholarships ("
            "id SERIAL PRIMARY KEY, "
            "title TEXT NOT NULL, "
            "body TEXT NOT NULL, "
            "link TEXT, "
            "push_sent BOOLEAN NOT NULL DEFAULT FALSE, "
            "push_detail TEXT NOT NULL DEFAULT '', "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
    _init_fcm()
    log.info("[scholarships] DB pool ready, table ensured; push %s",
             "ENABLED" if _fcm_creds else "off (no FIREBASE_SERVICE_ACCOUNT_JSON)")


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _require_pool(rid):
    if _pool is None:
        log.error(f"[{rid}] /scholarships: DB pool not initialized (DATABASE_URL missing?)")
        raise HTTPException(status_code=500,
                             detail={"error": "server missing DATABASE_URL", "request_id": rid})


def check_admin_key(incoming, rid="-"):
    """Constant-time check of X-Admin-Key against WREN_ADMIN_KEY. An unset
    server key disables the admin routes entirely (never open by default)."""
    inc = (incoming or "").strip()
    if not ADMIN_KEY or not hmac.compare_digest(inc.encode(), ADMIN_KEY.encode()):
        log.warning(f"[{rid}] /admin/scholarships: admin key rejected")
        raise HTTPException(status_code=401,
                             detail={"error": "invalid or missing admin key", "request_id": rid})


# ── Models ───────────────────────────────────────────────────────────────
class PublishRequest(BaseModel):
    text: str
    title: Optional[str] = None
    link: Optional[str] = None
    notify: bool = True


class ScholarshipItem(BaseModel):
    id: int
    title: str
    body: str
    link: Optional[str] = None
    created_at: str


class ScholarshipListResponse(BaseModel):
    scholarships: List[ScholarshipItem]


class PublishResponse(BaseModel):
    id: int
    title: str
    push_sent: bool
    push_detail: str


class OkResponse(BaseModel):
    ok: bool = True


# ── Text helpers ─────────────────────────────────────────────────────────
def derive_title(text):
    """First non-empty line, bullets/markdown stripped, max 80 chars."""
    for line in text.splitlines():
        t = re.sub(r"^[\s#*>\-\u2022\u00b7_]+", "", line).strip(" *_")
        if t:
            return (t[:77] + "\u2026") if len(t) > 80 else t
    return "New scholarship"


def derive_preview(text, title, limit=140):
    """Notification body: what follows the title line, flattened."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    rest = lines[1:] if lines and derive_title(lines[0]) == title else lines
    flat = re.sub(r"[#*_>`]+", "", " ".join(rest))
    flat = re.sub(r"\s+", " ", flat).strip()
    if not flat:
        return "Tap to read the details."
    return (flat[: limit - 1] + "\u2026") if len(flat) > limit else flat


def _row_to_item(r):
    return ScholarshipItem(id=r["id"], title=r["title"], body=r["body"],
                           link=r["link"], created_at=r["created_at"].isoformat())


# ── FCM push (optional, HTTP v1, topic broadcast) ────────────────────────
def _init_fcm():
    global _fcm_creds, _fcm_project_id
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        return
    try:
        from google.oauth2 import service_account
        info = json.loads(raw)
        _fcm_creds = service_account.Credentials.from_service_account_info(
            info, scopes=[FCM_SCOPE])
        _fcm_project_id = info["project_id"]
    except Exception as e:
        _fcm_creds = None
        log.error(f"[scholarships] FCM init failed: {e!r}")


def _fcm_access_token():
    from google.auth.transport.requests import Request as GReq
    if not _fcm_creds.valid:
        _fcm_creds.refresh(GReq())          # blocking - call via to_thread
    return _fcm_creds.token


async def _send_push(item_id, title, body, rid):
    if not _fcm_creds:
        return False, "push not configured (no FIREBASE_SERVICE_ACCOUNT_JSON)"
    try:
        token = await asyncio.to_thread(_fcm_access_token)
        msg = {"message": {
            "topic": FCM_TOPIC,
            "notification": {"title": title, "body": body},
            "data": {"type": "scholarship", "id": str(item_id)},
            "android": {"priority": "HIGH",
                        "notification": {"channel_id": ANDROID_CHANNEL_ID}},
        }}
        async with httpx.AsyncClient(timeout=15) as cli:
            r = await cli.post(
                f"https://fcm.googleapis.com/v1/projects/{_fcm_project_id}/messages:send",
                headers={"Authorization": f"Bearer {token}"}, json=msg)
        if r.status_code == 200:
            return True, "sent to topic 'scholarships'"
        log.error(f"[{rid}] /scholarships: FCM {r.status_code}: {r.text[:300]}")
        return False, f"FCM {r.status_code}: {r.text[:200]}"
    except Exception as e:
        log.error(f"[{rid}] /scholarships: FCM error {e!r}")
        return False, f"push error: {e!r}"


# ── Operations (called from main.py routes) ──────────────────────────────
async def scholarships_publish(payload: PublishRequest, rid) -> PublishResponse:
    _require_pool(rid)
    text = (payload.text or "").strip()
    if len(text) < MIN_TEXT_CHARS:
        raise HTTPException(status_code=400,
                             detail={"error": "text too short", "request_id": rid})
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400,
                             detail={"error": f"text too long (max {MAX_TEXT_CHARS} characters)",
                                     "request_id": rid})
    title = (payload.title or "").strip() or derive_title(text)
    link = (payload.link or "").strip() or None
    if link and not re.match(r"^https?://", link):
        raise HTTPException(status_code=400,
                             detail={"error": "link must start with http:// or https://",
                                     "request_id": rid})

    async with _pool.acquire() as conn:
        item_id = await conn.fetchval(
            "INSERT INTO scholarships (title, body, link) VALUES ($1, $2, $3) RETURNING id",
            title, text, link)

    sent, detail = False, "saved only (push unticked)"
    if payload.notify:
        sent, detail = await _send_push(item_id, title, derive_preview(text, title), rid)
        async with _pool.acquire() as conn:
            await conn.execute(
                "UPDATE scholarships SET push_sent = $2, push_detail = $3 WHERE id = $1",
                item_id, sent, detail)
    log.info(f"[{rid}] /scholarships: published #{item_id} push_sent={sent} ({detail})")
    return PublishResponse(id=item_id, title=title, push_sent=sent, push_detail=detail)


async def scholarships_list(rid, after_id=0, limit=50) -> ScholarshipListResponse:
    _require_pool(rid)
    limit = max(1, min(int(limit), 100))
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, body, link, created_at FROM scholarships "
            "WHERE id > $1 ORDER BY id DESC LIMIT $2", int(after_id), limit)
    return ScholarshipListResponse(scholarships=[_row_to_item(r) for r in rows])


async def scholarships_get(item_id, rid) -> ScholarshipItem:
    _require_pool(rid)
    async with _pool.acquire() as conn:
        r = await conn.fetchrow(
            "SELECT id, title, body, link, created_at FROM scholarships WHERE id = $1",
            int(item_id))
    if not r:
        raise HTTPException(status_code=404,
                             detail={"error": "scholarship not found", "request_id": rid})
    return _row_to_item(r)


async def scholarships_delete(item_id, rid) -> OkResponse:
    _require_pool(rid)
    async with _pool.acquire() as conn:
        res = await conn.execute("DELETE FROM scholarships WHERE id = $1", int(item_id))
    if not res.endswith("1"):
        raise HTTPException(status_code=404,
                             detail={"error": "scholarship not found", "request_id": rid})
    log.info(f"[{rid}] /scholarships: deleted #{item_id}")
    return OkResponse()


def admin_page() -> HTMLResponse:
    return HTMLResponse(_ADMIN_PAGE_HTML)


# ── Admin page (all actions go through the X-Admin-Key routes) ───────────
_ADMIN_PAGE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Wren - Scholarships</title>
<style>
 :root{--p:#6c3fd1;--bg:#f6f4fb;--card:#fff;--tx:#1d1a27;--mut:#6f6a80;--bd:#e3dff0}
 @media(prefers-color-scheme:dark){:root{--bg:#14111c;--card:#1d1927;--tx:#eeeaf8;--mut:#a39db8;--bd:#2f2943}}
 *{box-sizing:border-box}body{margin:0;font:16px system-ui,sans-serif;background:var(--bg);color:var(--tx)}
 main{max-width:720px;margin:0 auto;padding:16px}
 h1{font-size:20px;margin:8px 0 16px}
 .card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px;margin-bottom:14px}
 label{display:block;font-size:13px;color:var(--mut);margin:10px 0 4px}
 input[type=text],input[type=password],textarea{width:100%;padding:10px;border-radius:10px;border:1px solid var(--bd);background:transparent;color:var(--tx);font:inherit}
 textarea{min-height:220px;resize:vertical}
 button{background:var(--p);color:#fff;border:0;border-radius:10px;padding:12px 16px;font:inherit;font-weight:600;cursor:pointer}
 button.sec{background:transparent;color:var(--mut);border:1px solid var(--bd);padding:6px 10px;font-weight:500}
 button:disabled{opacity:.5}
 .row{display:flex;gap:10px;align-items:center;justify-content:space-between}
 .msg{margin-top:10px;font-size:14px;white-space:pre-wrap}.ok{color:#18a058}.err{color:#d03050}
 .item{border-top:1px solid var(--bd);padding:10px 0}.item:first-child{border-top:0}
 .item b{display:block}.item small{color:var(--mut)}
</style></head><body><main>
<h1>Wren - Scholarships</h1>
<div class="card">
 <label>Admin key (kept in this tab only)</label>
 <input type="password" id="key" autocomplete="off" placeholder="WREN_ADMIN_KEY">
 <label>Scholarship text - paste everything</label>
 <textarea id="text" placeholder="Paste the scholarship here. The first line becomes the title."></textarea>
 <label>Title (optional - leave empty to use the first line)</label>
 <input type="text" id="title">
 <label>Apply link (optional)</label>
 <input type="text" id="link" placeholder="https://...">
 <div class="row" style="margin-top:14px">
  <label style="margin:0;display:flex;gap:8px;align-items:center;color:var(--tx)">
   <input type="checkbox" id="notify" checked> Also send phone push (if configured)</label>
  <button id="go">Publish</button>
 </div>
 <div class="msg" id="msg"></div>
</div>
<div class="card"><div class="row"><b>Recent</b><button class="sec" id="reload">Refresh</button></div><div id="list"></div></div>
<script>
const $=id=>document.getElementById(id);
$('key').value=sessionStorage.getItem('wk')||'';
const H=()=>({'Content-Type':'application/json','X-Admin-Key':$('key').value.trim()});
function say(t,ok){const m=$('msg');m.textContent=t;m.className='msg '+(ok?'ok':'err')}
async function load(){
 sessionStorage.setItem('wk',$('key').value.trim());
 const r=await fetch('/admin/scholarships/api/list',{headers:H()});
 const L=$('list');L.textContent='';
 if(!r.ok){L.textContent=r.status==401?'Enter your admin key to see the list.':'Could not load.';return}
 (await r.json()).scholarships.forEach(s=>{
  const d=document.createElement('div');d.className='item';
  const b=document.createElement('b');b.textContent=s.title;
  const sm=document.createElement('small');sm.textContent='#'+s.id+' - '+new Date(s.created_at).toLocaleString()+' ';
  const x=document.createElement('button');x.className='sec';x.textContent='Delete';
  x.onclick=async()=>{if(!confirm('Delete "'+s.title+'" for everyone?'))return;
   await fetch('/admin/scholarships/api/'+s.id,{method:'DELETE',headers:H()});load()};
  sm.appendChild(x);d.append(b,sm);L.appendChild(d)})}
$('reload').onclick=load;
$('go').onclick=async()=>{
 const text=$('text').value.trim();if(text.length<20){say('Paste the scholarship text first.',false);return}
 $('go').disabled=true;say('Publishing...',true);
 try{
  const r=await fetch('/admin/scholarships/api',{method:'POST',headers:H(),
   body:JSON.stringify({text,title:$('title').value,link:$('link').value,notify:$('notify').checked})});
  const j=await r.json();
  if(!r.ok){say((j.detail&&j.detail.error)||j.detail||'Failed',false)}
  else{say('Published as #'+j.id+': '+j.title+'\nPhone push: '+(j.push_sent?'sent':j.push_detail),true);
   $('text').value='';$('title').value='';$('link').value='';load()}
 }catch(e){say('Network error: '+e,false)}
 $('go').disabled=false};
if($('key').value)load();
</script></main></body></html>"""
