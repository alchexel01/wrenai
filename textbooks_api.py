"""
Textbook Library backend — FastAPI router for Wren AI's Textbook Library.

HOW YOU ADD A TEXTBOOK
-----------------------
There's no upload endpoint or admin panel — the folder IS the catalog.
Just drop a .pdf file into TEXTBOOKS_DIR (optionally inside a subject
subfolder):

    textbooks/
      Chemistry/
        Organic Chemistry Basics.pdf
      Mathematics/
        Further Maths Vol 1.pdf

That gives you:
    subject = "Chemistry", title = "Organic Chemistry Basics"
    subject = "Mathematics", title = "Further Maths Vol 1"

A PDF placed directly in TEXTBOOKS_DIR (no subfolder) gets subject ''.
The book's `id` is a stable hash of its path relative to TEXTBOOKS_DIR,
so ids don't change across restarts as long as you don't rename or
move the file afterward.

IMPORTANT IF YOU'RE ON RENDER
-------------------------------
Render's default filesystem is ephemeral — anything you upload
disappears on the next deploy or restart. Point TEXTBOOKS_DIR (and
TEXTBOOKS_CACHE_DIR) at a Render persistent disk's mount path
(dashboard -> your service -> Disks -> Add Disk), or the textbooks
you drop in will vanish the moment you next deploy.

HOW PAGES GET TO THE APP
--------------------------
PyMuPDF (fitz) runs fine here — this is your normal Linux server, not
python-for-android, so none of the on-device build-recipe problems
apply. A page is rendered to a JPEG the first time it's requested and
cached to disk after that, so repeat reads (a student re-opening the
same book) never re-render.

INTEGRATING THIS
------------------
If you already have a FastAPI app elsewhere in your backend:
    from textbooks_api import router as textbooks_router
    app.include_router(textbooks_router)

To run this as its own standalone service instead, the __main__ block
at the bottom does that — see the run instructions there.

Install: pip install fastapi uvicorn pymupdf
"""

import os
import hashlib
import threading
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse

# ── Config (env vars so this matches however you deploy) ───────────────────
WREN_APP_SECRET   = os.environ.get('WREN_APP_SECRET', '')
TEXTBOOKS_DIR     = Path(os.environ.get('TEXTBOOKS_DIR', './textbooks')).resolve()
CACHE_DIR         = Path(os.environ.get('TEXTBOOKS_CACHE_DIR', './textbooks_cache')).resolve()
RENDER_DPI        = int(os.environ.get('TEXTBOOKS_RENDER_DPI', '150'))

TEXTBOOKS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# fitz page rendering isn't guaranteed safe to call concurrently across
# documents from multiple threads — serialize actual renders. Cache
# hits (the common case once a book's been read once) never touch
# this lock at all.
_render_lock = threading.Lock()


def _check_secret(x_app_secret):
    """Same shared-secret scheme the app already sends as
    X-App-Secret on every request (see _backend_headers() in
    wren.py) — nothing new to add on the app side."""
    if not WREN_APP_SECRET or x_app_secret != WREN_APP_SECRET:
        raise HTTPException(status_code=401, detail='Invalid or missing X-App-Secret')


def _book_id(rel_path: Path) -> str:
    return hashlib.sha1(str(rel_path).encode('utf-8')).hexdigest()[:16]


def _scan_books():
    """Walks TEXTBOOKS_DIR for .pdf files and returns
    {id: {"path": Path, "title": str, "subject": str}}. Re-scanning
    per-request is cheap enough for a personal-scale library (tens to
    low hundreds of books); swap in a cached index or a tiny SQLite
    table if this ever grows much larger."""
    books = {}
    for pdf_path in TEXTBOOKS_DIR.rglob('*.pdf'):
        rel = pdf_path.relative_to(TEXTBOOKS_DIR)
        bid = _book_id(rel)
        subject = rel.parent.name if rel.parent != Path('.') else ''
        title = pdf_path.stem.replace('_', ' ').replace('-', ' ').strip()
        books[bid] = {'path': pdf_path, 'title': title, 'subject': subject}
    return books


def _page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def _page_cache_path(book_id: str, page: int) -> Path:
    d = CACHE_DIR / book_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f'page_{page:04d}.jpg'


def _render_page(pdf_path: Path, page: int, dest: Path):
    """Renders one page to dest as a JPEG. Writes to a .part file and
    renames into place at the end (same pattern as TextbookManager's
    downloads on the app side) so a request that reads dest mid-write
    can never see a truncated image."""
    tmp = dest.with_suffix(dest.suffix + '.part')
    with _render_lock:
        with fitz.open(pdf_path) as doc:
            if page < 1 or page > doc.page_count:
                raise HTTPException(status_code=404, detail='Page out of range')
            pix = doc[page - 1].get_pixmap(dpi=RENDER_DPI)
            pix.save(str(tmp))
    tmp.replace(dest)


@router.get('/textbooks')
def list_textbooks(x_app_secret: str = Header(default=None)):
    _check_secret(x_app_secret)
    out = []
    for bid, meta in _scan_books().items():
        try:
            pages = _page_count(meta['path'])
        except Exception:
            # Skip a corrupt/unreadable PDF rather than 500ing the
            # whole list for every other book.
            continue
        out.append({
            'id': bid,
            'title': meta['title'],
            'subject': meta['subject'],
            'page_count': pages,
        })
    out.sort(key=lambda b: b['title'].casefold())
    return {'textbooks': out}


@router.get('/textbooks/{book_id}/page/{page}')
def get_textbook_page(book_id: str, page: int, x_app_secret: str = Header(default=None)):
    _check_secret(x_app_secret)
    books = _scan_books()
    meta = books.get(book_id)
    if not meta:
        raise HTTPException(status_code=404, detail='Textbook not found')

    dest = _page_cache_path(book_id, page)
    if not dest.exists():
        _render_page(meta['path'], page, dest)

    return FileResponse(dest, media_type='image/jpeg')


# ── Standalone mode ──────────────────────────────────────────────────────
# Only relevant if you ever want to run this file on its own instead of
# mounting `router` into your existing backend app. Does NOT run on
# import — importing `router` elsewhere (the normal case) never builds
# a second app. Start standalone with:
#   WREN_APP_SECRET=yourvalue uvicorn textbooks_api:standalone_app --host 0.0.0.0 --port 8000
if __name__ == '__main__':
    import uvicorn
    from fastapi import FastAPI
    standalone_app = FastAPI()
    standalone_app.include_router(router)
    uvicorn.run(standalone_app, host='0.0.0.0', port=8000)
