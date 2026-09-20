"""
Textbook Library backend logic for Wren AI.

This module is plain Python — no FastAPI decorators or auth in here,
same split as rag_engine.py: main.py owns the routes, request
logging (rid), and the shared _check_app_secret() gate; this module
just knows how to find textbooks and turn a page into an image.

HOW YOU ADD A TEXTBOOK
-----------------------
There's no upload endpoint or admin panel — the folder IS the catalog.
Just drop a .pdf file into TEXTBOOKS_DIR (optionally inside a subject
subfolder), commit it to the repo, and push:

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

Because these PDFs are committed to the same GitHub repo Render
redeploys from (same as syllabus_data/), they always come back after
a redeploy — Render's ephemeral filesystem only affects things
written at *runtime*, which here is just the rendered-page cache
(CACHE_DIR). Losing that cache on redeploy is harmless: the first
student to open a page after a redeploy just triggers one re-render,
then it's cached again.

HOW PAGES GET TO THE APP
--------------------------
PyMuPDF (fitz) runs fine here — this is your normal Render server,
not python-for-android, so none of the on-device build-recipe
problems apply. A page is rendered to a JPEG the first time it's
requested and cached to disk after that.

Install: pip install pymupdf   (add to requirements.txt)
"""

import os
import hashlib
import logging
import threading
import traceback
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger("wren-backend")

TEXTBOOKS_DIR = Path(os.environ.get("TEXTBOOKS_DIR", "./textbooks")).resolve()
CACHE_DIR     = Path(os.environ.get("TEXTBOOKS_CACHE_DIR", "./textbooks_cache")).resolve()
RENDER_DPI    = int(os.environ.get("TEXTBOOKS_RENDER_DPI", "150"))

TEXTBOOKS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# fitz page rendering isn't guaranteed safe to call concurrently across
# documents from multiple threads/tasks — serialize actual renders.
# Cache hits (the common case once a book's been read once) never
# touch this lock at all.
_render_lock = threading.Lock()


class TextbookNotFound(Exception):
    """Raised when book_id doesn't match any PDF under TEXTBOOKS_DIR.
    main.py catches this and turns it into a 404."""
    pass


class PageOutOfRange(Exception):
    """Raised when the requested page number isn't in the document.
    main.py catches this and turns it into a 404."""
    pass


class PageRenderError(Exception):
    """Raised when the PDF opens fine (it's in the /textbooks listing)
    but rendering *this* page to a JPEG fails — e.g. a scanned page in
    a CMYK/indexed/exotic colorspace that PyMuPDF can't hand straight
    to a JPEG encoder. main.py catches this and turns it into a 500
    with an actual message instead of a bare, unlogged crash."""
    pass


def _book_id(rel_path: Path) -> str:
    return hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:16]


def _scan_books():
    """Walks TEXTBOOKS_DIR for .pdf files and returns
    {id: {"path": Path, "title": str, "subject": str}}. Re-scanning
    per-call is cheap enough for a personal-scale library (tens to low
    hundreds of books); swap in a cached index if this grows much
    larger later."""
    books = {}
    for pdf_path in TEXTBOOKS_DIR.rglob("*.pdf"):
        rel = pdf_path.relative_to(TEXTBOOKS_DIR)
        bid = _book_id(rel)
        subject = rel.parent.name if rel.parent != Path(".") else ""
        title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
        books[bid] = {"path": pdf_path, "title": title, "subject": subject}
    return books


def list_textbooks():
    """Returns [{id, title, subject, page_count}, ...] — one entry per
    PDF found under TEXTBOOKS_DIR. Called from main.py's /textbooks
    route, the same way rag_engine.list_exam_bodies() is."""
    out = []
    for bid, meta in _scan_books().items():
        try:
            with fitz.open(meta["path"]) as doc:
                pages = doc.page_count
        except Exception:
            # Skip a corrupt/unreadable PDF rather than failing the
            # whole list for every other book.
            continue
        out.append({
            "id": bid,
            "title": meta["title"],
            "subject": meta["subject"],
            "page_count": pages,
        })
    out.sort(key=lambda b: b["title"].casefold())
    return out


def get_page_path(book_id: str, page: int) -> Path:
    """Returns the local path to `page` (1-indexed) of the given
    textbook, rendering and caching it first if needed. Raises
    TextbookNotFound / PageOutOfRange rather than any FastAPI
    exception, so this module stays framework-agnostic — main.py
    is what turns those into HTTPExceptions with the right rid."""
    meta = _scan_books().get(book_id)
    if not meta:
        raise TextbookNotFound(book_id)

    dest = CACHE_DIR / book_id / f"page_{page:04d}.jpg"
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with _render_lock:
            with fitz.open(meta["path"]) as doc:
                if page < 1 or page > doc.page_count:
                    raise PageOutOfRange(page)
                # Force RGB explicitly. Without this, get_pixmap() can
                # hand back a pixmap in whatever colorspace the source
                # page uses (CMYK, indexed, an embedded ICC profile —
                # all common in scanned textbook PDFs), and pix.save()
                # to .jpg then throws deep inside MuPDF since JPEG
                # can't encode most of those directly. That throw was
                # previously uncaught here, producing the bare 500
                # you're seeing with no logged reason.
                pix = doc[page - 1].get_pixmap(dpi=RENDER_DPI,
                                                colorspace=fitz.csRGB,
                                                alpha=False)
                pix.save(str(tmp))
    except PageOutOfRange:
        raise
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        tb = traceback.format_exc()
        log.exception(f"textbooks: failed to render {book_id} page {page}")
        # Full traceback rides in the exception message (not just log.exception)
        # because Render's log tab isn't live for this dev — main.py puts this
        # straight into the HTTP response body, and the app's TextbookManager
        # forwards it into Settings > Developer Options, so the real MuPDF
        # error is visible on-device without touching server logs at all.
        raise PageRenderError(
            f"{type(e).__name__}: {e}\n---\n{tb[-1200:]}") from e
    tmp.replace(dest)
    return dest
