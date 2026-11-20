"""
Textbook Library backend logic for Wren AI (faster version).

Same public API as before (list_textbooks, get_page_path, TextbookNotFound,
PageOutOfRange), so main.py does not need to change.

HOW YOU ADD A TEXTBOOK
-----------------------
The folder IS the catalog. Drop a .pdf into TEXTBOOKS_DIR (optionally inside
a subject subfolder), commit, and push:

    textbooks/
      Chemistry/
        Organic Chemistry Basics.pdf
      Mathematics/
        Further Maths Vol 1.pdf

The book id is a stable hash of its path relative to TEXTBOOKS_DIR, so don't
rename or move a file after publishing it.

WHAT MAKES THIS VERSION FASTER
-------------------------------
1. The book index (folder scan) is cached in memory for a short time instead
   of walking the disk on every request.
2. Page counts are cached, so /textbooks no longer opens every PDF each time.
3. Open PDF handles are reused (small LRU) instead of re-opening and
   re-parsing the whole PDF for every single page.
4. Pages are rendered as smaller JPEGs (lower DPI and quality), which also
   means less data to download on a phone.
5. The next 4 pages after whatever page was requested are rendered in the
   background, so page turns are usually cache hits. Jumping straight to
   page 190 renders page 190 immediately (nothing before it is needed),
   then prefetches 191 to 194.
6. warm_up() can pre-render the first pages of every book at startup.

Install: pip install pymupdf   (add to requirements.txt)
"""

import os
import time
import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import pymupdf  # PyMuPDF >= 1.24.3
except ImportError:  # older releases only ship `fitz`
    import fitz as pymupdf

TEXTBOOKS_DIR = Path(os.environ.get("TEXTBOOKS_DIR", "./textbooks")).resolve()
CACHE_DIR = Path(os.environ.get("TEXTBOOKS_CACHE_DIR", "./textbooks_cache")).resolve()

# 110 DPI is sharp enough for a phone screen and renders much faster than 150.
# Raise it with the env var if text looks soft on large tablets.
RENDER_DPI = int(os.environ.get("TEXTBOOKS_RENDER_DPI", "110"))
JPEG_QUALITY = int(os.environ.get("TEXTBOOKS_JPEG_QUALITY", "72"))

INDEX_TTL = float(os.environ.get("TEXTBOOKS_INDEX_TTL", "60"))   # seconds
PREFETCH_PAGES = int(os.environ.get("TEXTBOOKS_PREFETCH_PAGES", "4"))
MAX_OPEN_DOCS = int(os.environ.get("TEXTBOOKS_MAX_OPEN_DOCS", "4"))

TEXTBOOKS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# PyMuPDF is not safe to call concurrently on the same process state, so all
# renders (and the shared open-document handles) go through this one lock.
# Cache hits never touch it.
_render_lock = threading.RLock()

# One background worker: prefetching must never compete with real requests
# for CPU on a small server.
_prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tb-prefetch")
_prefetch_inflight = set()
_prefetch_guard = threading.Lock()


class TextbookNotFound(Exception):
    """Raised when book_id doesn't match any PDF under TEXTBOOKS_DIR.
    main.py catches this and turns it into a 404."""
    pass


class PageOutOfRange(Exception):
    """Raised when the requested page number isn't in the document.
    main.py catches this and turns it into a 404."""
    pass


# ---------------------------------------------------------------------------
# Book index (cached)
# ---------------------------------------------------------------------------

def _book_id(rel_path: Path) -> str:
    return hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:16]


_index_lock = threading.Lock()
_index = {}
_index_time = 0.0


def _scan_books_uncached():
    books = {}
    for pdf_path in TEXTBOOKS_DIR.rglob("*.pdf"):
        rel = pdf_path.relative_to(TEXTBOOKS_DIR)
        bid = _book_id(rel)
        subject = rel.parent.name if rel.parent != Path(".") else ""
        title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
        books[bid] = {"path": pdf_path, "title": title, "subject": subject}
    return books


def _scan_books(force: bool = False):
    """Returns {id: {"path", "title", "subject"}}, rescanning the disk at
    most once every INDEX_TTL seconds."""
    global _index, _index_time
    now = time.monotonic()
    if not force and _index and (now - _index_time) < INDEX_TTL:
        return _index
    with _index_lock:
        now = time.monotonic()
        if force or not _index or (now - _index_time) >= INDEX_TTL:
            _index = _scan_books_uncached()
            _index_time = now
        return _index


# ---------------------------------------------------------------------------
# Open-document handles and page counts (cached)
# ---------------------------------------------------------------------------

_docs = OrderedDict()          # path -> open pymupdf.Document (LRU)
_page_counts = {}              # (path, mtime_ns, size) -> int


def _get_doc(path: Path):
    """Must be called while holding _render_lock."""
    key = str(path)
    doc = _docs.get(key)
    if doc is not None:
        _docs.move_to_end(key)
        return doc
    doc = pymupdf.open(path)
    _docs[key] = doc
    while len(_docs) > MAX_OPEN_DOCS:
        _, old = _docs.popitem(last=False)
        try:
            old.close()
        except Exception:
            pass
    return doc


def _page_count(path: Path) -> int:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    cached = _page_counts.get(key)
    if cached is not None:
        return cached
    with _render_lock:
        count = _get_doc(path).page_count
    _page_counts[key] = count
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_textbooks():
    """Returns [{id, title, subject, page_count}, ...], one entry per PDF
    found under TEXTBOOKS_DIR. Called from main.py's /textbooks route."""
    out = []
    for bid, meta in _scan_books().items():
        try:
            pages = _page_count(meta["path"])
        except Exception:
            # Skip a corrupt/unreadable PDF rather than failing the whole list.
            continue
        out.append({
            "id": bid,
            "title": meta["title"],
            "subject": meta["subject"],
            "page_count": pages,
        })
    out.sort(key=lambda b: b["title"].casefold())
    return out


def _page_dest(book_id: str, page: int) -> Path:
    return CACHE_DIR / book_id / f"page_{page:04d}.jpg"


def _render_page(meta: dict, book_id: str, page: int) -> Path:
    """Renders one page to the cache. Raises PageOutOfRange if needed."""
    dest = _page_dest(book_id, page)
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.{threading.get_ident()}.tmp")

    with _render_lock:
        # A concurrent request may have rendered it while we waited.
        if dest.exists():
            return dest
        try:
            doc = _get_doc(meta["path"])
            if page < 1 or page > doc.page_count:
                raise PageOutOfRange(page)
            # alpha=False skips the transparency channel (smaller, faster).
            pix = doc[page - 1].get_pixmap(dpi=RENDER_DPI, alpha=False)
            try:
                data = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)
            except TypeError:  # very old PyMuPDF without jpg_quality
                data = pix.tobytes("jpeg")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        except PageOutOfRange:
            raise
        except Exception:
            # A cached handle can go bad (file replaced on redeploy, etc).
            # Drop it so the next request re-opens the file cleanly.
            stale = _docs.pop(str(meta["path"]), None)
            if stale is not None:
                try:
                    stale.close()
                except Exception:
                    pass
            raise
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    return dest


def _prefetch(book_id: str, start: int, count: int):
    meta = _scan_books().get(book_id)
    if not meta:
        return
    for p in range(start, start + count):
        key = (book_id, p)
        if _page_dest(book_id, p).exists():
            continue
        with _prefetch_guard:
            if key in _prefetch_inflight:
                continue
            _prefetch_inflight.add(key)
        try:
            _render_page(meta, book_id, p)
        except PageOutOfRange:
            return  # reached the end of the book
        except Exception:
            return  # prefetch is best effort, never raise
        finally:
            with _prefetch_guard:
                _prefetch_inflight.discard(key)


def get_page_path(book_id: str, page: int) -> Path:
    """Returns the local path to `page` (1-indexed) of the given textbook,
    rendering and caching it first if needed. Raises TextbookNotFound /
    PageOutOfRange (main.py turns those into 404s)."""
    meta = _scan_books().get(book_id)
    if not meta:
        # The index may be a few seconds stale (new book just deployed).
        meta = _scan_books(force=True).get(book_id)
        if not meta:
            raise TextbookNotFound(book_id)

    if page < 1:
        raise PageOutOfRange(page)

    dest = _render_page(meta, book_id, page)

    # Warm the next pages in the background so the next swipe is instant.
    if PREFETCH_PAGES > 0 and not _page_dest(book_id, page + 1).exists():
        try:
            _prefetch_pool.submit(_prefetch, book_id, page + 1, PREFETCH_PAGES)
        except RuntimeError:
            pass  # pool shut down during server exit
    return dest


def warm_up(first_pages: int = 3):
    """Optional. Pre-renders the first few pages of every book so the first
    student after a redeploy doesn't wait. Call it from a background thread
    at startup, for example in main.py:

        import threading, textbooks
        threading.Thread(target=textbooks.warm_up, daemon=True).start()
    """
    for bid in list(_scan_books().keys()):
        _prefetch(bid, 1, first_pages)
