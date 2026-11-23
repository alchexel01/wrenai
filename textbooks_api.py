"""
Textbook Library backend logic for Wren AI (fast + crash-resistant version).

Same public API as before (list_textbooks, get_page_path, warm_up,
TextbookNotFound, PageOutOfRange, PageSuperseded), so main.py does not need
to change.

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

SPEED FEATURES (all kept from the previous version)
----------------------------------------------------
1. The book index (folder scan) is cached in memory for a short time.
2. Page counts are cached, so /textbooks never re-opens every PDF.
3. Open PDF handles are reused (small LRU).
4. Pages are small JPEGs (110 DPI, quality 72), capped at 1000 px wide.
5. The next 4 pages (and 1 page back) around the requested page are rendered
   in the background, so page turns are usually cache hits.
6. As soon as a book is opened, EVERY page of it is rendered in the
   background, nearest pages first, re-targeting whenever the student jumps.
7. The first pages of every book are pre-rendered a few seconds after the
   server starts, whenever nobody is reading.

WHY THE OLD VERSION CRASHED ON FAST SCROLLING, AND WHAT CHANGED
----------------------------------------------------------------
Old: every waiting page request sat in a loop re-checking a shared condition,
and every finished render woke ALL of them (hundreds of threads, each scanning
all the others). That burned the CPU on a small server, starved the rest of
the API of threads, and made renders slower, which made the pile grow.

New:
 * ONE render worker thread does all rendering (student pages first, then
   nearby pages, then the rest of the book, then warm-up). No locks to fight
   over, no polling loops, no sleeping-and-rechecking threads.
 * A request thread just sleeps on its own private Event until its page is
   ready. Nothing wakes it up except its own page being finished.
 * Student requests are served NEWEST FIRST. When someone drags the page
   slider, the newest request is the page they stopped on, so that is the one
   rendered first. This also works with many students at once, because it
   needs no per-student bookkeeping.
 * The student queue is capped (MAX_WAITING). If fast scrolling creates more
   waiting requests than that, the OLDEST ones (pages already flown past) are
   answered immediately with PageSuperseded, which main.py already turns into
   a 404. The page the student stopped on is never dropped, and dropped pages
   are just rendered later if the student comes back to them.
 * Duplicate requests for the same page share one render.
 * Background rendering pauses briefly between pages so the chat endpoints
   still get CPU, and MuPDF's internal memory cache is trimmed after renders.
 * If a page always fails to render, background work gives up on it instead
   of retrying forever.

Install: pip install pymupdf   (add to requirements.txt)

NOTE for main.py: the page route must be a plain `def`, not `async def`
(FastAPI then runs it in its thread pool). An `async def` route that calls
get_page_path directly would block the whole server while a page renders.
"""

import os
import time
import hashlib
import threading
from collections import OrderedDict, deque
from pathlib import Path

try:
    import pymupdf  # PyMuPDF >= 1.24.3
except ImportError:  # older releases only ship `fitz`
    import fitz as pymupdf

TEXTBOOKS_DIR = Path(os.environ.get("TEXTBOOKS_DIR", "./textbooks")).resolve()
CACHE_DIR = Path(os.environ.get("TEXTBOOKS_CACHE_DIR", "./textbooks_cache")).resolve()

# 110 DPI is sharp enough for a phone screen and renders much faster than 150.
RENDER_DPI = int(os.environ.get("TEXTBOOKS_RENDER_DPI", "110"))
JPEG_QUALITY = int(os.environ.get("TEXTBOOKS_JPEG_QUALITY", "72"))
# Phones never need more than about 1000 px of width.
MAX_WIDTH_PX = int(os.environ.get("TEXTBOOKS_MAX_WIDTH_PX", "1000"))

INDEX_TTL = float(os.environ.get("TEXTBOOKS_INDEX_TTL", "60"))   # seconds
MAX_OPEN_DOCS = int(os.environ.get("TEXTBOOKS_MAX_OPEN_DOCS", "4"))

# Pages prepared around the one the student is on.
PREFETCH_PAGES = int(os.environ.get("TEXTBOOKS_PREFETCH_PAGES", "4"))
PREFETCH_BEHIND = int(os.environ.get("TEXTBOOKS_PREFETCH_BEHIND", "1"))

# Render EVERY page of the book being read, nearest first. Stops if nobody has
# asked for that book for FULL_BOOK_IDLE_STOP seconds.
FULL_BOOK = os.environ.get("TEXTBOOKS_PRERENDER_BOOK", "1") == "1"
FULL_BOOK_IDLE_STOP = float(os.environ.get("TEXTBOOKS_PRERENDER_IDLE_STOP", "900"))

# Max student requests allowed to wait for a render at once. Beyond this, the
# OLDEST waiting requests (pages the student already scrolled past) get a
# quick PageSuperseded (404) instead of piling up and starving the server.
# Keep it well below your server's thread pool (FastAPI's default is 40).
# Set to 0 to disable the cap (not recommended).
MAX_WAITING = int(os.environ.get("TEXTBOOKS_MAX_WAITING", "24"))

# When a student jumps (slider / fast scroll), their OWN queued requests for
# pages more than this many pages away from the new one are cancelled: those
# are the pages they flew past, and rendering them would delay the pages
# around where they landed. Only the same student's requests are cancelled
# (matched by the app's `sid`), so other readers of the same book are never
# affected. 0 turns this off.
JUMP_DROP_PAGES = int(os.environ.get("TEXTBOOKS_JUMP_DROP_PAGES", "12"))

# A request that has waited this long gives up instead of hanging forever.
REQUEST_TIMEOUT = float(os.environ.get("TEXTBOOKS_REQUEST_TIMEOUT", "20"))

# Pause (seconds) after each background page so the rest of the API gets CPU.
BG_PAUSE = float(os.environ.get("TEXTBOOKS_BG_PAUSE", "0.1"))

# MuPDF keeps an internal cache that can grow large on image-heavy pages.
# Above this size (MB) it is emptied after a render. Keeps memory flat.
MUPDF_STORE_MB = int(os.environ.get("TEXTBOOKS_MUPDF_STORE_MB", "48"))

# Pre-render the first pages of every book shortly after start-up, but only
# while nobody has requested a page for WARMUP_IDLE seconds.
AUTO_WARMUP = os.environ.get("TEXTBOOKS_AUTO_WARMUP", "1") == "1"
AUTO_WARMUP_PAGES = int(os.environ.get("TEXTBOOKS_AUTO_WARMUP_PAGES", "1"))
AUTO_WARMUP_DELAY = float(os.environ.get("TEXTBOOKS_AUTO_WARMUP_DELAY", "5"))
WARMUP_IDLE = 20.0

# Timing lines go to stdout so they show up in Render's log viewer.
# Set TEXTBOOKS_DEBUG=0 to silence them.
DEBUG = os.environ.get("TEXTBOOKS_DEBUG", "1") == "1"

TEXTBOOKS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str):
    if DEBUG:
        print(f"[textbooks] {msg}", flush=True)


class TextbookNotFound(Exception):
    """Raised when book_id doesn't match any PDF under TEXTBOOKS_DIR.
    main.py catches this and turns it into a 404."""
    pass


class PageOutOfRange(Exception):
    """Raised when the requested page number isn't in the document.
    main.py catches this and turns it into a 404."""
    pass


class PageSuperseded(PageOutOfRange):
    """The student scrolled far past this page while it was waiting.
    Subclasses PageOutOfRange so main.py's existing handler returns a 404.
    Asking again later works normally."""
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

# PyMuPDF is not safe to call from several threads at once. Only the render
# worker renders, and this lock additionally protects the shared handles
# against the (rare, cached) page-count lookups made by request threads.
_render_lock = threading.RLock()

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
        doc = _docs.get(str(path))
        if doc is not None:
            count = doc.page_count
        else:
            # Open briefly and close, so counting pages never pushes the
            # book the student is reading out of the handle cache.
            with pymupdf.open(path) as tmp_doc:
                count = tmp_doc.page_count
    _page_counts[key] = count
    return count


def _trim_mupdf_store():
    """Empties MuPDF's internal cache when it grows past MUPDF_STORE_MB.
    Call while holding _render_lock."""
    try:
        tools = pymupdf.TOOLS
        if tools.store_size > MUPDF_STORE_MB * 1024 * 1024:
            tools.store_shrink(100)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Which pages are already on disk (kept in memory, so nobody lists the
# cache folder over and over)
# ---------------------------------------------------------------------------

_have = {}  # book_id -> set of page numbers on disk (or not worth retrying)
_have_lock = threading.Lock()


def _have_pages(book_id: str) -> set:
    with _have_lock:
        s = _have.get(book_id)
        if s is None:
            s = set()
            try:
                for n in os.listdir(CACHE_DIR / book_id):
                    if n.startswith("page_") and n.endswith(".jpg"):
                        try:
                            s.add(int(n[5:-4]))
                        except ValueError:
                            pass
            except FileNotFoundError:
                pass
            _have[book_id] = s
        return s


def _page_dest(book_id: str, page: int) -> Path:
    return CACHE_DIR / book_id / f"page_{page:04d}.jpg"


# ---------------------------------------------------------------------------
# Rendering (only ever called from the render worker thread)
# ---------------------------------------------------------------------------

def _render_to_cache(meta: dict, book_id: str, page: int, source: str) -> Path:
    """Renders one page into the cache. source is "user", "prefetch", "book"
    or "warmup". Raises PageOutOfRange if the page isn't in the document."""
    dest = _page_dest(book_id, page)
    if dest.exists():
        _have_pages(book_id).add(page)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.{threading.get_ident()}.tmp")
    t0 = time.perf_counter()
    temp_doc = None
    try:
        with _render_lock:
            t_locked = time.perf_counter()
            if source == "warmup":
                # Warm-up touches many books: use a throwaway handle so it
                # does not evict the book a student is reading.
                temp_doc = pymupdf.open(meta["path"])
                doc = temp_doc
            else:
                doc = _get_doc(meta["path"])
            t_open = time.perf_counter()
            if page < 1 or page > doc.page_count:
                raise PageOutOfRange(page)
            pg = doc[page - 1]
            zoom = RENDER_DPI / 72.0
            if pg.rect.width * zoom > MAX_WIDTH_PX:
                zoom = MAX_WIDTH_PX / pg.rect.width
            pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            t_render = time.perf_counter()
            try:
                data = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)
            except TypeError:  # very old PyMuPDF without jpg_quality
                data = pix.tobytes("jpeg")
            pix = None
            pg = None
            t_encode = time.perf_counter()
            _trim_mupdf_store()
        # Disk write happens outside the lock.
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        t_done = time.perf_counter()
        _log(
            f"{source} {book_id[:6]} p{page}: "
            f"open {t_open - t_locked:.2f}s, render {t_render - t_open:.2f}s, "
            f"encode {t_encode - t_render:.2f}s, write {t_done - t_encode:.2f}s, "
            f"total {t_done - t0:.2f}s, {len(data) // 1024} KB"
        )
    except PageOutOfRange:
        raise
    except Exception as e:
        _log(f"{source} {book_id[:6]} p{page}: FAILED {type(e).__name__}: {e}")
        # A cached handle can go bad (file replaced on redeploy, etc).
        # Drop it so the next request re-opens the file cleanly.
        with _render_lock:
            stale = _docs.pop(str(meta["path"]), None)
        if stale is not None:
            try:
                stale.close()
            except Exception:
                pass
        raise
    finally:
        if temp_doc is not None:
            try:
                temp_doc.close()
            except Exception:
                pass
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    _have_pages(book_id).add(page)
    return dest


# ---------------------------------------------------------------------------
# Scheduler: student requests, newest first, served by one worker thread
# ---------------------------------------------------------------------------

_QUEUED, _RUNNING, _DONE = 0, 1, 2


class _Job:
    """One page a student is waiting for. Any number of request threads can
    wait on the same job; each sleeps on the shared Event."""
    __slots__ = ("book_id", "page", "seq", "state", "event", "error", "student")

    def __init__(self, book_id, page, seq, student=""):
        self.student = student
        self.book_id = book_id
        self.page = page
        self.seq = seq              # higher = requested more recently
        self.state = _QUEUED
        self.event = threading.Event()
        self.error = None


# _cv guards _jobs, _focus and the counters. Its lock is re-entrant.
_cv = threading.Condition()
_jobs = {}                         # (book_id, page) -> _Job (queued or running)
_seq = [0]
_focus = {"book": None, "page": 1, "t": -1e9}   # where a student last looked
_focus_version = [0]
_book_last_request = {}            # book_id -> monotonic time of last request
_START = time.monotonic()


def _note_request(book_id: str, page: int):
    """Records where the student is. Background work follows this."""
    with _cv:
        now = time.monotonic()
        _focus["book"] = book_id
        _focus["page"] = page
        _focus["t"] = now
        _focus_version[0] += 1
        _book_last_request[book_id] = now
        _cv.notify()


def _drop(job: _Job):
    """Answers a still-queued job with PageSuperseded. Hold _cv."""
    job.error = PageSuperseded(job.page)
    job.state = _DONE
    key = (job.book_id, job.page)
    if _jobs.get(key) is job:
        del _jobs[key]
    job.event.set()


def _submit(book_id: str, page: int, student: str = "") -> _Job:
    key = (book_id, page)
    with _cv:
        _note_request(book_id, page)
        _seq[0] += 1
        job = _jobs.get(key)
        if student and JUMP_DROP_PAGES > 0:
            # This student jumped: cancel THEIR queued requests for pages
            # far from here (never a render that is already running).
            for old in [j for j in _jobs.values()
                        if j.state == _QUEUED and j.student == student
                        and j.book_id == book_id and j is not job
                        and abs(j.page - page) > JUMP_DROP_PAGES]:
                _log(f"user {old.book_id[:6]} p{old.page}: "
                     f"cancelled (student jumped to p{page})")
                _drop(old)
        if job is None:
            job = _Job(book_id, page, _seq[0], student)
            _jobs[key] = job
            if MAX_WAITING > 0:
                queued = [j for j in _jobs.values() if j.state == _QUEUED]
                excess = len(queued) - MAX_WAITING
                if excess > 0:
                    queued.sort(key=lambda j: j.seq)   # oldest first
                    for old in queued[:excess]:
                        _log(f"user {old.book_id[:6]} p{old.page}: "
                             f"dropped (student scrolled past)")
                        _drop(old)
        else:
            job.seq = _seq[0]     # asked again: it matters now
        _cv.notify()
        return job


def _wait_for(job: _Job):
    if not job.event.wait(REQUEST_TIMEOUT):
        with _cv:
            if job.state == _QUEUED:
                _drop(job)
        job.event.wait(10)  # a render that is already running will finish
        if not job.event.is_set():
            raise PageSuperseded(job.page)
    if job.error is not None:
        raise job.error


def _pick_user_job():
    """Newest queued student request, or None. Hold _cv."""
    best = None
    for j in _jobs.values():
        if j.state == _QUEUED and (best is None or j.seq > best.seq):
            best = j
    return best


def _run_user_job(job: _Job):
    key = (job.book_id, job.page)
    try:
        meta = _scan_books().get(job.book_id)
        if meta is None:
            raise TextbookNotFound(job.book_id)
        _render_to_cache(meta, job.book_id, job.page, "user")
    except Exception as e:
        job.error = e
    finally:
        with _cv:
            job.state = _DONE
            if _jobs.get(key) is job:
                del _jobs[key]
        job.event.set()


def _outward(center: int, total: int):
    """Page numbers nearest to `center` first, looking about twice as far
    ahead as behind."""
    center = max(1, min(center, total))
    yield center
    ahead, behind = center + 1, center - 1
    while ahead <= total or behind >= 1:
        for _ in range(2):
            if ahead <= total:
                yield ahead
                ahead += 1
        if behind >= 1:
            yield behind
            behind -= 1


_warm = {"todo": deque(), "built": False}


def warm_up(first_pages: int = 2):
    """Queues the first pages of every book for pre-rendering. The render
    worker does them while nobody is reading. Runs automatically a few
    seconds after start-up (see AUTO_WARMUP), so you normally never call
    this."""
    todo = deque()
    for bid, meta in list(_scan_books().items()):
        for p in range(1, first_pages + 1):
            todo.append((bid, p, meta))
    with _cv:
        _warm["todo"] = todo
        _warm["built"] = True
        _cv.notify()
    _ensure_worker()


def _pick_background(focus: dict, now: float):
    """Next background page to render, or None.
    Order: pages around the student, then the rest of the book, then warm-up.
    Returns (source, book_id, page, meta)."""
    book_id = focus["book"]
    if book_id:
        meta = _scan_books().get(book_id)
        if meta:
            try:
                total = _page_count(meta["path"])
            except Exception:
                total = 0
            if total:
                have = _have_pages(book_id)
                page = focus["page"]

                near = [page + i for i in range(1, PREFETCH_PAGES + 1)]
                near += [page - i for i in range(1, PREFETCH_BEHIND + 1)]
                for p in near:
                    if 1 <= p <= total and p not in have:
                        return ("prefetch", book_id, p, meta)

                if FULL_BOOK and (now - _book_last_request.get(book_id, 0)
                                  <= FULL_BOOK_IDLE_STOP):
                    for p in _outward(page, total):
                        if p not in have:
                            return ("book", book_id, p, meta)

    if now - focus["t"] > WARMUP_IDLE:
        if AUTO_WARMUP and not _warm["built"] and now - _START >= AUTO_WARMUP_DELAY:
            warm_up(AUTO_WARMUP_PAGES)
        todo = _warm["todo"]
        while todo:
            bid, p, meta = todo.popleft()
            if p not in _have_pages(bid):
                return ("warmup", bid, p, meta)
    return None


def _worker():
    """The only thread that renders. Never exits, never busy-loops."""
    while True:
        try:
            with _cv:
                job = _pick_user_job()
                if job is not None:
                    job.state = _RUNNING
                focus = dict(_focus)
                version = _focus_version[0]

            if job is not None:
                _run_user_job(job)
                continue

            task = _pick_background(focus, time.monotonic())
            if task is None:
                with _cv:
                    # Sleep only if nothing arrived since we looked.
                    if _pick_user_job() is None and _focus_version[0] == version:
                        _cv.wait(2.0)
                continue

            source, bid, page, meta = task
            ok = False
            try:
                _render_to_cache(meta, bid, page, source)
                ok = True
            except Exception:
                pass  # already logged; PageOutOfRange needs no log
            if not ok:
                # Never retry a failing page in the background (students
                # asking for it directly still get a fresh attempt).
                _have_pages(bid).add(page)
            if source != "prefetch" or not ok:
                time.sleep(BG_PAUSE)   # let the rest of the API breathe
        except Exception as e:
            _log(f"worker error {type(e).__name__}: {e}")
            time.sleep(0.5)


_worker_lock = threading.Lock()
_worker_thread = None


def _ensure_worker():
    """Starts the render worker if it isn't running (also covers servers that
    fork worker processes after this module was imported)."""
    global _worker_thread
    t = _worker_thread
    if t is not None and t.is_alive():
        return
    with _worker_lock:
        t = _worker_thread
        if t is None or not t.is_alive():
            _worker_thread = threading.Thread(
                target=_worker, daemon=True, name="tb-render")
            _worker_thread.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_textbooks():
    """Returns [{id, title, subject, page_count}, ...], one entry per PDF
    found under TEXTBOOKS_DIR. Called from main.py's /textbooks route."""
    _ensure_worker()
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


def ready_pages(book_id: str):
    """Page numbers of this book that are already rendered and waiting on
    disk, sorted. Cheap (one folder listing) and never renders anything, so
    the app can poll it to download pages the moment they are ready."""
    if not _scan_books().get(book_id) and not _scan_books(force=True).get(book_id):
        raise TextbookNotFound(book_id)
    out = []
    try:
        for n in os.listdir(CACHE_DIR / book_id):
            if n.startswith("page_") and n.endswith(".jpg"):
                try:
                    out.append(int(n[5:-4]))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    out.sort()
    return out


def get_page_path(book_id: str, page: int, track: bool = True,
                  student: str = "") -> Path:
    """Returns the local path to `page` (1-indexed) of the given textbook,
    rendering and caching it first if needed. Raises TextbookNotFound /
    PageOutOfRange (main.py turns those into 404s).

    track=False is for the app's background downloader: it only ever gets
    pages that are already rendered, never triggers a render, and does not
    move the "where is the student" marker the background rendering follows."""
    _ensure_worker()

    meta = _scan_books().get(book_id)
    if not meta:
        # The index may be a few seconds stale (new book just deployed).
        meta = _scan_books(force=True).get(book_id)
        if not meta:
            raise TextbookNotFound(book_id)

    if page < 1:
        raise PageOutOfRange(page)
    try:
        total = _page_count(meta["path"])
    except Exception:
        total = None  # let the render report the real problem
    if total is not None and page > total:
        raise PageOutOfRange(page)

    dest = _page_dest(book_id, page)
    if dest.exists():
        if track:
            _note_request(book_id, page)   # background work follows the student
            _log(f"user {book_id[:6]} p{page}: cache hit")
        return dest
    if not track:
        raise PageSuperseded(page)         # not ready yet: don't render it

    t_req = time.perf_counter()
    job = _submit(book_id, page, student)
    _wait_for(job)
    if not dest.exists():
        raise PageOutOfRange(page)
    _log(f"user {book_id[:6]} p{page}: served after "
         f"{time.perf_counter() - t_req:.2f}s")
    return dest


_ensure_worker()
