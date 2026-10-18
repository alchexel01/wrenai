"""
Wren News and Updates - the folder IS the catalog
==============================================
No admin page, no database. To post a news item: create a text file in
the news/ folder of the GitHub repo and commit it. Render
auto-deploys the commit (same as syllabus_data/ and textbooks/), and the
app shows the new item the next time the News and Updates screen opens.

    news/
      Undergraduate/
        2026-10-16 - Exam Timetable Released.txt
      Postgraduate/
        JAMB Registration Update.txt
      Anything.txt                      <- no folder is fine too

HOW A FILE IS READ
------------------
  * Title     = the file name (minus .txt/.md and an optional date prefix).
  * Category  = the folder it sits in ('' if directly in news/).
  * Body      = the file's text, exactly as you wrote it.
  * Date      = used to sort newest-first. Taken from, in order:
                a "Date: YYYY-MM-DD" line at the top of the file, a
                "YYYY-MM-DD - " prefix in the file name, else the file's
                modified time (which on Render is the deploy time - so add
                a date if you care about the order).

OPTIONAL HEADER LINES (at the very top of the file, before the text):
    Title: Custom title that overrides the file name
    Link: https://where-to-apply.example
    Date: 2026-10-16
(then a blank line, then the news text)

SKIPPED: empty files, files whose name starts with "_" or "." (use
"_draft.txt" to keep a draft in the repo without publishing it), and
README files.

This module is plain Python - main.py owns the routes, request logging
(rid) and the shared _check_app_secret() gate, same split as
textbooks_api.py / rag_engine.py.
"""

import os
import re
import hashlib
import logging
import datetime as dt
from pathlib import Path
from typing import Optional, List

from fastapi import HTTPException
from pydantic import BaseModel

log = logging.getLogger("wren-backend")

NEWS_DIR = Path(os.environ.get(
    "NEWS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "news"),
)).resolve()
NEWS_DIR.mkdir(parents=True, exist_ok=True)

_EXTS = {".txt", ".md"}
_HEADER_RE = re.compile(r"^\s*(title|link|date)\s*:\s*(.*?)\s*$", re.I)
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*[-_ ]\s*(.+)$")


class NewsItem(BaseModel):
    id: str
    title: str
    body: str
    link: Optional[str] = None
    category: str = ""
    created_at: str


class NewsListResponse(BaseModel):
    news: List[NewsItem]


def _item_id(rel_path: Path) -> str:
    return hashlib.sha1(rel_path.as_posix().encode("utf-8")).hexdigest()[:16]


def _parse_date(s):
    try:
        return dt.datetime.strptime((s or "").strip()[:10], "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _read_item(path: Path):
    """Returns a NewsItem for one file, or None to skip it."""
    rel = path.relative_to(NEWS_DIR)
    name = path.name
    if (path.suffix.lower() not in _EXTS or name.startswith(("_", "."))
            or path.stem.lower().startswith("readme")):
        return None
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as e:
        log.warning(f"[news] could not read {rel}: {e}")
        return None

    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    meta = {}
    while i < len(lines):
        m = _HEADER_RE.match(lines[i])
        if not m:
            break
        meta[m.group(1).lower()] = m.group(2)
        i += 1
    body = "\n".join(lines[i:]).strip()
    if not body:
        return None

    stem = path.stem
    prefix_date = None
    pm = _DATE_PREFIX_RE.match(stem)
    if pm and _parse_date(pm.group(1)):
        prefix_date, stem = pm.group(1), pm.group(2)
    title = (meta.get("title") or "").strip() or stem.replace("_", " ").strip() or "News"

    when = _parse_date(meta.get("date")) or _parse_date(prefix_date)
    if when is None:
        try:
            when = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except Exception:
            when = dt.datetime.now(dt.timezone.utc)

    link = (meta.get("link") or "").strip()
    if not re.match(r"^https?://", link):
        link = None

    return NewsItem(
        id=_item_id(rel), title=title, body=body, link=link,
        category=rel.parent.name if rel.parent != Path(".") else "",
        created_at=when.isoformat(),
    )


def _scan():
    items = []
    for p in NEWS_DIR.rglob("*"):
        if p.is_file():
            it = _read_item(p)
            if it:
                items.append(it)
    items.sort(key=lambda x: (x.created_at, x.title.casefold()), reverse=True)
    return items


def list_news(limit: int = 50) -> NewsListResponse:
    limit = max(1, min(int(limit), 100))
    return NewsListResponse(news=_scan()[:limit])


def get_news_item(item_id: str, rid: str) -> NewsItem:
    for it in _scan():
        if it.id == item_id:
            return it
    raise HTTPException(status_code=404,
                         detail={"error": "news item not found", "request_id": rid})