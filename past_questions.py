"""
Past Questions
==============
Serves past exam questions (JAMB etc.) from JSON files that live in the
repo, so new papers can be added without shipping a new APK. Same design
as scholarships.py / news.py: plain module here, thin routes in main.py.

Folder convention (next to main.py):

    past_questions_data/
        physics_2019.json
        chemistry_2020.json
        use_of_english_2018.json      <- spaces in a subject become "_"
        raw/                           <- OCR .txt files (git-ignored)

Each JSON file is a list of questions produced by tools/parse_questions.py:

    {
      "id": "physics-2019-14",
      "subject": "physics",
      "year": 2019,
      "number": 14,
      "question": "The graph ...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "B",
      "explanation": "Short reason why B is correct",
      "has_diagram": false,
      "image": "physics/1983_p2_1.png",
      "images": ["physics/1983_p2_1.png"]
    }

Diagram pictures live in past_questions_data/images/<subject>/<file>.png and
are served by /past-questions/images/<subject>/<file>.png (see image_path()).

To add a paper: run the parser, commit the new .json, push. Render
redeploys and the new paper shows up in /past-questions/papers.

ANSWERS can live in a separate file, so no script has to be run on a PC:

    past_questions_data/answers/physics_1983.json

        {"year": 1983, "answers": [
            {"number": 1, "answer": "C", "explanation": "Short reason."},
            {"number": 3, "answer": null, "explanation": null}
        ]}

FIXES let you correct questions that came out of the PDF wrongly (blank options,
a question merged into another one's options, ...) without touching the main
files and without re-running anything on a PC:

    past_questions_data/fixes/physics_1998.json

        {"year": 1998, "questions": [
            {"number": 11, "options": {"A": "cos e", "B": "sin e / sin b", "C": "...", "D": "..."}},
            {"number": 12, "question": "The pressure of a mass of gas changes ...",
             "options": {"A": "5 : 4", "B": "5 : 2", "C": "4 : 5", "D": "2 : 5"}},
            {"number": 30, "delete": true}
        ]}

A fix replaces only the fields it contains, so images, id and so on stay as they
were. A number that does not exist yet is added; "delete": true removes one.
Fixes are applied first, then answers.

Answers and fixes are both merged into the questions when a paper is served.
An answer in an answers file replaces the one in the question file; null or
missing keeps whatever the question file already has. Upload a new answers or
fixes file to GitHub and Render redeploys: done.
"""

import json
import logging
import pathlib
import random
import re
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

log = logging.getLogger("wren-backend")

DATA_DIR = pathlib.Path(__file__).parent / "past_questions_data"

# path -> ((mtime of questions, answers, fixes), questions). Files are small;
# this just avoids re-parsing the same JSON on every request while still
# picking up a redeploy.
_CACHE: dict = {}


class PastQuestion(BaseModel):
    id: str
    subject: str
    year: int
    number: int
    question: str
    options: dict[str, str]
    answer: Optional[str] = None
    explanation: Optional[str] = None
    has_diagram: bool = False
    options_are_images: bool = False
    image: Optional[str] = None            # first picture, e.g. "physics/1983_p2_1.png"
    images: list[str] = []                 # every picture of the question, in order


class PapersResponse(BaseModel):
    # {"physics": [2018, 2019], "chemistry": [2020]}
    papers: dict[str, list[int]]


class QuestionsResponse(BaseModel):
    subject: str
    year: int
    total: int
    questions: list[PastQuestion]


def _slug(subject: str) -> str:
    """'Use of English' -> 'use_of_english'. Also blocks path tricks like '../'."""
    return re.sub(r"[^a-z0-9]+", "_", (subject or "").lower()).strip("_")


def _read_json(path: pathlib.Path):
    text = path.read_text(encoding="utf-8").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()   # tolerate a pasted code block
    return json.loads(text)


def _answers_for(slug: str, year: int) -> dict:
    """{number: (letter, explanation)} from answers/<slug>_<year>.json, or {}."""
    path = DATA_DIR / "answers" / f"{slug}_{year}.json"
    if not path.is_file():
        return {}
    data = _read_json(path)
    blocks = data if isinstance(data, list) else [data]
    out = {}
    for blk in blocks:
        answers = blk.get("answers", []) if isinstance(blk, dict) else []
        if isinstance(answers, dict):
            answers = [{"number": k, "answer": v} for k, v in answers.items()]
        for a in answers:
            try:
                num = int(a.get("number"))
            except (TypeError, ValueError, AttributeError):
                continue
            letter = str(a.get("answer") or "").strip().upper()[:1]
            if letter not in ("A", "B", "C", "D", "E"):
                continue
            expl = " ".join(str(a.get("explanation") or "").split()) or None
            out[num] = (letter, expl)
    return out


def _read_fixes(slug: str, year: int) -> list:
    path = DATA_DIR / "fixes" / f"{slug}_{year}.json"
    if not path.is_file():
        return []
    data = _read_json(path)
    if isinstance(data, dict):
        items = data.get("questions", [])
    else:
        items = []
        for blk in data:
            items += blk.get("questions", []) if isinstance(blk, dict) and "questions" in blk else [blk]
    return [q for q in items if isinstance(q, dict)]


def _apply_fixes(data: list, fixes: list, slug: str, year: int) -> list:
    by_num = {q["number"]: q for q in data if isinstance(q.get("number"), int)}
    for fx in fixes:
        try:
            n = int(fx.get("number"))
        except (TypeError, ValueError):
            continue
        if fx.get("delete") is True:
            by_num.pop(n, None)
            continue
        base = by_num.get(n) or {
            "id": f"{slug.replace('_', '-')}-{year}-{n}", "subject": slug, "year": year,
            "number": n, "question": "", "options": {}, "answer": None,
            "explanation": None, "has_diagram": False, "options_are_images": False,
            "image": None, "images": []}
        merged = {**base, **{k: v for k, v in fx.items() if k not in ("number", "delete")}}
        merged["number"] = n
        if not isinstance(merged.get("options"), dict) or not isinstance(merged.get("question"), str):
            log.error(f"past_questions: ignoring a bad fix for {slug} {year} Q{n}")
            continue
        merged["options"] = {str(k): str(v) for k, v in merged["options"].items()}
        by_num[n] = merged
    return [by_num[k] for k in sorted(by_num)]


def _load(subject: str, year: int) -> list:
    slug = _slug(subject)
    path = DATA_DIR / f"{slug}_{year}.json"
    if not slug or not path.is_file():
        return []
    apath = DATA_DIR / "answers" / f"{slug}_{year}.json"
    fpath = DATA_DIR / "fixes" / f"{slug}_{year}.json"
    stamp = (path.stat().st_mtime, apath.stat().st_mtime if apath.is_file() else 0,
             fpath.stat().st_mtime if fpath.is_file() else 0)
    hit = _CACHE.get(str(path))
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        data = _read_json(path)
    except Exception as e:
        log.error(f"past_questions: could not read {path.name}: {e!r}")
        return []
    try:
        fixes = _read_fixes(slug, year)
        if fixes:
            data = _apply_fixes(data, fixes, slug, year)
    except Exception as e:
        # a broken fixes file must never take the questions down with it
        log.error(f"past_questions: could not apply fixes for {slug} {year}: {e!r}")
    try:
        key = _answers_for(slug, year)
    except Exception as e:
        # a broken answers file must never take the questions down with it
        log.error(f"past_questions: could not read answers for {slug} {year}: {e!r}")
        key = {}
    if key:
        for q in data:
            hit_ans = key.get(q.get("number"))
            if hit_ans:
                q["answer"], q["explanation"] = hit_ans
    _CACHE[str(path)] = (stamp, data)
    return data


def image_path(subject: str, filename: str) -> pathlib.Path:
    """Safe path to one diagram picture, or 404. Only plain *.png names pass,
    so '../' tricks cannot reach other files."""
    slug = _slug(subject)
    if not slug or not re.fullmatch(r"[A-Za-z0-9_.\-]+\.png", filename or ""):
        raise HTTPException(status_code=404, detail={"error": "image not found"})
    path = DATA_DIR / "images" / slug / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"error": "image not found"})
    return path


def list_papers() -> PapersResponse:
    papers: dict[str, list[int]] = {}
    if DATA_DIR.is_dir():
        for f in sorted(DATA_DIR.glob("*.json")):
            subject, _, year = f.stem.rpartition("_")
            if subject and year.isdigit():
                papers.setdefault(subject, []).append(int(year))
    for years in papers.values():
        years.sort(reverse=True)
    return PapersResponse(papers=papers)


def get_questions(subject: str, year: int, limit: int, offset: int,
                  shuffle: bool, rid: str) -> QuestionsResponse:
    qs = _load(subject, year)
    if not qs:
        log.warning(f"[{rid}] /past-questions: no paper for {subject!r} {year}")
        raise HTTPException(status_code=404,
                            detail={"error": "paper not found", "request_id": rid})
    qs = list(qs)
    if shuffle:
        random.shuffle(qs)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    return QuestionsResponse(subject=_slug(subject), year=year, total=len(qs),
                             questions=qs[offset:offset + limit])


# ══════════════════════════════════════════════════════════════════════════
#  FLAGGED QUESTIONS
#  The app sends a question here when a student flags it, so you can see
#  which ones look wrong and fix them with a fixes/<subject>_<year>.json file.
#
#    POST   /past-questions/flag              (app; same X-App-Secret as the rest)
#    GET    /past-questions/flags             (you; header X-Admin-Key = WREN_ADMIN_KEY)
#    DELETE /past-questions/flags/{id}        (you; header X-Admin-Key; clears it after fixing)
#
#  STORAGE: reports are appended to flags.jsonl inside PQ_FLAGS_DIR. Render's
#  normal disk is wiped on every deploy (including every time you push a new
#  paper), so for the reports to survive either attach a Render Disk and set
#  PQ_FLAGS_DIR to a folder on it (e.g. /var/data/pq_flags), or read them
#  before you deploy. Every report is ALSO written to the Render logs as a
#  single "PQ_FLAG {...}" line, as a backup.
# ══════════════════════════════════════════════════════════════════════════
import os
import tempfile
import threading
import time
from collections import Counter

_FLAG_LOCK = threading.Lock()


class FlagIn(BaseModel):
    id: str = ""
    subject: str
    year: int
    number: int
    question: str = ""
    options: dict[str, str] = {}
    answer: Optional[str] = None
    explanation: Optional[str] = None
    images: list[str] = []
    chosen: Optional[str] = None
    reason: str = "flagged"


def _flags_file() -> pathlib.Path:
    d = os.environ.get("PQ_FLAGS_DIR") or (
        "/var/data/pq_flags" if os.path.isdir("/var/data")
        else os.path.join(tempfile.gettempdir(), "pq_flags"))
    path = pathlib.Path(d)
    path.mkdir(parents=True, exist_ok=True)
    return path / "flags.jsonl"


def _flag_rows() -> list:
    f = _flags_file()
    if not f.is_file():
        return []
    rows = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def save_flag(body: FlagIn, device: str = "", rid: str = "-") -> dict:
    slug = _slug(body.subject)
    if not slug or not _load(slug, body.year):
        raise HTTPException(status_code=400, detail={"error": "unknown paper"})
    qid = (body.id or f"{slug.replace('_', '-')}-{body.year}-{body.number}")[:120]
    row = {
        "ts": int(time.time()), "id": qid, "subject": slug, "year": body.year,
        "number": body.number, "question": body.question[:3000],
        "options": {str(k)[:3]: str(v)[:500] for k, v in list(body.options.items())[:6]},
        "answer": (body.answer or None), "explanation": (body.explanation or None),
        "images": body.images[:10], "chosen": (body.chosen or None),
        "reason": body.reason[:40], "device": str(device or "")[:64],
    }
    line = json.dumps(row, ensure_ascii=False)
    log.warning(f"[{rid}] PQ_FLAG {line}")
    try:
        with _FLAG_LOCK:
            with _flags_file().open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as e:
        log.error(f"past_questions: could not store flag: {e!r}")   # still in the logs above
    return {"ok": True}


def list_flags(limit: int = 200) -> dict:
    """Flagged questions grouped by question, most-reported first. Each entry
    shows what the app showed, what the server has now, and a ready-to-edit
    fix entry for fixes/<subject>_<year>.json."""
    groups: dict = {}
    for r in _flag_rows():
        g = groups.setdefault(r.get("id"), {"rows": []})
        g["rows"].append(r)
    out = []
    for qid, g in groups.items():
        rows = g["rows"]
        last = max(rows, key=lambda r: r.get("ts", 0))
        slug, year, num = last.get("subject"), last.get("year"), last.get("number")
        current = next((q for q in _load(slug, year) if q.get("number") == num), None)
        shown = current or last
        out.append({
            "id": qid, "subject": slug, "year": year, "number": num,
            "reports": len(rows),
            "devices": len({r.get("device") for r in rows if r.get("device")}),
            "last_reported": last.get("ts"),
            "chosen_by_students": dict(Counter(r.get("chosen") or "none" for r in rows)),
            "question_now": (current or {}).get("question"),
            "options_now": (current or {}).get("options"),
            "answer_now": (current or {}).get("answer"),
            "images": (current or last).get("images"),
            "fix_file": f"past_questions_data/fixes/{slug}_{year}.json",
            "fix_template": {"number": num, "question": shown.get("question"),
                             "options": shown.get("options")},
        })
    out.sort(key=lambda e: (-e["reports"], -(e["last_reported"] or 0)))
    return {"total": len(out), "flags": out[:max(1, min(limit, 500))]}


def clear_flags(qid: str) -> dict:
    """Remove every report for one question (call after you have fixed it)."""
    with _FLAG_LOCK:
        keep = [r for r in _flag_rows() if r.get("id") != qid]
        f = _flags_file()
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                     encoding="utf-8")
    return {"ok": True}
