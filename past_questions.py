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
      "image": null
    }

To add a paper: run the parser, commit the new .json, push. Render
redeploys and the new paper shows up in /past-questions/papers.
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

# path -> (mtime, questions). Files are small; this just avoids re-parsing
# the same JSON on every request while still picking up a redeploy.
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
    image: Optional[str] = None


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


def _load(subject: str, year: int) -> list:
    slug = _slug(subject)
    path = DATA_DIR / f"{slug}_{year}.json"
    if not slug or not path.is_file():
        return []
    mtime = path.stat().st_mtime
    hit = _CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"past_questions: could not read {path.name}: {e!r}")
        return []
    _CACHE[str(path)] = (mtime, data)
    return data


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
