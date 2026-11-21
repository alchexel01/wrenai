"""
parse_questions.py  -  run ONCE on your PC, not on Render.
=============================================================
Turns the text your AI extracted from a past-question PDF into the JSON
files the backend serves (one file per year).

Put this file at:  tools/parse_questions.py   (in the backend repo root)

It expects the text to follow the extraction prompt:

    YEAR: 1983                      <- starts each year's paper
    1. Question text...
    [IMAGE]                         <- diagram sits between question and option A
    A. option
    B. option
    ...
    ANSWER KEY                      <- optional, after the questions of that year
    1. B | one short sentence explaining why
    2. C | ...

    YEAR: 1984
    1. ...

If the whole file is ONE paper with no YEAR lines, pass --year 2019.

Run from the repo ROOT (the folder that contains main.py):

    python tools/parse_questions.py physics past_questions_data/raw/physics.txt
    python tools/parse_questions.py physics past_questions_data/raw/physics.txt --year 2019
    python tools/parse_questions.py physics past_questions_data/raw/physics.txt --key past_questions_data/raw/physics_key.txt

--key is only needed when answers are in a separate file. That file can use
the same "YEAR: 1983" headings, or be a single year together with --year.

Output:  past_questions_data/physics_1983.json, physics_1984.json, ...

After it runs, read the report it prints. Every line starting with CHECK is
something to fix by hand in the JSON (or in the raw text, then re-run).
"""

import argparse
import json
import pathlib
import re

OUT_DIR = pathlib.Path("past_questions_data")

IMG = "[IMAGE]"
UNCLEAR = "[UNCLEAR]"

YEAR_RE = re.compile(r"(?im)^[ \t]*YEAR[ \t]*[:\-]?[ \t]*((?:19|20)\d{2})\b.*$")
KEY_RE = re.compile(r"(?im)^[ \t]*ANSWER[ \t]*KEYS?\b.*$")
NUM_RE = re.compile(r"(?m)^[ \t]*(\d{1,3})[.)](?:[ \t]+|[ \t]*$)")
OPT_RE = re.compile(r"(?m)^[ \t]*\(?([A-E])[.)](?:[ \t]+|[ \t]*$)")


def slug(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")


def clean(s: str) -> str:
    return " ".join(s.split())


# ── splitting the file ───────────────────────────────────────────────────
def split_years(text: str, default_year):
    """-> {year: text}. Sections with the same YEAR line are joined, so a
    repeated page header does not break a paper in two."""
    marks = list(YEAR_RE.finditer(text))
    if not marks:
        return {default_year: text} if default_year else {}
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.setdefault(int(m.group(1)), "")
        out[int(m.group(1))] += "\n" + text[m.end():end]
    return out


def split_questions(text: str):
    """Split on lines that start with '12.' or '12)'. A candidate is accepted
    only if its number is larger than the last one and at most 3 ahead, so a
    stray '2. 5 m' inside an option is not mistaken for a new question."""
    starts = []
    last = 0
    for m in NUM_RE.finditer(text):
        n = int(m.group(1))
        if last < n <= last + 3:
            starts.append((n, m.start(), m.end()))
            last = n
    blocks = []
    for i, (n, _s, body_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        blocks.append((n, text[body_start:end]))
    return blocks


def split_inline_options(text: str):
    """Fallback for 'question? A. 11 km B. 12 km C. 13 km D14 km' typed on one
    line. -> (stem, {letter: text}) or None."""
    m = re.search(r"(?:(?<=\s)|^)A[.)]\s*(?=\S)", text)
    if not m:
        return None
    marks = [("A", m.start(), m.end())]
    pos = m.end()
    for letter in "BCDE":
        mm = re.compile(r"(?<=\s)%s(?:[.)]\s*(?=\S)|(?=\d))" % letter).search(text, pos)
        if not mm:
            break
        marks.append((letter, mm.start(), mm.end()))
        pos = mm.end()
    if len(marks) < 3:
        return None
    opts = {}
    for i, (letter, _s, e) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        opts[letter] = clean(text[e:end])
    return text[: marks[0][1]], opts


# ── parsing one year ─────────────────────────────────────────────────────
def parse_year(text: str, subject: str, year: int, notes: list):
    km = KEY_RE.search(text)
    body, key_text = (text[:km.start()], text[km.end():]) if km else (text, "")

    questions = []
    for num, block in split_questions(body.replace("\r", "")):
        pieces = OPT_RE.split(block)
        stem = pieces[0]
        options = {}
        for j in range(1, len(pieces) - 1, 2):
            letter = pieces[j]
            if letter not in options:          # keep the first one if OCR repeats a letter
                options[letter] = clean(pieces[j + 1])

        inline_note = False
        if len(options) <= 1:
            alt = split_inline_options(" ".join(block.split()))
            if alt:
                stem, options = clean(alt[0].replace(IMG, " ")), alt[1]
                if IMG in alt[0]:
                    block = IMG + block
                inline_note = True

        # [IMAGE] belongs between the question and option A. If it landed at
        # the end of the last option instead, move it back to the question.
        has_diagram = IMG in stem
        if options:
            last_letter = list(options)[-1]
            v = options[last_letter]
            if v != IMG and v.endswith(IMG):
                options[last_letter] = clean(v[: -len(IMG)])
                has_diagram = True
                notes.append((year, num, "[IMAGE] was after the options; treated as a diagram"))
        stem = clean(stem.replace(IMG, " "))
        options_are_images = any(v == IMG for v in options.values())
        if not options and has_diagram:
            # the options are drawn inside the picture itself
            options_are_images = True
            notes.append((year, num, "options are inside the diagram image; check the crop"))
        if inline_note:
            notes.append((year, num, "options were on one line and were split automatically; check them"))

        questions.append({
            "id": f"{slug(subject).replace('_', '-')}-{year}-{num}",
            "subject": slug(subject),
            "year": year,
            "number": num,
            "question": stem,
            "options": options,
            "answer": None,
            "explanation": None,
            "has_diagram": has_diagram,
            "options_are_images": options_are_images,
            "image": None,
        })
    return questions, key_text


KEY_LINE = re.compile(
    r"(?m)^[ \t]*(\d{1,3})[.):\-]?[ \t]*([A-Ea-e?])(?![A-Za-z0-9])[ \t]*(?:\|[ \t]*(.*?))?[ \t]*$")


def read_key(key_text: str) -> dict:
    """-> {question number: (letter or None, explanation or None)}.
    Accepts   "14. C | Because ..."   (answer + short explanation)
    and       "14. C"                 (answer only)
    and the old several-per-line form "1. B 2. C 3. A"."""
    key = {}
    for n, a, expl in KEY_LINE.findall(key_text):
        letter = a.upper() if a != "?" else None
        key.setdefault(int(n), (letter, clean(expl) if (letter and expl) else None))
    if not key:                                   # old format: several answers per line
        for n, a in re.findall(r"(\d{1,3})\s*[.\-):]?\s*([A-Ea-e])\b", key_text):
            key.setdefault(int(n), (a.upper(), None))
    return key


# ── checks ───────────────────────────────────────────────────────────────
def check(questions, expect_answers=True):
    bad = []
    for q in questions:
        p = []
        n_opts = len(q["options"])
        if n_opts == 0 and q["options_are_images"]:
            pass                                   # options live inside the diagram
        elif n_opts not in (4, 5):
            p.append(f"{n_opts} options")
        elif n_opts and list(q["options"]) != list("ABCDE")[:n_opts]:
            p.append(f"options out of order: {''.join(q['options'])}")
        if not q["answer"]:
            if expect_answers:
                p.append("no answer")
        elif q["answer"] not in q["options"]:
            p.append("answer not in options")
        if not q["question"]:
            p.append("empty question")
        if any(len(v) > 200 for v in q["options"].values()):
            p.append("an option is very long (the next question's number may be missing)")
        if UNCLEAR in q["question"] or any(UNCLEAR in v for v in q["options"].values()):
            p.append("contains [UNCLEAR]")
        if not q["has_diagram"] and re.search(r"\bFig(ure)?\.?\s*\d|diagram shown|shown below", q["question"], re.I):
            p.append("mentions a figure but no [IMAGE]")
        if p:
            bad.append((q["number"], p))
    return bad


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Convert extracted exam text to JSON, one file per year.")
    ap.add_argument("subject")
    ap.add_argument("text_file")
    ap.add_argument("--year", type=int, help="use when the file has no 'YEAR: xxxx' lines")
    ap.add_argument("--key", help="separate answer-key file (optional)")
    args = ap.parse_args()

    text = pathlib.Path(args.text_file).read_text(encoding="utf-8")
    sections = split_years(text, args.year)
    if not sections:
        raise SystemExit("No 'YEAR: xxxx' lines found. Add them, or pass --year 2019.")

    key_by_year = {}
    if args.key:
        ktext = pathlib.Path(args.key).read_text(encoding="utf-8")
        for y, t in split_years(ktext, args.year).items():
            key_by_year[y] = read_key(t)

    OUT_DIR.mkdir(exist_ok=True)
    total = 0
    for year in sorted(sections):
        notes = []
        qs, inline_key_text = parse_year(sections[year], args.subject, year, notes)
        key = read_key(inline_key_text)
        key.update(key_by_year.get(year, {}))
        for q in qs:
            ans, expl = key.get(q["number"], (None, None))
            q["answer"] = ans
            q["explanation"] = expl

        out = OUT_DIR / f"{slug(args.subject)}_{year}.json"
        out.write_text(json.dumps(qs, indent=1, ensure_ascii=False), encoding="utf-8")
        total += len(qs)

        nums = [q["number"] for q in qs]
        imgs = sum(q["has_diagram"] or q["options_are_images"] for q in qs)
        print(f"{year}: {len(qs)} questions, {imgs} with diagrams -> {out}")
        missing = [n for n in range(1, (max(nums) if nums else 0) + 1) if n not in nums]
        if missing:
            print(f"  CHECK missing question numbers: {missing}")
        for _y, num, why in notes:
            print(f"  NOTE Q{num}: {why}")
        has_any = any(q["answer"] for q in qs)
        n_expl = sum(1 for q in qs if q["explanation"])
        if has_any:
            print(f"  {sum(1 for q in qs if q['answer'])} answers, {n_expl} explanations")
        if not has_any:
            print("  NOTE no answers for this year yet (add an ANSWER KEY or use --key)")
        for num, probs in check(qs, expect_answers=has_any):
            print(f"  CHECK Q{num}: {', '.join(probs)}")
    print(f"\nDone: {total} questions across {len(sections)} year(s).")


if __name__ == "__main__":
    main()
