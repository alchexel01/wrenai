"""
publish_questions.py  -  PDF in, questions on GitHub out. One command.
======================================================================
Keep this file in the same folder as pdf_to_text.py and parse_questions.py,
and run it from the repo ROOT (the folder with main.py, and .git if you want
the upload step).

EASIEST: just run it and answer the two questions.

    python publish_questions.py

    PDF number or file name:   JAMB Physics Past Question 1983 - 2004.pdf
    Subject:                   physics

It then does, in order:
  1. reads the PDF (pdf_to_text.py)        -> <pdf name>.txt + diagrams_<subject>/
  2. turns it into JSON (parse_questions.py) -> past_questions_data/
  3. shows the report; if there are CHECK lines it asks before going on
  4. git add / commit / push past_questions_data, so Render redeploys

If this folder is not a git repo, steps 1 to 3 still run and it tells you to
upload past_questions_data on the GitHub website instead.

Without questions (old style, still works):

    python publish_questions.py physics "JAMB Physics Past Question 1983 - 2004.pdf"
    python publish_questions.py physics physics.txt --images diagrams_physics
    python publish_questions.py physics physics.txt --key ans.json --year 2019

Options:  --year N   --key FILE   --images FOLDER   --yes (skip CHECK question)
          --no-push (parse and commit only)
"""

import argparse
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
READER = HERE / "pdf_to_text.py"
PARSER = HERE / "parse_questions.py"
DATA_DIR = "past_questions_data"


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, **kw)


def ask(prompt):
    """Input with stray quotes removed (Windows adds them when you paste a path)."""
    return input(prompt).strip().strip('"').strip("'").strip()


def yes_no(prompt, default_yes=True):
    hint = "[Y/n]" if default_yes else "[y/N]"
    a = input(f"{prompt} {hint} ").strip().lower()
    return default_yes if not a else a.startswith("y")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def pick_pdf():
    pdfs = sorted(p.name for p in pathlib.Path(".").glob("*.pdf"))
    while True:
        if pdfs:
            print("PDF files in this folder:")
            for i, name in enumerate(pdfs, 1):
                print(f"  {i}. {name}")
        raw = ask("PDF number or file name (just Enter to quit): ")
        if not raw:
            sys.exit("Cancelled.")
        if raw.isdigit() and 1 <= int(raw) <= len(pdfs):
            return pdfs[int(raw) - 1]
        name = raw if raw.lower().endswith(".pdf") else raw + ".pdf"
        if pathlib.Path(name).is_file():
            return name
        print(f"  Can't find '{name}' in {pathlib.Path('.').resolve()}. Check the spelling.\n")


def pick_subject():
    while True:
        raw = ask("Subject (for example physics, chemistry, use of english): ")
        s = slug(raw)
        if s:
            if s != raw:
                print(f"  Using subject name: {s}")
            return s
        print("  Please type a subject.")


def main():
    ap = argparse.ArgumentParser(description="Read a past-question PDF, parse it, push it to GitHub.")
    ap.add_argument("subject", nargs="?")
    ap.add_argument("source", nargs="?", help="a .pdf (read first) or an already-read .txt")
    ap.add_argument("--year", type=int)
    ap.add_argument("--key", help="answer file (.txt or .json)")
    ap.add_argument("--images", help="folder for the diagram crops")
    ap.add_argument("--yes", action="store_true", help="push even if the report has CHECK lines")
    ap.add_argument("--no-push", action="store_true", help="parse and commit only")
    args = ap.parse_args()

    for tool in (READER, PARSER):
        if not tool.is_file():
            sys.exit(f"Missing {tool.name}. Put it in the same folder as this script ({HERE}).")

    has_git = pathlib.Path(".git").exists()
    interactive = not (args.subject and args.source)

    if interactive:
        print("=== Past questions: PDF to GitHub ===\n")
        source = pick_pdf()
        subject = pick_subject()
        if has_git:
            push = yes_no("Upload to GitHub when it is done?")
            args.no_push = not push
        else:
            print("\n(This folder has no .git, so I will make the files but cannot upload them.)")
        print(f"\nPDF:     {source}\nSubject: {subject}")
        if not yes_no("Start?"):
            sys.exit("Cancelled.")
        print()
    else:
        source, subject = args.source, slug(args.subject)

    is_pdf = source.lower().endswith(".pdf")
    total = 4 if is_pdf else 3
    step = 0
    imgdir = args.images or (f"diagrams_{subject}" if is_pdf else None)

    # 1) PDF -> text
    if is_pdf:
        step += 1
        txt = str(pathlib.Path(source).with_suffix(".txt"))
        print(f"== {step}/{total}  Reading the PDF ==")
        cmd = [sys.executable, str(READER), source, txt, "--images", imgdir]
        if run(cmd).returncode != 0:
            sys.exit("Reading the PDF failed; nothing else was done.")
    else:
        txt = source

    # 2) text -> JSON
    step += 1
    print(f"\n== {step}/{total}  Parsing ==")
    cmd = [sys.executable, str(PARSER), subject, txt]
    if args.year:
        cmd += ["--year", str(args.year)]
    if args.key:
        cmd += ["--key", args.key]
    if imgdir:
        cmd += ["--images", imgdir]
    res = run(cmd, capture_output=True)
    if res.returncode != 0 and "--year" in (res.stderr or "") and interactive:
        print("This PDF has no year headings, so I need the year.")
        while True:
            y = ask("Year of this paper (for example 2019): ")
            if y.isdigit() and len(y) == 4:
                break
        res = run(cmd + ["--year", y], capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        sys.exit(res.stderr or "The parser failed; nothing was uploaded.")

    checks = [l for l in res.stdout.splitlines() if l.strip().startswith("CHECK")]
    if checks:
        print(f"== {len(checks)} CHECK line(s) above ==")
        if not args.yes:
            if input("Continue anyway? [y/N] ").strip().lower() != "y":
                sys.exit("Stopped. Fix the raw text or JSON, then run it again.")

    if not has_git:
        print(f"\nDone. The files are in {DATA_DIR}.")
        print("This folder is not a git repo, so upload it on GitHub:")
        print("  Add file -> Upload files, into past_questions_data (JSON files),")
        print("  and past_questions_data/images/<subject> (pictures, up to 100 at a time).")
        return

    # 3) commit
    step += 1
    print(f"== {step}/{total}  Committing ==")
    # --ignore-removal: never delete files that exist only on GitHub (for example the
    # answers you uploaded there); the raw OCR text is never uploaded.
    run(["git", "add", "--ignore-removal", "--", DATA_DIR,
         f":(exclude){DATA_DIR}/raw"], check=True)
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        sys.exit("Nothing changed, so there is nothing to upload.")
    run(["git", "commit", "-m", f"Update {subject} past questions"], check=True)

    if args.no_push:
        sys.exit("Committed locally (no upload). Run `git push` when you are ready.")

    # 4) push
    step += 1
    print(f"== {step}/{total}  Pushing ==")
    if run(["git", "push"]).returncode != 0:
        sys.exit("Push failed. If it says 'rejected', run `git pull --rebase` and then `git push`.\n"
                 "Otherwise check you are signed in to GitHub.")
    print("\nDone. Render will redeploy in a minute or two.")


if __name__ == "__main__":
    main()
