"""
publish_questions.py  -  one command: parse, then upload to GitHub.
===================================================================
Put this file next to parse_questions.py (tools/publish_questions.py).
Run it from the repo ROOT (the folder that contains main.py and .git):

    python tools/publish_questions.py physics past_questions_data/raw/physics.txt
    python tools/publish_questions.py physics physics.txt --images diagrams_physics
    python tools/publish_questions.py physics physics.txt --key physics_key.txt --year 2019

It does three things:
  1. runs parse_questions.py with the same options (writes the JSON + images
     into past_questions_data/)
  2. shows you the report; if there are CHECK lines it asks before going on
  3. git add / commit / push past_questions_data, so Render redeploys

Needs: git installed, and this repo already able to `git push` from your PC.
"""

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PARSER = HERE / "parse_questions.py"
DATA_DIR = "past_questions_data"


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, **kw)


def main():
    ap = argparse.ArgumentParser(description="Parse past questions, then push them to GitHub.")
    ap.add_argument("subject")
    ap.add_argument("text_file")
    ap.add_argument("--year", type=int)
    ap.add_argument("--key")
    ap.add_argument("--images")
    ap.add_argument("--yes", action="store_true", help="push even if the report has CHECK lines")
    ap.add_argument("--no-push", action="store_true", help="parse and commit only")
    args = ap.parse_args()

    if not pathlib.Path(".git").exists():
        sys.exit("Run this from the repo root (the folder that has .git and main.py).")

    cmd = [sys.executable, str(PARSER), args.subject, args.text_file]
    for flag in ("year", "key", "images"):
        v = getattr(args, flag)
        if v:
            cmd += [f"--{flag}", str(v)]

    print("== 1/3  Parsing ==")
    res = run(cmd, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        sys.exit(res.stderr or "The parser failed; nothing was uploaded.")

    checks = [l for l in res.stdout.splitlines() if l.strip().startswith("CHECK")]
    if checks:
        print(f"== {len(checks)} CHECK line(s) above ==")
        if not args.yes:
            if input("Upload anyway? [y/N] ").strip().lower() != "y":
                sys.exit("Stopped. Fix the raw text or JSON, then run it again.")

    print("== 2/3  Committing ==")
    run(["git", "add", DATA_DIR], check=True)
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        sys.exit("Nothing changed, so there is nothing to upload.")
    run(["git", "commit", "-m", f"Update {args.subject} past questions"], check=True)

    if args.no_push:
        sys.exit("Committed locally (--no-push). Run `git push` when ready.")
    print("== 3/3  Pushing ==")
    if run(["git", "push"]).returncode != 0:
        sys.exit("Push failed. Check you are signed in to GitHub, then run `git push`.")
    print("\nDone. Render will redeploy in a minute or two.")


if __name__ == "__main__":
    main()
