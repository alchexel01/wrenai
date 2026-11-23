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

Options:  --year N (or 2018,2019,2020 / 2010-2018, PDFs only)   --key FILE   --images FOLDER   --yes (skip CHECK question)
          --no-push (parse and commit only)

SAFETY (so this can never damage the backend repo)
  * Only files inside past_questions_data (never raw/) are ever committed.
    Anything else that was already staged is un-staged first, so deletions of
    textbooks/, syllabus_data/, textbooks_api.py etc. can never ride along.
  * A commit that would delete anything, or touch anything outside
    past_questions_data, is refused before it is made.
  * It never force-pushes and never deletes anything on GitHub.
  * Safe problems are fixed for you (half-finished rebase/merge, a conflicted
    file, a stale index.lock, a rejected push, missing git name/email); each
    asks first when it could discard your work. Anything else stops with a
    plain explanation.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
READER = HERE / "pdf_to_text.py"
PARSER = HERE / "parse_questions.py"
DATA_DIR = "past_questions_data"
EXPECTED_REPO = "alchexel01/wrenai"     # the folder's "origin" must be this repo
MAX_FILE_MB = 95                        # GitHub rejects files over 100 MB


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, **kw)


def ask(prompt):
    """Input with stray quotes removed (Windows adds them when you paste a path)."""
    return input(prompt).strip().strip('"').strip("'").strip()


def yes_no(prompt, default_yes=True):
    hint = "[Y/n]" if default_yes else "[y/N]"
    a = input(f"{prompt} {hint} ").strip().lower()
    return default_yes if not a else a.startswith("y")


def git_out(*a):
    return subprocess.run(["git", *a], text=True, capture_output=True).stdout


# ──────────────────────────────────────────────────────────────────────────
#  Git safety net
# ──────────────────────────────────────────────────────────────────────────
def git(*a):
    return subprocess.run(["git", *a], text=True, capture_output=True,
                          encoding="utf-8", errors="replace")


def run_live(cmd):
    """Run a command, show its output as it happens, and also return it."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    lines = []
    for ln in p.stdout:
        print(ln, end="")
        lines.append(ln)
    p.wait()
    return p.returncode, "".join(lines)


def explain(text):
    """Turn a raw git error into a plain sentence. True if it was recognised."""
    t = (text or "").lower()
    rules = [
        (("index.lock",),
         "Another Git program is using this repo (or it crashed and left a lock). "
         "Close other Git windows and run this again."),
        (("authentication failed", "could not read username", "permission denied",
          "invalid username or password", "returned error: 403"),
         "GitHub did not accept your sign-in. Sign in again (GitHub Desktop or Git "
         "Credential Manager), then run this again. Your commit is saved locally."),
        (("could not resolve host", "failed to connect", "unable to access",
          "timed out", "network is unreachable"),
         "Cannot reach GitHub. Check your internet, then run this again. "
         "Your commit is saved locally."),
        (("gh001", "large files detected", "file size limit"),
         "A file is over GitHub's 100 MB limit, so GitHub refused it."),
        (("please tell me who you are", "unable to auto-detect email"),
         "Git does not know your name/email yet."),
        (("dubious ownership",),
         'Git does not trust this folder. Run once:  git config --global --add '
         'safe.directory "%CD%"'),
        (("unrelated histories",),
         "This folder and GitHub do not share history. Do not force anything; send "
         "the output of 'git remote -v' and 'git log --oneline -3'."),
    ]
    for keys, msg in rules:
        if any(k in t for k in keys):
            print("\n" + msg)
            return True
    return False


def staged_changes():
    """[(status, path), ...] for everything currently staged."""
    out = git_out("-c", "core.quotepath=false", "diff", "--cached", "--name-status", "-z")
    tok, i, res = out.split("\0"), 0, []
    while i < len(tok) and tok[i]:
        st = tok[i]
        n = 2 if st[0] in "RC" else 1
        for p in tok[i + 1:i + 1 + n]:
            res.append((st, p))
        i += 1 + n
    return res


def in_data(path):
    return path.startswith(DATA_DIR + "/") and not path.startswith(DATA_DIR + "/raw/")


def backup_work(label):
    """Copy every locally changed or conflicted file (except the question data this
    script manages) into .publish_backups/<time>/ so nothing can ever be lost.
    Returns the backup folder, or None if there was nothing to copy."""
    out = git_out("-c", "core.quotepath=false", "status", "--porcelain", "-z")
    paths = []
    for ent in out.split("\0"):
        if len(ent) > 3 and ent[:2] != "??" and ent[:2].strip() != "D":
            paths.append(ent[3:])
    paths = [p for p in paths if not in_data(p) and pathlib.Path(p).is_file()]
    if not paths:
        return None
    dest = pathlib.Path(".publish_backups") / (time.strftime("%Y%m%d-%H%M%S") + "-" + label)
    for p in paths:
        try:
            t = dest / p
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
        except OSError:
            pass
    try:                                     # keep backups out of git, forever
        ex = pathlib.Path(git_out("rev-parse", "--git-dir").strip()) / "info" / "exclude"
        cur = ex.read_text(encoding="utf-8") if ex.exists() else ""
        if ".publish_backups/" not in cur:
            ex.parent.mkdir(parents=True, exist_ok=True)
            ex.write_text(cur.rstrip("\n") + "\n.publish_backups/\n", encoding="utf-8")
    except OSError:
        pass
    print(f"  Safety copy of your {len(paths)} edited file(s) saved in {dest}")
    return dest


def preflight():
    """Check the repo is healthy and is the right one, fixing the safe things.
    Returns the current branch name. Never touches textbooks/ or backend files."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        sys.exit("Git is not installed (or not on PATH). Install Git for Windows, "
                 "then run this again.")
    r = git("rev-parse", "--is-inside-work-tree")
    if r.returncode != 0:
        if not explain(r.stderr):
            print(r.stderr)
        sys.exit("This folder is not a usable git repo. Nothing was changed.")

    remote = git_out("remote", "get-url", "origin").strip()
    if EXPECTED_REPO not in remote.lower():
        sys.exit(f"This folder's GitHub address is '{remote or 'none'}', not "
                 f"github.com/{EXPECTED_REPO}. Wrong folder? Nothing was changed.")

    gitdir = pathlib.Path(git_out("rev-parse", "--git-dir").strip())

    # a half-finished rebase / merge left over from an earlier problem.
    # Your files are copied to .publish_backups first, and a merge is cleared with
    # 'git reset' (which never touches the files on disk).
    for op, marks, abort in (
            ("rebase", ("rebase-merge", "rebase-apply"), ["rebase", "--abort"]),
            ("merge", ("MERGE_HEAD",), None),
            ("cherry-pick", ("CHERRY_PICK_HEAD",), None),
            ("revert", ("REVERT_HEAD",), None)):
        if any((gitdir / m).exists() for m in marks):
            print(f"\nA {op} was left half-finished in this repo.")
            backup_work(op)
            if not yes_no(f"Clear it so uploading can continue? (your files are not deleted)", True):
                sys.exit("Stopped. Finish or cancel it, then run this again.")
            if abort:
                ok = git(*abort).returncode == 0
            else:
                ok = git("reset", "-q").returncode == 0
                for m in marks:                      # leftover state files, if any
                    try:
                        (gitdir / m).unlink()
                    except OSError:
                        pass
            if not ok:
                sys.exit(f"Could not clear the {op}. Your files are safe in .publish_backups. "
                         "Run 'git status' and send me what it says.")
            print(f"  {op} cleared.")

    # files flagged as conflicted (the '!' in VS Code). They are NOT reverted: the
    # script only uploads question data, so their contents do not matter here. We
    # copy them aside, then un-flag them with 'git reset' (files on disk untouched).
    un = [p for p in git_out("-c", "core.quotepath=false", "diff", "--name-only",
                             "--diff-filter=U").splitlines() if p]
    if un:
        print("\nThese files are flagged as an unresolved conflict:")
        for p in un:
            print("  " + p)
        backup_work("conflict")
        if git("reset", "-q").returncode != 0:
            sys.exit("Could not clear the conflict flags. Your files are safe in "
                     ".publish_backups. Run 'git status' and send me what it says.")
        print("  Flags cleared. Your files on disk were NOT changed or reverted.")
        for p in un:
            try:
                if "<<<<<<< " in pathlib.Path(p).read_text(encoding="utf-8", errors="ignore"):
                    print(f"  WARNING: {p} still has <<<<<<< conflict markers inside it. "
                          "Search for them and fix them before you run the app.")
            except OSError:
                pass

    # a stale lock left by a crashed git
    lock = gitdir / "index.lock"
    if lock.exists():
        print("\nGit left a lock file behind (.git/index.lock).")
        if not yes_no("Make sure no other Git program is running, then delete the lock?", False):
            sys.exit("Stopped. Close other Git programs and run this again.")
        try:
            lock.unlink()
        except OSError as e:
            sys.exit(f"Could not delete the lock: {e}")

    # commits need a name and email
    if not git_out("config", "user.email").strip():
        print("\nGit needs your name and email for commits (saved for this repo only).")
        name, email = ask("Name: "), ask("Email: ")
        if not (name and email):
            sys.exit("Stopped: Git needs a name and email to commit.")
        git("config", "user.name", name)
        git("config", "user.email", email)

    branch = git_out("symbolic-ref", "--short", "-q", "HEAD").strip()
    if not branch:
        sys.exit("Git is not on a branch right now (detached HEAD). Run 'git checkout main', "
                 "then run this again.")
    if branch != "main" and not yes_no(
            f"You are on branch '{branch}', not 'main'. Render deploys 'main'. Continue anyway?", False):
        sys.exit("Stopped. Run 'git checkout main', then run this again.")

    gone = git_out("-c", "core.quotepath=false", "ls-files", "--deleted").splitlines()
    if gone:
        print(f"\nNOTE: {len(gone)} file(s) that GitHub has are missing from this folder "
              f"(for example {gone[0]}).")
        print("      That is safe: this script only uploads past_questions_data and never "
              "deletes anything on GitHub.")
    return branch


def stage_safely():
    """Stage ONLY the question data, and prove nothing else is in the commit."""
    pre = staged_changes()
    if pre:
        dels = sum(1 for st, _ in pre if st.startswith("D"))
        print(f"NOTE: {len(pre)} change(s) were already staged from earlier"
              f"{f' ({dels} deletions)' if dels else ''}. I un-staged them so they cannot be "
              "uploaded by accident. Your files on disk are untouched.")
    git("reset", "-q")                         # empty index; the working folder is not touched
    r = git("add", "--ignore-removal", "--", DATA_DIR, f":(exclude){DATA_DIR}/raw")
    if r.returncode != 0:
        if not explain(r.stderr):
            print(r.stderr)
        sys.exit("Could not stage the question files. Nothing was committed.")

    bad, big = [], []
    for st, p in staged_changes():
        if not (st[0] in "AM" and in_data(p)):
            bad.append(f"{st}  {p}")
        elif pathlib.Path(p).is_file() and pathlib.Path(p).stat().st_size > MAX_FILE_MB * 1024 * 1024:
            big.append(p)
    if bad:
        git("reset", "-q")
        print("\nSTOPPED: the commit would contain changes outside the question data:")
        for b in bad[:15]:
            print("  " + b)
        sys.exit("Nothing was committed or uploaded.")
    if big:
        git("reset", "-q", "--", *big)
        print(f"\nWARNING: {len(big)} file(s) are over {MAX_FILE_MB} MB (GitHub's limit is 100 MB) "
              "and were left out:")
        for b in big[:5]:
            print("  " + b)


def unsafe_local_changes(branch):
    """Lines describing anything in the not-yet-pushed commits that is not a plain
    add/modify of question data. [] means safe (or nothing to compare against)."""
    if git("rev-parse", "--verify", "-q", f"origin/{branch}").returncode != 0:
        return []
    ours = git_out("-c", "core.quotepath=false", "diff", "--name-status",
                   f"origin/{branch}...HEAD").splitlines()
    return [l for l in ours if not (l[:1] in ("A", "M") and in_data(l.split("\t", 1)[-1]))]


def unpushed_count(branch):
    if git("rev-parse", "--verify", "-q", f"origin/{branch}").returncode != 0:
        return 0
    return int(git_out("rev-list", "--count", f"origin/{branch}..HEAD").strip() or 0)


def recover_rejected(branch):
    """GitHub has newer commits. Put our question commits on top of them in a
    separate temporary folder, so your working files (main.py etc.) are never
    stashed, rebased, merged or conflicted."""
    print("\nGitHub has newer changes than this PC. Combining them in a separate "
          "temporary folder (your files are not touched)...")
    r = git("fetch", "origin")
    if r.returncode != 0:
        if not explain(r.stderr):
            print(r.stderr)
        return False
    if not refuse_if_unsafe(branch):
        return False
    backup_work("before-sync")
    gitdir = pathlib.Path(git_out("rev-parse", "--git-dir").strip()).resolve()
    tmp = gitdir / "publish_tmp_worktree"
    git("worktree", "prune")
    if tmp.exists():
        git("worktree", "remove", "--force", str(tmp))
        shutil.rmtree(tmp, ignore_errors=True)
    if git("worktree", "add", "--detach", str(tmp), f"origin/{branch}").returncode != 0:
        print("Could not make the temporary folder. Nothing was pushed.")
        return False
    ok = False
    try:
        commits = git_out("rev-list", "--reverse", f"origin/{branch}..HEAD").split()
        code, _ = run_live(["git", "-C", str(tmp), "cherry-pick", *commits])
        if code != 0:
            git("-C", str(tmp), "cherry-pick", "--abort")
            print("\nCould not combine automatically (the same question file changed on "
                  "GitHub too). Nothing was pushed and your files are untouched.")
            print("Send me the lines above and I will sort it out.")
            return False
        code, _ = run_live(["git", "-C", str(tmp), "push", "origin", f"HEAD:{branch}"])
        ok = code == 0
    finally:
        git("worktree", "remove", "--force", str(tmp))
        shutil.rmtree(tmp, ignore_errors=True)
        git("worktree", "prune")
    if ok:
        # point this PC's history at GitHub's. --mixed never touches files on disk.
        git("fetch", "origin")
        git("reset", "--mixed", "-q", f"origin/{branch}")
        # bring in the other question files GitHub had (question data only)
        git("checkout", "HEAD", "--", DATA_DIR)
    return ok


def refuse_if_unsafe(branch):
    bad = unsafe_local_changes(branch)
    if not bad:
        return True
    print("\nSTOPPED: a commit waiting on this PC changes more than question data "
          "(for example it deletes files that exist on GitHub):")
    for b in bad[:12]:
        print("  " + b)
    if len(bad) > 12:
        print(f"  ... and {len(bad) - 12} more")
    print("I will not push it. To throw that commit away but KEEP all your files on disk, run")
    print("these two, in this order (the first one gets GitHub's latest state):")
    print("  git fetch origin")
    print(f"  git reset --mixed origin/{branch}")
    print("then run this script again.")
    return False


def push_safely(branch):
    """Push once; if GitHub says 'fetch first', combine safely and push again.
    Never forces."""
    git("fetch", "origin")                       # safe: only reads GitHub's latest state
    if not refuse_if_unsafe(branch):
        return False
    for attempt in (1, 2):
        print("  Uploading... (pictures can take a minute or two)")
        code, text = run_live(["git", "push", "origin", branch])
        if code == 0:
            return True
        low = text.lower()
        rejected = "[rejected]" in low or "non-fast-forward" in low or "fetch first" in low
        if rejected and attempt == 1:
            if not recover_rejected(branch):
                return False
            continue
        if not explain(text):
            print("\nPush failed (see the message above). Your commit is saved locally.")
        return False
    return False


def summarize():
    """Say exactly what is being uploaded, so nothing is a surprise."""
    cnt = {"questions": [0, 0], "images": [0, 0], "other": [0, 0]}      # [new, updated]
    for ln in git_out("diff", "--cached", "--name-status", "--", DATA_DIR).splitlines():
        st, _, path = ln.partition("\t")
        if st.startswith("D"):
            continue
        if "/images/" in path:
            key = "images"
        elif path.endswith(".json") and "/answers/" not in path:
            key = "questions"
        else:
            key = "other"
        cnt[key][0 if st.startswith("A") else 1] += 1
    q, im = cnt["questions"], cnt["images"]
    print(f"This upload:  question files {q[0]} new, {q[1]} updated   |   pictures {im[0]} new, {im[1]} updated")

    tracked = {l for l in git_out("ls-files", f"{DATA_DIR}/images").splitlines() if l.lower().endswith(".png")}
    local = sorted(p.as_posix() for p in pathlib.Path(DATA_DIR, "images").rglob("*.png"))
    print(f"Pictures on GitHub after this upload: {len(tracked)}   (on this PC: {len(local)})")
    missing = [p for p in local if p not in tracked]
    if missing:
        print(f"  WARNING: {len(missing)} picture(s) on this PC will NOT be uploaded. First one: {missing[0]}")
        why = git_out("check-ignore", "-v", missing[0]).strip()
        print("  Git is ignoring it because of this rule:  " + why if why else
              "  Git is not ignoring it, so check the path with: git status")
    stray = [l for l in git_out("ls-files", f"{DATA_DIR}/raw").splitlines() if l.endswith(".json")]
    if stray:
        print(f"  NOTE: GitHub has {len(stray)} .json file(s) inside {DATA_DIR}/raw. The app never reads them there;")
        print("        they look like a mis-placed upload and can be deleted on GitHub.")


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


def parse_years(raw):
    """'2018,2019, 2020' or '2010-2018' -> [2018, 2019, 2020]. None if invalid."""
    raw = re.sub(r"\s*-\s*", "-", (raw or "").strip())
    years = []
    for part in re.split(r"[,;\s]+", raw):
        if not part:
            continue
        m = re.fullmatch(r"(\d{4})-(\d{4})", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b or b - a > 60:
                return None
            nums = list(range(a, b + 1))
        elif re.fullmatch(r"\d{4}", part):
            nums = [int(part)]
        else:
            return None
        for n in nums:
            if n not in years:
                years.append(n)
    return years or None


def years_in_text(txt):
    """Years that have a 'YEAR: xxxx' line in the text file written by the reader."""
    try:
        t = pathlib.Path(txt).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return sorted({int(m) for m in re.findall(r"(?m)^YEAR:\s*(\d{4})", t)})


def main():
    ap = argparse.ArgumentParser(description="Read a past-question PDF, parse it, push it to GitHub.")
    ap.add_argument("subject", nargs="?")
    ap.add_argument("source", nargs="?", help="a .pdf (read first) or an already-read .txt")
    ap.add_argument("--year", help="one year, or several: 2018,2019,2020 or 2010-2018")
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

    branch = preflight() if has_git else None

    is_pdf = source.lower().endswith(".pdf")
    total = 4 if is_pdf else 3
    step = 0
    imgdir = args.images or (f"diagrams_{subject}" if is_pdf else None)

    years = None
    if args.year:
        years = parse_years(args.year)
        if not years:
            sys.exit("--year must look like 2019, or 2018,2019,2020, or 2010-2018.")
    if years and len(years) > 1 and not is_pdf:
        sys.exit("Several years only work with a PDF. For a .txt file, put 'YEAR: 2019' lines in it.")

    # 1) PDF -> text
    if is_pdf:
        step += 1
        txt = str(pathlib.Path(source).with_suffix(".txt"))
        print(f"== {step}/{total}  Reading the PDF ==")
        cmd = [sys.executable, str(READER), source, txt, "--images", imgdir]
        if years:
            cmd += ["--years", ",".join(map(str, years))]
        if run(cmd).returncode != 0:
            sys.exit("Reading the PDF failed; nothing else was done.")

        if interactive and not years and not years_in_text(txt):
            print("\nNo year headings were found in this PDF (see the message above).")
            print("Type ONE year if the PDF holds a single year, or SEVERAL separated by commas")
            print("(for example 2018,2019,2020 or 2010-2018). Press Ctrl+C to cancel.")
            while True:
                years = parse_years(ask("Year(s) of this paper: "))
                if years:
                    break
                print("  Please type years like 2019 or 2018,2019,2020.")
            print("\nReading the PDF again with those years ...")
            cmd = [sys.executable, str(READER), source, txt, "--images", imgdir,
                   "--years", ",".join(map(str, years))]
            if run(cmd).returncode != 0:
                sys.exit("Reading the PDF failed; nothing else was done.")

        if years and len(years) > 1:
            found = years_in_text(txt)
            lost = [y for y in years if y not in found]
            if lost:
                print("\nCould not find where these year(s) start: " + ", ".join(map(str, lost)))
                if not found:
                    sys.exit("No year start was found, so nothing was done. Check the years you typed.")
                if not (args.yes or (interactive and yes_no(
                        "Continue with only the years that were found?", False))):
                    sys.exit("Stopped. Check the years you typed.")
    else:
        txt = source

    # 2) text -> JSON
    step += 1
    print(f"\n== {step}/{total}  Parsing ==")
    cmd = [sys.executable, str(PARSER), subject, txt]
    if years and len(years) == 1:
        cmd += ["--year", str(years[0])]
    if args.key:
        cmd += ["--key", args.key]
    if imgdir:
        cmd += ["--images", imgdir]
    res = run(cmd, capture_output=True)
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
    stage_safely()
    summarize()
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        waiting = unpushed_count(branch)
        if not waiting:
            sys.exit("Nothing changed, so there is nothing to upload.")
        print(f"No new changes, but {waiting} earlier commit(s) are waiting to be uploaded.")
    else:
        code, text = run_live(["git", "commit", "-m", f"Update {subject} past questions"])
        if code != 0:
            if not explain(text):
                print("\nThe commit failed (see the message above). Nothing was uploaded.")
            sys.exit(1)

    if args.no_push:
        sys.exit("Committed locally (no upload). Run `git push` when you are ready.")

    # 4) push
    step += 1
    print(f"== {step}/{total}  Pushing ==")
    if not push_safely(branch):
        sys.exit("Not uploaded. Your questions are committed on this PC; run this again "
                 "after fixing the problem above.")
    print("\nDone. Render will redeploy in a minute or two.")


if __name__ == "__main__":
    main()
