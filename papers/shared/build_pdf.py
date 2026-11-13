"""Compiles a paper with a minimal TeX Live (TinyTeX) installed only inside this repository.

TinyTeX lives in `.tools/tinytex/`, ignored by git. Nothing goes into the system PATH: the
official installer runs `tlmgr path add`, and this script does not. The installer is the Windows
binary of TinyTeX, since that is this project's machine; on another system, install TinyTeX with
the official script and point `TEX_BIN` to it.

    uv run python papers/shared/build_pdf.py install                 # TinyTeX + paper packages
    uv run python papers/shared/build_pdf.py build papers/mssp-2026  # builds main.pdf
    uv run python papers/shared/build_pdf.py build papers/mssp-2026 --submission
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / ".tools"
TEX = TOOLS / "tinytex"
VERSION = "2026.11"
ASSET = f"TinyTeX-1-windows-v{VERSION}.exe"
URL = f"https://github.com/rstudio/tinytex-releases/releases/download/v{VERSION}/{ASSET}"
TEX_BIN = Path(os.environ.get("TEX_BIN", TEX / "bin" / "windows"))
# packages that elsarticle, todonotes and the paper's preamble need on top of TinyTeX-1
PACKAGES = [
    "elsarticle",
    "todonotes",
    "pgf",
    "xcolor",
    "xkeyval",
    "siunitx",
    "booktabs",
    "natbib",
    "hyperref",
    "amsmath",
    "amsfonts",
    "graphics",
    "grfext",
    "tools",
    "lineno",
    "geometry",
    "etoolbox",
]


def tlmgr(*args: str) -> None:
    bat = TEX_BIN / "tlmgr.bat"
    subprocess.run(["cmd", "/c", str(bat), *args], check=True)


def install() -> None:
    if (TEX_BIN / "pdflatex.exe").exists():
        print(f"TinyTeX is already in {TEX}")
    else:
        dl = TOOLS / "_download"
        if dl.exists():
            shutil.rmtree(dl)
        dl.mkdir(parents=True)
        exe = dl / ASSET
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "tinytex-local-install"})
        with urllib.request.urlopen(req) as r, exe.open("wb") as f:
            shutil.copyfileobj(r, f)
        # self-extracting archive: -y extracts the TinyTeX folder into the cwd, without asking
        subprocess.run([str(exe), "-y"], cwd=dl, check=True)
        if TEX.exists():
            shutil.rmtree(TEX)
        shutil.move(str(dl / "TinyTeX"), str(TEX))
        shutil.rmtree(dl)
        print(f"TinyTeX extracted to {TEX}")
    tlmgr("update", "--self")  # the bundled tlmgr is usually behind the repository
    tlmgr("install", *PACKAGES)


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    # on Windows, CreateProcess looks up the executable in the parent's PATH, not the child's
    exe = shutil.which(cmd[0], path=env["PATH"]) or cmd[0]
    return subprocess.run([exe, *cmd[1:]], cwd=cwd, env=env, capture_output=True).returncode


def build(paper: Path, main: str = "main", submission: bool = False) -> int:
    if not (TEX_BIN / "pdflatex.exe").exists() and not shutil.which("pdflatex"):
        print("TinyTeX is missing: run `build_pdf.py install` first.")
        return 2
    env = os.environ.copy()
    env["PATH"] = str(TEX_BIN) + os.pathsep + env.get("PATH", "")
    # submission version: same sources, with todonotes disabled (no list and no boxes)
    job = f"{main}-submission" if submission else main
    source = f"{main}.tex"
    if submission:
        source = "\\PassOptionsToPackage{disable}{todonotes}\\input{" + source + "}"
    latex = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-file-line-error",
        f"-jobname={job}",
        source,
    ]
    steps = [latex, ["bibtex", job], latex, latex]
    codes = [run(cmd, paper, env) for cmd in steps]
    log = (paper / f"{job}.log").read_text(encoding="latin-1", errors="replace")
    errors = [line for line in log.splitlines() if line.startswith("!") or ".tex:" in line]
    missing = re.findall(r"File `([^']+)' not found", log)
    undef = len(re.findall(r"(Citation|Reference) `[^']+' on page", log))
    blg = paper / f"{job}.blg"
    bib_warn = blg.read_text(encoding="latin-1").count("Warning--") if blg.exists() else 0
    pages = re.search(r"Output written on \S+ \((\d+) pages", log)
    print(f"exit codes: {codes}")
    print(
        f"pages: {pages.group(1) if pages else '?'}; undefined references: {undef}; "
        f"BibTeX warnings: {bib_warn}"
    )
    if missing:
        print("missing files (install them with tlmgr):", sorted(set(missing)))
    for line in errors[:15]:
        print("  ", line)
    pdf = paper / f"{job}.pdf"
    ok = pdf.exists() and codes[-1] == 0
    print(f"{'PDF built' if ok else 'failed'}: {pdf}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install")
    b = sub.add_parser("build")
    b.add_argument("paper", type=Path)
    b.add_argument(
        "--submission",
        action="store_true",
        help="builds main-submission.pdf, without the pending-items list or the todonotes boxes",
    )
    args = ap.parse_args()
    if args.cmd == "install":
        install()
        return 0
    paper = (ROOT / args.paper).resolve() if not args.paper.is_absolute() else args.paper
    return build(paper, submission=args.submission)


if __name__ == "__main__":
    sys.exit(main())
