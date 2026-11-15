"""Builds the files to upload to the journal from the versioned sources.

usage: uv run python papers/shared/pack_submission.py papers/mssp-2026

Writes <paper>/submission/, which git ignores:
- Manuscript.pdf: the submission PDF (main-submission.pdf, rebuilt first);
- LaTeX_source.zip: the LaTeX sources in a single folder, because Elsevier's LaTeX compiler cannot
  process subfolders, with main.bbl and a refs.bib reduced to the cited entries and to the fields
  BibTeX uses (the repository's own fields, such as `verified` and `annote`, stay out);
- Highlights.txt: the highlights, checked against the journal's limits (3 to 5, of up to 85
  characters);
- Cover_letter.pdf: cover_letter.md, typeset (the cover letter is kept out of the public
  repository, where the package comes without it).

The flat sources are compiled twice in a temporary folder, once with the shipped main.bbl and no
BibTeX and once with BibTeX. The script stops if either compile differs from main-submission.pdf in
pages, undefined references, overfull boxes or errors, or if BibTeX on the reduced refs.bib does
not give back the same main.bbl.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pdf  # noqa: E402

DROP_FIELDS = {"annote", "cluster", "verified"}
HIGHLIGHTS = (3, 5, 85)  # minimum and maximum count, maximum characters per highlight
INPUT_SECTION = re.compile(r"\\input\{sections/([^}]+)\}")
INPUT_TABLE = re.compile(r"\\input\{tables/([^}]+)\}")
GRAPHICS = re.compile(r"\\includegraphics(\[[^\]]*\])?\{figures/([^}]+)\}")
LETTER = r"""\documentclass[11pt,a4paper]{article}
\usepackage[hmargin=2cm,vmargin=1.8cm]{geometry}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}
\pagestyle{empty}
\begin{document}
@BODY@
\end{document}
"""
LIST_SPACING = r"\setlength{\topsep}{0pt}\setlength{\itemsep}{2pt}\setlength{\parsep}{0pt}"
TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def fail(message: str) -> None:
    print(f"error: {message}")
    sys.exit(1)


def tex_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(build_pdf.TEX_BIN) + os.pathsep + env.get("PATH", "")
    return env


def log_stats(log: str) -> dict[str, object]:
    pages = re.search(r"Output written on \S+ \((\d+) pages?", log)
    return {
        "pages": int(pages.group(1)) if pages else None,
        "undefined": len(re.findall(r"(Citation|Reference) `[^']+' on page", log)),
        "overfull": len(re.findall(r"^Overfull", log, flags=re.M)),
        "errors": [line for line in log.splitlines() if line.startswith("!")],
    }


def stage_sources(paper: Path, stage: Path) -> list[str]:
    """Copies the sources into one folder, rewriting the paths; returns the copied names."""
    main = (paper / "main.tex").read_text(encoding="utf-8")
    lines = main.splitlines()
    first = next(i for i, line in enumerate(lines) if not line.startswith("%"))
    main = "\n".join(lines[first:]) + "\n"
    for old, new in (
        ("\\usepackage[colorinlistoftodos]{todonotes}\n", ""),
        ("\\listoftodos\n", ""),
        ("\\bibliography{../../literature/refs}", "\\bibliography{refs}"),
    ):
        if old not in main:
            fail(f"main.tex no longer contains {old.strip()!r}; update pack_submission.py")
        main = main.replace(old, new)
    sections = INPUT_SECTION.findall(main)
    main = INPUT_SECTION.sub(r"\\input{\1}", main)
    header = (
        "% Manuscript submitted to Mechanical Systems and Signal Processing.\n"
        "% Compile with pdflatex, bibtex, pdflatex, pdflatex; main.bbl is included.\n"
    )
    copied = {"main.tex": header + main}
    for name in sections:
        text = (paper / "sections" / f"{name}.tex").read_text(encoding="utf-8")
        for table in INPUT_TABLE.findall(text):
            copied[table] = (paper / "tables" / table).read_text(encoding="utf-8")
        for _, figure in GRAPHICS.findall(text):
            copied[figure] = paper / "figures" / figure
        text = INPUT_TABLE.sub(r"\\input{\1}", text)
        text = GRAPHICS.sub(
            lambda m: "\\includegraphics" + (m.group(1) or "") + "{" + m.group(2) + "}", text
        )
        if f"{name}.tex" in copied:
            fail(f"two files would be named {name}.tex in the flat folder")
        copied[f"{name}.tex"] = text
    for name, content in copied.items():
        if isinstance(content, Path):
            shutil.copy2(content, stage / name)
        else:
            if re.search(r"\{(sections|tables|figures)/", content):
                fail(f"{name} still points to a subfolder")
            (stage / name).write_text(content, encoding="utf-8")
    return sorted(copied)


def cited_keys(aux: Path) -> list[str]:
    keys: list[str] = []
    for group in re.findall(r"\\citation\{([^}]*)\}", aux.read_text(encoding="latin-1")):
        for key in (k.strip() for k in group.split(",")):
            if key == "*":
                fail("\\nocite{*} is not supported")
            if key and key not in keys:
                keys.append(key)
    return keys


def split_top(body: str) -> list[str]:
    """Splits a BibTeX entry body at the commas outside braces and quotes."""
    parts, current, depth, quoted = [], [], 0, False
    for ch in body:
        if ch == '"' and depth == 0:
            quoted = not quoted
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def reduced_bib(bib: str, keys: list[str]) -> str:
    entries = {}
    for m in re.finditer(r"^@(\w+)\s*\{", bib, flags=re.M):
        start, depth, i = m.end() - 1, 0, m.end() - 1
        while i < len(bib):
            depth += {"{": 1, "}": -1}.get(bib[i], 0)
            if depth == 0:
                break
            i += 1
        parts = split_top(bib[start + 1 : i])
        entries[parts[0].strip()] = (m.group(1).lower(), parts[1:])
    missing = [k for k in keys if k not in entries]
    if missing:
        fail(f"cited keys missing from refs.bib: {missing}")
    out = []
    for key in keys:
        kind, fields = entries[key]
        kept = []
        for field in fields:
            name = field.split("=", 1)[0].strip()
            if name and name.lower() not in DROP_FIELDS:
                kept.append("  " + " ".join(field.split()))
        out.append(f"@{kind}{{{key},\n" + ",\n".join(kept) + "\n}\n")
    return "\n".join(out)


def compile_flat(stage: Path, bibtex: bool) -> tuple[dict[str, object], str]:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for f in stage.iterdir():
            shutil.copy2(f, work / f.name)
        latex = ["pdflatex", "-interaction=nonstopmode", "-file-line-error", "main.tex"]
        steps = [latex, ["bibtex", "main"], latex, latex] if bibtex else [latex, latex]
        for cmd in steps:
            build_pdf.run(cmd, work, tex_env())
        log = (work / "main.log").read_text(encoding="latin-1", errors="replace")
        return log_stats(log), (work / "main.bbl").read_text(encoding="latin-1")


def letter_tex(md: str) -> str:
    def inline(s: str) -> str:
        s = "".join(TEX_SPECIAL.get(c, c) for c in s)
        s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
        s = re.sub(r'"([^"]+)"', r"``\1''", s)
        s = re.sub(r" (\d+)×", r"~\1×", s)  # keeps a factor such as 4x on the line of its word
        return s.replace("×", r"$\times$")

    lines = [line for line in md.splitlines() if not line.startswith(("#", ">"))]
    blocks, current = [], []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    parts = []
    for n, block in enumerate(blocks):
        if re.match(r"\d+\. ", block[0]):
            items: list[list[str]] = []
            for line in block:
                if re.match(r"\d+\. ", line):
                    items.append([line.split(". ", 1)[1].strip()])
                else:
                    items[-1].append(line.strip())
            body = "\n".join("\\item " + inline(" ".join(item)) for item in items)
            parts.append("\\begin{enumerate}" + LIST_SPACING + "\n" + body + "\n\\end{enumerate}")
        elif n == len(blocks) - 1:  # signature: one line each, ragged right
            lines_ = " \\\\\n".join(inline(line.strip()) for line in block)
            parts.append("{\\raggedright\n" + lines_ + "\\par}")
        else:
            parts.append(inline(" ".join(line.strip() for line in block)))
    return LETTER.replace("@BODY@", "\n\n".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", type=Path)
    args = ap.parse_args()
    paper = args.paper if args.paper.is_absolute() else (build_pdf.ROOT / args.paper).resolve()
    if build_pdf.build(paper, submission=True) != 0:
        fail("main-submission.pdf did not build")
    reference = log_stats((paper / "main-submission.log").read_text(encoding="latin-1"))
    if reference["errors"]:
        fail(f"main-submission.pdf has LaTeX errors: {reference['errors'][:3]}")
    out = paper / "submission"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        names = stage_sources(paper, stage)
        keys = cited_keys(paper / "main-submission.aux")
        bib = (build_pdf.ROOT / "literature" / "refs.bib").read_text(encoding="utf-8")
        (stage / "refs.bib").write_text(reduced_bib(bib, keys), encoding="utf-8")
        shipped_bbl = (paper / "main-submission.bbl").read_text(encoding="latin-1")
        (stage / "main.bbl").write_text(shipped_bbl, encoding="latin-1")
        names += ["refs.bib", "main.bbl"]

        for bibtex in (False, True):
            stats, bbl = compile_flat(stage, bibtex)
            label = "with BibTeX" if bibtex else "with the shipped main.bbl"
            if stats != reference:
                fail(f"flat sources {label} differ from main-submission: {stats} vs {reference}")
            if bibtex and bbl.replace("\r\n", "\n") != shipped_bbl.replace("\r\n", "\n"):
                fail("BibTeX on the reduced refs.bib does not give back the same main.bbl")
            print(f"flat sources {label}: {stats['pages']} pages, same as main-submission.pdf")

        with zipfile.ZipFile(out / "LaTeX_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(names):
                z.write(stage / name, arcname=name)

    shutil.copy2(paper / "main-submission.pdf", out / "Manuscript.pdf")

    lo, hi, width = HIGHLIGHTS
    highlights = [
        line.strip()
        for line in (paper / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lo <= len(highlights) <= hi or max(map(len, highlights)) > width:
        fail(f"highlights must be {lo} to {hi} lines of up to {width} characters")
    (out / "Highlights.txt").write_text("\n".join(highlights) + "\n", encoding="utf-8")

    # the cover letter stays out of the public repository, which then packs without it
    letter = paper / "cover_letter.md"
    if not letter.exists():
        print("cover_letter.md not found: package without a cover letter")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            md = letter.read_text(encoding="utf-8")
            (work / "letter.tex").write_text(letter_tex(md), encoding="utf-8")
            build_pdf.run(["pdflatex", "-interaction=nonstopmode", "letter.tex"], work, tex_env())
            stats = log_stats((work / "letter.log").read_text(encoding="latin-1", errors="replace"))
            if stats["errors"] or not (work / "letter.pdf").exists():
                fail(f"cover letter did not build: {stats['errors'][:3]}")
            shutil.copy2(work / "letter.pdf", out / "Cover_letter.pdf")
            print(f"cover letter: {stats['pages']} page(s), {stats['overfull']} overfull boxes")

    print(f"LaTeX_source.zip: {len(names)} files, no subfolders")
    print(f"highlights: {len(highlights)}, longest {max(map(len, highlights))} characters")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:20s} {f.stat().st_size:>9,d} bytes")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
