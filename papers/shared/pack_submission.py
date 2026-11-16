"""Builds the files to upload to the journal from the versioned sources.

usage: uv run python papers/shared/pack_submission.py papers/mssp-2026

Writes <paper>/submission/, which git ignores:
- Manuscript.pdf: the submission PDF (main-submission.pdf, rebuilt first);
- LaTeX_source.zip: the LaTeX sources in a single folder, because Elsevier's LaTeX compiler cannot
  process subfolders, with main.bbl and a refs.bib reduced to the cited entries and to the fields
  BibTeX uses (the repository's own fields, such as `verified` and `annote`, stay out);
- Highlights.txt: the highlights, checked against the journal's limits (3 to 5, of up to 85
  characters);
- Cover_letter.pdf: cover_letter.md, typeset;
- declarationStatement.docx: the Word file of Elsevier's declaration tool, copied as it was saved
  in <paper>/, where git ignores it;
- suggested_reviewers.md: copied from <paper>/;
- Checklist.md: submission_checklist.md, the steps of the submission system, with the title,
  abstract, keywords, authors and statements of main.tex filled in as plain text for the web forms.

The cover letter, the suggested reviewers and the checklist are kept out of the public repository,
where the package comes without them.

The flat sources are compiled twice in a temporary folder, once with the shipped main.bbl and no
BibTeX and once with BibTeX. The script stops if either compile differs from main-submission.pdf in
pages, undefined references, overfull boxes or errors, or if BibTeX on the reduced refs.bib does
not give back the same main.bbl.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pdf  # noqa: E402

DROP_FIELDS = {"annote", "cluster", "verified"}
HIGHLIGHTS = (3, 5, 85)  # minimum and maximum count, maximum characters per highlight
CHECKLIST = "submission_checklist.md"
DECLARATION = "declarationStatement*.docx"  # name given by declarations.elsevier.com
REVIEWERS = "suggested_reviewers.md"
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
ACCENTS = {"'": "́", "`": "̀", "^": "̂", "~": "̃", '"': "̈"}
CEDILLA = "̧"
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


def fail(message: str) -> NoReturn:
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


def braced(text: str, start: int) -> tuple[str, int]:
    """Content of the brace group that opens at text[start], and the index after it."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1 : i], i + 1
    fail("unbalanced braces in main.tex")


def arguments(tex: str, command: str) -> list[tuple[str, str]]:
    """Optional and braced arguments of every \\command in tex, in order."""
    found = []
    for m in re.finditer(r"\\" + command + r"(?![A-Za-z])(?:\[([^\]]*)\])?\s*(?=\{)", tex):
        found.append((m.group(1) or "", braced(tex, m.end())[0]))
    return found


def plain(tex: str, labels: dict[str, str]) -> str:
    """LaTeX of the front matter and of the statements as plain text, for the web forms."""

    def ref(m: re.Match[str]) -> str:
        if m.group(1) not in labels:
            fail(f"label {m.group(1)!r} is not in main-submission.aux")
        return labels[m.group(1)]

    s = re.sub(r"\s*\\cite[pt]?\{[^}]*\}", "", tex)
    s = re.sub(r"\\ref\{([^}]*)\}", ref, s)
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:url|textbf|textit|emph)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("\\times", "×").replace(" ", ""), s)
    s = re.sub(r"\\([`'^~\"])\{?([A-Za-z])\}?", lambda m: m.group(2) + ACCENTS[m.group(1)], s)
    s = re.sub(r"\\c\{([A-Za-z])\}", lambda m: m.group(1) + CEDILLA, s)
    for old, new in (("---", "—"), ("--", "–"), ("\\%", "%"), ("\\&", "&"), ("~", " ")):
        s = s.replace(old, new)
    s = unicodedata.normalize("NFC", " ".join(s.split()))
    if re.search(r"[\\${}]", s):
        fail(f"no plain-text rule for {s[:80]!r}; extend plain() in pack_submission.py")
    return s


def section(tex: str, title: str) -> str:
    """Body of the unnumbered section with this title."""
    m = re.search(
        r"\\section\*\{" + re.escape(title) + r"\}(.*?)(?=\\section|\\appendix|\\end\{document\})",
        tex,
        flags=re.S,
    )
    if not m:
        fail(f"main.tex has no section {title!r}")
    return m.group(1)


def name_parts(full: str) -> tuple[str, str, bool]:
    """First and last names as the Zenodo record splits them; False when the split is a guess."""
    for path in (build_pdf.ROOT / "release" / "zenodo.json", build_pdf.ROOT / ".zenodo.json"):
        if path.exists():
            for creator in json.loads(path.read_text(encoding="utf-8")).get("creators", []):
                last, _, first = creator.get("name", "").partition(", ")
                if f"{first} {last}" == full:
                    return first, last, True
    first, _, last = full.rpartition(" ")
    return first, last, False


def authors_block(tex: str, labels: dict[str, str]) -> str:
    """The authors in the order of main.tex, with the fields of the submission form."""
    places = {}
    for label, body in arguments(tex, "affiliation"):
        parts = {}
        for m in re.finditer(r"(\w+)\s*=\s*(?=\{)", body):
            parts[m.group(1)] = plain(braced(body, m.end())[0], labels)
        city = " ".join(p for p in (parts.get("postcode"), parts.get("city")) if p)
        keys = (parts.get("organization"), parts.get("addressline"), city, parts.get("state"))
        places[label] = ", ".join(p for p in (*keys, parts.get("country")) if p)
    authors: list[dict[str, str]] = []
    for m in re.finditer(r"\\(author|ead)(?![A-Za-z])(?:\[([^\]]*)\])?\s*(?=\{)", tex):
        body = braced(tex, m.end())[0]
        if m.group(1) == "author":
            name = plain(re.sub(r"\\corref\{[^}]*\}", "", body), labels)
            role = " (corresponding author)" if "\\corref" in body else ""
            place = "; ".join(places.get(k.strip(), "?") for k in (m.group(2) or "").split(","))
            authors.append({"name": name, "role": role, "place": place, "email": "?"})
        elif authors:
            authors[-1]["email"] = body.strip()
    if not authors:
        fail("main.tex has no \\author")
    lines = []
    for n, a in enumerate(authors, 1):
        first, last, sure = name_parts(a["name"])
        lines += [
            f"{n}. **{a['name']}**{a['role']}",
            f"   - First name: {first}",
            f"   - Last name: {last}" + ("" if sure else " (split guessed: check it)"),
            f"   - Email: {a['email']}",
            f"   - Institution: {a['place']}",
        ]
    return "\n".join(lines)


def credit_block(tex: str, labels: dict[str, str]) -> str:
    body = section(tex, "CRediT authorship contribution statement")
    entries = re.findall(r"\\textbf\{([^}:]*):\}(.*?)(?=\\textbf\{|$)", body, flags=re.S)
    if not entries:
        fail("the CRediT section has no \\textbf{Name:} entries")
    return "\n".join(f"- {plain(n, labels)}: {plain(r, labels).rstrip('.')}" for n, r in entries)


def checklist(template: str, tex: str, labels: dict[str, str], pages: object) -> str:
    """The template with each {{field}} filled from main.tex; fails on an unknown field."""
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    keywords = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, flags=re.S)
    if not (abstract and keywords):
        fail("main.tex has no abstract or no keyword environment")
    fields: dict[str, Callable[[], str]] = {
        "date": lambda: datetime.date.today().isoformat(),
        "title": lambda: plain(arguments(tex, "title")[0][1], labels),
        "abstract": lambda: plain(abstract.group(1), labels),
        "abstract_words": lambda: str(len(plain(abstract.group(1), labels).split())),
        "keywords": lambda: "; ".join(plain(k, labels) for k in keywords.group(1).split("\\sep")),
        "authors": lambda: authors_block(tex, labels),
        "credit": lambda: credit_block(tex, labels),
        "competing": lambda: plain(section(tex, "Declaration of competing interest"), labels),
        "funding": lambda: plain(section(tex, "Funding"), labels),
        "data": lambda: plain(section(tex, "Data and code availability"), labels),
        "pages": lambda: str(pages),
    }

    def fill(m: re.Match[str]) -> str:
        if m.group(1) not in fields:
            fail(f"{CHECKLIST} asks for an unknown field {{{{{m.group(1)}}}}}")
        return fields[m.group(1)]()

    return PLACEHOLDER.sub(fill, template)


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

    # the Word file of Elsevier's declaration tool goes as it was downloaded, without conversion
    declarations = sorted(paper.glob(DECLARATION), key=lambda p: p.stat().st_mtime)
    if not declarations:
        print(f"no {DECLARATION} in {paper.name}/: package without the declaration of interests")
    else:
        newest = declarations[-1]
        shutil.copy2(newest, out / newest.name)
        if len(declarations) > 1:
            print(f"{len(declarations)} declaration files: copied the newest, {newest.name}")

    reviewers = paper / REVIEWERS
    if reviewers.exists():
        shutil.copy2(reviewers, out / REVIEWERS)
    else:
        print(f"{REVIEWERS} not found: package without the suggested reviewers")

    template = paper / CHECKLIST
    if not template.exists():
        print(f"{CHECKLIST} not found: package without a checklist")
    else:
        lines = (paper / "main.tex").read_text(encoding="utf-8").splitlines()
        tex = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in lines)  # drops comments
        aux = (paper / "main-submission.aux").read_text(encoding="latin-1")
        labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))
        source = template.read_text(encoding="utf-8")
        (out / "Checklist.md").write_text(
            checklist(source, tex, labels, reference["pages"]), encoding="utf-8"
        )
        print(f"checklist: {len(PLACEHOLDER.findall(source))} fields filled from main.tex")

    print(f"LaTeX_source.zip: {len(names)} files, no subfolders")
    print(f"highlights: {len(highlights)}, longest {max(map(len, highlights))} characters")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:26s} {f.stat().st_size:>9,d} bytes")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
