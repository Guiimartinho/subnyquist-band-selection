"""Relabels the result files of the released experiments, and inverts the relabelling exactly.

The labels that the runs wrote into their result files were Portuguese words; `relabel_map.py`
maps each one to its English name. This tool applies that map to every tracked text result file
of the released experiments (``experiments/E07..E26-*/results*/``: csv, the content of csv.gz,
json, txt and md) and writes ``tools/relabel_record.json``, from which the original bytes of every
file can be rebuilt (the analysis files record hashes of some of them, ``metrics_sha256``).

Three passes, in this order, each replacing only the spans it matches and no other byte:

1. **SENTENCES**, on whole values: the JSON strings (keys included) of a .json, the fields of a
   .csv (quoting kept, or added if the new value needs it), the lines of a .txt or .md.
2. **TOKENS**, on runs of ``[A-Za-z0-9_]``: a run equal to a key is replaced, so that ``bruto``
   becomes ``raw`` while ``bruto_x`` and ``xbruto`` stay.
3. **PATH_TOKENS**, on substrings: the name fragments of the renamed directories and files.

The paths in ``PATHS`` must be renamed (``git mv``) before running the tool, which reads every
file at its new path and records its original path.

**The record.** For every file: the SHA-256 of its content before and after (for a .csv.gz, of
the decompressed content; for a file with CRLF line ends, also of the content with LF line ends,
which is what git stores and what a fresh checkout has), and, per pass, the replacements made:
``"from"`` gives, for each new string, its old strings with their counts. A pass is inverted by
replacing each new string by its old one. Where that would be ambiguous, because the new string
also occurred before the pass (``raw`` in ``datasets/raw/``, or ``ceiling`` inside an English
sentence) or because two old strings share a new one, ``"seq"`` says, in run-length form
(``label*count``, separated by spaces), which occurrences of the new string, in the order of the
file after that pass, came from which old string (label i = the i-th entry of ``"from"``) and
which were already there (label 0). No per-cell copy of the data is stored.

Usage, from the repository root::

    uv run python tools/relabel_results.py --dry-run          # what would change, per file
    uv run python tools/relabel_results.py                    # relabel and write the record
    uv run python tools/relabel_results.py --invert --check   # rebuild every original in memory
    uv run python tools/relabel_results.py --invert --out DIR # write the originals under DIR

Besides the hashes, ``--invert --check`` parses both versions of each file and checks that every
JSON number, boolean and null and every CSV field without a label is unchanged, and that each
string that changed is exactly the relabelling of the original one.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_map import PATH_TOKENS, PATHS, SENTENCES, TOKENS  # noqa: E402

RECORD = ROOT / "tools" / "relabel_record.json"
EXPERIMENTS = ("E07", "E15", "E20", "E21", "E22", "E23", "E24", "E25", "E26")
RESULT_FILE = re.compile(
    r"^experiments/(?:" + "|".join(EXPERIMENTS) + r")-[^/]+/results[^/]*/"
    r".+\.(?:csv|csv\.gz|json|txt|md)$"
)
WORD = re.compile(r"[A-Za-z0-9_]+")
JSON_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.S)
CSV_QUOTED = re.compile(r'"[^"]*(?:""[^"]*)*"')
CSV_PLAIN = re.compile(r"[^,\r\n]*")
LINE = re.compile(r"[^\r\n]*")
SENTENCE_RX = [(re.compile(p), r) for p, r in SENTENCES]
ANY_SENTENCE = re.compile("|".join(f"(?:{p})" for p, _ in SENTENCES))
PATH_RX = re.compile("|".join(re.escape(k) for k in sorted(PATH_TOKENS, key=len, reverse=True)))
PATH_OUT_RX = re.compile(
    "|".join(re.escape(v) for v in sorted(set(PATH_TOKENS.values()), key=len, reverse=True))
)
PASSES = ("sentences", "tokens", "path_tokens")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- values
def kind_of(path: str) -> str:
    if path.endswith((".csv", ".csv.gz")):
        return "csv"
    return "json" if path.endswith(".json") else "lines"


def csv_spans(text: str) -> Iterator[tuple[int, int]]:
    """Spans of the CSV fields, quotes included (the dialect written by Python's csv module)."""
    i, n = 0, len(text)
    while i < n:
        m = (CSV_QUOTED if text[i] == '"' else CSV_PLAIN).match(text, i)
        if m is None:
            raise ValueError(f"unterminated quoted CSV field at offset {i}")
        yield m.span()
        j = m.end()
        if j < n and text[j] == ",":
            i = j + 1
            if i == n:  # trailing empty field
                yield n, n
        elif text.startswith("\r\n", j):
            i = j + 2
        elif j < n and text[j] in "\r\n":
            i = j + 1
        elif j < n:
            raise ValueError(f"stray character after a CSV field at offset {j}")
        else:
            i = j


def line_spans(text: str) -> Iterator[tuple[int, int]]:
    for m in LINE.finditer(text):
        if m.start() == m.end() and m.start() > 0 and text[m.start() - 1] not in "\r\n":
            continue  # the empty match that follows each line
        yield m.span()


def value_spans(text: str, kind: str) -> Iterator[tuple[int, int]]:
    """Spans of the whole values of a file: JSON strings, CSV fields or lines."""
    if kind == "json":
        return (m.span() for m in JSON_STRING.finditer(text))
    return csv_spans(text) if kind == "csv" else line_spans(text)


def decode_value(raw: str, kind: str) -> str:
    if kind == "json":
        value = json.loads(raw)
        assert isinstance(value, str)
        return value
    if kind == "csv" and raw.startswith('"'):
        return raw[1:-1].replace('""', '"')
    return raw


def encode_value(new: str, raw_old: str, kind: str) -> str:
    """Encodes ``new`` in the style of the raw value it replaces."""
    if kind == "json":
        out = json.dumps(new, ensure_ascii=raw_old.isascii())
        assert json.loads(out) == new
        return out
    if kind == "csv" and (raw_old.startswith('"') or any(c in new for c in ',"\r\n')):
        return '"' + new.replace('"', '""') + '"'
    return new


def apply_sentences(value: str) -> str:
    if not ANY_SENTENCE.search(value):
        return value
    for rx, rep in SENTENCE_RX:
        value = rx.sub(rep, value)
    return value


def relabel_value(value: str) -> str:
    """The relabelling of one decoded value, as the semantic check expects it."""
    value = apply_sentences(value)
    value = WORD.sub(lambda m: TOKENS.get(m.group(0), m.group(0)), value)
    return PATH_RX.sub(lambda m: PATH_TOKENS[m.group(0)], value)


# ----------------------------------------------------------------------------- passes
@dataclass
class Pass:
    """A substitution pass: where its candidate spans are and what each one becomes."""

    name: str
    scan: Callable[[str], Iterator[tuple[int, int]]]  # candidates before the pass
    forward: Callable[[str], str | None]  # raw span -> new raw span (None: keep it)
    scan_out: Callable[[str], Iterator[tuple[int, int]]] | None = None  # after (default: scan)

    def spans_after(self, text: str) -> Iterator[tuple[int, int]]:
        return (self.scan_out or self.scan)(text)


@dataclass
class Edits:
    """Per new string: its old strings with counts and, if ambiguous, the per-occurrence labels."""

    sources: dict[str, dict[str, int]] = field(default_factory=dict)
    seq: dict[str, list[int]] = field(default_factory=dict)  # new -> label per occurrence

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for new, olds in self.sources.items():
            entry: dict[str, Any] = {"from": olds}
            if new in self.seq:
                entry["seq"] = rle(self.seq[new])
            out[new] = entry
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Edits:
        e = cls()
        for new, entry in data.items():
            e.sources[new] = dict(entry["from"])
            if "seq" in entry:
                e.seq[new] = unrle(entry["seq"])
        return e


def rle(labels: list[int]) -> str:
    runs: list[list[int]] = []
    for lab in labels:
        if runs and runs[-1][0] == lab:
            runs[-1][1] += 1
        else:
            runs.append([lab, 1])
    return " ".join(f"{lab}*{n}" for lab, n in runs)


def unrle(text: str) -> list[int]:
    out: list[int] = []
    for run in text.split():
        lab, n = run.split("*")
        out += [int(lab)] * int(n)
    return out


def sentence_pass(kind: str) -> Pass:
    def forward(raw: str) -> str | None:
        value = decode_value(raw, kind)
        new = apply_sentences(value)
        return None if new == value else encode_value(new, raw, kind)

    return Pass("sentences", lambda t: value_spans(t, kind), forward)


TOKEN_PASS = Pass("tokens", lambda t: (m.span() for m in WORD.finditer(t)), TOKENS.get)
PATH_PASS = Pass(
    "path_tokens",
    lambda t: (m.span() for m in PATH_RX.finditer(t)),
    PATH_TOKENS.get,
    lambda t: (m.span() for m in PATH_OUT_RX.finditer(t)),
)


def passes_for(path: str) -> list[Pass]:
    return [sentence_pass(kind_of(path)), TOKEN_PASS, PATH_PASS]


def run_forward(p: Pass, text: str) -> tuple[str, Edits]:
    """Applies a pass; records the old string of each replaced occurrence of each new string."""
    pieces: list[str] = []
    edits = Edits()
    made: dict[int, str] = {}  # start in the output -> old string
    last = out_len = 0
    for a, b in p.scan(text):
        raw = text[a:b]
        new = p.forward(raw)
        if new is None or new == raw:
            continue
        pieces.append(text[last:a])
        out_len += a - last
        made[out_len] = raw
        pieces.append(new)
        out_len += len(new)
        last = b
        olds = edits.sources.setdefault(new, {})
        olds[raw] = olds.get(raw, 0) + 1
    pieces.append(text[last:])
    out = "".join(pieces)
    if not edits.sources:
        return out, edits
    labels: dict[str, list[int]] = {new: [] for new in edits.sources}
    for a, b in p.spans_after(out):
        new = out[a:b]
        if new in labels:
            old = made.get(a)
            labels[new].append(0 if old is None else 1 + list(edits.sources[new]).index(old))
    for new, seq in labels.items():
        if len(edits.sources[new]) > 1 or 0 in seq:
            edits.seq[new] = seq
        if sum(1 for lab in seq if lab) != sum(edits.sources[new].values()):
            raise ValueError(f"{p.name}: the replacements of {new!r} are not found again")
    return out, edits


def run_inverse(p: Pass, text: str, edits: Edits) -> str:
    if not edits.sources:
        return text
    pos = dict.fromkeys(edits.seq, 0)
    pieces: list[str] = []
    last = 0
    for a, b in p.spans_after(text):
        new = text[a:b]
        if new not in edits.sources:
            continue
        olds = list(edits.sources[new])
        if new in edits.seq:
            if pos[new] >= len(edits.seq[new]):
                raise ValueError(f"more occurrences of {new!r} than the record lists")
            lab = edits.seq[new][pos[new]]
            pos[new] += 1
            if lab == 0:
                continue
            old = olds[lab - 1]
        else:
            old = olds[0]
        pieces += [text[last:a], old]
        last = b
    pieces.append(text[last:])
    for new, i in pos.items():
        if i != len(edits.seq[new]):
            raise ValueError(f"fewer occurrences of {new!r} than the record lists")
    return "".join(pieces)


def relabel_text(path: str, text: str) -> tuple[str, dict[str, Any]]:
    edits: dict[str, Any] = {}
    for p in passes_for(path):
        text, e = run_forward(p, text)
        if e.sources:
            edits[p.name] = e.to_json()
    return text, edits


def invert_text(path: str, text: str, entry: dict[str, Any]) -> str:
    for p in reversed(passes_for(path)):
        if p.name in entry:
            text = run_inverse(p, text, Edits.from_json(entry[p.name]))
    return text


# ----------------------------------------------------------------------------- semantic check
def same_json(a: Any, b: Any, where: str) -> None:
    if isinstance(a, str):
        if not isinstance(b, str) or relabel_value(a) != b:
            raise ValueError(f"{where}: {a!r} -> {b!r}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or len(a) != len(b):
            raise ValueError(f"{where}: the object differs")
        for (ka, va), (kb, vb) in zip(a.items(), b.items(), strict=True):
            same_json(ka, kb, where)
            same_json(va, vb, f"{where}/{ka}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError(f"{where}: the array differs")
        for i, (va, vb) in enumerate(zip(a, b, strict=True)):
            same_json(va, vb, f"{where}[{i}]")
    elif type(a) is not type(b) or (a != b and not (a != a and b != b)):  # NaN == NaN here
        raise ValueError(f"{where}: {a!r} != {b!r}")


def semantic_check(path: str, before: str, after: str) -> None:
    """Parses both versions: only strings change, each one into its relabelling."""
    kind = kind_of(path)
    if kind == "json":
        same_json(json.loads(before), json.loads(after), path)
        return
    if kind == "csv":
        ra = list(csv.reader(io.StringIO(before, newline="")))
        rb = list(csv.reader(io.StringIO(after, newline="")))
    else:
        ra = [[x] for x in before.splitlines()]
        rb = [[x] for x in after.splitlines()]
    if len(ra) != len(rb):
        raise ValueError(f"{path}: {len(ra)} rows before, {len(rb)} after")
    for i, (xa, xb) in enumerate(zip(ra, rb, strict=True)):
        if len(xa) != len(xb) or any(relabel_value(u) != v for u, v in zip(xa, xb, strict=True)):
            raise ValueError(f"{path}: row {i} differs beyond the relabelling")


# ----------------------------------------------------------------------------- files
def read_content(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.name.endswith(".gz") else data


def write_gzip_like(path: Path, content: bytes) -> None:
    """Rewrites a .gz with the same header (flags, mtime, file name, OS), maximum compression."""
    old = path.read_bytes()
    if old[:3] != b"\x1f\x8b\x08" or old[3] & ~0x08:
        raise ValueError(f"{path}: unexpected gzip header")
    end = old.index(b"\x00", 10) + 1 if old[3] & 0x08 else 10
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    body = comp.compress(content) + comp.flush()
    size = len(content) & 0xFFFFFFFF
    path.write_bytes(
        old[:end] + body + zlib.crc32(content).to_bytes(4, "little") + size.to_bytes(4, "little")
    )


def tracked_results() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return sorted(p for p in out.splitlines() if RESULT_FILE.match(p))


def original_path(path: str) -> str:
    """Inverse of PATHS: the path of the file when its run wrote it."""
    hits = [(o, n) for o, n in PATHS.items() if path == n or path.startswith(n + "/")]
    if not hits:
        return path
    old, new = max(hits, key=lambda on: len(on[1]))
    return old + path[len(new) :]


def hashes(tag: str, data: bytes) -> dict[str, str]:
    out = {f"sha256_{tag}": sha256(data)}
    if b"\r\n" in data:
        out[f"sha256_{tag}_lf"] = sha256(data.replace(b"\r\n", b"\n"))
    return out


# ----------------------------------------------------------------------------- modes
def apply(dry_run: bool, force: bool) -> int:
    pending = [old for old in PATHS if (ROOT / old).exists()]
    if pending and not dry_run:
        raise SystemExit(f"rename these first (git mv): {pending}")
    if RECORD.exists() and not (dry_run or force):
        raise SystemExit(f"{RECORD.name} exists: the files are already relabelled")
    files: dict[str, Any] = {}
    n_changed = 0
    for rel in tracked_results():
        path = ROOT / rel
        raw = read_content(path)
        text = raw.decode("utf-8")
        new_text, edits = relabel_text(rel, text)
        new = new_text.encode("utf-8")
        if invert_text(rel, new_text, edits).encode("utf-8") != raw:
            raise SystemExit(f"{rel}: the inverse does not rebuild the original")
        semantic_check(rel, text, new_text)
        entry: dict[str, Any] = {}
        if original_path(rel) != rel:
            entry["original_path"] = original_path(rel)
        entry |= hashes("before", raw) | hashes("after", new)
        if rel.endswith(".gz"):
            entry["gzip_sha256_before"] = sha256(path.read_bytes())
        if new != raw:
            n_changed += 1
            if not dry_run:
                if rel.endswith(".gz"):
                    write_gzip_like(path, new)
                    entry["gzip_sha256_after"] = sha256(path.read_bytes())
                else:
                    path.write_bytes(new)
        entry |= edits
        files[rel] = entry
        if dry_run and edits:
            short = {
                p: {k: (v["from"], "seq" in v) for k, v in e.items()} for p, e in edits.items()
            }
            print(rel, json.dumps(short, ensure_ascii=False)[:800])
    record = {
        "about": (
            "Relabelling of the result files by tools/relabel_results.py with tools/"
            "relabel_map.py; see the docstring of the tool for the format."
        ),
        "map_sha256": sha256((ROOT / "tools" / "relabel_map.py").read_bytes()),
        "passes": list(PASSES),
        "files": files,
    }
    print(f"{len(files)} result files, {n_changed} {'to change' if dry_run else 'changed'}")
    if not dry_run:
        RECORD.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", "utf-8")
        print(f"-> {RECORD.relative_to(ROOT).as_posix()}")
    return 0


def invert(check: bool, out_dir: Path | None) -> int:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    tracked = set(tracked_results())
    missing = sorted(set(record["files"]) - tracked)
    extra = sorted(tracked - set(record["files"]))
    bad = []
    for rel, entry in record["files"].items():
        path = ROOT / rel
        if not path.exists():
            continue
        cur = read_content(path)
        if sha256(cur) not in (entry["sha256_after"], entry.get("sha256_after_lf")):
            bad.append(f"{rel}: changed since it was relabelled")
            continue
        text = cur.decode("utf-8")
        orig = invert_text(rel, text, entry).encode("utf-8")
        if sha256(orig) not in (entry["sha256_before"], entry.get("sha256_before_lf")):
            bad.append(f"{rel}: the inverse does not match sha256_before")
            continue
        if check:
            try:
                if relabel_text(rel, orig.decode("utf-8"))[0] != text:
                    raise ValueError(f"{rel}: relabelling the original does not give the file")
                semantic_check(rel, orig.decode("utf-8"), text)
            except ValueError as exc:
                bad.append(str(exc))
        if out_dir is not None:
            dst = out_dir / entry.get("original_path", rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(gzip.compress(orig, mtime=0) if rel.endswith(".gz") else orig)
    for rel in missing:
        print(f"FAIL {rel}: in the record, not tracked")
    for rel in extra:
        print(f"note {rel}: tracked, not in the record")
    for b in bad:
        print("FAIL", b)
    n = len(record["files"])
    ok = n - len(bad) - len(missing)
    print(f"{ok}/{n} files rebuilt to their recorded sha256_before")
    return 1 if bad or missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="report what would change")
    ap.add_argument("--force", action="store_true", help="relabel even if the record exists")
    ap.add_argument("--invert", action="store_true", help="rebuild the original contents")
    ap.add_argument("--check", action="store_true", help="with --invert: verify in memory")
    ap.add_argument("--out", type=Path, default=None, help="with --invert: write originals here")
    args = ap.parse_args()
    if args.invert:
        if not (args.check or args.out):
            ap.error("--invert needs --check or --out")
        return invert(args.check, args.out)
    return apply(args.dry_run, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
