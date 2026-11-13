"""Compila um paper com uma TeX Live mínima (TinyTeX) instalada só dentro deste repositório.

O TinyTeX fica em `.tools/tinytex/`, ignorado pelo git. Nada vai para o PATH do sistema: o
instalador oficial roda `tlmgr path add`, e este script não roda. O instalador é o binário de
Windows do TinyTeX, a máquina deste projeto; em outro sistema, instale o TinyTeX pelo script
oficial e aponte `TEX_BIN` para ele.

    uv run python papers/shared/build_pdf.py install                 # TinyTeX + pacotes do paper
    uv run python papers/shared/build_pdf.py build papers/mssp-2026  # gera main.pdf
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
# pacotes que o elsarticle, o todonotes e o preâmbulo do paper pedem além do TinyTeX-1
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
        print(f"TinyTeX já está em {TEX}")
    else:
        dl = TOOLS / "_download"
        if dl.exists():
            shutil.rmtree(dl)
        dl.mkdir(parents=True)
        exe = dl / ASSET
        print(f"baixando {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "tinytex-local-install"})
        with urllib.request.urlopen(req) as r, exe.open("wb") as f:
            shutil.copyfileobj(r, f)
        # arquivo autoextraível: -y extrai a pasta TinyTeX no diretório corrente, sem perguntar
        subprocess.run([str(exe), "-y"], cwd=dl, check=True)
        if TEX.exists():
            shutil.rmtree(TEX)
        shutil.move(str(dl / "TinyTeX"), str(TEX))
        shutil.rmtree(dl)
        print(f"TinyTeX extraído em {TEX}")
    tlmgr("update", "--self")  # o tlmgr do pacote costuma estar atrás do repositório
    tlmgr("install", *PACKAGES)


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    # no Windows o CreateProcess procura o executável no PATH do pai, não no do filho
    exe = shutil.which(cmd[0], path=env["PATH"]) or cmd[0]
    return subprocess.run([exe, *cmd[1:]], cwd=cwd, env=env, capture_output=True).returncode


def build(paper: Path, main: str = "main") -> int:
    if not (TEX_BIN / "pdflatex.exe").exists() and not shutil.which("pdflatex"):
        print("TinyTeX ausente: rode primeiro `build_pdf.py install`.")
        return 2
    env = os.environ.copy()
    env["PATH"] = str(TEX_BIN) + os.pathsep + env.get("PATH", "")
    latex = ["pdflatex", "-interaction=nonstopmode", "-file-line-error", f"{main}.tex"]
    steps = [latex, ["bibtex", main], latex, latex]
    codes = [run(cmd, paper, env) for cmd in steps]
    log = (paper / f"{main}.log").read_text(encoding="latin-1", errors="replace")
    errors = [line for line in log.splitlines() if line.startswith("!") or ".tex:" in line]
    missing = re.findall(r"File `([^']+)' not found", log)
    undef = len(re.findall(r"(Citation|Reference) `[^']+' on page", log))
    blg = paper / f"{main}.blg"
    bib_warn = blg.read_text(encoding="latin-1").count("Warning--") if blg.exists() else 0
    pages = re.search(r"Output written on \S+ \((\d+) pages", log)
    print(f"códigos de saída: {codes}")
    print(
        f"páginas: {pages.group(1) if pages else '?'}; referências indefinidas: {undef}; "
        f"avisos do BibTeX: {bib_warn}"
    )
    if missing:
        print("arquivos ausentes (instale com tlmgr):", sorted(set(missing)))
    for line in errors[:15]:
        print("  ", line)
    pdf = paper / f"{main}.pdf"
    ok = pdf.exists() and codes[-1] == 0
    print(f"{'PDF gerado' if ok else 'falhou'}: {pdf}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install")
    b = sub.add_parser("build")
    b.add_argument("paper", type=Path)
    args = ap.parse_args()
    if args.cmd == "install":
        install()
        return 0
    return build((ROOT / args.paper).resolve() if not args.paper.is_absolute() else args.paper)


if __name__ == "__main__":
    sys.exit(main())
