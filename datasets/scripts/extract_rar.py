"""Extracts the KAt-DataCenter .rar files with unrar-cffi, refusing paths outside the target."""

import sys
from pathlib import Path

from unrar.cffi import rarfile

arch, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
for a in sorted(arch.glob("*.rar")):
    rf = rarfile.RarFile(str(a))
    n = 0
    infos = rf.infolist()
    names = [i.filename.replace("\\", "/").rstrip("/") for i in infos]
    dirs = {x for x in names if any(y.startswith(x + "/") for y in names)}
    for info, name in zip(infos, names, strict=True):
        if name in dirs or getattr(info, "is_dir", lambda: False)():
            continue
        target = (out / name).resolve()
        if out not in target.parents:
            raise SystemExit(f"path outside the target directory in {a.name}: {name}")
        if target.exists() and target.stat().st_size == info.file_size:
            n += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = rf.read(info)
        if len(data) != info.file_size:
            raise SystemExit(f"wrong size in {a.name}: {name}")
        target.write_bytes(data)
        n += 1
    print(f"ok {a.stem} {n}", flush=True)
print("EXTRACTION_DONE", flush=True)
