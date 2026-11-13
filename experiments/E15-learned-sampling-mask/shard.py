"""Parallel shard of E15: the same runs as ``run.py``, in another process, in the same ``runs/``.

**What it is for.** On Paderborn, training is limited by one CPU core, and the GPU sits at about
6 % utilization (measured with NVML on 2026-11-12). The serial run has 48 cells (4 folds x 3
seeds x 4 budgets) of about half an hour each.

This script runs, in another process, the cells it receives in ``--cells`` and writes each run
to ``runs/`` with the same ``run_key`` as ``run.py``. The key covers the config, ``run.py``, the
package code and the CLI options that change the result. When the main process reaches a cell
that is already written, it resumes it (``execute``), and its manifest marks it as
``"resumed"``.

**What does not change:**
- **No computation.** The script calls ``run.run_method`` with the same context. The data seed
  comes from ``derive_seed`` per cell, and cudnn is deterministic.
- **No final result.** The script exits before ``summarize``, so it does not write
  ``metrics.csv``, ``summary.csv``, ``hypotheses.csv``, ``masks.json`` or the manifest. Those are
  written by the main process, which goes through all the cells.

**What it adds:**
- **Cache of the classical stage.** It is the list of ``classical_task`` results per record, in
  a file whose name includes the hash of the config, of ``run.py`` and of the packages.
  ``--only-cache`` computes, writes and exits; the following shards only read it.
- **Atomic write of each run** (temporary file + ``os.replace``). This way the main process
  never reads a half-written JSON.

Usage, with the ``run.py`` CLI after ``--``:

    python shard.py --cache C.pkl --only-cache -- --config config/paderborn.yaml --workers 16
    python shard.py --cache C.pkl --cells 2.0.4 2.0.8 -- --config config/paderborn.yaml

A cell is ``split.seed.budget``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import run  # noqa: E402  (the experiment's run.py, unmodified)


class ShardDone(Exception):  # noqa: N818  (normal end, not an error)
    """End of the shard: exits before any final result is written."""


def cache_key(config: Path) -> str:
    h = hashlib.sha256()
    h.update(config.read_bytes())
    h.update((EXP / "run.py").read_bytes())
    h.update(str(run.source_sha256()[0]).encode())
    return h.hexdigest()[:16]


def make_pool(cache: Path, only_cache: bool, key: str):
    class CachedPool:
        """Replaces the classical stage's ProcessPoolExecutor (its only use in run.py)."""

        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, tasks):
            if cache.exists():
                saved = pickle.loads(cache.read_bytes())
                if saved.get("key") != key:
                    raise SystemExit(
                        f"cache from another config or code: {saved.get('key')} != {key}"
                    )
                out = saved["cls"]
                print(f"[shard] classical stage read from {cache.name} ({key})", flush=True)
            else:
                t0 = time.time()
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    out = list(pool.map(fn, tasks))
                tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
                tmp.write_bytes(pickle.dumps({"key": key, "cls": out}))
                os.replace(tmp, cache)
                print(
                    f"[shard] classical stage written to {cache.name}: {time.time() - t0:.0f} s",
                    flush=True,
                )
            if only_cache:
                raise ShardDone
            return out

    return CachedPool


def make_execute(cells: set[tuple[int, int, int]]):
    def shard_execute(method, ctx, runs_dir, run_key, no_resume, manifest):
        rc = ctx.rc
        cell = (int(rc.split["index"]), int(rc.seed), int(rc.budget))
        if cell not in cells:
            return {"rows": [], "info": {}}  # cell of another process; nothing is written
        rel = Path(rc.ds) / rc.variant / f"b{rc.budget}" / f"split{rc.split['index']}"
        rel = rel / f"seed{rc.seed}" / f"{method}.json"
        run_file = runs_dir / rel
        if run_file.exists() and not no_resume:
            try:
                saved = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                saved = None  # another process writing non-atomically: recompute
            if saved and saved.get("run_key") == run_key:
                return saved
        t0 = time.time()
        out = run.run_wdcnn(ctx) if method == "wdcnn" else run.run_method(method, ctx)
        out["run_key"] = run_key
        run_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = run_file.with_name(f".{run_file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out, default=float), encoding="utf-8")
        os.replace(tmp, run_file)
        print(f"  [shard] {rel.as_posix()}: {time.time() - t0:.0f} s", flush=True)
        return out

    return shard_execute


def stop(*_a, **_k):
    raise ShardDone


def parse(argv: list[str]):
    if "--" not in argv:
        raise SystemExit("missing '--' before the run.py CLI")
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--only-cache", action="store_true")
    ap.add_argument("--cells", nargs="*", default=[])
    return ap.parse_args(argv[:i]), argv[i + 1 :]


def main() -> int:
    own, passthrough = parse(sys.argv[1:])
    cells = {tuple(int(v) for v in c.split(".")) for c in own.cells}
    if not own.only_cache and not cells:
        raise SystemExit("no --cells and no --only-cache: nothing to do")
    if "--config" not in passthrough:
        raise SystemExit("the run.py CLI needs --config")
    key = cache_key(Path(passthrough[passthrough.index("--config") + 1]))
    run.ProcessPoolExecutor = make_pool(own.cache, own.only_cache, key)
    run.execute = make_execute(cells)
    run.select_across_budgets = lambda *_a, **_k: []  # the shard's result is discarded
    run.summarize = stop  # exits before writing metrics, summary, hypotheses and manifest
    sys.argv = [str(EXP / "run.py"), *passthrough]
    try:
        run.main()
    except ShardDone:
        print(f"[shard] done: {len(cells)} cells; the final results come from the main process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
