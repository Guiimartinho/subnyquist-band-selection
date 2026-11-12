"""Pedaço paralelo do E15: as mesmas execuções do ``run.py``, em outro processo, no mesmo ``runs/``.

**Para que serve.** No Paderborn o treino é limitado por um núcleo de CPU, e a GPU fica em cerca
de 6 % de uso (medido por NVML em 2026-11-12). A rodada serial tem 48 células (4 dobras x 3
sementes x 4 orçamentos) de cerca de meia hora.

Este script roda, em outro processo, as células que recebe em ``--cells`` e grava cada execução
em ``runs/`` com o mesmo ``run_key`` do ``run.py``. A chave cobre o config, o ``run.py``, o
código dos pacotes e a CLI que muda o resultado. O processo principal, ao chegar numa célula já
gravada, a retoma (``execute``), e o manifest dele a marca como "retomado".

**O que não muda:**
- **Nenhum cálculo.** O script chama ``run.run_method`` com o mesmo contexto. A semente de dados
  sai de ``derive_seed`` por célula, e o cudnn é determinístico.
- **Nenhum resultado final.** O script sai antes de ``summarize``, então não escreve
  ``metrics.csv``, ``summary.csv``, ``hypotheses.csv``, ``masks.json`` nem o manifest. Quem os
  escreve é o processo principal, que passa por todas as células.

**O que acrescenta:**
- **Cache do estágio clássico.** É a lista de ``classical_task`` por registro, num arquivo cujo
  nome inclui o hash do config, do ``run.py`` e dos pacotes. ``--only-cache`` calcula, grava e sai;
  os pedaços seguintes só leem.
- **Gravação atômica de cada execução** (arquivo temporário + ``os.replace``). Assim o processo
  principal nunca lê um JSON pela metade.

Uso, com a CLI do ``run.py`` depois de ``--``:

    python shard.py --cache C.pkl --only-cache -- --config config/paderborn.yaml --workers 16
    python shard.py --cache C.pkl --cells 2.0.4 2.0.8 -- --config config/paderborn.yaml

Uma célula é ``divisão.semente.orçamento``.
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

import run  # noqa: E402  (o run.py do experimento, sem alteração)


class ShardDone(Exception):  # noqa: N818  (fim normal, não erro)
    """Fim do pedaço: sai antes de qualquer escrita de resultado final."""


def cache_key(config: Path) -> str:
    h = hashlib.sha256()
    h.update(config.read_bytes())
    h.update((EXP / "run.py").read_bytes())
    h.update(str(run.source_sha256()[0]).encode())
    return h.hexdigest()[:16]


def make_pool(cache: Path, only_cache: bool, key: str):
    class CachedPool:
        """Substitui o ProcessPoolExecutor do estágio clássico (o único uso no run.py)."""

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
                        f"cache de outro config ou código: {saved.get('key')} != {key}"
                    )
                out = saved["cls"]
                print(f"[pedaço] estágio clássico lido de {cache.name} ({key})", flush=True)
            else:
                t0 = time.time()
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    out = list(pool.map(fn, tasks))
                tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
                tmp.write_bytes(pickle.dumps({"key": key, "cls": out}))
                os.replace(tmp, cache)
                print(
                    f"[pedaço] estágio clássico gravado em {cache.name}: {time.time() - t0:.0f} s",
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
            return {"rows": [], "info": {}}  # célula de outro processo; nada é gravado
        rel = Path(rc.ds) / rc.variant / f"b{rc.budget}" / f"split{rc.split['index']}"
        rel = rel / f"seed{rc.seed}" / f"{method}.json"
        run_file = runs_dir / rel
        if run_file.exists() and not no_resume:
            try:
                saved = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                saved = None  # outro processo gravando sem atomicidade: recalcula
            if saved and saved.get("run_key") == run_key:
                return saved
        t0 = time.time()
        out = run.run_wdcnn(ctx) if method == "wdcnn" else run.run_method(method, ctx)
        out["run_key"] = run_key
        run_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = run_file.with_name(f".{run_file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out, default=float), encoding="utf-8")
        os.replace(tmp, run_file)
        print(f"  [pedaço] {rel.as_posix()}: {time.time() - t0:.0f} s", flush=True)
        return out

    return shard_execute


def stop(*_a, **_k):
    raise ShardDone


def parse(argv: list[str]):
    if "--" not in argv:
        raise SystemExit("faltou '--' antes da CLI do run.py")
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
        raise SystemExit("sem --cells e sem --only-cache: nada a fazer")
    if "--config" not in passthrough:
        raise SystemExit("a CLI do run.py precisa de --config")
    key = cache_key(Path(passthrough[passthrough.index("--config") + 1]))
    run.ProcessPoolExecutor = make_pool(own.cache, own.only_cache, key)
    run.execute = make_execute(cells)
    run.select_across_budgets = lambda *_a, **_k: []  # resultado do pedaço é descartado
    run.summarize = stop  # sai antes de escrever metrics, summary, hypotheses e manifest
    sys.argv = [str(EXP / "run.py"), *passthrough]
    try:
        run.main()
    except ShardDone:
        print(f"[pedaço] terminou: {len(cells)} células; resultados finais são do principal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
