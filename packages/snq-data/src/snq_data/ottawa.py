"""Dataset de Ottawa 2018 — vibração de rolamento sob velocidade variável.

200 kSPS, acelerômetro no canal 1 e **encoder no canal 2**. É o único dos candidatos
com referência angular, que Smith & Randall dizem ser *"essential"* para velocidade
variável — e sem a qual a análise de ordem não opera.

Três fatos que **não estão na página do dataset** e foram determinados por dry-run
em 2026-09-03:

- **O encoder tem 1024 pulsos por revolução.** Nenhum valor está documentado; 1024 é
  o único candidato que faz a ordem 1 (eixo) aparecer no espectro de ordem junto com
  a frequência de falha.
- **A banda informativa está em 6,8–10 kHz**, com curtose de 34. Os 97% de energia
  abaixo de 50 Hz são conteúdo estrutural, não falha.
- **A folga nominal de 200 kSPS é enganosa.** Só 0,1% da energia está acima de
  20 kHz, então a subamostragem só começa a testar alguma coisa a partir do Nyquist
  efetivo da banda informativa, em torno de 20 kSPS.

Licença CC BY 4.0. Acesso programático pela API pública do Mendeley, sem cadastro.
"""

from __future__ import annotations

import http.client
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FS_HZ = 200_000.0
PULSES_PER_REV = 1024
"""Determinado empiricamente; ver o docstring do módulo."""

API = "https://data.mendeley.com/public-api/datasets/v43hmbwxpm/files?folder_id=root&version=1"
USER_AGENT = "subnyquist-thesis/0.1 (academic use)"

CONDITIONS = {"H": "healthy", "I": "inner_race", "O": "outer_race"}
SCENARIOS = ("A", "B", "C", "D")
"""A: acelerando · B: desacelerando · C: acelera então desacelera · D: o inverso."""

# Ordens de falha desta bancada, derivadas do dry-run: a BPFI medida em 5,43 implica
# 9 elementos e razao ~0,2067, e dai sai a BPFO. Conferir contra a documentacao do
# dataset antes de publicar.
N_ELEMENTS = 9
GEOM_RATIO = 0.2067
FAULT_ORDERS = {
    "inner_race": (N_ELEMENTS / 2) * (1 + GEOM_RATIO),
    "outer_race": (N_ELEMENTS / 2) * (1 - GEOM_RATIO),
    "cage": 0.5 * (1 - GEOM_RATIO),
}


@dataclass(frozen=True)
class OttawaRecord:
    """Um registro, sempre carregando a própria procedência."""

    name: str
    condition: str
    scenario: str
    trial: int
    vibration: NDArray[np.float64]
    encoder: NDArray[np.float64]
    fs_hz: float = FS_HZ

    @property
    def duration_s(self) -> float:
        return self.vibration.size / self.fs_hz

    @property
    def fault_order(self) -> float | None:
        return FAULT_ORDERS.get(self.condition)

    def cite(self) -> str:
        return f"Ottawa {self.name} ({self.condition}, cenário {self.scenario})"

    def speed_profile_hz(self, ppr: int = PULSES_PER_REV) -> NDArray[np.float64]:
        """Rotação instantânea em Hz, amostrada na mesma grade da vibração.

        Extraída dos flancos de subida do encoder e interpolada. É o que alimenta
        `snq_recon.angular_resample`.
        """
        enc = self.encoder
        thr = 0.5 * (float(enc.max()) + float(enc.min()))
        edges = np.flatnonzero((enc[:-1] < thr) & (enc[1:] >= thr)) / self.fs_hz
        if edges.size < 3:
            raise ValueError(f"{self.name}: encoder sem pulsos suficientes")
        rot = 1.0 / (np.diff(edges) * ppr)
        t = np.arange(self.vibration.size) / self.fs_hz
        return np.interp(t, edges[:-1], rot)

    def speed_range_rpm(self, ppr: int = PULSES_PER_REV) -> tuple[float, float]:
        r = self.speed_profile_hz(ppr) * 60.0
        return float(r.min()), float(r.max())


def _file_index() -> dict[str, str]:
    """Nome do arquivo → URL de download, pela API pública do Mendeley."""
    req = urllib.request.Request(API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310
        data = json.loads(r.read())
    return {f["filename"]: f["content_details"]["download_url"] for f in data}


def download(name: str, dest_dir: str | Path, *, tries: int = 4) -> Path:
    """Baixa um registro pelo nome, por exemplo ``"I-A-1.mat"``."""
    dest = Path(dest_dir) / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _file_index()[name]

    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310
                expected = r.headers.get("Content-Length")
                blob = r.read()
            if expected and len(blob) != int(expected):
                raise OSError(f"truncado: {len(blob)} de {expected}")
            if not blob.startswith(b"MATLAB"):
                raise OSError("resposta não é um arquivo MAT")
            dest.write_bytes(blob)
            return dest
        # IncompleteRead herda de HTTPException, nao de OSError (ver snq_data.cwru).
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(3.0 * (attempt + 1))
    raise OSError(f"{name}: falhou em {tries} tentativas ({last})")


def load(name: str, data_dir: str | Path) -> OttawaRecord:
    """Carrega um registro já baixado."""
    from scipy.io import loadmat

    path = Path(data_dir) / name
    mat = loadmat(io.BytesIO(path.read_bytes()))
    stem = name.replace(".mat", "")
    cond, scen, trial = stem.split("-")
    return OttawaRecord(
        name=stem,
        condition=CONDITIONS[cond],
        scenario=scen,
        trial=int(trial),
        vibration=np.ravel(mat["Channel_1"]).astype(float),
        encoder=np.ravel(mat["Channel_2"]).astype(float),
    )


def record_names(
    conditions: tuple[str, ...] = ("H", "I", "O"),
    scenarios: tuple[str, ...] = SCENARIOS,
    trials: tuple[int, ...] = (1, 2, 3),
) -> list[str]:
    """Nomes de registro para o subconjunto pedido, em ordem determinística."""
    return [f"{c}-{s}-{t}.mat" for c in conditions for s in scenarios for t in trials]
