"""MAFAULDA - Machinery Fault Database (UFRJ), subconjuntos de rolamento.

Fatos conferidos na pagina oficial em 2026-11-11
(https://www02.smt.ufrj.br/~offshore/mfs/page_01.html):

- Sequencias em CSV, 8 colunas, 250.000 amostras. A pagina declara **50 kHz** e 5 s, e diz
  que os modulos NI 9234 amostram a 51,2 kHz.

**A taxa real e 51,2 kHz, nao 50 kHz.** Isto foi inferido em 2026-11-11 sobre as 1120
sequencias (``FS_HZ``; a declarada fica em ``DECLARED_FS_HZ``), por duas evidencias
independentes:

1. **Tacometro contra o nome do arquivo**, que e a rotacao nominal. A razao mediana e
   0,9769 assumindo 50 kHz (vies de -2,3 %) e 1,0004 assumindo 51,2 kHz (IQR 0,9988 a
   1,0022). Os nomes sao multiplos de 1,024 = 51,2/50.
2. **Linha de rede (120 Hz, 2o harmonico de 60 Hz) no microfone.** Ela cai em 117,18 Hz no
   eixo de 50 kHz, exatamente 120 x 50/51,2. Isso vale para 863 das 1120 sequencias a
   +-0,15 Hz, contra 3 a +-0,15 Hz de 120 Hz.

Consequencias: cada sequencia dura 4,883 s. Frequencias em Hz calculadas com 50 kHz saem 2,3 %
baixas. A analise de ordem pelo tacometro nao e afetada, porque tacometro e vibracao
compartilham o relogio.
- Colunas: tacometro; acelerometro do mancal *underhang* (axial, radial, tangencial);
  acelerometro do mancal *overhang* (axial, radial, tangencial); microfone.
- Rolamento: 8 esferas, esfera de 0,7145 cm, gaiola de 2,8519 cm, angulo de contato
  nao declarado. As ordens documentadas (FTF 0,3750; BPFO 2,9980; BPFI 5,0020;
  BSF 1,8710) batem com essa geometria e angulo zero (``computed_orders``).
- Nao ha falha de pista interna: as falhas de rolamento sao ``outer_race``,
  ``ball_fault`` e ``cage_fault``, nos mancais ``underhang`` e ``overhang``, com massas
  de desbalanceamento adicionais de 0, 6, 20 e 35 g.
- Licenca e termos de uso: **nao declarados** na pagina.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FS_HZ = 51_200.0
"""Taxa real, inferida pelo tacometro e pela linha de rede (ver o docstring do modulo)."""
DECLARED_FS_HZ = 50_000.0
"""Taxa declarada na pagina oficial; usar so para comparar com trabalhos que a adotaram."""
BASE_URL = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda"
ARCHIVES = ("normal.tgz", "underhang.tgz", "overhang.tgz")

N_ELEMENTS = 8
BALL_DIAMETER_CM = 0.7145
PITCH_DIAMETER_CM = 2.8519
GEOM_RATIO = BALL_DIAMETER_CM / PITCH_DIAMETER_CM
DOCUMENTED_ORDERS = {"FTF": 0.3750, "BPFO": 2.9980, "BPFI": 5.0020, "BSF": 1.8710}

COLUMNS = (
    "tachometer",
    "underhang_axial",
    "underhang_radial",
    "underhang_tangential",
    "overhang_axial",
    "overhang_radial",
    "overhang_tangential",
    "microphone",
)
BEARINGS = ("underhang", "overhang")
MAX_ROTATION_HZ = 70.0
"""Acima da maior rotacao do dataset (61,44 Hz); define o bloqueio do tacometro."""
FAULT_FAMILY = {"outer_race": "BPFO", "ball_fault": "BSF", "cage_fault": "FTF"}

_LOAD = re.compile(r"^(\d+)g$")


def computed_orders(geom_ratio: float = GEOM_RATIO, n: int = N_ELEMENTS) -> dict[str, float]:
    """Ordens de falha pela geometria, com angulo de contato zero."""
    return {
        "FTF": 0.5 * (1 - geom_ratio),
        "BPFO": 0.5 * n * (1 - geom_ratio),
        "BPFI": 0.5 * n * (1 + geom_ratio),
        "BSF": (1 - geom_ratio**2) / (2 * geom_ratio),
    }


def parse_path(rel_path: str) -> tuple[str | None, str, int | None, float]:
    """``(mancal, falha, massa_g, rotacao_nominal_hz)`` a partir do caminho relativo.

    Exemplos: ``normal/12.288.csv`` e ``underhang/outer_race/6g/30.1056.csv``. O nome do
    arquivo e a rotacao nominal em Hz.
    """
    parts = Path(rel_path).parts
    rotation = float(Path(rel_path).stem)
    if parts[0] == "normal":
        return None, "normal", None, rotation
    if parts[0] not in BEARINGS or len(parts) < 4:
        raise ValueError(f"caminho fora do esquema MAFAULDA: {rel_path}")
    m = _LOAD.match(parts[2])
    if parts[1] not in FAULT_FAMILY or m is None:
        raise ValueError(f"caminho fora do esquema MAFAULDA: {rel_path}")
    return parts[0], parts[1], int(m.group(1)), rotation


@dataclass(frozen=True)
class MafauldaRecord:
    """Uma sequencia, sempre carregando a propria procedencia."""

    path: str
    bearing: str | None
    fault: str
    load_g: int | None
    nominal_rotation_hz: float
    data: NDArray[np.float32]
    fs_hz: float = FS_HZ

    def channel(self, name: str) -> NDArray[np.float64]:
        return self.data[:, COLUMNS.index(name)].astype(np.float64)

    def vibration(self, axis: str = "radial", side: str | None = None) -> NDArray[np.float64]:
        """Acelerometro do mancal com falha (ou ``side``), no eixo pedido."""
        bearing = side or self.bearing or "underhang"
        return self.channel(f"{bearing}_{axis}")

    @property
    def fault_order(self) -> float | None:
        family = FAULT_FAMILY.get(self.fault)
        return DOCUMENTED_ORDERS[family] if family else None

    def tachometer_edges_s(self) -> NDArray[np.float64]:
        """Instantes dos flancos de subida do tacometro (um pulso por revolucao).

        Usa histerese (sobe acima de 60 % da excursao e so rearma abaixo de 40 %) e um tempo
        de bloqueio fisico: depois de cada flanco, ignora novos flancos por
        ``1/(2*MAX_ROTATION_HZ)``. Sem isso, ruido no tacometro gera flancos espurios: uma
        sequencia chegou a dar 6,3 x a rotacao nominal com o limiar simples. Um filtro pela
        mediana dos intervalos falha quando ha um glitch por pulso, porque a mediana passa a
        ser o intervalo do glitch.
        """
        tach = self.channel("tachometer")
        lo_v, hi_v = float(tach.min()), float(tach.max())
        hi = lo_v + 0.6 * (hi_v - lo_v)
        lo = lo_v + 0.4 * (hi_v - lo_v)
        holdoff = int(self.fs_hz / (2.0 * MAX_ROTATION_HZ))
        edges: list[int] = []
        armed = bool(tach[0] < lo)
        for i in np.flatnonzero((tach >= hi) | (tach <= lo)):
            if edges and i - edges[-1] < holdoff:
                continue  # dentro do bloqueio nada conta: nem flanco nem rearme (glitch)
            if armed and tach[i] >= hi:
                edges.append(int(i))
                armed = False
            elif not armed and tach[i] <= lo:
                armed = True
        e = np.asarray(edges, dtype=np.int64)
        if e.size < 3:
            raise ValueError(f"{self.path}: tacometro sem pulsos suficientes")
        return e / self.fs_hz

    def tachometer_rotation_hz(self) -> float:
        """Rotacao media pelos pulsos, para conferir contra ``nominal_rotation_hz``."""
        edges = self.tachometer_edges_s()
        return float((edges.size - 1) / (edges[-1] - edges[0]))

    def speed_profile_hz(self) -> NDArray[np.float64]:
        """Rotacao instantanea em Hz, na grade da vibracao (interpolada entre pulsos)."""
        edges = self.tachometer_edges_s()
        rot = 1.0 / np.diff(edges)
        t = np.arange(self.data.shape[0]) / self.fs_hz
        return np.interp(t, edges[:-1], rot)


def list_records(root: str | Path, subsets: tuple[str, ...] = ("normal", *BEARINGS)) -> list[str]:
    """Caminhos relativos de todas as sequencias dos subconjuntos, em ordem deterministica."""
    root = Path(root)
    out: list[str] = []
    for sub in subsets:
        out.extend(p.relative_to(root).as_posix() for p in (root / sub).rglob("*.csv"))
    return sorted(out)


def load(rel_path: str, root: str | Path, cache_dir: str | Path | None = None) -> MafauldaRecord:
    """Carrega uma sequencia. Com ``cache_dir``, guarda/le uma copia ``.npy`` (float32)."""
    bearing, fault, load_g, rotation = parse_path(rel_path)
    cache = Path(cache_dir) / Path(rel_path).with_suffix(".npy") if cache_dir else None
    if cache is not None and cache.exists():
        data = np.load(cache)
    else:
        data = np.loadtxt(Path(root) / rel_path, delimiter=",", dtype=np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, data)
    if data.ndim != 2 or data.shape[1] != len(COLUMNS):
        raise ValueError(f"{rel_path}: esperava {len(COLUMNS)} colunas, veio {data.shape}")
    return MafauldaRecord(
        path=rel_path,
        bearing=bearing,
        fault=fault,
        load_g=load_g,
        nominal_rotation_hz=rotation,
        data=data,
    )
