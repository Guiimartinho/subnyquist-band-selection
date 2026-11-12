"""Paderborn - KAt-DataCenter (Universidade de Paderborn), rolamentos 6203.

Fatos conferidos em 2026-11-12 nas páginas oficiais e no artigo PHME16 (Lessmeier, Kimotho,
Zimmer e Sextro, 2016). O detalhe está em ``datasets/manifests/paderborn.yaml``.

- **Licença:** CC BY-NC 4.0, com citação obrigatória. O dado bruto não é versionado.
- **Rolamentos:** 32, sendo 6 saudáveis (K001-K006), 12 com dano artificial e 14 com dano real
  de ensaio de vida acelerado (tabelas 4 e 5 do PHME16).
- **Medições:** 4 ajustes de operação (``SETTINGS``), com 20 medições de 4 s cada.
- **Arquivo:** ``<código>/<ajuste>_<código>_<n>.mat``, em MATLAB 5. A variável única leva o nome
  do arquivo e é um struct com ``X`` (eixos de tempo) e ``Y`` (sinais).
  - ``vibration_1`` e as duas correntes de fase estão a 64 kHz (256001 amostras).
  - ``speed`` (rpm), ``force`` e ``torque`` estão a 4 kHz (16001 amostras).
- **Geometria:** tabela 1 do PHME16 (FAG), com 8 esferas, d = 6,75 mm, D = 28,55 mm e ângulo de
  contato 0°.
  - Pelo PHME16, os fabricantes usados (FAG, MTK, IBU/IBB) variam "não mais que 1-2 %" nas
    frequências características.
  - A BPFO (3,054) e a BPFI (4,946) ficam a 0,054 ordem das harmônicas 3 e 5 do eixo, então a
    tolerância de ordem tem de ficar abaixo de ~0,027.

**Base de tempo (conferida em 2026-11-12, nos 2560 arquivos).**
- O número de amostras de 64 kHz varia: 256001 em 1042 arquivos, até 262062 nos mais longos.
- O eixo ``HostService`` vai sempre de 0 a 4 s, o que sugeriria taxa variável.
- **A taxa é de 64 kHz fixos.** O PMSM tem 4 pares de polos, então a corrente gira a exatamente 4×
  a rotação medida no canal de 4 kHz. Com 64 kHz fixos, a razão entre as duas tem mediana
  0,99998 (P1 0,99936; P99 1,00057), e só 1 arquivo passa de 0,5 %. Pela hipótese do eixo de
  tempo, 185 arquivos passam de 0,5 %, até 8 %.
- Os registros mais longos são só mais longos. O carregador corta a vibração nas primeiras 256001
  amostras (4 s), alinhadas com o canal de rotação.
- **Um arquivo não abre no scipy:** ``KA08/N15_M01_F10_KA08_2.mat`` ("Expecting matrix here").
  O carregador levanta ``ValueError``, e quem chama pula a medição.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS_HZ = 64_000.0
SPEED_FS_HZ = 4_000.0
N_ELEMENTS = 8
BALL_DIAMETER_MM = 6.75
PITCH_DIAMETER_MM = 28.55
GEOM_RATIO = BALL_DIAMETER_MM / PITCH_DIAMETER_MM

SETTINGS: dict[str, dict[str, float]] = {
    "N15_M07_F10": {"rpm": 1500.0, "torque_nm": 0.7, "radial_force_n": 1000.0},
    "N09_M07_F10": {"rpm": 900.0, "torque_nm": 0.7, "radial_force_n": 1000.0},
    "N15_M01_F10": {"rpm": 1500.0, "torque_nm": 0.1, "radial_force_n": 1000.0},
    "N15_M07_F04": {"rpm": 1500.0, "torque_nm": 0.7, "radial_force_n": 400.0},
}
MEASUREMENTS = tuple(range(1, 21))

# código -> (origem, componente, nível de extensão). PHME16, tabelas 4 e 5.
BEARINGS: dict[str, tuple[str, str, int]] = {
    **{f"K00{i}": ("healthy", "none", 0) for i in range(1, 7)},
    "KA01": ("artificial", "OR", 1),
    "KA03": ("artificial", "OR", 2),
    "KA05": ("artificial", "OR", 1),
    "KA06": ("artificial", "OR", 2),
    "KA07": ("artificial", "OR", 1),
    "KA08": ("artificial", "OR", 2),
    "KA09": ("artificial", "OR", 2),
    "KI01": ("artificial", "IR", 1),
    "KI03": ("artificial", "IR", 1),
    "KI05": ("artificial", "IR", 1),
    "KI07": ("artificial", "IR", 2),
    "KI08": ("artificial", "IR", 2),
    "KA04": ("real", "OR", 1),
    "KA15": ("real", "OR", 1),
    "KA16": ("real", "OR", 2),
    "KA22": ("real", "OR", 1),
    "KA30": ("real", "OR", 1),
    "KB23": ("real", "IR+OR", 2),
    "KB24": ("real", "IR+OR", 3),
    "KB27": ("real", "OR+IR", 1),
    "KI04": ("real", "IR", 1),
    "KI14": ("real", "IR", 1),
    "KI16": ("real", "IR", 3),
    "KI17": ("real", "IR", 1),
    "KI18": ("real", "IR", 2),
    "KI21": ("real", "IR", 1),
}

_NAME = re.compile(r"^(N\d\d_M\d\d_F\d\d)_(K[0-9A-Z]{3})_(\d{1,2})$")


def computed_orders(geom_ratio: float = GEOM_RATIO, n: int = N_ELEMENTS) -> dict[str, float]:
    """Ordens de falha pela geometria, com ângulo de contato zero."""
    return {
        "FTF": 0.5 * (1 - geom_ratio),
        "BPFO": 0.5 * n * (1 - geom_ratio),
        "BPFI": 0.5 * n * (1 + geom_ratio),
        "BSF": (1 - geom_ratio**2) / (2 * geom_ratio),
    }


def parse_name(stem: str) -> tuple[str, str, int]:
    """``N15_M07_F10_KA01_3`` -> (ajuste, código, medição)."""
    m = _NAME.match(stem)
    if not m or m.group(1) not in SETTINGS or m.group(2) not in BEARINGS:
        raise ValueError(f"nome fora do esquema do KAt-DataCenter: {stem}")
    return m.group(1), m.group(2), int(m.group(3))


def measurement_path(root: str | Path, code: str, setting: str, n: int) -> Path:
    if code not in BEARINGS or setting not in SETTINGS or n not in MEASUREMENTS:
        raise ValueError(f"medição inexistente: {code} {setting} {n}")
    return Path(root) / code / f"{setting}_{code}_{n}.mat"


def _flat(v):
    while isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        v = v.ravel()[0]
    return v


@dataclass(frozen=True)
class Measurement:
    code: str
    setting: str
    index: int
    vibration: np.ndarray
    speed_rpm: np.ndarray
    fs_hz: float = FS_HZ
    speed_fs_hz: float = SPEED_FS_HZ

    def speed_profile_hz(self) -> np.ndarray:
        """Rotação em Hz no eixo de tempo da vibração (interpolação linear do canal de 4 kHz)."""
        t_v = np.arange(self.vibration.size) / self.fs_hz
        t_s = np.arange(self.speed_rpm.size) / self.speed_fs_hz
        return np.interp(t_v, t_s, self.speed_rpm) / 60.0


N_VIB = 256_001  # 4 s a 64 kHz, alinhados com as 16001 amostras de rotação a 4 kHz


def load_measurement(root: str | Path, code: str, setting: str, n: int) -> Measurement:
    """Lê uma medição, com a vibração cortada em 4 s a 64 kHz fixos (ver o docstring do módulo).

    Confere o nome da variável, os rasters, a duração do canal de rotação e o comprimento mínimo
    da vibração. Arquivo ilegível vira ``ValueError``.
    """
    from scipy.io import loadmat

    path = measurement_path(root, code, setting, n)
    try:
        m = loadmat(path, squeeze_me=False, struct_as_record=False)
    except (TypeError, ValueError) as exc:  # KA08/N15_M01_F10_KA08_2.mat
        raise ValueError(f"{path.name}: ilegível pelo scipy ({exc})") from exc
    keys = [k for k in m if not k.startswith("__")]
    if keys != [path.stem]:
        raise ValueError(f"{path.name}: variáveis inesperadas {keys}")
    s = m[keys[0]][0, 0]
    sig = {}
    for i in range(s.Y.shape[1]):
        e = s.Y[0, i]
        sig[str(_flat(e.Name))] = (str(_flat(e.Raster)), np.asarray(e.Data, dtype=float).ravel())
    for name, raster in (("vibration_1", "HostService"), ("speed", "Mech_4kHz")):
        if name not in sig or sig[name][0] != raster:
            raise ValueError(f"{path.name}: canal {name} ausente ou fora do raster {raster}")
    vib, spd = sig["vibration_1"][1], sig["speed"][1]
    dur_s = (spd.size - 1) / SPEED_FS_HZ
    if not 3.99 < dur_s < 4.01:
        raise ValueError(f"{path.name}: rotação com {dur_s:.4f} s, esperado 4 s")
    if vib.size < N_VIB - 2:
        raise ValueError(f"{path.name}: vibração com {vib.size} amostras, menos de 4 s a 64 kHz")
    return Measurement(code, setting, n, vib[:N_VIB], spd)
