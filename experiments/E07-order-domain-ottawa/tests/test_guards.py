"""Verificação executável dos três guardas do E07 e da extensão.

A extensão (`config/extended.yaml`) relaxa dois guardas para que os fatores 96, 128 e
160 passem a produzir medição. Quantas medições cada fator admite é a **premissa** da
extensão, e era uma conta feita fora do repositório. Aqui ela vira execução:

- `test_primeiro_guarda_por_fator` roda só aritmética, sem dado, e fixa a tabela do
  README (qual guarda dispara primeiro em cada fator, sob cada config);
- `test_extended_relaxa_exatamente_dois_guardas` impede que um terceiro confundidor
  entre no config estendido sem ser notado;
- `test_admissao_base_reproduz_o_csv_medido` compara o modelo dos guardas com a
  rodada que está em `results/metrics.csv` — é o que autoriza usar o modelo para
  falar de fatores ainda não rodados;
- `test_admissao_extended_e_n_cheio` executa a afirmação "72 de 72 nos doze fatores".

Os dois últimos leem os `.mat` de Ottawa, que não são versionados; sem eles o teste é
pulado, não falha.

    uv run pytest experiments/E07-order-domain-ottawa/tests -q

O caminho precisa ser explícito porque `testpaths` cobre `packages/`, e isso é
deliberado: nada em `packages/` importa `experiments/`. Este teste vive dentro do
próprio experimento e só importa o `run.py` dele.
"""

from __future__ import annotations

import csv
import importlib.util
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

EXP = Path(__file__).resolve().parents[1]
RAW = EXP / "datasets/raw/ottawa"
FS_HZ = 200_000.0

_spec = importlib.util.spec_from_file_location("e07_run", EXP / "run.py")
assert _spec is not None and _spec.loader is not None
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def load_cfg(name: str) -> dict:
    return yaml.safe_load((EXP / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


BASE = load_cfg("base")
EXTENDED = load_cfg("extended")

# Rotação de pior caso DENTRO de um segmento analisado: 1674,1 rpm = 27,9017 Hz, o
# máximo de fs_eff/max(v) sobre as 72 janelas. É o valor que o guarda enxerga —
# `run.py` aplica `np.max` ao perfil decimado do segmento, não ao registro inteiro.
WORST_SPEED_HZ = 1674.107142878 / 60.0


def geometry(factor: int, cfg: dict) -> tuple[int, float, float]:
    """`(n_amostras, fs_eff_hz, pontos_por_revolucao)` no pior segmento."""
    n = math.ceil(int(cfg["segment_samples"]) / factor)
    fs_eff = FS_HZ / factor
    return n, fs_eff, fs_eff / WORST_SPEED_HZ


@pytest.mark.parametrize(
    ("factor", "sob_base", "sob_extended"),
    [
        (48, None, None),
        # 64: `samples_per_rev` derruba parte das janelas no base (112,00 < 128) e
        # nenhuma no estendido (112,00 >= 32).
        (64, "samples_per_rev", None),
        (96, "samples_per_rev", None),
        # 128: os dois guardas relaxados o zeram no base; `min_samples` dispara antes.
        (128, "min_samples", None),
        (160, "min_samples", None),
        # 192 passa no estendido; a razão de a varredura parar em 160 é a largura da
        # janela de busca do kurtograma, não um guarda — ver o README.
        (192, "min_samples", None),
        # 256 é o primeiro fator que o estendido também recusa: ceil(500000/256)=1954.
        (256, "min_samples", "min_samples"),
    ],
)
def test_primeiro_guarda_por_fator(factor: int, sob_base: str | None, sob_extended: str | None):
    assert run.guard_verdict(*geometry(factor, BASE), BASE) == sob_base
    assert run.guard_verdict(*geometry(factor, EXTENDED), EXTENDED) == sob_extended


def test_f_min_hz_nao_e_o_guarda_que_zera_96_e_128():
    """O piso de 500 Hz só passa a vincular a partir do fator 200."""
    for factor in (96, 128, 160, 192):
        _, fs_eff, _ = geometry(factor, BASE)
        assert fs_eff / 2.0 > float(BASE["kurtogram"]["f_min_hz"])
    assert float(BASE["kurtogram"]["f_min_hz"]) == FS_HZ / 200 / 2.0


def test_extended_relaxa_exatamente_dois_guardas():
    """Terceiro confundidor entrando no config estendido quebra este teste."""
    esperado = {"min_samples", "samples_per_rev", "sweep", "output"}
    difere = {k for k in BASE.keys() | EXTENDED.keys() if BASE.get(k) != EXTENDED.get(k)}
    assert difere == esperado
    assert EXTENDED["min_samples"] == 2048 and BASE["min_samples"] == 4096
    assert EXTENDED["samples_per_rev"] == 16 and BASE["samples_per_rev"] == 64
    # A lista estendida contém a do base inteira: são os nove fatores de controle que
    # medem o degrau de guarda sob taxa idêntica.
    assert set(BASE["sweep"]["undersampling_factor"]) - {96, 128} <= set(
        EXTENDED["sweep"]["undersampling_factor"]
    )
    assert EXTENDED["output"]["results_dir"] != BASE["output"]["results_dir"]


# ----------------------------------------------------------------------------
# Verificação com os dados reais: pulada quando os .mat não estão presentes.
# ----------------------------------------------------------------------------
needs_data = pytest.mark.skipif(
    not (RAW / "I-A-1.mat").exists(), reason=f"dados de Ottawa ausentes em {RAW}"
)


@pytest.fixture(scope="module")
def segment_geometry() -> dict[int, list[tuple[int, float, float]]]:
    """Para cada fator, a geometria real das 72 janelas (24 registros × 3 segmentos)."""
    from snq_data import effective_rate_hz, undersample
    from snq_data.ottawa import load, record_names

    factors = sorted(
        set(BASE["sweep"]["undersampling_factor"]) | set(EXTENDED["sweep"]["undersampling_factor"])
    )
    seg_n = int(BASE["segment_samples"])
    out: dict[int, list[tuple[int, float, float]]] = {f: [] for f in factors}
    names = record_names(
        tuple(BASE["conditions"]), tuple(BASE["scenarios"]), tuple(int(t) for t in BASE["trials"])
    )
    for name in names:
        rec = load(name, RAW)
        speed = rec.speed_profile_hz(int(BASE["pulses_per_rev"]))
        for seg in range(int(BASE["segments_per_record"])):
            a, b = seg * seg_n, (seg + 1) * seg_n
            if b > rec.vibration.size:
                break
            spd = speed[a:b]
            for factor in factors:
                v = undersample(spd, factor)
                fs_eff = effective_rate_hz(FS_HZ, factor)
                n = undersample(rec.vibration[a:b], factor).size
                out[factor].append((n, fs_eff, fs_eff / float(np.max(v))))
    return out


def admitidas(geom: dict[int, list[tuple[int, float, float]]], cfg: dict) -> dict[int, int]:
    c = Counter()
    for factor in cfg["sweep"]["undersampling_factor"]:
        c[int(factor)] = sum(run.guard_verdict(*g, cfg) is None for g in geom[int(factor)])
    return dict(c)


@needs_data
def test_geometria_das_janelas(segment_geometry):
    assert all(len(v) == 72 for v in segment_geometry.values())
    # A rotação de pior caso usada na aritmética acima é a medida, não uma escolha.
    pior = min(pts for _, _, pts in segment_geometry[64])
    assert pior == pytest.approx(112.0, abs=0.01)


@needs_data
def test_admissao_base_reproduz_o_csv_medido(segment_geometry):
    """O modelo dos guardas tem de reproduzir a rodada que está no CSV."""
    csv_path = EXP / "results/metrics.csv"
    if not csv_path.exists():
        pytest.skip("results/metrics.csv ausente")
    medido = Counter()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            medido[int(row["factor"])] += 1
    modelo = admitidas(segment_geometry, BASE)
    # Três modos por medição admitida.
    assert {f: n * 3 for f, n in modelo.items() if n} == dict(medido)
    assert modelo == {
        1: 72,
        2: 72,
        4: 72,
        8: 72,
        16: 72,
        24: 72,
        32: 72,
        48: 72,
        64: 45,
        96: 0,
        128: 0,
    }


@needs_data
def test_admissao_extended_e_n_cheio(segment_geometry):
    """ "72 de 72 nos doze fatores" — executado, e só sobre a admissão de segmento.

    Este teste não diz nada sobre acerto: ele diz que nenhum ponto da curva estendida
    entra com `n` reduzido, e portanto que uma diferença entre fatores não pode ser
    troca de denominador.
    """
    modelo = admitidas(segment_geometry, EXTENDED)
    assert modelo == dict.fromkeys(EXTENDED["sweep"]["undersampling_factor"], 72)
    assert len(modelo) == 12
