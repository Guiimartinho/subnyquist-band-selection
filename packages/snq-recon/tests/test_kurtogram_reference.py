"""Kurtograma contra a fonte: Antoni, MSSP 21 (2007) 108-124.

Conferido no PDF (`literature/pdf/antoni2007kurtogram.pdf`):

- Eq. (5)-(6): no nível ``k`` a banda ``i`` tem centro ``(i + 1/2)·2^(-k-1)`` e largura
  ``2^(-k-1)``, em frequência normalizada por ``fs``. A Fig. 5 acrescenta os níveis
  1/3-binários com larguras 1/6, 1/12, 1/24.
- Eq. (7): ``K = <|c|^4> / <|c|^2>^2 - 2`` sobre o envelope complexo.
- Seção 1.1, propriedade (ii): a SK de um processo gaussiano estacionário é nula.
- Nota 3: os primeiros coeficientes, afetados pelo transitório dos filtros, não entram
  na curtose.

O que este módulo trava como **desvio documentado** (não é Antoni): com ``f_min_hz`` ou
``f_max_hz`` a partição é a da faixa de busca, e não a de ``[0, fs/2]``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from snq_core import alias_frequency_hz, fold_band
from snq_recon import fast_kurtogram, spectral_kurtosis
from snq_sim import impulse_train


def _structural_background(n: int, fs_hz: float, seed: int) -> np.ndarray:
    """Ruído gaussiano estacionário com quase toda a energia abaixo de ~200 Hz.

    É recortado do meio de uma realização três vezes maior, para que o registro **não**
    seja periódico: a extensão periódica tem um salto na emenda, como num registro real.
    Sintetizar por ``irfft`` daria um sinal exatamente periódico e esconderia o defeito.
    """
    rng = np.random.default_rng(seed)
    b, a = signal.butter(4, 200.0, fs=fs_hz)
    y = signal.lfilter(b, a, rng.normal(size=3 * n))[n : 2 * n]
    return y / np.std(y)


# ----------------------------------------------------------------------
# geometria: Eq. (5)-(6) e Fig. 5
# ----------------------------------------------------------------------
def test_pavimentacao_sem_faixa_de_busca_e_a_de_antoni():
    """Sem ``f_min_hz``/``f_max_hz``, nível ``L`` tem ``2^L`` bandas de largura ``2^(-L-1)``.

    O nível 0 (a banda inteira) não é avaliado: desvio documentado, não erro.
    """
    rng = np.random.default_rng(0)
    k = fast_kurtogram(rng.normal(size=4096), 1.0, max_level=4)
    levels = sorted({cell[0] for cell in k.grid})
    assert levels == pytest.approx(
        [1.0, np.log2(3.0), 2.0, 1.0 + np.log2(3.0), 3.0, 2.0 + np.log2(3.0), 4.0]
    )
    for level in levels:
        cells = sorted(cell for cell in k.grid if cell[0] == level)
        n_bands = int(round(2.0**level))
        assert len(cells) == n_bands
        width = 2.0 ** (-level - 1.0)
        for i, (_, lo, hi, _) in enumerate(cells):
            assert lo == pytest.approx(i * width)
            assert hi == pytest.approx((i + 1) * width)
            assert 0.5 * (lo + hi) == pytest.approx((i + 0.5) * width)  # Eq. (5)


def test_faixa_de_busca_reancora_a_particao_e_a_torna_independente_de_fs():
    """Desvio de Antoni, travado aqui porque o E20 depende dele.

    A partição é a de ``[f_min, min(f_max, fs/2)]``. Com o teto de 25 kHz do E07, os fatores
    1, 2 e 4 sobre 200 kSPS veem **as mesmas** bandas candidatas; só a 8x (Nyquist de
    12,5 kHz) a partição muda. Em Antoni, ancorada em ``[0, fs/2]``, ela mudaria a cada fator.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=8192)

    def cells(fs_hz: float) -> list[tuple[float, float, float]]:
        k = fast_kurtogram(x, fs_hz, max_level=3, f_min_hz=500.0, f_max_hz=min(25e3, fs_hz / 2))
        return sorted((c[0], c[1], c[2]) for c in k.grid)

    full, half, quarter, eighth = (cells(fs) for fs in (200e3, 100e3, 50e3, 25e3))
    assert full == half == quarter
    assert eighth != quarter
    level1 = [(lo, hi) for lev, lo, hi in full if lev == 1.0]
    assert level1 == [(500.0, 12750.0), (12750.0, 25000.0)]


# ----------------------------------------------------------------------
# estimador: Eq. (7) e propriedade (ii)
# ----------------------------------------------------------------------
def test_curtose_espectral_de_tom_puro_e_menos_um():
    """Envelope constante: ``<|c|^4>/<|c|^2>^2 = 1``, logo ``K = -1`` pela Eq. (7)."""
    z = np.exp(2j * np.pi * 0.1 * np.arange(4096))
    assert spectral_kurtosis(z) == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sk_de_gaussiano_estacionario_colorido_e_nula_em_toda_celula(seed):
    """Propriedade (ii) de Antoni (2007): processo gaussiano estacionário dá SK nula.

    O fundo tem o espectro de uma máquina real, quase toda a energia em baixa frequência.
    Regressão: com filtragem circular sem tratamento de borda, o salto na emenda da
    extensão periódica virava um "transiente" único, e a SK passava de 100 em bandas
    sem transiente algum. Medido antes da correção: até ~4000 sem a guarda de energia.
    """
    fs = 48_000.0
    x = _structural_background(int(fs), fs, seed) + 0.03 * np.random.default_rng(seed + 10).normal(
        size=int(fs)
    )
    k = fast_kurtogram(x, fs, max_level=4, min_energy_frac=0.0)
    worst = max(abs(cell[3]) for cell in k.grid)
    assert worst < 0.5, f"SK de ruido gaussiano chegou a {worst:.2f}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kurtograma_acha_a_ressonancia_sob_fundo_estrutural(seed):
    """Rajada impulsiva numa ressonância conhecida (6 kHz), sob fundo realista.

    Antes da correção, nestas sementes a banda vencedora era [500, 4417] ou [500, 3438] Hz,
    com K entre 130 e 430: era a emenda do registro, não a falha.
    """
    fs = 48_000.0
    n = int(fs)
    rng = np.random.default_rng(seed)
    fault = impulse_train(1.0, fs, rate_hz=104.0, resonance_hz=6000.0, damping=0.02, seed=seed)
    x = _structural_background(n, fs, seed) + 0.02 * rng.normal(size=n)
    x = x + 0.01 * fault / np.std(fault)
    k = fast_kurtogram(x, fs, max_level=4, f_min_hz=500.0)
    assert k.f_lo_hz <= 6000.0 <= k.f_hi_hz, k.summary()
    assert 0.5 < k.kurtosis < 10.0, k.summary()


def test_transiente_colado_na_borda_nao_entra_na_curtose():
    """Custo da correção, documentado como teste negativo.

    Como na nota 3 de Antoni, as amostras a menos de ``4·fs/Δf`` das bordas ficam fora da
    curtose. Um transiente curto inteiro ali é invisível; o mesmo transiente no meio, não.
    A menor guarda desta busca é ``4·48000/2000 = 96`` amostras (banda de 2 kHz); o
    transiente decai em 5 amostras. Um transiente com cauda mais longa que a guarda
    continuaria parcialmente visível.
    """
    fs = 48_000.0
    n = 48_000
    rng = np.random.default_rng(5)
    ring = np.exp(-np.arange(60) / 5.0) * np.sin(2 * np.pi * 6000.0 * np.arange(60) / fs)

    def one_burst(at: int) -> float:
        x = 0.01 * rng.normal(size=n)
        x[at : at + ring.size] += ring
        return fast_kurtogram(x, fs, max_level=3, f_min_hz=4000.0, f_max_hz=8000.0).kurtosis

    assert one_burst(n // 2) > 20.0
    assert one_burst(0) < 1.0


# ----------------------------------------------------------------------
# o mecanismo do ADR 0006, em sinal sintético controlado
# ----------------------------------------------------------------------
FS_ORIG = 200_000.0
SEARCH = {"max_level": 4, "f_min_hz": 500.0, "f_max_hz": 25e3}


def _fault_and_interferer(kind: str, seed: int = 11) -> np.ndarray:
    """Falha em 5 kHz (dentro da busca) e, opcionalmente, interferente em 76-84 kHz.

    O interferente é limitado em banda por um passa-faixa de fase zero, para que em taxa
    cheia **nada** dele caia na faixa de busca: só a dobra pode trazê-lo para dentro.
    ``kind``: ``"none"``, ``"impulsive"`` (rajadas a 9 Hz) ou ``"gaussian"`` (mesma potência,
    estacionário).
    """
    duration = 0.5
    n = int(FS_ORIG * duration)
    rng = np.random.default_rng(seed)
    fault = impulse_train(
        duration, FS_ORIG, rate_hz=104.0, resonance_hz=5000.0, damping=0.02, seed=1
    )
    x = fault / np.std(fault) + 0.3 * rng.normal(size=n)
    if kind == "none":
        return x
    sos = signal.butter(8, [76e3, 84e3], btype="bandpass", fs=FS_ORIG, output="sos")
    raw = (
        impulse_train(duration, FS_ORIG, rate_hz=9.0, resonance_hz=80e3, damping=0.01, seed=2)
        if kind == "impulsive"
        else rng.normal(size=n)
    )
    interferer = signal.sosfiltfilt(sos, raw)
    return x + 2.0 * interferer / np.std(interferer)


def _band(x: np.ndarray, factor: int) -> tuple[float, float]:
    return fast_kurtogram(x[::factor], FS_ORIG / factor, **SEARCH).band_hz  # decimacao pura


def test_rajada_fora_da_busca_captura_o_kurtograma_onde_o_mapa_de_alias_preve():
    """O mecanismo do ADR 0006, isolado: dobra de conteúdo **impulsivo** fora da busca.

    A 1x o interferente não muda a escolha. A 4x (``x[::4]``, 50 kSPS) ele cai em
    ``|80 - 2·50| = 20`` kHz, na zona 3, invertida, e a banda escolhida passa a ser a que
    contém a dobra, [12,75; 25] kHz, a mesma do vale do E07.
    Isso prova que o código reproduz o mecanismo. Não prova que é ele que atua no Ottawa:
    isso é o E20.
    """
    clean, hit = _fault_and_interferer("none"), _fault_and_interferer("impulsive")
    assert _band(hit, 1) == _band(clean, 1)

    fs_eff = FS_ORIG / 4
    alias = float(alias_frequency_hz(80e3, fs_eff))
    assert alias == pytest.approx(20e3)
    assert fold_band(76e3, 84e3, fs_eff).inverted  # zona impar: o espectro chega invertido

    lo, hi = _band(hit, 4)
    assert lo <= alias <= hi
    assert (lo, hi) != _band(clean, 4)
    assert (lo, hi) == (12750.0, 25000.0)


def test_interferente_potente_mas_gaussiano_dobra_sem_capturar():
    """Contraprova: a mesma dobra, com a mesma potência, de ruído estacionário, não captura.

    É a distinção que o ADR 0006 reivindica: o que derruba o seletor é conteúdo dobrado
    **impulsivo**, não potente.
    """
    gauss, clean = _fault_and_interferer("gaussian"), _fault_and_interferer("none")
    assert _band(gauss, 4) == _band(clean, 4)
