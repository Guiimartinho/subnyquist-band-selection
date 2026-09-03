"""Cadeia de aquisição física — o simulador honesto.

Não modelar como ``sinal -> decima -> pronto``. O que decide a viabilidade do
projeto são justamente os blocos que um modelo ingênuo omite::

    transdutor (banda + ressonância)
        v
    condicionamento + AAF  <- PASSA-FAIXA, nao passa-baixas
        v
    track-and-hold (banda de entrada finita)   <- parede frequentemente esquecida
        v
    relogio de amostragem (jitter)             <- amplificado por f0/fs
        v
    quantizador (12 bits, saturacao)
        v
    amostras

As três paredes físicas do projeto aparecem aqui explicitamente. Um experimento
que assume amostrador ideal não diz nada sobre o hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import signal


@dataclass(frozen=True)
class Transducer:
    """Resposta do transdutor. O que ele não transduz, nenhum algoritmo recupera."""

    f_lo_hz: float
    f_hi_hz: float
    resonance_hz: float | None = None
    q_factor: float = 10.0
    name: str = ""


@dataclass(frozen=True)
class AntiAliasFilter:
    """Filtro anti-aliasing.

    ``kind='bandpass'`` é o modo que este projeto exige: um AAF passa-baixas
    comercial destrói exatamente a informação super-Nyquist que se quer medir.
    """

    kind: str = "bandpass"
    f_lo_hz: float = 140e3
    f_hi_hz: float = 160e3
    order: int = 4


@dataclass(frozen=True)
class Sampler:
    """Amostrador. ``th_bandwidth_hz`` é a parede a medir em bancada (Medida 0)."""

    fs_hz: float
    th_bandwidth_hz: float = 200e3
    jitter_rms_s: float = 0.0
    offsets_s: NDArray[np.float64] | None = None
    """Instantes não-uniformes (multi-coset). ``None`` = grade uniforme."""


@dataclass(frozen=True)
class Quantizer:
    n_bits: int = 12
    full_scale_v: float = 1.0
    saturate: bool = True


@dataclass
class AcquisitionChain:
    """Cadeia completa. Determinística dada uma seed."""

    transducer: Transducer
    aaf: AntiAliasFilter
    sampler: Sampler
    quantizer: Quantizer = field(default_factory=Quantizer)
    seed: int = 0

    def acquire(self, x_analog: NDArray[np.float64], fs_analog_hz: float) -> NDArray[np.float64]:
        """Passa ``x_analog`` (amostrado finamente) por toda a cadeia.

        ``fs_analog_hz`` precisa ser ao menos 10x a maior frequência de interesse
        para que a simulação da parte analógica seja fiel.
        """
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x_analog, dtype=float)

        x = self._apply_transducer(x, fs_analog_hz)
        x = self._apply_aaf(x, fs_analog_hz)
        x = self._apply_track_and_hold(x, fs_analog_hz)
        x = self._sample(x, fs_analog_hz, rng)
        return self._quantize(x)

    # ------------------------------------------------------------------
    def _apply_transducer(self, x: NDArray[np.float64], fs: float) -> NDArray[np.float64]:
        nyq = fs / 2.0
        lo = max(self.transducer.f_lo_hz / nyq, 1e-6)
        hi = min(self.transducer.f_hi_hz / nyq, 0.999)
        sos = signal.butter(4, [lo, hi], btype="bandpass", output="sos")
        y = signal.sosfiltfilt(sos, x)
        if self.transducer.resonance_hz is not None:
            w0 = self.transducer.resonance_hz / nyq
            if 0.0 < w0 < 1.0:
                b, a = signal.iirpeak(w0, self.transducer.q_factor)
                y = signal.filtfilt(b, a, y)
        return y

    def _apply_aaf(self, x: NDArray[np.float64], fs: float) -> NDArray[np.float64]:
        nyq = fs / 2.0
        if self.aaf.kind == "bandpass":
            wn = [max(self.aaf.f_lo_hz / nyq, 1e-6), min(self.aaf.f_hi_hz / nyq, 0.999)]
            sos = signal.butter(self.aaf.order, wn, btype="bandpass", output="sos")
        elif self.aaf.kind == "lowpass":
            sos = signal.butter(
                self.aaf.order, min(self.aaf.f_hi_hz / nyq, 0.999), btype="low", output="sos"
            )
        elif self.aaf.kind == "none":
            return x
        else:
            raise ValueError(f"AAF desconhecido: {self.aaf.kind!r}")
        return signal.sosfiltfilt(sos, x)

    def _apply_track_and_hold(self, x: NDArray[np.float64], fs: float) -> NDArray[np.float64]:
        """Polo único na banda do T/H. É a parede que mata subamostragem ingênua."""
        nyq = fs / 2.0
        wn = min(self.sampler.th_bandwidth_hz / nyq, 0.999)
        sos = signal.butter(1, wn, btype="low", output="sos")
        return signal.sosfilt(sos, x)

    def _sample(
        self, x: NDArray[np.float64], fs_analog: float, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        t_analog = np.arange(x.size) / fs_analog
        duration = t_analog[-1]

        if self.sampler.offsets_s is None:
            t = np.arange(0.0, duration, 1.0 / self.sampler.fs_hz)
        else:
            base = np.arange(0.0, duration, 1.0 / self.sampler.fs_hz)
            offs = np.asarray(self.sampler.offsets_s, dtype=float)
            t = np.sort((base[:, None] + offs[None, :]).ravel())
            t = t[t <= duration]

        if self.sampler.jitter_rms_s > 0.0:
            t = t + rng.normal(0.0, self.sampler.jitter_rms_s, size=t.size)
            t = np.clip(t, 0.0, duration)

        return np.interp(t, t_analog, x)

    def _quantize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        fs_v = self.quantizer.full_scale_v
        if self.quantizer.saturate:
            x = np.clip(x, -fs_v, fs_v)
        levels = 2 ** (self.quantizer.n_bits - 1)
        return np.round(x / fs_v * levels) / levels * fs_v


def impulse_train(
    duration_s: float,
    fs_hz: float,
    rate_hz: float,
    resonance_hz: float,
    damping: float = 0.05,
    jitter_frac: float = 0.01,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Trem de impactos amortecidos — o modelo canônico de falha de rolamento.

    Impulsos periódicos a ``rate_hz`` (BPFO/BPFI) excitando uma ressonância
    estrutural em ``resonance_hz``. O ``jitter_frac`` (escorregamento de ~1-2%)
    é o que torna o sinal **ciclo-estacionário e não periódico**, e é a razão de
    a análise de envelope funcionar melhor que a espectral direta.

    É também exatamente um sinal com taxa de inovação finita: dois parâmetros
    por impacto, e não ``2B`` amostras por segundo.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs_hz)
    t = np.arange(n) / fs_hz
    x = np.zeros(n)

    period = 1.0 / rate_hz
    n_impacts = int(duration_s / period)

    # O escorregamento perturba cada INTERVALO, nao a posicao absoluta. Somar os
    # intervalos da um passeio aleatorio (desvio ~ sqrt(i)); escalar a posicao por
    # (1 + eps) daria erro crescendo com i e destruiria a periodicidade em poucos
    # segundos. Bug encontrado em 2026-09-03 ao montar o baseline de envelope.
    intervals = period * (1.0 + rng.normal(0.0, jitter_frac, size=n_impacts))
    starts = np.cumsum(intervals) - intervals[0]

    for t0 in starts:
        if not 0.0 <= t0 < duration_s:
            continue
        idx = t >= t0
        dt = t[idx] - t0
        decay = np.exp(-2.0 * np.pi * damping * resonance_hz * dt)
        x[idx] += decay * np.sin(2.0 * np.pi * resonance_hz * dt)
    return x
