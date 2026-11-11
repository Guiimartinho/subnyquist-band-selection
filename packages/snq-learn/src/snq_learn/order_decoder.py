"""Decodificador desdobrado do E15: o pipeline clássico de ordem, com peças aprendíveis.

**Não é uma CNN genérica** (ADR 0006). Cada etapa é uma etapa do pipeline do E07/E22,
reescrita de forma diferenciável (algorithm unrolling, `monga2021unrolling`):

1. **sinal mascarado e preenchido com zeros**, ``z = x·m`` (a máscara vem de
   `snq_learn.masks`);
2. **banco de filtros complexos aprendíveis** sobre ``Z = FFT(z)``:
   ``Y_c(f) = Σ_r G_{c,r}(f)·Z(f - r·fs/L)``. Só com ``r = 0`` o banco é LTI e, na máscara
   uniforme, iniciado nas bandas do kurtograma, reproduz o pipeline clássico (decimar, filtrar a
   banda dobrada, demodular). Com vários deslocamentos ele é periódico no tempo com período
   ``L`` (LPTV), a forma da reconstrução linear multi-coset de uma banda: combina réplicas que
   o padrão não uniforme separou, e com isso pode anular a réplica de um interferente;
3. **envelope ao quadrado** ``|y_c|²`` (ou ``|y_c|``, para comparar com o `snq_recon`);
4. **redução da taxa do envelope** por passa-baixas ideal (truncamento da FFT) e
   **reamostragem angular** por interpolação linear na fase do tacômetro ou do encoder. A
   interpolação é linear nos valores do envelope, então o gradiente passa; a fase é dado, não
   parâmetro;
5. **espectro de ordem**: janela de Hann e FFT com zeros até ``n_fft``, numa grade fixa de
   ordens, e **logits por bin**: ``β·log(Σ_c a_c·S_c/mediana(S_c))``.

A seleção de canal ``a_c = softmax(τ·SK_c + b_c)`` é o kurtograma desdobrado: o clássico toma o
argmax da curtose espectral, e aqui o argmax vira uma softmax com temperatura aprendida.

**O que o modelo não pode fazer, por construção:** não há termo aprendido por bin de ordem. O
logit de uma ordem é função do espectro medido naquela ordem, então o modelo não pinta um pico
onde o dado não tem; ele só escolhe onde olhar (bandas, réplicas) e como combinar canais. Não há
reconstrução de forma de onda, então a decomposição espaço-imagem/núcleo não se define; o controle
de alucinação do E15 é o conjunto saudável (ver o runner).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn

__all__ = [
    "AngularPlan",
    "ComplexFilterBank",
    "OrderAxis",
    "OrderDecoder",
    "SampledOrderModel",
    "angular_plan",
    "envelope_sample_positions",
    "kurtogram_bands_hz",
    "order_cross_entropy",
    "soft_order_target",
    "stack_plans",
]


# ----------------------------------------------------------------------
# eixo de ordens e plano angular
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OrderAxis:
    """Grade fixa de ordens: bins ``k·spr/n_fft`` dentro de ``[order_min, order_max]``."""

    samples_per_rev: int
    n_fft: int
    order_min: float
    order_max: float

    def __post_init__(self) -> None:
        if self.samples_per_rev < 2 or self.n_fft < 2:
            raise ValueError("samples_per_rev e n_fft precisam ser >= 2")
        if not 0.0 <= self.order_min < self.order_max:
            raise ValueError("precisa 0 <= order_min < order_max")
        if self.order_max > self.samples_per_rev / 2.0:
            raise ValueError(
                f"order_max={self.order_max} acima do Nyquist angular {self.samples_per_rev / 2}"
            )

    @property
    def step(self) -> float:
        """Espaçamento entre bins, em ordens."""
        return self.samples_per_rev / self.n_fft

    @property
    def bin_range(self) -> tuple[int, int]:
        """``[lo, hi)`` dos bins do ``rfft`` de ``n_fft`` pontos que caem no intervalo de ordens."""
        lo = math.ceil(self.order_min / self.step - 1e-9)
        hi = math.floor(self.order_max / self.step + 1e-9) + 1
        return lo, min(hi, self.n_fft // 2 + 1)

    @property
    def orders(self) -> NDArray[np.float64]:
        lo, hi = self.bin_range
        return np.arange(lo, hi) * self.step


def envelope_sample_positions(n_samples: int, n_env: int) -> NDArray[np.float64]:
    """Posições, em amostras da grade cheia, das ``n_env`` amostras do envelope de taxa reduzida.

    O truncamento da FFT de ``N`` pontos para ``n_env`` devolve a interpolação de banda limitada
    nos instantes ``k·N/n_env``; é aí que a fase precisa ser avaliada.
    """
    return np.arange(int(n_env)) * (float(n_samples) / float(n_env))


def angular_plan(phase_rev_env: ArrayLike, samples_per_rev: int, n_fft: int) -> dict[str, object]:
    """Pesos da interpolação linear do envelope para a grade angular uniforme.

    ``phase_rev_env`` é a fase, em revoluções, nas amostras do envelope reduzido. A grade
    angular é ``θ_0 + j/spr`` cobrindo ``[θ_0, θ_fim)``, como em `snq_recon.angular_resample`.
    A janela de Hann tem o comprimento válido; o resto, até ``n_fft``, é preenchimento com zeros.
    """
    th = np.asarray(phase_rev_env, dtype=np.float64).ravel()
    if th.size < 2 or np.any(np.diff(th) <= 0.0):
        raise ValueError("fase precisa ser estritamente crescente")
    total = float(th[-1] - th[0])
    if total < 1.0:
        raise ValueError(f"janela cobre so {total:.2f} revolucao")
    grid = th[0] + np.arange(0.0, total, 1.0 / float(samples_per_rev))
    if grid.size > n_fft:
        raise ValueError(f"grade angular de {grid.size} pontos excede n_fft={n_fft}")
    pos = np.interp(grid, th, np.arange(th.size, dtype=np.float64))
    idx = np.minimum(np.floor(pos).astype(np.int64), th.size - 2)
    return {
        "index": idx,
        "frac": (pos - idx).astype(np.float32),
        "window": np.hanning(grid.size).astype(np.float32),
        "n_valid": int(grid.size),
        "n_env": int(th.size),
    }


@dataclass
class AngularPlan:
    """Planos angulares empilhados, ``(B, A)``, com preenchimento de peso zero."""

    index: torch.Tensor
    frac: torch.Tensor
    window: torch.Tensor
    n_valid: torch.Tensor
    n_env: int

    def to(self, device: torch.device | str) -> AngularPlan:
        return AngularPlan(
            self.index.to(device),
            self.frac.to(device),
            self.window.to(device),
            self.n_valid.to(device),
            self.n_env,
        )


def stack_plans(plans: Sequence[dict[str, object]]) -> AngularPlan:
    """Empilha planos de janelas com o mesmo ``n_env`` (comprimentos angulares podem variar)."""
    n_env = {int(p["n_env"]) for p in plans}  # type: ignore[call-overload]
    if len(n_env) != 1:
        raise ValueError(f"planos com n_env diferentes: {sorted(n_env)}")
    a = max(int(p["n_valid"]) for p in plans)  # type: ignore[call-overload]
    b = len(plans)
    index = np.zeros((b, a), dtype=np.int64)
    frac = np.zeros((b, a), dtype=np.float32)
    window = np.zeros((b, a), dtype=np.float32)
    for i, p in enumerate(plans):
        n = int(p["n_valid"])  # type: ignore[call-overload]
        index[i, :n] = p["index"]  # type: ignore[index]
        frac[i, :n] = p["frac"]  # type: ignore[index]
        window[i, :n] = p["window"]  # type: ignore[index]
    n_valid = torch.tensor([float(p["n_valid"]) for p in plans])  # type: ignore[arg-type]
    return AngularPlan(
        torch.from_numpy(index),
        torch.from_numpy(frac),
        torch.from_numpy(window),
        n_valid,
        n_env.pop(),
    )


def kurtogram_bands_hz(
    f_lo_hz: float, f_hi_hz: float, levels: Sequence[float]
) -> list[tuple[float, float]]:
    """Bandas dos níveis 1/3-binários de Antoni (2007) sobre ``[f_lo, f_hi]``.

    Nível ``λ`` divide a faixa em ``round(2^λ)`` bandas iguais (2, 3, 4, 6, 8, ... para
    ``λ = 1, log2 3, 2, ...``), a mesma partição de `snq_recon.fast_kurtogram` com
    ``f_min_hz``/``f_max_hz``. É a inicialização do banco: na máscara uniforme, o decodificador
    começa onde o kurtograma clássico procura.
    """
    if not f_hi_hz > f_lo_hz >= 0.0:
        raise ValueError("faixa de bandas vazia")
    out: list[tuple[float, float]] = []
    for lv in levels:
        edges = np.linspace(f_lo_hz, f_hi_hz, int(round(2.0 ** float(lv))) + 1)
        out += [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return out


# ----------------------------------------------------------------------
# banco de filtros
# ----------------------------------------------------------------------
def _band_response(f_frac: NDArray[np.float64], lo: float, hi: float, transition: float) -> NDArray:
    """Resposta analítica: 2 na banda positiva ``[lo, hi)`` (fração de fs), 0 fora.

    Bordas em cosseno levantado de largura ``transition·(hi - lo)``, centradas nas bordas
    nominais; ``transition = 0`` dá a máscara ideal do `snq_recon.bandpass_analytic`.
    """
    if transition <= 0.0:
        return 2.0 * ((f_frac >= lo) & (f_frac < hi)).astype(np.float64)
    t = transition * (hi - lo)

    def ramp(s: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.clip(s, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * s)

    resp = ramp((f_frac - lo) / t + 0.5) * ramp((hi - f_frac) / t + 0.5)
    resp[f_frac >= 0.5] = 0.0  # frequências negativas: nada, o sinal é analítico
    return 2.0 * resp


class ComplexFilterBank(nn.Module):
    """``C`` canais × ``R`` deslocamentos de resposta complexa livre.

    Cada resposta é dada em ``n_control`` pontos sobre ``[0, fs)`` (circular) e interpolada
    linearmente até os bins da FFT do sinal. Deslocamento ``r`` combina ``Z(f - r·fs/L)``.
    """

    def __init__(self, n_channels: int, shifts: Sequence[int], n_control: int) -> None:
        super().__init__()
        self.shifts = tuple(int(s) for s in shifts)
        if 0 not in self.shifts:
            raise ValueError("o deslocamento 0 precisa estar no banco")
        if len(set(self.shifts)) != len(self.shifts):
            raise ValueError("deslocamentos repetidos")
        self.n_control = int(n_control)
        self.control = nn.Parameter(
            torch.zeros(int(n_channels), len(self.shifts), self.n_control, 2)
        )

    @property
    def n_channels(self) -> int:
        return int(self.control.shape[0])

    def init_bands(self, bands_frac: Sequence[tuple[float, float]], transition_frac: float) -> None:
        """Canal ``c`` passa a banda ``bands_frac[c]`` (fração de fs) no deslocamento 0."""
        if len(bands_frac) != self.n_channels:
            raise ValueError(f"{len(bands_frac)} bandas para {self.n_channels} canais")
        f = np.arange(self.n_control) / self.n_control
        ctrl = torch.zeros_like(self.control)
        r0 = self.shifts.index(0)
        for c, (lo, hi) in enumerate(bands_frac):
            ctrl[c, r0, :, 0] = torch.from_numpy(_band_response(f, lo, hi, transition_frac))
        with torch.no_grad():
            self.control.copy_(ctrl)

    def _interp(self, n_bins: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        u = torch.arange(n_bins, device=device, dtype=torch.float64) * (self.n_control / n_bins)
        base = torch.floor(u)
        i0 = base.long() % self.n_control
        i1 = (i0 + 1) % self.n_control
        a = (u - base).to(torch.float32)
        return i0, i1, a

    def responses(self, n_bins: int) -> torch.Tensor:
        """``(C, R, n_bins)`` complexo: as respostas interpoladas nos bins de uma FFT."""
        i0, i1, a = self._interp(n_bins, self.control.device)
        ctrl = torch.view_as_complex(self.control)
        return ctrl[..., i0] * (1.0 - a) + ctrl[..., i1] * a

    def forward(self, spectrum: torch.Tensor, period_ticks: int) -> torch.Tensor:
        """``spectrum`` ``(B, N)`` complexo (FFT cheia de ``z``) → ``(B, C, N)`` complexo."""
        n = spectrum.shape[-1]
        if any(self.shifts) and n % int(period_ticks):
            raise ValueError(f"N={n} precisa ser multiplo de L={period_ticks} para deslocar")
        i0, i1, a = self._interp(n, spectrum.device)
        ctrl = torch.view_as_complex(self.control)
        out: torch.Tensor | None = None
        for r_i, r in enumerate(self.shifts):
            g = ctrl[:, r_i, i0] * (1.0 - a) + ctrl[:, r_i, i1] * a  # (C, N)
            zr = (
                torch.roll(spectrum, shifts=r * (n // int(period_ticks)), dims=-1)
                if r
                else spectrum
            )
            term = g.unsqueeze(0) * zr.unsqueeze(1)
            out = term if out is None else out + term
        assert out is not None
        return out


# ----------------------------------------------------------------------
# decodificador
# ----------------------------------------------------------------------
def _inv_softplus(y: float) -> float:
    return float(y + math.log(-math.expm1(-y)))


class OrderDecoder(nn.Module):
    """Banco de filtros → envelope → ordem → logits por bin de ordem (ver docstring do módulo)."""

    def __init__(
        self,
        *,
        fs_hz: float,
        fs_env_hz: float,
        axis: OrderAxis,
        bands_hz: Sequence[tuple[float, float]],
        shifts: Sequence[int] = (0,),
        period_ticks: int = 1,
        n_control: int = 1024,
        transition_frac: float = 0.1,
        envelope_power: int = 2,
        selection: str = "sk",
        sk_guard_frac: float = 0.05,
        beta_init: float = 2.0,
    ) -> None:
        super().__init__()
        if envelope_power not in (1, 2):
            raise ValueError("envelope_power precisa ser 1 (|y|) ou 2 (|y|^2)")
        if selection not in ("sk", "static"):
            raise ValueError(f"selecao desconhecida: {selection}")
        self.fs_hz = float(fs_hz)
        self.fs_env_hz = float(fs_env_hz)
        self.axis = axis
        self.period_ticks = int(period_ticks)
        self.envelope_power = int(envelope_power)
        self.selection = selection
        self.sk_guard_frac = float(sk_guard_frac)
        self.bank = ComplexFilterBank(len(bands_hz), shifts, n_control)
        self.bank.init_bands([(lo / fs_hz, hi / fs_hz) for lo, hi in bands_hz], transition_frac)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.sel_bias = nn.Parameter(torch.zeros(len(bands_hz)))
        self.raw_beta = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        self.register_buffer("orders", torch.from_numpy(axis.orders.astype(np.float32)))

    def n_env(self, n_samples: int) -> int:
        """Amostras do envelope de taxa reduzida para uma janela de ``n_samples``."""
        return max(2, int(round(n_samples * self.fs_env_hz / self.fs_hz)))

    def spectra(self, z: torch.Tensor, plan: AngularPlan) -> tuple[torch.Tensor, torch.Tensor]:
        """Espectros de ordem por canal ``(B, C, n_bins)`` e a curtose espectral ``(B, C)``."""
        b, n = z.shape
        n_env = self.n_env(n)
        if plan.n_env != n_env:
            raise ValueError(f"plano feito para n_env={plan.n_env}, decodificador usa {n_env}")
        y = torch.fft.ifft(self.bank(torch.fft.fft(z), self.period_ticks))
        p2 = y.real**2 + y.imag**2
        g0 = int(n * self.sk_guard_frac)
        core = p2[..., g0 : n - g0] if n - 2 * g0 > 1 else p2
        m2 = core.mean(-1)
        sk = (core**2).mean(-1) / m2.clamp_min(1e-20) ** 2 - 2.0
        env = p2 if self.envelope_power == 2 else torch.sqrt(p2 + 1e-20)
        env = env - env.mean(-1, keepdim=True)
        spec_t = torch.fft.rfft(env)
        env_d = torch.fft.irfft(spec_t[..., : n_env // 2 + 1], n=n_env) * (n_env / n)
        c = env_d.shape[1]
        idx = plan.index.unsqueeze(1).expand(-1, c, -1)
        e0 = torch.gather(env_d, 2, idx)
        e1 = torch.gather(env_d, 2, idx + 1)
        ang = (e0 + plan.frac.unsqueeze(1) * (e1 - e0)) * plan.window.unsqueeze(1)
        mag = torch.fft.rfft(ang, n=self.axis.n_fft).abs() * (2.0 / plan.n_valid.view(b, 1, 1))
        lo, hi = self.axis.bin_range
        return mag[..., lo:hi], sk

    def forward(
        self, z: torch.Tensor, plan: AngularPlan
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spec, sk = self.spectra(z, plan)
        norm = spec / spec.median(dim=-1, keepdim=True).values.clamp_min(1e-20)
        if self.selection == "sk":
            weights = torch.softmax(torch.exp(self.log_tau) * sk + self.sel_bias, dim=-1)
        else:
            weights = torch.softmax(self.sel_bias, dim=-1).expand(z.shape[0], -1)
        combined = (weights.unsqueeze(-1) * norm).sum(1)
        logits = torch.nn.functional.softplus(self.raw_beta) * torch.log(combined + 1e-6)
        return logits, {"weights": weights, "sk": sk}

    def estimate(self, logits: torch.Tensor) -> NDArray[np.float64]:
        """Ordem do maior logit em ``[order_min, order_max]``: o ``estimate_without_prior``."""
        return self.orders[logits.argmax(dim=-1)].detach().cpu().numpy().astype(np.float64)


class SampledOrderModel(nn.Module):
    """Máscara + decodificador.

    A escala vem do RMS das amostras **adquiridas**, não da taxa cheia: o nó não conhece o
    sinal que não converteu.
    """

    def __init__(self, sampler: nn.Module, decoder: OrderDecoder) -> None:
        super().__init__()
        self.sampler = sampler
        self.decoder = decoder

    def forward(
        self, x: torch.Tensor, plan: AngularPlan
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        m = self.sampler(x)
        z = x * m
        count = m.sum(-1, keepdim=True).clamp_min(1.0)
        rms = torch.sqrt((z * z).sum(-1, keepdim=True) / count).clamp_min(1e-12)
        logits, aux = self.decoder(z / rms, plan)
        aux["mask_rate"] = m.detach().mean(-1)
        return logits, aux

    def penalty(self) -> torch.Tensor:
        pen = getattr(self.sampler, "last_penalty", None)
        return pen if isinstance(pen, torch.Tensor) else torch.zeros(())


def soft_order_target(orders: torch.Tensor, truth: torch.Tensor, sigma: float) -> torch.Tensor:
    """Alvo suave: gaussiana de desvio ``sigma`` na ordem verdadeira, normalizada por linha."""
    d = orders.unsqueeze(0) - truth.unsqueeze(1)
    q = torch.exp(-0.5 * (d / float(sigma)) ** 2)
    return q / q.sum(-1, keepdim=True).clamp_min(1e-20)


def order_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Entropia cruzada com alvo suave, média no lote."""
    return -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
