"""Decodificador desdobrado do E15: ordem sintética, gradiente e consistência interna.

A concordância com `snq_recon.envelope_order_spectrum` fica nos testes do E15, porque este
pacote não importa snq-recon (regra de dependência do CLAUDE.md).
"""

import numpy as np
import pytest
import torch
from snq_learn import (
    BlockAdaptiveMask,
    ComplexFilterBank,
    FeasibleSet,
    FixedMask,
    GumbelTopKMask,
    LoupeMask,
    OrderAxis,
    OrderDecoder,
    SampledOrderModel,
    angular_plan,
    envelope_sample_positions,
    kurtogram_bands_hz,
    order_cross_entropy,
    soft_order_target,
    stack_plans,
)

FS = 20_000.0
ORDER = 5.43  # ordem de falha não inteira, como BPFI do Ottawa


def synthetic_bearing(n: int, seed: int = 0, order: float = ORDER):
    """Trem de impactos numa ordem fixa, cada um excitando uma ressonância de 4 kHz.

    Rotação em rampa de 15 a 20 Hz (o espectro em Hz borra; o de ordem não), mais ruído
    branco e uma componente forte na ordem 1. Devolve sinal, fase em revoluções e rotação.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    v = 15.0 + 5.0 * t / t[-1]
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) / FS)])
    x = np.zeros(n)
    hit = np.interp(np.arange(1, int(theta[-1] * order)) / order, theta, t)
    ht = np.arange(int(0.004 * FS)) / FS
    h = np.exp(-ht / 0.0008) * np.sin(2 * np.pi * 4000.0 * ht)
    for ti in hit:
        i0 = int(round(ti * FS))
        seg = h[: max(0, min(h.size, n - i0))]
        x[i0 : i0 + seg.size] += seg
    x += 0.3 * rng.normal(size=n) + 2.0 * np.sin(2 * np.pi * theta)
    return x, theta, v


def plan_for(theta: np.ndarray, n: int, dec: OrderDecoder, batch: int = 1):
    th = np.interp(envelope_sample_positions(n, dec.n_env(n)), np.arange(n), theta)
    p = angular_plan(th, dec.axis.samples_per_rev, dec.axis.n_fft)
    return stack_plans([p] * batch)


def make_decoder(period: int = 1, shifts=(0,), f_hi: float = 10_000.0) -> OrderDecoder:
    return OrderDecoder(
        fs_hz=FS,
        fs_env_hz=2048.0,
        axis=OrderAxis(256, 32768, 2.0, 32.0),
        bands_hz=kurtogram_bands_hz(500.0, f_hi, [1, 1.585, 2]),
        shifts=shifts,
        period_ticks=period,
    )


def test_eixo_de_ordens_cobre_a_faixa_pedida():
    axis = OrderAxis(256, 32768, 2.0, 32.0)
    o = axis.orders
    assert o[0] == pytest.approx(2.0) and o[-1] == pytest.approx(32.0)
    assert np.allclose(np.diff(o), 256 / 32768)
    with pytest.raises(ValueError, match="Nyquist angular"):
        OrderAxis(32, 1024, 2.0, 32.0)


def test_plano_angular_de_rotacao_constante_e_regular():
    n_env, spr = 2000, 64
    th = np.arange(n_env) * 0.01  # 100 amostras por revolução
    p = angular_plan(th, spr, 4096)
    assert p["n_valid"] == int(np.ceil((th[-1] - th[0]) * spr))
    pos = p["index"] + p["frac"]
    assert np.allclose(np.diff(pos), 100 / spr, atol=1e-4)
    with pytest.raises(ValueError, match="excede"):
        angular_plan(th, spr, 128)


def test_decodificador_com_mascara_cheia_acha_a_ordem_sintetica():
    """Iniciado nas bandas do kurtograma, o decodificador é o pipeline clássico desdobrado:
    sem treino nenhum ele já tem de achar a ordem de um trem de impactos sintético."""
    n = 40960
    x, theta, _ = synthetic_bearing(n)
    dec = make_decoder()
    model = SampledOrderModel(FixedMask([0], 1), dec).eval()
    with torch.no_grad():
        logits, aux = model(torch.tensor(x[None, :], dtype=torch.float32), plan_for(theta, n, dec))
    assert abs(dec.estimate(logits)[0] - ORDER) < 0.06
    # a selecao por curtose espectral prefere uma banda que contem a ressonancia de 4 kHz
    bands = kurtogram_bands_hz(500.0, 10_000.0, [1, 1.585, 2])
    best = bands[int(aux["weights"][0].argmax())]
    assert best[0] <= 4000.0 <= best[1]


def test_gradiente_chega_a_mascara_e_aos_filtros():
    torch.manual_seed(0)
    n = 16 * 1280
    x, theta, _ = synthetic_bearing(n, seed=1)
    xb = torch.tensor(np.stack([x, x[::-1].copy()]), dtype=torch.float32)
    fs = FeasibleSet(16, 4, 2)
    for sampler in (
        GumbelTopKMask(fs, seed=1),
        LoupeMask(16, 0.25, seed=1),
        BlockAdaptiveMask(fs, n_blocks=2, n_features=4, hidden=4, seed=1),
    ):
        dec = make_decoder(period=16, shifts=(-1, 0, 1), f_hi=FS / 8)
        model = SampledOrderModel(sampler, dec).train()
        logits, _ = model(xb, plan_for(theta, n, dec, batch=2))
        target = soft_order_target(dec.orders, torch.tensor([ORDER, ORDER]), 0.03)
        loss = order_cross_entropy(logits, target) + model.penalty()
        loss.backward()
        mask_param = sampler.logits0 if isinstance(sampler, BlockAdaptiveMask) else sampler.logits
        assert mask_param.grad is not None and float(mask_param.grad.abs().sum()) > 0.0
        ctrl = dec.bank.control.grad
        assert ctrl is not None and float(ctrl.abs().sum()) > 0.0
        # o gradiente chega tambem aos deslocamentos r != 0, iniciados em zero
        assert float(ctrl[:, 0].abs().sum()) > 0.0 and float(ctrl[:, 2].abs().sum()) > 0.0


def test_mascara_uniforme_com_zeros_equivale_a_decimar():
    """Na máscara uniforme, filtrar o sinal com zeros numa banda da 1a zona de Nyquist da taxa
    reduzida é o mesmo que decimar e filtrar a banda dobrada: o espectro de ordem coincide."""
    n, factor = 40960, 4
    x, theta, _ = synthetic_bearing(n, seed=2)
    xt = torch.tensor(x[None, :], dtype=torch.float32)
    dec_full = make_decoder(period=factor, f_hi=FS / (2 * factor))
    masked = SampledOrderModel(FixedMask([0], factor), dec_full).eval()
    with torch.no_grad():
        z = xt * masked.sampler(xt)
        spec_zero_filled, _ = dec_full.spectra(z, plan_for(theta, n, dec_full))
    # decimado: mesma banda em Hz, grade de amostragem fs/4, fase nas mesmas posições de tempo
    dec_dec = OrderDecoder(
        fs_hz=FS / factor,
        fs_env_hz=2048.0,
        axis=OrderAxis(256, 32768, 2.0, 32.0),
        bands_hz=kurtogram_bands_hz(500.0, FS / (2 * factor), [1, 1.585, 2]),
    )
    nd = n // factor
    th_dec = theta[::factor]
    with torch.no_grad():
        spec_dec, _ = dec_dec.spectra(xt[:, ::factor].contiguous(), plan_for(th_dec, nd, dec_dec))
    a = spec_zero_filled[0].numpy()
    b = spec_dec[0].numpy()
    for c in range(a.shape[0]):
        corr = np.corrcoef(a[c], b[c])[0, 1]
        assert corr > 0.99, f"canal {c}: correlacao {corr:.4f}"
        assert abs(int(np.argmax(a[c])) - int(np.argmax(b[c]))) <= 1


def test_deslocamentos_iniciados_em_zero_nao_mudam_o_banco_lti():
    torch.manual_seed(0)
    z = torch.fft.fft(torch.randn(2, 64))
    lti = ComplexFilterBank(3, (0,), 32)
    lptv = ComplexFilterBank(3, (-1, 0, 1), 32)
    bands = [(0.05, 0.15), (0.15, 0.3), (0.3, 0.45)]
    lti.init_bands(bands, 0.1)
    lptv.init_bands(bands, 0.1)
    assert torch.allclose(lti(z, 8), lptv(z, 8), atol=1e-5)


def test_alvo_suave_e_entropia_cruzada():
    orders = torch.linspace(2.0, 32.0, 3841)
    q = soft_order_target(orders, torch.tensor([5.43, 3.39]), 0.03)
    assert torch.allclose(q.sum(-1), torch.ones(2))
    assert float(orders[q[0].argmax()]) == pytest.approx(5.43, abs=0.01)
    good = 50.0 * q.log().clamp_min(-50.0)
    bad = torch.roll(good, 300, dims=-1)
    assert order_cross_entropy(good, q) < order_cross_entropy(bad, q)


def test_plano_de_n_env_diferente_e_recusado():
    n = 40960
    _, theta, _ = synthetic_bearing(n)
    dec = make_decoder()
    plan = plan_for(theta, n, dec)
    with pytest.raises(ValueError, match="n_env"):
        dec.spectra(torch.zeros(1, n // 2), plan)
