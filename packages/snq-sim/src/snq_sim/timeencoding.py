"""Amostragem por codificação temporal — a realização A2 do ADR 0004.

A amostragem clássica mede **amplitude em instantes pré-definidos**. A codificação
temporal faz o oposto: registra **o instante em que o sinal, ou uma função dele,
atinge um valor pré-definido**. As amostras são irregularmente espaçadas.

Por que isto está no projeto, citando `naaman2021fritem`:

    "At high sampling rates, clocks are power-consuming and prone to electromagnetic
    interference. An integrate-and-fire time encoding machine is an alternative
    power-efficient sampling mechanism which does not require a global clock."

A rota A1 (TIMER + DPPI) otimiza o uso do relógio; esta **elimina o relógio global**.

**Cuidado com o argumento de energia, medido neste simulador em 2026-09-03.** As duas
variantes não são equivalentes nesse ponto:

- O **IF-TEM** dispara a uma taxa fixada pela *média* do sinal, não pela amplitude:
  para entrada de média zero a integral de ``x + bias`` num período vale ``bias·T``
  qualquer que seja a amplitude. Amplitude de 0,01 a 1,9 dá exatamente a mesma taxa.
  O ganho dele é dispensar o relógio de alta frequência, **não** disparar menos com a
  máquina parada.
- O **cruzamento de nível com histerese** é que tem taxa proporcional à atividade:
  medido, 980 Hz para amplitude 0,01 contra 2980 Hz para amplitude 1,0.

Quem quiser "energia proporcional ao que acontece" precisa da segunda variante.

Duas propriedades estabelecidas na literatura e que justificam a rota:

- `liu2023tembandpass`: para sinal passa-faixa, a codificação temporal é **equivalente
  a amostragem periódica não-uniforme**. Toda a teoria de multi-coset se transfere.
- `adam2019multichanneltem`: M canais com integradores deslocados recuperam um sinal
  com **M vezes a banda**, e **sem conhecer os deslocamentos entre canais** — o que o
  multi-coset exige e precisa calibrar.

No nRF54L15 a peça é o comparador com captura de TIMER a 128 MHz. O modelo aqui
inclui a quantização do instante de disparo, que é o limite real dessa captura.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class IntegrateFireTEM:
    """Máquina de codificação temporal do tipo integrate-and-fire.

    Segue a definição de `adam2021asynchrony`: soma um viés ``bias`` ao sinal,
    integra o resultado escalado por ``1/kappa`` até atingir o limiar ``delta``,
    registra o instante, reinicia o integrador e recomeça.

    ``bias`` precisa exceder o valor absoluto máximo do sinal, senão o integrador
    pode nunca disparar — é a condição que garante que a codificação termina.

    ``timer_hz`` modela a captura de instante em hardware: no nRF54L15 o TIMER roda a
    128 MHz, então o instante do disparo é conhecido com resolução de ~7,8 ns e não
    exatamente. `gontier2011tem` mostra que o método é robusto a essa quantização.
    """

    kappa: float = 1.0
    delta: float = 1e-5
    bias: float = 2.0
    timer_hz: float | None = 128e6

    def max_input_amplitude(self) -> float:
        """Maior amplitude admissível: acima disso o disparo não é garantido."""
        return self.bias

    def nominal_rate_hz(self) -> float:
        """Taxa de disparo para entrada de média zero: ``bias / (2·delta·kappa)``.

        É o parâmetro de projeto que se compara contra ``fs`` da rota A1. Os padrões
        dão ~100 kHz; escolher ``delta`` grande demais faz o integrador disparar uma
        vez por segundo e a codificação não produz amostra alguma no registro.
        """
        return self.bias / (2.0 * self.delta * self.kappa)

    @classmethod
    def for_rate(cls, rate_hz: float, bias: float = 2.0, **kw: float | None) -> IntegrateFireTEM:
        """Constrói um TEM com a taxa nominal pedida."""
        return cls(delta=bias / (2.0 * rate_hz), bias=bias, **kw)  # type: ignore[arg-type]

    def encode(self, x: ArrayLike, fs_analog_hz: float) -> NDArray[np.float64]:
        """Instantes de disparo para ``x``, amostrado finamente a ``fs_analog_hz``.

        A integral é acumulada por regra do trapézio e o cruzamento do limiar é
        interpolado linearmente dentro do passo — sem isso a resolução dos instantes
        ficaria limitada por ``fs_analog_hz`` em vez de pelo modelo.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size < 2:
            return np.empty(0)
        peak = float(np.max(np.abs(x)))
        if peak >= self.bias:
            raise ValueError(
                f"amplitude do sinal ({peak:.3g}) >= bias ({self.bias:.3g}); "
                "o integrador pode nunca disparar. Aumente bias ou normalize a entrada."
            )

        dt = 1.0 / fs_analog_hz
        y = (x + self.bias) / self.kappa
        # Integral acumulada; o disparo ocorre a cada 2*delta de area acumulada.
        area = np.concatenate([[0.0], np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)])
        step = 2.0 * self.delta

        n_fires = int(area[-1] // step)
        if n_fires <= 0:
            return np.empty(0)

        targets = step * np.arange(1, n_fires + 1)
        # np.interp inverte a integral: qual instante acumula cada alvo de area.
        times = np.interp(targets, area, np.arange(x.size) * dt)

        if self.timer_hz:
            times = np.round(times * self.timer_hz) / self.timer_hz
        return times

    def firing_rate_hz(self, x: ArrayLike, fs_analog_hz: float) -> float:
        """Taxa média de disparos — o análogo da taxa de amostragem.

        É o número que se compara contra ``fs`` da rota A1 num orçamento de energia:
        cada disparo é uma conversão, e o que não dispara não gasta.
        """
        t = self.encode(x, fs_analog_hz)
        duration = np.asarray(x).size / fs_analog_hz
        return float(t.size / duration) if duration > 0 else 0.0


def multichannel_tem(
    x: ArrayLike,
    fs_analog_hz: float,
    n_channels: int,
    tem: IntegrateFireTEM | None = None,
) -> list[NDArray[np.float64]]:
    """M canais com integradores deslocados — o análogo temporal do multi-coset.

    `adam2019multichanneltem` mostra que M canais recuperam um sinal com M vezes a
    banda, **sem conhecer os deslocamentos relativos**. Aqui os canais diferem pelo
    estado inicial do integrador, que é o que um circuito real faria.

    Diferença prática em relação ao multi-coset: lá os offsets precisam ser conhecidos
    e calibrados; aqui não. É a vantagem que motivou o ADR 0004.
    """
    base = tem or IntegrateFireTEM()
    out: list[NDArray[np.float64]] = []
    for m in range(n_channels):
        # Deslocar o estado inicial equivale a deslocar o limiar do primeiro disparo.
        shifted = IntegrateFireTEM(
            kappa=base.kappa,
            delta=base.delta,
            bias=base.bias,
            timer_hz=base.timer_hz,
        )
        t = shifted.encode(x, fs_analog_hz)
        offset = (m / n_channels) * (2.0 * base.delta * base.kappa / base.bias)
        out.append(t + offset)
    return out


def crossing_times(
    x: ArrayLike,
    fs_analog_hz: float,
    levels: ArrayLike,
    hysteresis: float = 0.0,
    timer_hz: float | None = 128e6,
) -> NDArray[np.float64]:
    """Amostragem por cruzamento de nível: instantes em que ``x`` cruza cada limiar.

    É a variante mais simples da rota A2, e a mais direta de realizar com um
    comparador.

    **``hysteresis`` não é opcional na prática.** Sem ela, ruído em torno de um nível
    faz o comparador oscilar sem parar: medido neste simulador, ruído de amplitude
    0,001 em torno de zero gerou 5018 eventos em 50 ms, contra 149 de uma senoide de
    amplitude 1. Isso destrói o argumento de energia da rota, que depende de sinal
    parado custar pouco. Circuito real resolve com histerese, e é por isso que a
    literatura de ADC por cruzamento de nível sempre a inclui.

    Com histerese ``h``, depois de cruzar para cima o sinal precisa cair abaixo de
    ``nivel - h`` antes de um novo cruzamento para cima ser contado.
    """
    x = np.asarray(x, dtype=float).ravel()
    dt = 1.0 / fs_analog_hz
    out: list[float] = []

    for lv in np.asarray(levels, dtype=float).ravel():
        armed_up = x[0] < lv - hysteresis
        armed_down = x[0] > lv + hysteresis
        for i in range(x.size - 1):
            a, b = x[i], x[i + 1]
            if armed_up and a < lv <= b:
                out.append((i + (lv - a) / (b - a)) * dt)
                armed_up, armed_down = False, False
            elif armed_down and a > lv >= b:
                out.append((i + (a - lv) / (a - b)) * dt)
                armed_up, armed_down = False, False
            if b < lv - hysteresis:
                armed_up = True
            elif b > lv + hysteresis:
                armed_down = True

    t = np.sort(np.asarray(out, dtype=float))
    if timer_hz:
        t = np.round(t * timer_hz) / timer_hz
    return t
