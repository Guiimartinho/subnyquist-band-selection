"""Figura E07 — a linha de falha não se move quando a taxa desmorona.

Pilha de espectros de ordem do mesmo registro, à taxa cheia e subamostrado. É a
evidência visual do argumento central: a banda de ressonância dobra para posições
diferentes a cada taxa, mas a **ordem** da falha fica onde está, porque ela vive na
modulação do envelope e não na portadora.

Painel da esquerda: onde a energia está em frequência, e onde o kurtograma pousa.
Painel da direita: o espectro de ordem correspondente, com a ordem de falha marcada.

    uv run python papers/shared/figures/fig_e07_spectra.py --record I-B-1.mat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Type 3 e o default do matplotlib e costuma reprovar no PDF eXpress do IEEE (T0.8).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from snq_data import effective_rate_hz, undersample  # noqa: E402
from snq_data.ottawa import FS_HZ, load  # noqa: E402
from snq_recon import envelope_order_spectrum, fast_kurtogram  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/E07-order-domain-ottawa"
OUT = EXP / "results/figures"
ACCENT = "#0F6E78"
WARM = "#B4502B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="I-B-1.mat")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 8, 32, 64])
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP / "config/base.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((EXP / "results/manifest.json").read_text(encoding="utf-8"))
    truth = float(manifest["calibrated_per_record"][args.record.replace(".mat", "")])

    rec = load(args.record, Path(cfg["data_dir"]))
    n = int(cfg["segment_samples"])
    vib, spd = rec.vibration[:n], rec.speed_profile_hz(int(cfg["pulses_per_rev"]))[:n]
    kg, spr = cfg["kurtogram"], int(cfg["samples_per_rev"])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = len(args.factors)
    fig, axes = plt.subplots(rows, 2, figsize=(7.16, 1.35 * rows + 0.5), constrained_layout=True)

    for i, factor in enumerate(args.factors):
        x, v = undersample(vib, factor), undersample(spd, factor)
        fs = effective_rate_hz(FS_HZ, factor)
        k = fast_kurtogram(
            x,
            fs,
            max_level=int(kg["max_level"]),
            f_min_hz=float(kg["f_min_hz"]),
            f_max_hz=min(float(kg["f_max_hz"]), fs / 2.0),
        )
        spec = envelope_order_spectrum(x, fs, v, k.band_hz, samples_per_rev=spr)

        # --- esquerda: densidade espectral e a banda que o kurtograma escolheu ---
        axl = axes[i, 0]
        f, p = _welch(x, fs)
        axl.semilogy(f / 1000.0, p, color="0.35", lw=0.7)
        axl.axvspan(k.f_lo_hz / 1000, k.f_hi_hz / 1000, color=WARM, alpha=0.18, lw=0)
        axl.set_xlim(0, fs / 2000.0)
        axl.set_ylabel(f"{factor}×\n{fs / 1000:.1f} kSPS", fontsize=8)
        axl.tick_params(labelsize=7)
        axl.text(
            0.98,
            0.88,
            f"banda {k.f_lo_hz / 1000:.1f}–{k.f_hi_hz / 1000:.1f} kHz · K={k.kurtosis:.0f}",
            transform=axl.transAxes,
            ha="right",
            fontsize=6.6,
            color=WARM,
        )

        # --- direita: espectro de ordem, com a ordem de falha marcada ---
        axr = axes[i, 1]
        sel = (spec.freqs_hz >= 0.5) & (spec.freqs_hz <= 14.0)
        axr.plot(spec.freqs_hz[sel], spec.magnitude[sel], color=ACCENT, lw=0.8)
        for h in (1, 2):
            axr.axvline(truth * h, color=WARM, lw=0.9, ls="--", alpha=0.8, zorder=0)
        axr.set_xlim(0.5, 14.0)
        axr.tick_params(labelsize=7)
        axr.set_yticks([])
        if i == 0:
            axr.text(
                truth,
                axr.get_ylim()[1] * 0.98,
                f" ordem {truth:.2f}",
                fontsize=6.8,
                color=WARM,
                va="top",
            )

    axes[0, 0].set_title("(a) densidade espectral e banda escolhida", fontsize=9, loc="left")
    axes[0, 1].set_title("(b) espectro de ordem do envelope", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("frequência [kHz]", fontsize=8)
    axes[-1, 1].set_xlabel("ordem da rotação", fontsize=8)
    fig.suptitle(
        f"{rec.cite()} — a banda muda de lugar a cada taxa; a ordem, não",
        fontsize=8.5,  # sem y explicito: o constrained_layout reserva o espaco
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            OUT / f"e07_spectra.{ext}", dpi=200
        )  # sem bbox tight: a largura fica nos 7,16 in
    print(f"escrito: {OUT / 'e07_spectra.pdf'}")


def _welch(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy import signal

    nper = min(4096, x.size)
    return signal.welch(x, fs=fs, nperseg=nper)


if __name__ == "__main__":
    main()
