"""Paderborn: a raia de falha se separa da harmonica do eixo? (taxa cheia, so diagnostico do dado)

Para cada medicao: kurtograma rapido (nivel 4, piso 500 Hz, como o E15), espectro de envelope em
ordens (64 amostras por volta, anti-alias angular), e o pico na janela de +-7 % da ordem nominal.
Reporta a ordem do pico, a distancia a harmonica inteira mais proxima e as amplitudes na nominal e
na harmonica (maximo em +-0,015 ordem), relativas a mediana da janela.
"""

import glob
import sys

import numpy as np
from scipy.io import loadmat
from snq_recon.highres import envelope_order_spectrum
from snq_recon.kurtogram import fast_kurtogram

ROOT = sys.argv[1]
FS = 64000.0
RATIO = 6.75 / 28.55  # d/D, FAG 6203 (Lessmeier et al., PHME16, tabela 1)
NOMINAL = {"OR": 4 * (1 - RATIO), "IR": 4 * (1 + RATIO)}
CASES = [
    ("K001", "OR"),
    ("K001", "IR"),
    ("KA01", "OR"),
    ("KA05", "OR"),
    ("KA07", "OR"),
    ("KA04", "OR"),
    ("KA15", "OR"),
    ("KA22", "OR"),
    ("KI01", "IR"),
    ("KI05", "IR"),
    ("KI04", "IR"),
    ("KI14", "IR"),
    ("KI18", "IR"),
]


def flat(v):
    while isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        v = v.ravel()[0]
    return v


def signals(path):
    m = loadmat(path, squeeze_me=False, struct_as_record=False)
    s = m[[k for k in m if not k.startswith("__")][0]][0, 0]
    return {
        str(flat(s.Y[0, i].Name)): np.asarray(s.Y[0, i].Data, float).ravel()
        for i in range(s.Y.shape[1])
    }


def amp_near(orders, mag, o, tol=0.015):
    sel = np.abs(orders - o) <= tol
    return float(mag[sel].max()) if sel.any() else float("nan")


print(f"BPFO nominal {NOMINAL['OR']:.4f}, BPFI nominal {NOMINAL['IR']:.4f}")
print(
    "rolamento tipo medicao  banda_kHz      eixo_Hz  pico   dist_inteiro"
    "  A(nominal)/med  A(inteiro)/med"
)
for code, kind in CASES:
    nom = NOMINAL[kind]
    harm = round(nom)
    for f in sorted(glob.glob(f"{ROOT}/{code}/N15_M07_F10_{code}_*.mat"))[:3]:
        sig = signals(f)
        x, shaft = sig["vibration_1"], float(np.mean(sig["speed"])) / 60.0
        k = fast_kurtogram(x, FS, max_level=4, f_min_hz=500.0)
        spec = envelope_order_spectrum(x, FS, shaft, k.band_hz, samples_per_rev=64)
        o, a = np.asarray(spec.freqs_hz), np.asarray(spec.magnitude)
        win = (o >= nom * 0.93) & (o <= nom * 1.07)
        med = float(np.median(a[win]))
        peak = float(o[win][np.argmax(a[win])])
        tag = f.rsplit("_", 1)[1].split(".")[0]
        lo_k, hi_k = k.band_hz[0] / 1e3, k.band_hz[1] / 1e3
        print(
            f"{code:9s} {kind:4s} {tag:8s} {lo_k:5.1f}-{hi_k:5.1f}   {shaft:7.3f}"
            f"  {peak:6.3f}  {peak - round(peak):+6.3f}"
            f"        {amp_near(o, a, nom) / med:6.1f}          {amp_near(o, a, harm) / med:6.1f}"
        )
