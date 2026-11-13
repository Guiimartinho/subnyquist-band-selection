"""Exploratory (adversarial review, 2026-11-12, check C4): PWM switching in the vibration?

In E24, the full-rate kurtogram picks 11.0–21.5 kHz in 909 of 1305 segments, with no hit at all
there. The hypothesis is that the drive's 16 kHz PWM switching is the captor. This check looks
at measurement 2 of each of the 69 records (vibration and phase current at 64 kS/s) for:

- **Lines.** In the Welch PSD (Hann of 8192, 50 %), it looks for lines at 16 kHz ± 2k·f_e and at
  32 kHz − (2k+1)·f_e, with k = 0..5. f_e is the fundamental of the phase current, searched in
  50–150 Hz. A line counts if it is at least 10 dB above the PSD median within ±500 Hz.
- **Coherence** between the vibration and the phase current at those lines.
- **Impulsiveness.** It is the spectral kurtosis of the envelope (E|z|⁴/E[|z|²]² − 2, as in the
  kurtogram) in 15–17 kHz and in 11–15 kHz, by ideal FFT mask.

This alone does not decide the cause of the capture. The check that tests the cause is the
notch (C3).

    python experiments/E24-paderborn-spike-capture/exploratory/inverter_lines.py [PADERBORN_ROOT]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import coherence, welch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
FS = 64000.0
N = 255_936
SETTINGS = ["N15_M07_F10", "N15_M01_F10", "N15_M07_F04"]
BEARINGS = [
    "KA01",
    "KA03",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KA04",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KI01",
    "KI03",
    "KI05",
    "KI07",
    "KI08",
    "KI04",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
]


def channels(path: Path) -> dict[str, np.ndarray]:
    m = loadmat(path, squeeze_me=False, struct_as_record=False)
    s = m[path.stem][0, 0]
    out = {}
    for i in range(s.Y.shape[1]):
        e = s.Y[0, i]
        out[str(np.ravel(e.Name)[0])] = np.asarray(e.Data, dtype=float).ravel()
    return out


def band_sk(x: np.ndarray, lo: float, hi: float) -> float:
    xf = np.fft.fft(x)
    f = np.fft.fftfreq(x.size, 1.0 / FS)
    z = np.fft.ifft(np.where((f >= lo) & (f <= hi), 2.0 * xf, 0.0))
    e = np.abs(z) ** 2
    return float(np.mean(e**2) / np.mean(e) ** 2 - 2.0)


def analyse(path: Path) -> dict:
    ch = channels(path)
    vib = ch["vibration_1"][:N]
    cur = ch["phase_current_1"][:N]
    fv, pv = welch(vib, FS, window="hann", nperseg=8192, noverlap=4096)
    fc, pc = welch(cur, FS, window="hann", nperseg=8192, noverlap=4096)
    sel = (fc >= 50) & (fc <= 150)
    fe = float(fc[sel][np.argmax(pc[sel])])
    _, coh = coherence(vib, cur, FS, window="hann", nperseg=8192, noverlap=4096)
    lines = [16000.0 + s * 2 * k * fe for k in range(6) for s in ((1,) if k == 0 else (1, -1))]
    lines += [32000.0 - (2 * k + 1) * fe for k in range(6)]
    above, cohs = [], []
    for f0 in lines:
        i = int(np.argmin(np.abs(fv - f0)))
        loc = (fv >= f0 - 500) & (fv <= f0 + 500)
        db = 10 * np.log10(pv[max(0, i - 1) : i + 2].max() / np.median(pv[loc]))
        above.append(db)
        cohs.append(float(coh[max(0, i - 1) : i + 2].max()))
    above, cohs = np.array(above), np.array(cohs)
    return {
        "file": path.name,
        "fe_hz": round(fe, 2),
        "lines_10db": int(np.sum(above >= 10)),
        "max_line_db": round(float(above.max()), 1),
        "coh_at_10db_lines": round(float(np.max(cohs[above >= 10])), 3)
        if np.any(above >= 10)
        else "",
        "sk_15_17khz": round(band_sk(vib, 15000, 17000), 3),
        "sk_11_15khz": round(band_sk(vib, 11000, 15000), 3),
    }


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets/raw/paderborn"
    rows = []
    for b in BEARINGS:
        for s in SETTINGS:
            p = root / b / f"{s}_{b}_2.mat"
            try:
                rows.append({"bearing": b, "setting": s, **analyse(p)})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                rows.append({"bearing": b, "setting": s, "file": p.name, "error": str(exc)[:80]})
    ok = [r for r in rows if "error" not in r]
    out = EXP / "results" / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "inverter_lines.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "records": len(rows),
        "readable": len(ok),
        "fe_hz_median": float(np.median([r["fe_hz"] for r in ok])),
        "records_with_any_10db_line": sum(1 for r in ok if r["lines_10db"] > 0),
        "lines_10db_median": float(np.median([r["lines_10db"] for r in ok])),
        "coh_median": float(
            np.median([r["coh_at_10db_lines"] for r in ok if r["coh_at_10db_lines"] != ""])
        )
        if any(r["coh_at_10db_lines"] != "" for r in ok)
        else None,
        "sk_15_17_median": float(np.median([r["sk_15_17khz"] for r in ok])),
        "sk_11_15_median": float(np.median([r["sk_11_15khz"] for r in ok])),
        "sk_15_17_gt_11_15": sum(1 for r in ok if r["sk_15_17khz"] > r["sk_11_15khz"]),
    }
    (out / "inverter_lines.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
