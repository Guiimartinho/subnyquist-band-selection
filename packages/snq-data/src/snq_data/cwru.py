"""CWRU ingestion, with the Smith & Randall screening applied.

Rule from E02, and the reason this module exists: **measuring undersampling degradation
on a record that is not diagnosable at the full rate measures noise, not degradation.**
Smith & Randall (MSSP 2015) classified every record; using CWRU without that screening
produces a flat curve that would be read as a failure of the method.

The authors also explicitly ask that the **record number and measurement point be
reported**. Almost no learning paper on CWRU does this. Here it is mandatory
by construction: `CwruRecord` carries both.
"""

from __future__ import annotations

import http.client
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

BASE_URL = "https://engineering.case.edu/sites/default/files"
USER_AGENT = "subnyquist-thesis/0.1 (academic use)"
MEASUREMENT_POINTS = ("DE", "FE", "BA")
Category = Literal["classical", "undiagnosable", "other"]

_ID_RE = re.compile(r"^(\d+)(DE|FE|BA)?$")


def parse_record_id(rid: str | int) -> tuple[int, str | None]:
    """``"120DE"`` becomes ``(120, "DE")``; ``"130"`` becomes ``(130, None)``.

    Without a suffix, the classification applies to all measurement points of the record.
    """
    m = _ID_RE.match(str(rid).strip())
    if not m:
        raise ValueError(f"invalid record identifier: {rid!r}")
    return int(m.group(1)), m.group(2)


@dataclass(frozen=True)
class CwruCatalog:
    """Screening loaded from `datasets/catalogs/cwru-smith2015-categories.yaml`."""

    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> CwruCatalog:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @property
    def human_checked(self) -> bool:
        """Has the table already been checked by eye against the PDF?

        Extracting a multi-column table from a PDF is fallible. While this is ``False``,
        `screen` uses only the subsets listed in `reliable_subsets`.
        """
        return bool(self.raw.get("verification", {}).get("human_checked", False))

    @property
    def reliable_subsets(self) -> set[str]:
        return set(self.raw.get("verification", {}).get("reliable_subsets", []))

    def _members(self, table: str) -> set[tuple[int, str | None, str]]:
        out: set[tuple[int, str | None, str]] = set()
        for subset, faults in self.raw.get(table, {}).items():
            for ids in faults.values():
                for rid in ids or []:
                    n, point = parse_record_id(rid)
                    out.add((n, point, subset))
        return out

    def screen(
        self, record: int, point: str | None = None, *, require_reliable: bool = True
    ) -> Category:
        """Category of the record: ``classical``, ``undiagnosable`` or ``other``.

        With ``require_reliable``, subsets not yet checked visually are
        treated as ``other`` instead of being silently classified — it is better
        to leave a record out than to classify it wrongly.
        """
        for table, label in (("undiagnosable", "undiagnosable"), ("classical", "classical")):
            for n, p, subset in self._members(table):
                if n != record or (p is not None and point is not None and p != point):
                    continue
                if (
                    require_reliable
                    and not self.human_checked
                    and subset not in self.reliable_subsets
                ):
                    continue
                return label  # type: ignore[return-value]
        return "other"

    def records(self, table: str, subset: str | None = None) -> list[tuple[int, str | None]]:
        """Records of a table, optionally restricted to one subset."""
        return sorted({(n, p) for n, p, s in self._members(table) if subset is None or s == subset})

    # The fault-type key in the catalog maps to the characteristic frequency that the
    # geometry predicts. It is the link between the dataset label and the physical prior.
    FAULT_TO_FREQUENCY = {
        "inner_race": "BPFI",
        "ball": "BSF",
        "outer_race_centred": "BPFO",
        "outer_race_orthogonal": "BPFO",
        "outer_race_opposite": "BPFO",
    }

    def fault_of(self, record: int, point: str | None = None) -> tuple[str, str] | None:
        """Fault type of the record and the corresponding characteristic frequency.

        Returns ``("outer_race_centred", "BPFO")``, or ``None`` if not cataloged.
        """
        for table in ("classical", "undiagnosable"):
            for subset, faults in self.raw.get(table, {}).items():
                if not self.human_checked and subset not in self.reliable_subsets:
                    continue
                for fault, ids in faults.items():
                    for rid in ids or []:
                        n, p = parse_record_id(rid)
                        if n == record and (p is None or point is None or p == point):
                            return fault, self.FAULT_TO_FREQUENCY[fault]
        return None

    def selection(self, table: str, subset: str, point: str = "DE") -> list[tuple[int, str, str]]:
        """Records of a subset, as ``(number, point, expected frequency)``.

        Includes the identifiers without a suffix (which apply to all points) and those
        marked with the requested point. Records with clipping are excluded.
        """
        out: list[tuple[int, str, str]] = []
        for fault, ids in self.raw.get(table, {}).get(subset, {}).items():
            freq = self.FAULT_TO_FREQUENCY[fault]
            for rid in ids or []:
                n, p = parse_record_id(rid)
                if p in (None, point) and not self.is_clipped(n, point):
                    out.append((n, point, freq))
        return sorted(set(out))

    def is_clipped(self, record: int, point: str | None = None) -> bool:
        """Record with clipping in sections — a nonlinearity that contaminates the spectrum."""
        for rid in self.raw.get("clipped_in_sections", []):
            n, p = parse_record_id(rid)
            if n == record and (p is None or point is None or p == point):
                return True
        return False


@dataclass(frozen=True)
class CwruRecord:
    """A loaded record, always carrying its own provenance."""

    number: int
    point: str
    signal: NDArray[np.float64]
    fs_hz: float
    rpm: float | None
    category: Category

    @property
    def shaft_hz(self) -> float | None:
        return self.rpm / 60.0 if self.rpm else None

    def cite(self) -> str:
        """How to report it in the paper, as Smith & Randall ask."""
        rpm = f", {self.rpm:.0f} rpm" if self.rpm else ""
        return f"CWRU {self.number}{self.point} ({self.fs_hz / 1e3:.0f} kHz{rpm}) [{self.category}]"


def download_record(
    number: int, dest_dir: str | Path, *, overwrite: bool = False, tries: int = 4
) -> Path:
    """Downloads ``<number>.mat`` from the CWRU Bearing Data Center, with retries.

    The server fairly often cuts the connection in the middle of large files
    (``IncompleteRead``). Without retries, a single problematic record aborts the
    whole experiment; without checking ``Content-Length``, a truncated file would be
    written and would only show up as a strange error much later.
    """
    dest = Path(dest_dir) / f"{number}.mat"
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/{number}.mat", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310
                expected = r.headers.get("Content-Length")
                data = r.read()
            if expected and len(data) != int(expected):
                raise OSError(f"truncated: {len(data)} of {expected} bytes")
            if not data.startswith(b"MATLAB"):
                raise OSError("response is not a MAT file")
            dest.write_bytes(data)
            return dest
        # IncompleteRead inherits from HTTPException, not from OSError: without it in the
        # tuple, a cut in the middle of the file escaped the retry (remote server, 2026-11-11).
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise OSError(f"record {number}: failed after {tries} attempts ({last})")


def load_record(
    number: int,
    point: str,
    data_dir: str | Path,
    catalog: CwruCatalog | None = None,
    fs_hz: float = 12_000.0,
) -> CwruRecord:
    """Loads a record from the ``.mat``, choosing the channel of the measurement point.

    The CWRU keys follow the pattern ``X<num>_<POINT>_time``, with ``RPM`` separate.
    """
    from scipy.io import loadmat

    mat = loadmat(str(Path(data_dir) / f"{number}.mat"))
    suffix = {"DE": "DE_time", "FE": "FE_time", "BA": "BA_time"}[point]
    keys = [k for k in mat if k.endswith(suffix)]
    if not keys:
        raise KeyError(f"record {number} has no {point} channel; available: {sorted(mat)}")

    rpm_keys = [k for k in mat if k.endswith("RPM")]
    rpm = float(np.ravel(mat[rpm_keys[0]])[0]) if rpm_keys else None

    return CwruRecord(
        number=number,
        point=point,
        signal=np.ravel(mat[keys[0]]).astype(float),
        fs_hz=fs_hz,
        rpm=rpm,
        category=catalog.screen(number, point) if catalog else "other",
    )
