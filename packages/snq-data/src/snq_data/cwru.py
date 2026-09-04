"""Ingestão do CWRU, com a triagem de Smith & Randall aplicada.

Regra do E02, e a razão de este módulo existir: **medir degradação por subamostragem
sobre um registro que não é diagnosticável na taxa cheia mede ruído, não degradação.**
Smith & Randall (MSSP 2015) classificaram cada registro; usar o CWRU sem essa triagem
produz curva plana que seria lida como falha do método.

Os autores também pedem, explicitamente, que se **reporte número de registro e ponto
de medição**. Quase nenhum paper de aprendizado sobre CWRU faz isso. Aqui é obrigatório
por construção: `CwruRecord` carrega os dois.
"""

from __future__ import annotations

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
    """``"120DE"`` vira ``(120, "DE")``; ``"130"`` vira ``(130, None)``.

    Sem sufixo, a classificação vale para todos os pontos de medição do registro.
    """
    m = _ID_RE.match(str(rid).strip())
    if not m:
        raise ValueError(f"identificador de registro inválido: {rid!r}")
    return int(m.group(1)), m.group(2)


@dataclass(frozen=True)
class CwruCatalog:
    """Triagem carregada de `datasets/catalogs/cwru-smith2015-categories.yaml`."""

    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> CwruCatalog:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @property
    def human_checked(self) -> bool:
        """A tabela já foi conferida a olho contra o PDF?

        Extração de tabela multi-coluna de PDF é falível. Enquanto isto for ``False``,
        `screen` recusa subconjuntos marcados como não confiáveis.
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
        """Categoria do registro: ``classical``, ``undiagnosable`` ou ``other``.

        Com ``require_reliable``, subconjuntos ainda não conferidos visualmente são
        tratados como ``other`` em vez de silenciosamente classificados — é melhor
        deixar um registro de fora do que classificá-lo errado.
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
        """Registros de uma tabela, opcionalmente restritos a um subconjunto."""
        return sorted({(n, p) for n, p, s in self._members(table) if subset is None or s == subset})

    # A chave de tipo de falha no catalogo mapeia para a frequencia caracteristica
    # que a geometria preve. E o elo entre rotulo do dataset e prior fisico.
    FAULT_TO_FREQUENCY = {
        "inner_race": "BPFI",
        "ball": "BSF",
        "outer_race_centred": "BPFO",
        "outer_race_orthogonal": "BPFO",
        "outer_race_opposite": "BPFO",
    }

    def fault_of(self, record: int, point: str | None = None) -> tuple[str, str] | None:
        """Tipo de falha do registro e a frequência característica correspondente.

        Devolve ``("outer_race_centred", "BPFO")`` ou ``None`` se não catalogado.
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
        """Registros de um subconjunto, como ``(número, ponto, frequência esperada)``.

        Inclui os identificadores sem sufixo (que valem para todos os pontos) e os
        marcados com o ponto pedido. Registros com ceifamento são excluídos.
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
        """Registro com ceifamento em trechos — não-linearidade que contamina o espectro."""
        for rid in self.raw.get("clipped_in_sections", []):
            n, p = parse_record_id(rid)
            if n == record and (p is None or point is None or p == point):
                return True
        return False


@dataclass(frozen=True)
class CwruRecord:
    """Um registro carregado, sempre carregando a própria procedência."""

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
        """Como reportar no paper, conforme Smith & Randall pedem."""
        rpm = f", {self.rpm:.0f} rpm" if self.rpm else ""
        return f"CWRU {self.number}{self.point} ({self.fs_hz / 1e3:.0f} kHz{rpm}) [{self.category}]"


def download_record(
    number: int, dest_dir: str | Path, *, overwrite: bool = False, tries: int = 4
) -> Path:
    """Baixa ``<number>.mat`` do CWRU Bearing Data Center, com retentativa.

    O servidor corta a conexão no meio de arquivos grandes com alguma frequência
    (``IncompleteRead``). Sem retentativa, um único registro problemático aborta o
    experimento inteiro; sem conferir ``Content-Length``, um arquivo truncado seria
    gravado e só apareceria como erro estranho muito depois.
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
                raise OSError(f"truncado: {len(data)} de {expected} bytes")
            if not data.startswith(b"MATLAB"):
                raise OSError("resposta não é um arquivo MAT")
            dest.write_bytes(data)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise OSError(f"registro {number}: falhou em {tries} tentativas ({last})")


def load_record(
    number: int,
    point: str,
    data_dir: str | Path,
    catalog: CwruCatalog | None = None,
    fs_hz: float = 12_000.0,
) -> CwruRecord:
    """Carrega um registro do ``.mat``, escolhendo o canal do ponto de medição.

    As chaves do CWRU seguem o padrão ``X<num>_<PONTO>_time``, com ``RPM`` à parte.
    """
    from scipy.io import loadmat

    mat = loadmat(str(Path(data_dir) / f"{number}.mat"))
    suffix = {"DE": "DE_time", "FE": "FE_time", "BA": "BA_time"}[point]
    keys = [k for k in mat if k.endswith(suffix)]
    if not keys:
        raise KeyError(f"registro {number} não tem canal {point}; disponíveis: {sorted(mat)}")

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
