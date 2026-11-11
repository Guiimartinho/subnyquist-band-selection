"""Um corte no meio do arquivo (IncompleteRead) precisa cair na retentativa."""

from __future__ import annotations

import http.client

from snq_data import cwru

PAYLOAD = b"MATLAB 5.0 MAT-file" + b"\x00" * 64


class _Resp:
    headers = {"Content-Length": str(len(PAYLOAD))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return PAYLOAD


def test_cwru_retries_after_incomplete_read(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http.client.IncompleteRead(b"parcial", 1000)
        return _Resp()

    monkeypatch.setattr(cwru.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cwru.time, "sleep", lambda s: None)
    path = cwru.download_record(97, tmp_path, tries=3)
    assert calls["n"] == 2
    assert path.read_bytes() == PAYLOAD
