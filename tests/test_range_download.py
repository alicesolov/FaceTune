from __future__ import annotations

import fcntl
import hashlib
import io
from pathlib import Path
from typing import Self
from urllib.request import Request

import pytest

from ai_image_detector import range_download


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, *, start: int, end: int, total: int, status: int = 206):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _opener(data: bytes, starts: list[int]):
    def open_url(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout > 0
        range_header = request.headers["Range"]
        start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
        start, end = int(start_text), int(end_text)
        starts.append(start)
        return FakeResponse(data[start : end + 1], start=start, end=end, total=len(data))

    return open_url


def test_parallel_range_download_assembles_and_hashes_exact_bytes(tmp_path: Path) -> None:
    data = bytes(range(251)) * 17
    starts: list[int] = []
    target = tmp_path / "data.bin"

    result = range_download.download_range_file(
        "https://example.test/data.bin",
        target,
        expected_size=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        workers=3,
        part_size=257,
        chunk_size=31,
        opener=_opener(data, starts),
    )

    assert result == target
    assert target.read_bytes() == data
    assert len(starts) == len(range_download._ranges(len(data), 257))
    assert not target.with_name(target.name + ".range-parts").exists()


def test_range_download_resumes_partial_part(tmp_path: Path) -> None:
    data = bytes(range(199)) * 9
    target = tmp_path / "data.bin"
    parts = target.with_name(target.name + ".range-parts")
    parts.mkdir()
    part_size = 500
    start, end = range_download._ranges(len(data), part_size)[0]
    partial_size = 123
    part = parts / f"{start:020d}-{end:020d}.part"
    part.write_bytes(data[:partial_size])
    starts: list[int] = []

    range_download.download_range_file(
        "https://example.test/data.bin",
        target,
        expected_size=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        workers=2,
        part_size=part_size,
        opener=_opener(data, starts),
    )

    assert partial_size in starts
    assert target.read_bytes() == data


def test_range_download_rejects_server_that_ignores_range(tmp_path: Path) -> None:
    data = b"invalid-range-response"

    def ignores_range(request: Request, *, timeout: float) -> FakeResponse:
        return FakeResponse(data, start=0, end=len(data) - 1, total=len(data), status=200)

    with pytest.raises(ValueError, match="did not honor range"):
        range_download.download_range_file(
            "https://example.test/data.bin",
            tmp_path / "data.bin",
            expected_size=len(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            workers=1,
            part_size=len(data),
            opener=ignores_range,
        )


def test_range_download_rejects_concurrent_writer(tmp_path: Path) -> None:
    data = b"locked-content"
    target = tmp_path / "data.bin"
    lock_path = target.with_name(target.name + ".download.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            range_download.download_range_file(
                "https://example.test/data.bin",
                target,
                expected_size=len(data),
                expected_sha256=hashlib.sha256(data).hexdigest(),
                workers=1,
                part_size=len(data),
                opener=_opener(data, []),
            )
