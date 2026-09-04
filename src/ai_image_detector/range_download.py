"""Resumable parallel HTTP range downloader for large revision-pinned research files."""

from __future__ import annotations

import fcntl
import hashlib
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Final
from urllib.request import Request, urlopen

DEFAULT_PART_SIZE: Final = 256 * 1024 * 1024
DEFAULT_CHUNK_SIZE: Final = 1024 * 1024

OpenUrl = Callable[..., BinaryIO]


@contextmanager
def _exclusive_download_lock(destination: Path) -> Iterator[None]:
    """Prevent concurrent processes from appending to the same resumable part files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".download.lock")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another process is already downloading the same target: {destination}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ranges(total_size: int, part_size: int) -> list[tuple[int, int]]:
    if total_size <= 0 or part_size <= 0:
        raise ValueError("total_size and part_size must be positive")
    return [
        (start, min(total_size - 1, start + part_size - 1))
        for start in range(0, total_size, part_size)
    ]


def _download_part(
    *,
    url: str,
    path: Path,
    start: int,
    end: int,
    total_size: int,
    timeout: float,
    max_attempts: int,
    chunk_size: int,
    opener: OpenUrl,
) -> None:
    expected_size = end - start + 1
    for attempt in range(max_attempts):
        existing = path.stat().st_size if path.exists() else 0
        if existing == expected_size:
            return
        if existing > expected_size:
            raise ValueError(f"Range part exceeds its expected size: {path}")
        request_start = start + existing
        request = Request(
            url,
            headers={
                "Range": f"bytes={request_start}-{end}",
                "User-Agent": "ai-image-detector-research/0.1",
            },
        )
        try:
            with opener(request, timeout=timeout) as response:
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range")
                expected_content_range = f"bytes {request_start}-{end}/{total_size}"
                if status != 206 or content_range != expected_content_range:
                    raise ValueError(
                        f"Server did not honor range {request_start}-{end}: "
                        f"status={status}, Content-Range={content_range!r}"
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
        except OSError:
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(min(30.0, 1.0 * (2**attempt)))
            continue
        if path.stat().st_size == expected_size:
            return
        if attempt + 1 < max_attempts:
            time.sleep(min(30.0, 1.0 * (2**attempt)))
    raise RuntimeError(f"Range part remained incomplete after {max_attempts} attempts: {path}")


def download_range_file(
    url: str,
    target: str | Path,
    *,
    expected_size: int,
    expected_sha256: str,
    workers: int = 8,
    part_size: int = DEFAULT_PART_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: float = 60.0,
    max_attempts: int = 8,
    opener: OpenUrl = urlopen,
) -> Path:
    """Download fixed byte ranges, assemble once, and verify the complete content hash."""
    if (
        expected_size <= 0
        or len(expected_sha256) != 64
        or workers <= 0
        or part_size <= 0
        or chunk_size <= 0
        or timeout <= 0
        or max_attempts <= 0
    ):
        raise ValueError("Range download arguments are invalid")
    destination = Path(target)
    with _exclusive_download_lock(destination):
        return _download_range_file_locked(
            url,
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            workers=workers,
            part_size=part_size,
            chunk_size=chunk_size,
            timeout=timeout,
            max_attempts=max_attempts,
            opener=opener,
        )


def _download_range_file_locked(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    workers: int,
    part_size: int,
    chunk_size: int,
    timeout: float,
    max_attempts: int,
    opener: OpenUrl,
) -> Path:
    """Run one already locked range download."""
    if destination.is_file():
        if destination.stat().st_size == expected_size:
            digest_state = hashlib.sha256()
            with destination.open("rb") as handle:
                while chunk := handle.read(chunk_size):
                    digest_state.update(chunk)
            digest = digest_state.hexdigest()
            if digest == expected_sha256:
                return destination
        raise ValueError(f"Existing range-download target is invalid: {destination}")
    parts_dir = destination.with_name(destination.name + ".range-parts")
    ranges = _ranges(expected_size, part_size)

    def download(spec: tuple[int, int]) -> None:
        start, end = spec
        part = parts_dir / f"{start:020d}-{end:020d}.part"
        _download_part(
            url=url,
            path=part,
            start=start,
            end=end,
            total_size=expected_size,
            timeout=timeout,
            max_attempts=max_attempts,
            chunk_size=chunk_size,
            opener=opener,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(download, ranges))

    assembling = destination.with_name(destination.name + ".assembling")
    if assembling.exists():
        assembling.unlink()
    digest = hashlib.sha256()
    with assembling.open("wb") as output:
        for start, end in ranges:
            part = parts_dir / f"{start:020d}-{end:020d}.part"
            if part.stat().st_size != end - start + 1:
                raise ValueError(f"Range part size changed before assembly: {part}")
            with part.open("rb") as source:
                while chunk := source.read(chunk_size):
                    output.write(chunk)
                    digest.update(chunk)
    if assembling.stat().st_size != expected_size or digest.hexdigest() != expected_sha256:
        raise ValueError("Assembled range download differs from the pinned size or SHA-256")
    assembling.replace(destination)
    for start, end in ranges:
        (parts_dir / f"{start:020d}-{end:020d}.part").unlink()
    parts_dir.rmdir()
    return destination
