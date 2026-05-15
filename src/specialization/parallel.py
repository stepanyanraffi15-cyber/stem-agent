from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")

_DEFAULT_MAX_WORKERS: int = 10


def parallel_score_files(
    score_fn: Callable[[str], dict | None],
    file_paths: list[str],
    max_workers: int | None = None,
) -> list[dict]:
    """Run score_fn concurrently over file_paths. Returns results in stable insertion order.

    None returns from score_fn are filtered out — same contract as sequential loops.
    max_workers defaults to min(_DEFAULT_MAX_WORKERS, len(file_paths)) to avoid
    over-provisioning when the file list is small.
    """
    if not file_paths:
        return []

    workers = min(max_workers or _DEFAULT_MAX_WORKERS, len(file_paths))
    index: dict[str, int] = {fp: i for i, fp in enumerate(file_paths)}
    collected: dict[int, dict] = {}

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_fn, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("parallel_score_files.worker_error", path=fp, error=str(exc))
                continue
            if result is not None:
                collected[index[fp]] = result

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "parallel_score_files.done",
        total=len(file_paths),
        scored=len(collected),
        workers=workers,
        elapsed_ms=elapsed_ms,
    )
    return [collected[i] for i in sorted(collected)]


def parallel_map(
    fn: Callable[[T], dict | None],
    items: list[T],
    key_fn: Callable[[T], str],
    max_workers: int | None = None,
) -> list[dict]:
    """Generic parallel executor for non-file items (e.g. VariantResult dicts).

    key_fn extracts a string key used for ordering and error logging.
    Returns results in stable insertion order, None results filtered out.
    """
    if not items:
        return []

    workers = min(max_workers or _DEFAULT_MAX_WORKERS, len(items))
    index: dict[str, int] = {key_fn(item): i for i, item in enumerate(items)}
    collected: dict[int, dict] = {}

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): key_fn(item) for item in items}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("parallel_map.worker_error", key=key, error=str(exc))
                continue
            if result is not None:
                collected[index[key]] = result

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "parallel_map.done",
        total=len(items),
        results=len(collected),
        workers=workers,
        elapsed_ms=elapsed_ms,
    )
    return [collected[i] for i in sorted(collected)]
