from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.specialization.parallel import parallel_map, parallel_score_files


def _identity_fn(path: str) -> dict | None:
    return {"file_path": path, "score": len(path)}


def _none_fn(path: str) -> dict | None:
    return None


def _slow_fn(path: str) -> dict | None:
    time.sleep(0.01)
    return {"file_path": path}


def _raising_fn(path: str) -> dict | None:
    raise RuntimeError(f"boom: {path}")


def test_parallel_score_files_returns_same_results_as_sequential() -> None:
    paths = [f"file_{i:03d}.py" for i in range(20)]
    sequential = [_identity_fn(p) for p in paths]
    parallel = parallel_score_files(_identity_fn, paths)
    assert parallel == sequential


def test_parallel_score_files_preserves_insertion_order() -> None:
    paths = [f"z_{i}.py" for i in range(10)]
    result = parallel_score_files(_slow_fn, paths)
    assert [r["file_path"] for r in result] == paths


def test_parallel_score_files_filters_none_results() -> None:
    paths = ["a.py", "b.py", "c.py"]
    result = parallel_score_files(_none_fn, paths)
    assert result == []


def test_parallel_score_files_partial_none() -> None:
    paths = ["keep_0.py", "drop_1.py", "keep_2.py"]

    def mixed(p: str) -> dict | None:
        return None if "drop" in p else {"file_path": p}

    result = parallel_score_files(mixed, paths)
    assert len(result) == 2
    assert result[0]["file_path"] == "keep_0.py"
    assert result[1]["file_path"] == "keep_2.py"


def test_parallel_score_files_empty_input() -> None:
    assert parallel_score_files(_identity_fn, []) == []


def test_parallel_score_files_single_item() -> None:
    result = parallel_score_files(_identity_fn, ["only.py"])
    assert len(result) == 1
    assert result[0]["file_path"] == "only.py"


def test_parallel_score_files_worker_error_does_not_propagate() -> None:
    paths = ["ok_0.py", "bad_1.py", "ok_2.py"]

    def maybe_raise(p: str) -> dict | None:
        if "bad" in p:
            raise ValueError("intentional error")
        return {"file_path": p}

    result = parallel_score_files(maybe_raise, paths)
    returned_paths = [r["file_path"] for r in result]
    assert "ok_0.py" in returned_paths
    assert "ok_2.py" in returned_paths
    assert "bad_1.py" not in returned_paths


def test_parallel_score_files_max_workers_capped_to_file_count() -> None:
    paths = ["a.py", "b.py"]
    result = parallel_score_files(_identity_fn, paths, max_workers=100)
    assert len(result) == 2


def test_parallel_score_files_concurrent_execution_is_faster_than_sequential() -> None:
    paths = [f"file_{i}.py" for i in range(8)]

    def slow(p: str) -> dict | None:
        time.sleep(0.05)
        return {"file_path": p}

    start = time.monotonic()
    parallel_score_files(slow, paths, max_workers=8)
    elapsed = time.monotonic() - start

    assert elapsed < 0.3, f"expected <0.3s with concurrency, got {elapsed:.2f}s"


def test_parallel_map_returns_stable_order() -> None:
    items = [{"id": i, "value": str(i)} for i in range(10)]

    def fn(item: dict) -> dict | None:
        return {"result": item["id"] * 2}

    result = parallel_map(fn, items, key_fn=lambda x: str(x["id"]))
    assert [r["result"] for r in result] == [i * 2 for i in range(10)]


def test_parallel_map_filters_none() -> None:
    items = [{"id": i} for i in range(5)]

    def fn(item: dict) -> dict | None:
        return None if item["id"] % 2 == 0 else {"id": item["id"]}

    result = parallel_map(fn, items, key_fn=lambda x: str(x["id"]))
    assert all(r["id"] % 2 != 0 for r in result)


def test_get_llm_client_thread_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify concurrent calls to get_llm_client don't clobber os.environ."""
    call_log: list[str] = []
    lock = threading.Lock()

    original_init = None

    class TrackingClient:
        def __init__(self) -> None:
            model = os.environ.get("MODEL_NAME", "")
            with lock:
                call_log.append(model)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_PROVIDER", "openai")

    with patch("src.specialization.llm_factory.LLMClient", TrackingClient):
        from src.specialization.llm_factory import get_llm_client

        threads = [
            threading.Thread(target=get_llm_client, args=("agent_review",))
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(call_log) == 10
    cheap_model = os.environ.get("MODEL_NAME_CHEAP", "gpt-5.4-mini")
    assert all(m == cheap_model for m in call_log), (
        f"Some threads saw wrong MODEL_NAME: {call_log}"
    )


def test_get_max_tokens_returns_role_specific_values() -> None:
    from src.specialization.llm_factory import get_max_tokens

    assert get_max_tokens("agent_review") == 1024
    assert get_max_tokens("gradient") == 2048
    assert get_max_tokens("variant_gen") == 2048
    assert get_max_tokens("ewc_merge") == 2048
    assert get_max_tokens("stem_phase") == 4096


def test_get_max_tokens_raises_on_unknown_key() -> None:
    from src.specialization.llm_factory import get_max_tokens

    with pytest.raises(ValueError, match="Unknown model_key"):
        get_max_tokens("nonexistent_role")
