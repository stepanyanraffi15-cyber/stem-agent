from __future__ import annotations

import dataclasses

import pytest

from src.logging.console import (
    log_iteration_result,
    log_test_result,
    render_ascii_pareto,
    log_final_summary,
)


def test_log_iteration_result_runs() -> None:
    """log_iteration_result must not raise under all reward/flag combinations."""
    log_iteration_result(
        iteration=5,
        reward=0.721,
        best_reward=0.744,
        temperatures=[0.4, 0.6, 0.9],
        rollback_fired=False,
        gradient_recomputed=True,
    )
    log_iteration_result(
        iteration=1,
        reward=0.8,
        best_reward=0.7,
        temperatures=[0.6],
        rollback_fired=True,
        gradient_recomputed=False,
    )


def test_render_ascii_pareto_correct_dimensions() -> None:
    """ASCII grid must have one line per Y bucket plus 2 footer lines."""
    pareto_data = [(0, 0.3), (1, 0.5), (2, 0.7), (3, 0.65)]
    result = render_ascii_pareto(pareto_data, width=40)
    assert isinstance(result, str)
    lines = result.split("\n")
    assert len(lines) == 12, f"Expected 12 lines (10 rows + 2 footer), got {len(lines)}"
    assert "*" in result, "Must contain at least one data point marker"


def test_render_ascii_pareto_empty() -> None:
    """Empty pareto_data returns '(no data)' without raising."""
    result = render_ascii_pareto([])
    assert result == "(no data)"


def test_log_test_result_pass() -> None:
    """Pass result must not raise."""
    log_test_result(test_name="test_something", passed=True, duration_ms=12.5)


def test_log_test_result_fail() -> None:
    """Fail result with detail must not raise."""
    log_test_result(
        test_name="test_broken",
        passed=False,
        duration_ms=8.2,
        detail="AssertionError: 0 != 1",
    )


def test_log_final_summary_runs(mock_experiment_result: object) -> None:
    """log_final_summary must not raise when given dataclasses.asdict output."""
    result_dict = dataclasses.asdict(mock_experiment_result)
    log_final_summary(result_dict)
