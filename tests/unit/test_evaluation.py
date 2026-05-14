from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    MATCH_LINE_TOLERANCE,
    FileScore,
    bootstrap_ci,
    calibration_ece,
    generalization_gap,
    pareto_curve,
    precision_recall_f1,
    skill_growth_curve,
)
from src.verifier.models import GroundTruth


def _gt(bug_type: str = "undefined_variable", bug_line: int = 10) -> GroundTruth:
    return GroundTruth(
        file_path="test.py",
        bug_type=bug_type,
        bug_line=bug_line,
        bug_description="test",
        detectable_by_pylint=True,
        detectable_by_ast=False,
        detectable_by_execution=False,
    )


def _review(issues: list[dict], file_path: str = "test.py", reward: float = 1.0) -> dict:
    return {
        "file_path": file_path,
        "issues": issues,
        "overall_confidence": 0.9,
        "summary": "test",
        "reward": reward,
    }


def test_precision_recall_perfect_detection() -> None:
    """Agent finds exact bug type and line — precision=1, recall=1, f1=1."""
    reviews = [_review([{"line": 10, "bug_type": "undefined_variable", "description": "x", "confidence": 0.9}])]
    gts = [_gt("undefined_variable", 10)]
    scores = precision_recall_f1(reviews, gts)
    assert len(scores) == 1
    assert scores[0].precision == pytest.approx(1.0)
    assert scores[0].recall == pytest.approx(1.0)
    assert scores[0].f1 > 0.99


def test_precision_recall_no_detection() -> None:
    """Agent finds nothing — recall=0, f1~0."""
    reviews = [_review([])]
    gts = [_gt()]
    scores = precision_recall_f1(reviews, gts)
    assert scores[0].recall == pytest.approx(0.0)
    assert scores[0].f1 < 0.01


def test_precision_recall_false_positive() -> None:
    """Agent reports wrong bug type — FP=1, precision=0, recall=0."""
    reviews = [_review([{"line": 10, "bug_type": "bare_except", "description": "x", "confidence": 0.9}])]
    gts = [_gt("undefined_variable", 10)]
    scores = precision_recall_f1(reviews, gts)
    assert scores[0].precision == pytest.approx(0.0)
    assert scores[0].recall == pytest.approx(0.0)
    assert scores[0].false_positives == 1


def test_precision_recall_line_tolerance() -> None:
    """Bug on line 12, agent reports line 13 — should match (tolerance=2)."""
    reviews = [_review([{"line": 13, "bug_type": "undefined_variable", "description": "x", "confidence": 0.9}])]
    gts = [_gt("undefined_variable", 12)]
    scores = precision_recall_f1(reviews, gts)
    assert scores[0].recall == pytest.approx(1.0), "Should match within tolerance"


def test_generalization_gap_positive() -> None:
    """In-dist F1 > OOD F1 → positive gap."""

    def _score(f1: float) -> FileScore:
        return FileScore(
            file_path="x.py", precision=f1, recall=f1, f1=f1,
            issues_found=1, issues_expected=1, false_positives=0,
            reward=f1, is_silent_failure=False,
        )

    in_dist = [_score(0.8), _score(0.7)]
    ood = [_score(0.3), _score(0.2)]
    gap = generalization_gap(in_dist, ood)
    assert gap > 0.0
    assert gap == pytest.approx(0.5, abs=0.01)


def test_generalization_gap_zero() -> None:
    """Equal F1 → zero gap."""

    def _score(f1: float) -> FileScore:
        return FileScore(
            file_path="x.py", precision=f1, recall=f1, f1=f1,
            issues_found=1, issues_expected=1, false_positives=0,
            reward=f1, is_silent_failure=False,
        )

    scores = [_score(0.5), _score(0.5)]
    gap = generalization_gap(scores, scores)
    assert gap == pytest.approx(0.0, abs=0.001)


def test_calibration_ece_perfect() -> None:
    """Confidence 0.9 on 90% correct → ECE near 0."""
    n = 100
    confidences = [0.9] * n
    correct = [True] * 90 + [False] * 10
    ece = calibration_ece(confidences, correct)
    assert ece < 0.05


def test_calibration_ece_overconfident() -> None:
    """Confidence 0.9 on 50% correct → ECE > 0.3."""
    n = 100
    confidences = [0.9] * n
    correct = [True] * 50 + [False] * 50
    ece = calibration_ece(confidences, correct)
    assert ece > 0.3


def test_pareto_curve_length() -> None:
    """pareto_curve returns one (iter, score) tuple per input score."""
    history = [0.3, 0.5, 0.6, 0.55, 0.7]
    curve = pareto_curve(history)
    assert len(curve) == len(history)
    assert curve[0] == (0, 0.3)
    assert curve[-1] == (4, 0.7)


def test_bootstrap_ci_range() -> None:
    """CI lower < mean < CI upper."""
    values = [0.5, 0.6, 0.4, 0.55, 0.45, 0.7, 0.3, 0.5, 0.6, 0.5]
    mean = sum(values) / len(values)
    lower, upper = bootstrap_ci(values)
    assert lower < mean < upper
    assert lower >= 0.0
    assert upper <= 1.0


def test_print_comparison_table_runs_without_error(
    mock_experiment_result: object,
) -> None:
    """Rich table generation must not raise."""
    from src.evaluation.benchmark import print_comparison_table
    print_comparison_table(mock_experiment_result)
