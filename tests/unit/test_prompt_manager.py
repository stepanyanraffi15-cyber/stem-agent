from __future__ import annotations

from src.stem.llm_client import LLMClient
from src.stem.models import LLMResponse, Skill
from src.stem.prompt_manager import PromptManager


def _make_skill(name: str = "detect_undef", locked: bool = True) -> Skill:
    return Skill(
        name=name,
        description="Detect undefined variables",
        detection_pattern="check NameError",
        bug_types_covered=["undefined_variable"],
        confidence=0.9,
        locked=locked,
    )


def _pm_with_versions(scores: list[float]) -> PromptManager:
    pm = PromptManager()
    for i, score in enumerate(scores):
        pm.save_version(
            prompt=f"prompt_v{i}",
            score=score,
            iteration=i,
            skills_locked=[],
        )
    return pm


def test_save_version_stores_correctly() -> None:
    pm = PromptManager()
    v = pm.save_version(prompt="p0", score=0.5, iteration=0, skills_locked=["s1"])
    assert len(pm.versions) == 1
    assert v.prompt == "p0"
    assert v.score == 0.5
    assert v.iteration == 0
    assert v.skills_locked == ["s1"]
    assert v.improvement_over_prev == 0.0


def test_save_version_computes_improvement() -> None:
    pm = _pm_with_versions([0.4, 0.6])
    assert abs(pm.versions[1].improvement_over_prev - 0.2) < 1e-9


def test_get_best_version_returns_none_when_empty() -> None:
    pm = PromptManager()
    assert pm.get_best_version() is None


def test_get_best_version_returns_highest_score() -> None:
    pm = _pm_with_versions([0.3, 0.7, 0.5])
    best = pm.get_best_version()
    assert best is not None
    assert best.score == 0.7


def test_get_current_prompt_returns_none_when_empty() -> None:
    pm = PromptManager()
    assert pm.get_current_prompt() is None


def test_get_current_prompt_returns_latest() -> None:
    pm = _pm_with_versions([0.3, 0.7])
    assert pm.get_current_prompt() == "prompt_v1"


def test_detect_catastrophic_forgetting_false_when_insufficient_versions() -> None:
    pm = _pm_with_versions([0.5, 0.6])
    assert pm.detect_catastrophic_forgetting(0.1) is False


def test_detect_catastrophic_forgetting_false_on_normal_score() -> None:
    pm = _pm_with_versions([0.6, 0.65, 0.7])
    assert pm.detect_catastrophic_forgetting(0.68) is False


def test_detect_catastrophic_forgetting_true_on_hard_floor() -> None:
    pm = _pm_with_versions([0.8, 0.81, 0.82])
    assert pm.detect_catastrophic_forgetting(0.5) is True


def test_detect_catastrophic_forgetting_true_on_statistical_drop() -> None:
    pm = _pm_with_versions([0.7, 0.71, 0.72])
    assert pm.detect_catastrophic_forgetting(0.01) is True


def test_ewc_constrained_rollback_calls_complete(mocker) -> None:
    pm = _pm_with_versions([0.6, 0.7])
    mock_client = mocker.MagicMock(spec=LLMClient)
    mock_client.complete.return_value = LLMResponse(
        content="Merged prompt text.",
        model="mock",
        input_tokens=50,
        output_tokens=30,
        raw_response={},
    )

    locked = [_make_skill("detect_undef")]
    result = pm.ewc_constrained_rollback("new prompt", locked, mock_client)

    assert result == "Merged prompt text."
    mock_client.complete.assert_called_once()


def test_ewc_constrained_rollback_records_event(mocker) -> None:
    pm = _pm_with_versions([0.5, 0.75])
    mock_client = mocker.MagicMock(spec=LLMClient)
    mock_client.complete.return_value = LLMResponse(
        content="merged", model="mock", input_tokens=10, output_tokens=5, raw_response={}
    )

    pm.ewc_constrained_rollback("new prompt", [], mock_client)

    events = pm.rollback_events()
    assert len(events) == 1
    assert events[0]["best_score"] == 0.75


def test_ewc_constrained_rollback_no_best_version_returns_new_prompt(mocker) -> None:
    pm = PromptManager()
    mock_client = mocker.MagicMock(spec=LLMClient)
    result = pm.ewc_constrained_rollback("fallback prompt", [], mock_client)
    assert result == "fallback prompt"
    mock_client.complete.assert_not_called()


def test_compute_improvement_ci_returns_none_insufficient_data() -> None:
    pm = _pm_with_versions([0.5, 0.6])
    assert pm.compute_improvement_ci() is None


def test_compute_improvement_ci_returns_tuple_with_enough_data() -> None:
    pm = _pm_with_versions([0.4, 0.55, 0.65, 0.72])
    ci = pm.compute_improvement_ci()
    assert ci is not None
    lower, upper = ci
    assert lower <= upper


def test_should_stop_performance_threshold() -> None:
    pm = _pm_with_versions([0.9])
    should, reason = pm.should_stop()
    assert should is True
    assert reason == "performance_threshold_reached"


def test_should_stop_max_iterations() -> None:
    pm = _pm_with_versions([0.3 + i * 0.03 for i in range(15)])
    should, reason = pm.should_stop()
    assert should is True
    assert reason == "max_iterations_reached"


def test_should_stop_continue() -> None:
    pm = _pm_with_versions([0.4, 0.5])
    should, reason = pm.should_stop()
    assert should is False
    assert reason == "continue"


def test_rollback_events_returns_list() -> None:
    pm = PromptManager()
    assert pm.rollback_events() == []


def test_history_summary_structure() -> None:
    pm = _pm_with_versions([0.4, 0.55, 0.65])
    summary = pm.history_summary()
    assert summary["versions_count"] == 3
    assert summary["best_score"] == 0.65
    assert summary["best_iteration"] == 2
    assert summary["rollback_count"] == 0
    assert summary["current_score"] == 0.65
    assert len(summary["improvement_trend"]) == 3
