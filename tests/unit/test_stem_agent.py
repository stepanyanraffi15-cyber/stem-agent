from __future__ import annotations

import pytest

from src.stem.agent import StemAgent
from src.stem.llm_client import LLMClient
from src.stem.models import LLMResponse, StemConfig, TaskTheory, UncertaintyPrior

_THEORIZE_RESPONSE = {
    "difficulty_factors": ["subtle logic errors"],
    "common_patterns": ["check variable definitions"],
    "expert_strategies": ["read docstring first"],
    "information_needed": ["function intent"],
}

_CONFIGURE_RESPONSE = {
    "system_prompt": "You are a Python code reviewer...",
    "tools_selected": ["pylint_verifier"],
    "reasoning": "Pylint catches static errors deterministically.",
}

_UNCERTAINTY_RESPONSE = [
    {"bug_type": "undefined_variable", "confidence": 0.9, "reasoning": "Pylint catches this reliably."},
    {"bug_type": "off_by_one", "confidence": 0.3, "reasoning": "Requires reasoning about intent."},
]

_MOCK_LLM_RESPONSE = LLMResponse(
    content="Merged system prompt text here.",
    model="mock",
    input_tokens=100,
    output_tokens=50,
    raw_response={},
)


@pytest.fixture
def mock_llm_client(mocker):
    client = mocker.MagicMock(spec=LLMClient)
    client.complete_json.side_effect = [
        _THEORIZE_RESPONSE,
        _CONFIGURE_RESPONSE,
        _UNCERTAINTY_RESPONSE,
    ]
    client.complete.return_value = _MOCK_LLM_RESPONSE
    return client


@pytest.fixture
def stem_agent(mock_llm_client: LLMClient) -> StemAgent:
    return StemAgent(llm_client=mock_llm_client)


def test_theorize_returns_task_theory(stem_agent: StemAgent) -> None:
    result = stem_agent.theorize("Review Python code for bugs")
    assert isinstance(result, TaskTheory)
    assert result.difficulty_factors == ["subtle logic errors"]
    assert result.common_patterns == ["check variable definitions"]
    assert result.expert_strategies == ["read docstring first"]
    assert result.information_needed == ["function intent"]
    assert result.task_description == "Review Python code for bugs"


def test_theorize_calls_llm_once(stem_agent: StemAgent, mock_llm_client) -> None:
    stem_agent.theorize("Review Python code for bugs")
    assert mock_llm_client.complete_json.call_count == 1


def test_theorize_preserves_raw_output(stem_agent: StemAgent) -> None:
    result = stem_agent.theorize("Review Python code for bugs")
    assert "subtle logic errors" in result.raw_llm_output
    assert len(result.raw_llm_output) > 0


def test_configure_includes_theory_in_prompt(stem_agent: StemAgent, mock_llm_client) -> None:
    mock_llm_client.complete_json.side_effect = None
    mock_llm_client.complete_json.return_value = _CONFIGURE_RESPONSE

    theory = TaskTheory(
        task_description="Review Python code",
        difficulty_factors=["subtle logic errors"],
        common_patterns=["check variables"],
        expert_strategies=["read docstring"],
        information_needed=["function intent"],
        raw_llm_output="{}",
    )
    stem_agent.configure(theory)

    call_args = mock_llm_client.complete_json.call_args
    messages = call_args[0][0]
    user_message = next(m for m in messages if m.role == "user")
    assert "subtle logic errors" in user_message.content


def test_configure_returns_stem_config(stem_agent: StemAgent, mock_llm_client) -> None:
    mock_llm_client.complete_json.side_effect = None
    mock_llm_client.complete_json.return_value = _CONFIGURE_RESPONSE

    theory = TaskTheory(
        task_description="Review Python code",
        difficulty_factors=[],
        common_patterns=[],
        expert_strategies=[],
        information_needed=[],
        raw_llm_output="{}",
    )
    result = stem_agent.configure(theory)

    assert isinstance(result, StemConfig)
    assert len(result.system_prompt) > 0
    assert result.version == 0
    assert result.theory_used is theory


def test_uncertainty_returns_one_prior_per_bug_type(stem_agent: StemAgent, mock_llm_client) -> None:
    mock_llm_client.complete_json.side_effect = None
    mock_llm_client.complete_json.return_value = _UNCERTAINTY_RESPONSE

    config = StemConfig(
        system_prompt="You are a Python code reviewer...",
        tools_selected=["pylint_verifier"],
        reasoning="reason",
    )
    bug_types = ["undefined_variable", "off_by_one"]
    result = stem_agent.estimate_uncertainty(bug_types, config)

    assert len(result) == len(bug_types)


def test_uncertainty_confidence_in_range(stem_agent: StemAgent, mock_llm_client) -> None:
    mock_llm_client.complete_json.side_effect = None
    mock_llm_client.complete_json.return_value = [
        {"bug_type": "undefined_variable", "confidence": 1.5, "reasoning": "over"},
        {"bug_type": "off_by_one", "confidence": -0.2, "reasoning": "under"},
    ]

    config = StemConfig(
        system_prompt="You are a Python code reviewer...",
        tools_selected=[],
        reasoning="reason",
    )
    result = stem_agent.estimate_uncertainty(["undefined_variable", "off_by_one"], config)

    for prior in result:
        assert isinstance(prior, UncertaintyPrior)
        assert 0.0 <= prior.confidence <= 1.0


def test_run_stem_phase_returns_all_three(stem_agent: StemAgent) -> None:
    theory, config, priors = stem_agent.run_stem_phase()
    assert isinstance(theory, TaskTheory)
    assert isinstance(config, StemConfig)
    assert isinstance(priors, list)
    assert all(isinstance(p, UncertaintyPrior) for p in priors)


def test_run_stem_phase_calls_llm_three_times(stem_agent: StemAgent, mock_llm_client) -> None:
    stem_agent.run_stem_phase()
    assert mock_llm_client.complete_json.call_count == 3
