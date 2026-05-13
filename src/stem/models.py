from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskTheory:
    """
    Agent's self-generated understanding of the task class.
    Produced in the Stem phase before any specialization.
    Maps to the 'observe' behavior in the stem cell metaphor —
    agent reads signals from environment before transforming.
    """

    task_description: str
    difficulty_factors: list[str]
    common_patterns: list[str]
    expert_strategies: list[str]
    information_needed: list[str]
    raw_llm_output: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StemConfig:
    """
    Agent's initial self-configuration based on its TaskTheory.
    This is the first system prompt the agent writes for itself.
    """

    system_prompt: str
    tools_selected: list[str]
    reasoning: str
    version: int = 0
    theory_used: TaskTheory | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UncertaintyPrior:
    """
    Agent's self-assessed confidence per bug type before specialization.
    Used to weight curriculum ordering and track calibration over time.
    Grounds: Kuhn et al. 2023 semantic entropy (arXiv:2302.09664) —
    uncertainty should be tracked and updated as evidence accumulates.
    """

    bug_type: str
    confidence: float
    reasoning: str
    sample_count: int = 0


@dataclass
class PromptVersion:
    """
    A versioned snapshot of the agent's system prompt with its score.
    EWC-inspired: we track which skills are locked at each version
    to prevent catastrophic forgetting during specialization.
    Grounds: Kirkpatrick et al. 2017 (EWC) — preserve important weights.
    """

    prompt: str
    score: float
    iteration: int
    skills_locked: list[str]
    improvement_over_prev: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    stopping_reason: str | None = None


@dataclass
class Skill:
    """
    A discrete, reusable bug detection strategy.
    Grounds: Voyager (arXiv:2305.16291) — agent builds a growing skill
    library of reusable strategies rather than monolithic prompt updates.
    """

    name: str
    description: str
    detection_pattern: str
    bug_types_covered: list[str]
    confidence: float
    locked: bool = False
    evidence_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    raw_response: dict
