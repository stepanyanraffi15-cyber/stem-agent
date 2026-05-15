from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentReview(TypedDict):
    file_path: str
    issues: list[dict]
    overall_confidence: float
    summary: str


class VariantResult(TypedDict):
    variant_id: int
    prompt: str
    temperature: float
    score: float


class SFTState(TypedDict):
    task_theory: dict
    stem_config: dict
    experiment_id: str
    demonstrations: list[str]
    extracted_patterns: list[str]
    critiqued_patterns: list[str]
    rewrite_reasoning: str
    final_prompt: str
    skill_library: dict
    agent_reviews: Annotated[list[AgentReview], operator.add]


class RLState(TypedDict):
    task_theory: dict
    stem_config: dict
    uncertainty_priors: list[dict]
    experiment_id: str
    condition: str
    current_prompt: str
    iteration: int
    curriculum_index: int
    curriculum_order: list[str]
    performance_history: list[float]
    failure_memory: dict
    consecutive_no_improvement: int
    last_verbal_gradient: dict | None
    temperatures: list[float]
    variant_results: Annotated[list[VariantResult], operator.add]
    skill_library: dict
    prompt_manager: dict
    stopping_reason: str | None
    max_iterations: int
    rollback_events: list[dict]
    recent_failures: list[dict]
    agent_reviews: Annotated[list[AgentReview], operator.add]
    final_prompt: str | None


class VariantState(TypedDict):
    task_theory: dict
    stem_config: dict
    uncertainty_priors: list[dict]
    experiment_id: str
    condition: str
    current_prompt: str
    iteration: int
    curriculum_index: int
    curriculum_order: list[str]
    performance_history: list[float]
    failure_memory: dict
    consecutive_no_improvement: int
    last_verbal_gradient: dict | None
    temperatures: list[float]
    variant_results: Annotated[list[VariantResult], operator.add]
    skill_library: dict
    prompt_manager: dict
    stopping_reason: str | None
    max_iterations: int
    rollback_events: list[dict]
    recent_failures: list[dict]
    agent_reviews: Annotated[list[AgentReview], operator.add]
    final_prompt: str | None
    variant_id: int
    target_temperature: float
    variant_strategy: str


class OuterState(TypedDict):
    task_theory: dict
    stem_config: dict
    uncertainty_priors: list[dict]
    experiment_id: str
    condition: str
    final_prompt: str | None
    evaluation_results: dict | None
    # sft-specific
    demonstrations: list[str] | None
    extracted_patterns: list[str] | None
    critiqued_patterns: list[str] | None
    rewrite_reasoning: str | None
    # shared / rl-specific
    skill_library: dict | None
    performance_history: list[float] | None
    current_prompt: str | None
    iteration: int | None
    curriculum_index: int | None
    curriculum_order: list[str] | None
    failure_memory: dict | None
    consecutive_no_improvement: int | None
    last_verbal_gradient: dict | None
    temperatures: list[float] | None
    variant_results: list[dict] | None
    prompt_manager: dict | None
    stopping_reason: str | None
    max_iterations: int | None
    rollback_events: list[dict] | None
    recent_failures: list[dict] | None
    agent_reviews: list[dict] | None
