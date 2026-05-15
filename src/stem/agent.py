# The Stem phase — agent self-configuration from task description.
# The agent does NOT yet see any training examples. It reads only the task
# description and produces its own theory and initial system prompt.
#
# Maps to: "reads signals from environment and transforms" in stem cell metaphor.
# Grounds: PromptAgent (arXiv:2310.16427) — strategic prompt optimization
# begins with understanding the task, not immediately optimizing.
# Also: KAMI benchmark (arXiv:2512.07497) failure archetype 1 —
# "premature action without grounding." Our stem phase explicitly prevents this
# by forcing theory-first before any specialization.
from __future__ import annotations

import dataclasses
import json

import structlog

from src.stem.llm_client import LLMClient
from src.stem.models import LLMMessage, StemConfig, TaskTheory, UncertaintyPrior
from src.verifier.models import BUG_TYPE_TO_PYLINT_SYMBOL

logger = structlog.get_logger(__name__)

TASK_DESCRIPTION = "Review Python code for bugs"

THEORIZE_SYSTEM_PROMPT = """You are an expert AI researcher designing a code
review agent. Your job is to deeply analyze a task class and produce a structured
theory about how to solve it well. Be specific and technical. Think about failure
modes, not just success patterns."""

CONFIGURE_SYSTEM_PROMPT = """You are an AI agent that has just developed a theory
about a task class. Now write your own system prompt that you will use when performing
the task. Your system prompt should be concrete, specific, and actionable — not generic.
It should encode your theory as explicit instructions.
Return JSON with exactly two keys: "system_prompt" (a single plain-text string, NOT a \
JSON object or nested structure) and "reasoning" (a string). The value of "system_prompt" \
must be a flat prose string ready to be used verbatim as an LLM system message."""

UNCERTAINTY_SYSTEM_PROMPT = """You are an AI agent about to specialize on a code
review task. Assess your confidence for each bug type you might encounter. Be honest
about what you are likely to miss."""


class StemAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def theorize(self, task_description: str) -> TaskTheory:
        if not task_description.strip():
            raise ValueError("task_description must not be empty")

        messages = [
            LLMMessage(role="system", content=THEORIZE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=task_description),
        ]
        response_dict = self._llm.complete_json(messages)

        theory = TaskTheory(
            task_description=task_description,
            difficulty_factors=response_dict.get("difficulty_factors", []),
            common_patterns=response_dict.get("common_patterns", []),
            expert_strategies=response_dict.get("expert_strategies", []),
            information_needed=response_dict.get("information_needed", []),
            raw_llm_output=json.dumps(response_dict),
        )
        logger.info(
            "stem.theorize.done",
            difficulty_factors_count=len(theory.difficulty_factors),
            common_patterns_count=len(theory.common_patterns),
            expert_strategies_count=len(theory.expert_strategies),
        )
        return theory

    def configure(self, theory: TaskTheory) -> StemConfig:
        theory_json = json.dumps(dataclasses.asdict(theory), default=str, indent=2)
        messages = [
            LLMMessage(role="system", content=CONFIGURE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"Theory:\n{theory_json}"),
        ]
        response_dict = self._llm.complete_json(messages)

        raw_prompt = response_dict.get("system_prompt", "")
        if not isinstance(raw_prompt, str):
            raw_prompt = json.dumps(raw_prompt)
        if not raw_prompt.strip():
            raise ValueError("configure: LLM returned an empty system_prompt")

        config = StemConfig(
            system_prompt=raw_prompt,
            tools_selected=response_dict.get("tools_selected", []),
            reasoning=response_dict.get("reasoning", ""),
            version=0,
            theory_used=theory,
        )
        logger.info(
            "stem.configure.done",
            tools_count=len(config.tools_selected),
            prompt_length=len(config.system_prompt),
        )
        return config

    def estimate_uncertainty(
        self,
        bug_types: list[str],
        config: StemConfig,
    ) -> list[UncertaintyPrior]:
        if not bug_types:
            raise ValueError("bug_types must not be empty")

        bug_type_list = "\n".join(f"- {bt}" for bt in bug_types)
        messages = [
            LLMMessage(role="system", content=UNCERTAINTY_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Your current system prompt:\n{config.system_prompt}\n\n"
                    f"Rate your confidence for each bug type:\n{bug_type_list}"
                ),
            ),
        ]
        response = self._llm.complete_json(messages)

        raw_priors: list[dict] = response if isinstance(response, list) else []
        priors: list[UncertaintyPrior] = []
        for item in raw_priors:
            raw_conf = float(item.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, raw_conf))
            priors.append(
                UncertaintyPrior(
                    bug_type=item.get("bug_type", ""),
                    confidence=confidence,
                    reasoning=item.get("reasoning", ""),
                )
            )

        logger.info(
            "stem.uncertainty.done",
            bug_types_count=len(bug_types),
            priors_count=len(priors),
        )
        return priors

    def run_stem_phase(
        self,
    ) -> tuple[TaskTheory, StemConfig, list[UncertaintyPrior]]:
        bug_types = list(BUG_TYPE_TO_PYLINT_SYMBOL.keys())

        logger.info("stem.phase.start", task_description=TASK_DESCRIPTION)

        theory = self.theorize(TASK_DESCRIPTION)
        logger.info("stem.phase.theory_complete")

        config = self.configure(theory)
        logger.info("stem.phase.config_complete", version=config.version)

        priors = self.estimate_uncertainty(bug_types, config)
        logger.info("stem.phase.complete", priors_count=len(priors))

        return theory, config, priors
