# Prompt version control with catastrophic forgetting detection.
#
# EWC (Kirkpatrick et al. 2017): preserve "important weights" during updates.
# Here: locked skills are the important weights. Rollback merges new improvements
# while constraining the prompt to preserve locked skill language.
#
# AgenTracer (arXiv:2509.03312): decisive error = earliest action causing failure.
# Our forgetting detection identifies the decisive iteration where specialization
# went wrong — exactly the same concept applied to prompt optimization.
#
# IBM Silent Failures (arXiv:2511.04032): performance can degrade without
# an explicit error signal. The hard floor (15% below best) catches this.
from __future__ import annotations

import statistics

import numpy as np
import structlog

from src.stem.llm_client import LLMClient
from src.stem.models import LLMMessage, PromptVersion, Skill

logger = structlog.get_logger(__name__)

FORGETTING_THRESHOLD: float = 1.5
MIN_VERSIONS_FOR_CI: int = 3
IMPROVEMENT_EPSILON: float = 0.02
_BOOTSTRAP_SAMPLES: int = 1000


class PromptManager:
    def __init__(self) -> None:
        self.versions: list[PromptVersion] = []
        self.current_index: int = -1
        self._rollback_events: list[dict] = []

    def save_version(
        self,
        prompt: str,
        score: float,
        iteration: int,
        skills_locked: list[str],
    ) -> PromptVersion:
        if not prompt:
            raise ValueError("prompt must not be empty")

        improvement = score - self.versions[-1].score if self.versions else 0.0
        version = PromptVersion(
            prompt=prompt,
            score=score,
            iteration=iteration,
            skills_locked=list(skills_locked),
            improvement_over_prev=improvement,
        )
        self.versions.append(version)
        self.current_index = len(self.versions) - 1
        logger.info(
            "prompt_manager.version_saved",
            iteration=iteration,
            score=score,
            improvement=improvement,
        )
        return version

    def get_best_version(self) -> PromptVersion | None:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.score)

    def get_current_prompt(self) -> str | None:
        if not self.versions:
            return None
        return self.versions[-1].prompt

    def detect_catastrophic_forgetting(self, new_score: float) -> bool:
        if len(self.versions) < MIN_VERSIONS_FOR_CI:
            return False

        recent_scores = [v.score for v in self.versions[-3:]]
        mean = statistics.mean(recent_scores)
        std = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0

        if new_score < mean - FORGETTING_THRESHOLD * std:
            logger.warning(
                "prompt_manager.forgetting_detected.statistical",
                new_score=new_score,
                mean=mean,
                std=std,
            )
            return True

        best = self.get_best_version()
        if best is not None and new_score < best.score * 0.85:
            logger.warning(
                "prompt_manager.forgetting_detected.hard_floor",
                new_score=new_score,
                best_score=best.score,
            )
            return True

        return False

    def ewc_constrained_rollback(
        self,
        new_prompt: str,
        locked_skills: list[Skill],
        llm_client: LLMClient,
    ) -> str:
        best = self.get_best_version()
        if best is None:
            logger.warning("prompt_manager.rollback.no_best_version_found")
            return new_prompt

        locked_bullets = "\n".join(f"- {s.name}: {s.detection_pattern}" for s in locked_skills)
        merge_prompt = (
            f"Merge these two system prompts. Base prompt contains proven strategies "
            f"you must keep. New prompt contains improvements to incorporate. "
            f"You MUST preserve these behaviors:\n{locked_bullets}\n\n"
            f"BASE PROMPT:\n{best.prompt}\n\n"
            f"NEW PROMPT:\n{new_prompt}\n\n"
            f"Return only the merged system prompt text."
        )

        response = llm_client.complete(
            messages=[LLMMessage(role="user", content=merge_prompt)],
        )
        merged = response.content

        event: dict = {
            "iteration": len(self.versions),
            "reason": "catastrophic_forgetting",
            "best_score": best.score,
            "new_prompt_preview": new_prompt[:100],
        }
        self._rollback_events.append(event)
        logger.info("prompt_manager.ewc_rollback", **event)
        return merged

    def compute_improvement_ci(self) -> tuple[float, float] | None:
        if len(self.versions) < MIN_VERSIONS_FOR_CI:
            return None

        deltas = [v.improvement_over_prev for v in self.versions[-3:]]
        arr = np.array(deltas, dtype=float)
        rng = np.random.default_rng(seed=42)
        samples = rng.choice(arr, size=(_BOOTSTRAP_SAMPLES, len(arr)), replace=True)
        means = samples.mean(axis=1)
        lower = float(np.percentile(means, 2.5))
        upper = float(np.percentile(means, 97.5))
        return lower, upper

    def should_stop(self) -> tuple[bool, str]:
        best = self.get_best_version()

        if best is not None and best.score > 0.85:
            return True, "performance_threshold_reached"

        if len(self.versions) >= 5:
            ci = self.compute_improvement_ci()
            if ci is not None and ci[1] < IMPROVEMENT_EPSILON:
                return True, "diminishing_returns"

        if len(self.versions) >= 15:
            return True, "max_iterations_reached"

        return False, "continue"

    def rollback_events(self) -> list[dict]:
        return list(self._rollback_events)

    def history_summary(self) -> dict:
        best = self.get_best_version()
        return {
            "versions_count": len(self.versions),
            "best_score": best.score if best else None,
            "best_iteration": best.iteration if best else None,
            "rollback_count": len(self._rollback_events),
            "current_score": self.versions[-1].score if self.versions else None,
            "improvement_trend": [v.improvement_over_prev for v in self.versions],
        }
