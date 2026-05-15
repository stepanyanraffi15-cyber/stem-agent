from __future__ import annotations

import structlog

from src.specialization.graphs import build_outer_graph
from src.specialization.state import OuterState

logger = structlog.get_logger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        checkpoint_path: str = "experiments/checkpoints.db",
        human_review: bool = False,
    ) -> None:
        self.graph = build_outer_graph(
            checkpoint_path=checkpoint_path,
            human_review=human_review,
        )
        self._checkpoint_path = checkpoint_path

    def run(
        self,
        condition: str,
        task_theory: dict,
        stem_config: dict,
        uncertainty_priors: list[dict],
        experiment_id: str,
        max_iterations: int = 15,
        initial_state: OuterState | None = None,
    ) -> dict:
        """Run one condition through the outer graph.

        initial_state allows callers to inject pre-computed fields (e.g.
        cached_baseline_eval) without duplicating LLM calls. When None, a
        default OuterState is built from the explicit keyword arguments.
        """
        if condition not in ("sft", "rl"):
            raise ValueError(f"condition must be 'sft' or 'rl', got {condition!r}")
        if not experiment_id:
            raise ValueError("experiment_id must not be empty")

        config = {
            "configurable": {"thread_id": experiment_id},
            "tags": [f"condition:{condition}", f"experiment:{experiment_id}"],
            "metadata": {
                "condition": condition,
                "experiment_id": experiment_id,
                "project": "stem-agent",
            },
        }

        if initial_state is None:
            initial_state = OuterState(
                task_theory=task_theory,
                stem_config=stem_config,
                uncertainty_priors=uncertainty_priors,
                experiment_id=experiment_id,
                condition=condition,
                final_prompt=None,
                evaluation_results=None,
                cached_baseline_eval=None,
                demonstrations=None,
                extracted_patterns=None,
                critiqued_patterns=None,
                rewrite_reasoning=None,
                skill_library=None,
                performance_history=None,
                max_iterations=max_iterations,
            )

        final_state: dict = {}
        for chunk in self.graph.stream(initial_state, config=config, stream_mode="values"):
            if isinstance(chunk, dict):
                node_name = next(iter(chunk), "update")
                logger.info(
                    "node_complete",
                    node=node_name,
                    experiment=experiment_id,
                    condition=condition,
                )
                final_state = chunk

        return final_state

    def resume(self, experiment_id: str) -> dict:
        if not experiment_id:
            raise ValueError("experiment_id must not be empty")
        config = {"configurable": {"thread_id": experiment_id}}
        existing = self.graph.get_state(config)
        if existing is None or not existing.values:
            raise ValueError(f"No checkpoint found for experiment_id: {experiment_id!r}")
        logger.info("runner.resume", experiment_id=experiment_id)
        return self.graph.invoke(None, config=config)

    def get_state_at_iteration(
        self,
        experiment_id: str,
        iteration: int,
    ) -> dict:
        if not experiment_id:
            raise ValueError("experiment_id must not be empty")
        config = {"configurable": {"thread_id": experiment_id}}
        history = list(self.graph.get_state_history(config))
        target = [
            h for h in history
            if isinstance(h.values, dict) and h.values.get("iteration") == iteration
        ]
        return target[0].values if target else {}
