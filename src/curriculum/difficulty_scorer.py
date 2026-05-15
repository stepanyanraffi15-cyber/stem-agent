from __future__ import annotations

import os
from dataclasses import dataclass

import structlog

from src.verifier.ast_verifier import analyze

logger = structlog.get_logger(__name__)


@dataclass
class ScoredFile:
    """A Python file paired with its AST-derived difficulty score."""

    file_path: str
    difficulty: float


# Difficulty score formula (from AST analysis):
#   0.3 × cyclomatic_proxy  +  0.4 × function_count_norm  +  0.3 × nesting_depth_norm
# Function count is weighted highest because more functions = more independent code paths
# for the agent to reason about simultaneously. This ordering matters for RL convergence:
# easy files (single function, flat structure) provide a stable, low-noise gradient
# signal early when the verbal gradient direction is still uncertain. Hard files produce
# noisier rewards and are only meaningful once the agent has a working baseline strategy.
def score_directory(directory: str) -> list[ScoredFile]:
    """Score all .py files in directory by difficulty, sorted easy → hard."""
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory!r}")

    py_files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".py")
    )

    if not py_files:
        logger.warning("difficulty_scorer.empty_directory", directory=directory)
        return []

    scores: list[ScoredFile] = []
    for path in py_files:
        try:
            with open(path, encoding="utf-8") as fh:
                code = fh.read()
        except OSError as exc:
            logger.warning("difficulty_scorer.read_error", path=path, error=str(exc))
            continue

        analysis = analyze(code)
        scores.append(ScoredFile(file_path=path, difficulty=analysis.difficulty_score))

    scores.sort(key=lambda s: s.difficulty)
    logger.info(
        "difficulty_scorer.scored",
        directory=directory,
        count=len(scores),
        min_difficulty=scores[0].difficulty if scores else 0.0,
        max_difficulty=scores[-1].difficulty if scores else 0.0,
    )
    return scores
