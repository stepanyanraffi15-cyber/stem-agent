# Shaped reward combiner. Binary pass/fail loses information; shaped rewards
# accelerate learning (reward shaping literature). Three-signal design inspired by
# AgenTracer's multi-granular reward (arXiv:2509.03312) which uses both agent-level
# and step-level signals rather than a single binary outcome.
from __future__ import annotations

import structlog

from src.verifier.models import (
    BUG_TYPE_TO_PYLINT_SYMBOL,
    REWARD_WEIGHTS,
    SEVERITY_WEIGHTS,
    ExecutionResult,
    GroundTruth,
    PylintResult,
    VerifierOutput,
)

logger = structlog.get_logger(__name__)


def compute_reward(
    verifier_output: VerifierOutput,
    ground_truth: GroundTruth,
    weights: dict[str, float] = REWARD_WEIGHTS,
) -> float:
    """Combine verifier signals into a single shaped reward in [0.0, 1.0].

    Each component is gated on the corresponding detectable_by_* flag.
    execution_component is intentionally 0 when pylint already caught the bug
    to avoid double-counting the same failure.
    """
    pylint_component = _pylint_component(verifier_output.pylint, ground_truth)
    ast_component = _ast_component(verifier_output.ast, ground_truth)
    execution_component = _execution_component(
        verifier_output.pylint, verifier_output.execution, ground_truth
    )

    reward = (
        weights["pylint"] * pylint_component
        + weights["ast"] * ast_component
        + weights["execution"] * execution_component
    )
    reward = max(0.0, min(1.0, reward))

    logger.info(
        "shaped_reward.computed",
        pylint_component=pylint_component,
        ast_component=ast_component,
        execution_component=execution_component,
        reward=reward,
    )
    return reward


def classify_error_category(
    pylint: PylintResult,
    execution: ExecutionResult,
    ground_truth: GroundTruth,
) -> str:
    """Map verifier outputs to AgenTracer decisive error taxonomy.

    Categories (arXiv:2509.03312):
    - 'static_error': pylint E-severity caught the bug
    - 'warning_level': only pylint W-severity issues found
    - 'runtime_error': execution subprocess caught an error
    - 'silent_failure': bug exists but no verifier detected it
    - 'clean': no known bug
    """
    bug_exists = ground_truth.bug_type != ""
    pylint_symbol = BUG_TYPE_TO_PYLINT_SYMBOL.get(ground_truth.bug_type, "")

    if pylint_symbol:
        matching_e = [
            i for i in pylint.issues
            if i.symbol == pylint_symbol and i.severity == "E"
        ]
        matching_w = [
            i for i in pylint.issues
            if i.symbol == pylint_symbol and i.severity == "W"
        ]
        if matching_e:
            return "static_error"
        if matching_w:
            return "warning_level"

    if not execution.ran_successfully and not execution.timed_out and ground_truth.detectable_by_execution:
        return "runtime_error"

    if bug_exists:
        return "silent_failure"

    return "clean"


def _pylint_component(pylint: PylintResult, ground_truth: GroundTruth) -> float:
    """Recall on pylint-detectable bugs, weighted by severity."""
    if not ground_truth.detectable_by_pylint:
        return 0.0

    pylint_symbol = BUG_TYPE_TO_PYLINT_SYMBOL.get(ground_truth.bug_type, "")
    if not pylint_symbol:
        return 0.0

    matched_weight = 0.0
    for issue in pylint.issues:
        if issue.symbol == pylint_symbol:
            line_ok = abs(issue.line - ground_truth.bug_line) <= 2
            if line_ok:
                matched_weight += SEVERITY_WEIGHTS.get(issue.severity, 0.0)

    return min(matched_weight, 1.0)


def _ast_component(
    ast_analysis: "VerifierOutput.ast",  # type: ignore[name-defined]
    ground_truth: GroundTruth,
) -> float:
    """Binary: 1.0 if AST detected the bug type, 0.0 otherwise."""
    from src.verifier.models import ASTAnalysis

    if not isinstance(ast_analysis, ASTAnalysis):
        return 0.0
    if not ground_truth.detectable_by_ast:
        return 0.0
    pattern_names = [p.name for p in ast_analysis.patterns_found]
    return 1.0 if ground_truth.bug_type in pattern_names else 0.0


def _execution_component(
    pylint: PylintResult,
    execution: ExecutionResult,
    ground_truth: GroundTruth,
) -> float:
    """1.0 if execution caught the bug and pylint did not.

    Execution is gated on detectable_by_execution. Intentionally 0 when pylint
    already caught the bug to avoid double-counting.
    """
    if not ground_truth.detectable_by_execution:
        return 0.0

    pylint_symbol = BUG_TYPE_TO_PYLINT_SYMBOL.get(ground_truth.bug_type, "")
    pylint_caught = pylint_symbol and any(
        i.symbol == pylint_symbol for i in pylint.issues
    )
    if pylint_caught:
        return 0.0

    if not execution.ran_successfully and not execution.timed_out:
        return 1.0

    return 0.0
