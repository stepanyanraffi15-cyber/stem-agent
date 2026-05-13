from __future__ import annotations

import structlog

from src.verifier import ast_verifier, execution_verifier, pylint_verifier
from src.verifier.models import GroundTruth, VerifierOutput
from src.verifier.shaped_reward import classify_error_category, compute_reward

logger = structlog.get_logger(__name__)


def run_all_verifiers(code: str, ground_truth: GroundTruth) -> VerifierOutput:
    """Run all three verifiers, compute reward and error category, return VerifierOutput.

    This is the single entry point for the full verification pipeline.
    Fields on VerifierOutput are set exactly once at construction; no mutation after.
    """
    logger.info("pipeline.start", bug_type=ground_truth.bug_type)

    pylint_result = pylint_verifier.verify(code)
    ast_result = ast_verifier.analyze(code)
    execution_result = execution_verifier.execute(code)

    detected_issues = _build_detected_issues(pylint_result, ast_result, execution_result)

    output_partial = VerifierOutput(
        pylint=pylint_result,
        ast=ast_result,
        execution=execution_result,
        shaped_reward=0.0,
        detected_issues=detected_issues,
        decisive_error_category=None,
    )

    reward = compute_reward(output_partial, ground_truth)
    category = classify_error_category(pylint_result, execution_result, ground_truth)

    output = VerifierOutput(
        pylint=pylint_result,
        ast=ast_result,
        execution=execution_result,
        shaped_reward=reward,
        detected_issues=detected_issues,
        decisive_error_category=category,
    )

    logger.info(
        "pipeline.done",
        shaped_reward=reward,
        decisive_error_category=category,
        issue_count=len(detected_issues),
    )
    return output


def _build_detected_issues(
    pylint_result: "pylint_verifier.PylintResult",  # type: ignore[name-defined]
    ast_result: "ast_verifier.ASTAnalysis",  # type: ignore[name-defined]
    execution_result: "execution_verifier.ExecutionResult",  # type: ignore[name-defined]
) -> list[str]:
    """Build human-readable list of detected issues from all verifiers."""
    from src.verifier.models import ASTAnalysis, ExecutionResult, PylintResult

    issues: list[str] = []

    if isinstance(pylint_result, PylintResult):
        for issue in pylint_result.issues:
            issues.append(
                f"[pylint:{issue.severity}] {issue.symbol} at line {issue.line}: {issue.message}"
            )

    if isinstance(ast_result, ASTAnalysis):
        for pattern in ast_result.patterns_found:
            issues.append(
                f"[ast] {pattern.name} at line {pattern.line}: {pattern.description}"
            )

    if isinstance(execution_result, ExecutionResult):
        if not execution_result.ran_successfully:
            if execution_result.timed_out:
                issues.append("[execution] Timed out")
            elif execution_result.error_type:
                issues.append(
                    f"[execution:{execution_result.error_type}] {execution_result.error_message}"
                )

    return issues
