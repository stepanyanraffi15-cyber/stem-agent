from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from src.verifier.models import (
    ASTAnalysis,
    ExecutionResult,
    GroundTruth,
    PylintIssue,
    PylintResult,
    VerifierOutput,
)
from src.verifier import (
    ast_verifier,
    execution_verifier,
    pylint_verifier,
)
from src.verifier.shaped_reward import classify_error_category, compute_reward
from src.verifier.pipeline import run_all_verifiers

SIMPLE_FUNCTION = """\
def add(a: int, b: int) -> int:
    return a + b
"""

NESTED_LOOPS_CODE = """\
def search_matrix(matrix: list[list[int]], target: int) -> bool:
    for row in matrix:
        for cell in row:
            if cell == target:
                if cell > 0:
                    return True
    return False
"""

LARGE_FILE_CODE = "\n".join(
    [f"x_{i} = {i}" for i in range(500)]
)

INFINITE_LOOP_CODE = """\
while True:
    pass
"""

SYNTAX_ERROR_CODE = """\
def broken(:
    pass
"""

WRONG_RETURN_CODE = """\
def swap_values(a: int, b: int) -> tuple[int, int]:
    temp = a
    a = b
    b = temp
    return a, a
"""


# ---------------------------------------------------------------------------
# PYLINT TESTS
# ---------------------------------------------------------------------------


def test_pylint_catches_undefined_variable(undefined_variable_code: str) -> None:
    """Verifier must catch E0602. Core ground truth reliability check.

    If pylint cannot catch E0602, the reward signal for the most common
    Python bug category is broken and training will diverge silently.
    """
    result = pylint_verifier.verify(undefined_variable_code)
    symbols = [i.symbol for i in result.issues]
    assert "undefined-variable" in symbols


def test_pylint_passes_clean_code(clean_python_code: str) -> None:
    """Clean code must pass. False positives break precision measurement.

    A high false-positive rate inflates the pylint_component of the shaped
    reward, making the agent believe it has found bugs when there are none.
    """
    result = pylint_verifier.verify(clean_python_code)
    assert result.passed is True
    error_or_warning = [i for i in result.issues if i.severity in ("E", "W")]
    assert error_or_warning == []


def test_pylint_returns_structured_issues(undefined_variable_code: str) -> None:
    """Issues must be PylintIssue dataclasses with all fields populated.

    Downstream reward computation reads .symbol, .severity, and .line.
    Any None field will cause a silent KeyError or AttributeError.
    """
    result = pylint_verifier.verify(undefined_variable_code)
    assert len(result.issues) > 0
    for issue in result.issues:
        assert isinstance(issue, PylintIssue)
        assert issue.code
        assert issue.line > 0
        assert issue.severity in ("E", "W", "C", "R", "I", "F")
        assert issue.symbol
        assert issue.message


def test_pylint_severity_mapping() -> None:
    """E=error, W=warning, C=convention must map correctly to severity strings.

    The shaped reward weights (1.0 / 0.6 / 0.2 / 0.1) are keyed on severity.
    A mapping failure silently zeros out reward components.
    """
    error_code = "def f():\n    return undefined_var_xyz\n"
    result = pylint_verifier.verify(error_code)
    severities = {i.severity for i in result.issues}
    assert severities <= {"E", "W", "C", "R", "I", "F"}


def test_pylint_handles_syntax_error() -> None:
    """Syntactically invalid Python must return issues, not raise exceptions.

    The agent may generate syntactically broken code during exploration.
    A raised exception here would crash the entire training loop.
    """
    result = pylint_verifier.verify(SYNTAX_ERROR_CODE)
    assert isinstance(result, PylintResult)
    assert len(result.issues) > 0


def test_pylint_handles_empty_code() -> None:
    """Empty string input must not raise, must return PylintResult.

    Empty code is a degenerate case the agent can produce during cold start.
    """
    result = pylint_verifier.verify("")
    assert isinstance(result, PylintResult)


def test_pylint_handles_very_large_file() -> None:
    """500-line file must complete within 15 seconds and not raise.

    Pylint complexity scales with file size. A pathologically large file
    must not block the training loop indefinitely.
    """
    start = time.monotonic()
    result = pylint_verifier.verify(LARGE_FILE_CODE)
    elapsed = time.monotonic() - start
    assert isinstance(result, PylintResult)
    assert elapsed < 15.0


# ---------------------------------------------------------------------------
# AST TESTS
# ---------------------------------------------------------------------------


def test_ast_detects_mutable_default(mutable_default_code: str) -> None:
    """mutable_default_argument pattern must appear in patterns_found.

    This is a key curriculum bug: common in real code, detectable by AST,
    invisible to execution verifier. If missed, ast_component = 0 for this
    bug class and the curriculum split is broken.
    """
    result = ast_verifier.analyze(mutable_default_code)
    names = [p.name for p in result.patterns_found]
    assert "mutable_default_argument" in names


def test_ast_detects_bare_except(bare_except_code: str) -> None:
    """bare_except pattern must be detected with correct line number.

    Line number accuracy is required for the ±2 secondary matching filter
    in compute_reward. Wrong line = reward mismatch.
    """
    result = ast_verifier.analyze(bare_except_code)
    bare = [p for p in result.patterns_found if p.name == "bare_except"]
    assert len(bare) >= 1
    assert bare[0].line > 0


def test_ast_complexity_range(clean_python_code: str) -> None:
    """complexity_score must always be in [0.0, 1.0].

    Out-of-range values would make difficulty_score invalid and break
    curriculum ordering — simpler code would appear harder than complex code.
    """
    result = ast_verifier.analyze(clean_python_code)
    assert 0.0 <= result.complexity_score <= 1.0


def test_ast_difficulty_score_increases_with_complexity() -> None:
    """More complex code must produce higher difficulty score (curriculum ordering).

    The curriculum builder uses difficulty_score to order training examples.
    If monotonicity fails, the agent trains on hard examples first, violating
    the Bengio et al. 2009 curriculum learning principle.
    """
    simple_result = ast_verifier.analyze(SIMPLE_FUNCTION)
    complex_result = ast_verifier.analyze(NESTED_LOOPS_CODE)
    assert complex_result.difficulty_score >= simple_result.difficulty_score


def test_ast_handles_unparseable_code() -> None:
    """Syntax errors must return empty ASTAnalysis, not raise.

    Same rationale as pylint: agent-generated broken code must not crash
    the verifier pipeline.
    """
    result = ast_verifier.analyze(SYNTAX_ERROR_CODE)
    assert isinstance(result, ASTAnalysis)
    assert result.complexity_score == 0.0
    assert result.function_count == 0
    assert result.patterns_found == []


@pytest.mark.parametrize(
    "code,expected_min_complexity",
    [
        (SIMPLE_FUNCTION, 0.0),
        (NESTED_LOOPS_CODE, 0.1),
    ],
)
def test_ast_complexity_increases_with_nesting(
    code: str, expected_min_complexity: float
) -> None:
    """Complexity proxy must increase with nesting depth.

    Ensures the per-function complexity calculation actually measures
    branching rather than trivially returning a constant.
    """
    result = ast_verifier.analyze(code)
    assert result.complexity_score >= expected_min_complexity


# ---------------------------------------------------------------------------
# EXECUTION TESTS
# ---------------------------------------------------------------------------


def test_execution_runs_clean_code(clean_python_code: str) -> None:
    """Clean code must execute successfully with ran_successfully=True.

    False negatives in execution verifier inflate execution_component for
    clean code, introducing noise into the reward signal.
    """
    result = execution_verifier.execute(clean_python_code)
    assert result.ran_successfully is True
    assert result.timed_out is False
    assert result.error_type is None


def test_execution_catches_nameerror(undefined_variable_code: str) -> None:
    """Runtime NameError must be caught and classified correctly.

    NameError is the runtime manifestation of E0602. Verifying that execution
    catches it independently validates signal redundancy in the verifier stack.
    """
    result = execution_verifier.execute(undefined_variable_code)
    assert result.ran_successfully is False
    assert result.error_type == "NameError"


def test_execution_catches_syntax_error() -> None:
    """SyntaxError must set ran_successfully=False, error_type='SyntaxError'.

    SyntaxError is caught at compile time before execution starts.
    The verifier must handle this at the subprocess level, not as a timeout.
    """
    result = execution_verifier.execute(SYNTAX_ERROR_CODE)
    assert result.ran_successfully is False
    assert result.error_type == "SyntaxError"


def test_execution_timeout() -> None:
    """Infinite loop must time out within 7 seconds and set timed_out=True.

    The agent can generate non-terminating code. Without timeout enforcement,
    a single infinite loop stalls the entire training pipeline.
    """
    start = time.monotonic()
    result = execution_verifier.execute(INFINITE_LOOP_CODE)
    elapsed = time.monotonic() - start
    assert result.timed_out is True
    assert elapsed < 7.0


def test_execution_does_not_use_eval() -> None:
    """Verify subprocess is used, not eval/exec — enforced via AST check.

    eval/exec runs untrusted code in the parent process. A single malicious
    generated snippet would compromise the training environment.
    """
    source_path = (
        Path(__file__).parent.parent.parent / "src" / "verifier" / "execution_verifier.py"
    )
    source_code = source_path.read_text()
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec"), (
                    f"execution_verifier.py uses {node.func.id}() — forbidden"
                )


# ---------------------------------------------------------------------------
# SHAPED REWARD TESTS
# ---------------------------------------------------------------------------


def test_reward_is_zero_for_undetected_bug(
    off_by_one_code: str, ground_truth_logic: GroundTruth
) -> None:
    """Logic errors pylint cannot catch + execution doesn't crash = reward 0.

    This is the key invariant: the agent must LEARN to detect silent failures,
    not rely on the verifier. If reward > 0 here, the training signal is lying.
    Maps to 'silent failure' category in IBM Research (arXiv:2511.04032).
    """
    output = run_all_verifiers(off_by_one_code, ground_truth_logic)
    assert output.shaped_reward == 0.0


def test_reward_is_positive_for_detected_bug(
    undefined_variable_code: str, ground_truth_undefined: GroundTruth
) -> None:
    """Detected bugs must produce reward > 0.

    Positive reward for caught bugs is the primary learning signal.
    If this fails, the agent gets no gradient signal for correct detection.
    """
    output = run_all_verifiers(undefined_variable_code, ground_truth_undefined)
    assert output.shaped_reward > 0.0


def test_reward_range_always_valid() -> None:
    """Reward must always be in [0.0, 1.0] across diverse inputs.

    Out-of-range rewards corrupt policy gradient estimates and cause
    training instability. This is a hard constraint, not a soft preference.
    """
    ground_truth = GroundTruth(
        file_path="<test>",
        bug_type="undefined_variable",
        bug_line=1,
        bug_description="test",
        detectable_by_pylint=True,
        detectable_by_ast=False,
        detectable_by_execution=True,
    )
    pylint_r = PylintResult(issues=[], passed=True, raw_output="", exit_code=0)
    ast_r = ASTAnalysis(
        complexity_score=0.0,
        function_count=0,
        avg_function_length=0.0,
        max_nesting_depth=0,
        patterns_found=[],
        difficulty_score=0.0,
    )
    exec_r = ExecutionResult(
        ran_successfully=True,
        error_type=None,
        error_message=None,
        traceback=None,
        timed_out=False,
    )
    output = VerifierOutput(
        pylint=pylint_r,
        ast=ast_r,
        execution=exec_r,
        shaped_reward=0.0,
        detected_issues=[],
        decisive_error_category=None,
    )
    reward = compute_reward(output, ground_truth)
    assert 0.0 <= reward <= 1.0


def test_decisive_error_category_silent_failure(
    off_by_one_code: str, ground_truth_logic: GroundTruth
) -> None:
    """Logic errors undetectable by any verifier must be classified 'silent_failure'.

    Correct categorization is required by the AgenTracer decisive error
    taxonomy (arXiv:2509.03312). Misclassification corrupts the failure
    analysis and curriculum split.
    """
    output = run_all_verifiers(off_by_one_code, ground_truth_logic)
    assert output.decisive_error_category == "silent_failure"


def test_decisive_error_category_static_error(
    undefined_variable_code: str, ground_truth_undefined: GroundTruth
) -> None:
    """Pylint-caught errors must be classified 'static_error'.

    The AgenTracer taxonomy requires correct error category to identify
    the earliest point where the failure became inevitable.
    """
    output = run_all_verifiers(undefined_variable_code, ground_truth_undefined)
    assert output.decisive_error_category == "static_error"


def test_run_all_verifiers_returns_complete_output(
    clean_python_code: str, ground_truth_undefined: GroundTruth
) -> None:
    """run_all_verifiers must return fully populated VerifierOutput.

    All fields must be set — None on shaped_reward or decisive_error_category
    signals a partial pipeline execution that would silently corrupt training.
    """
    output = run_all_verifiers(clean_python_code, ground_truth_undefined)
    assert isinstance(output, VerifierOutput)
    assert isinstance(output.pylint, PylintResult)
    assert isinstance(output.ast, ASTAnalysis)
    assert isinstance(output.execution, ExecutionResult)
    assert 0.0 <= output.shaped_reward <= 1.0
    assert isinstance(output.detected_issues, list)


# ---------------------------------------------------------------------------
# INTEGRATION TESTS
# ---------------------------------------------------------------------------


def test_verifier_pipeline_on_five_bug_types() -> None:
    """Run the full pipeline on 5 known bug types.

    Assert: shaped_reward > 0 for pylint-detectable bugs.
    Assert: shaped_reward == 0 for logic-only bugs (silent failures).
    This maps directly to our curriculum split: pylint bugs in training,
    logic bugs in held-out OOD test set.
    """
    undefined_code = "def f():\n    return missing_var\n"
    mutable_default = "def f(x: list[int] = []) -> list[int]:\n    return x\n"
    bare_except = "def f():\n    try:\n        pass\n    except:\n        pass\n"
    off_by_one = (
        "def last(items: list[int]) -> int:\n"
        "    for i in range(len(items) - 1):\n"
        "        pass\n"
        "    return items[i]\n"
    )
    wrong_return = (
        "def swap(a: int, b: int) -> tuple[int, int]:\n"
        "    temp = a\n"
        "    a = b\n"
        "    b = temp\n"
        "    return a, a\n"
    )

    detectable_cases: list[tuple[str, GroundTruth]] = [
        (
            undefined_code,
            GroundTruth(
                file_path="<test>",
                bug_type="undefined_variable",
                bug_line=2,
                bug_description="missing_var undefined",
                detectable_by_pylint=True,
                detectable_by_ast=False,
                detectable_by_execution=True,
            ),
        ),
        (
            mutable_default,
            GroundTruth(
                file_path="<test>",
                bug_type="mutable_default_argument",
                bug_line=1,
                bug_description="mutable default []",
                detectable_by_pylint=True,
                detectable_by_ast=True,
                detectable_by_execution=False,
            ),
        ),
        (
            bare_except,
            GroundTruth(
                file_path="<test>",
                bug_type="bare_except",
                bug_line=4,
                bug_description="bare except clause",
                detectable_by_pylint=True,
                detectable_by_ast=True,
                detectable_by_execution=False,
            ),
        ),
    ]

    silent_cases: list[tuple[str, GroundTruth]] = [
        (
            off_by_one,
            GroundTruth(
                file_path="<test>",
                bug_type="off_by_one",
                bug_line=2,
                bug_description="range skips last element",
                detectable_by_pylint=False,
                detectable_by_ast=False,
                detectable_by_execution=False,
            ),
        ),
        (
            wrong_return,
            GroundTruth(
                file_path="<test>",
                bug_type="wrong_return_variable",
                bug_line=5,
                bug_description="returns a twice instead of (a, b)",
                detectable_by_pylint=False,
                detectable_by_ast=False,
                detectable_by_execution=False,
            ),
        ),
    ]

    for code, gt in detectable_cases:
        output = run_all_verifiers(code, gt)
        assert output.shaped_reward > 0.0, (
            f"Expected reward > 0 for {gt.bug_type}, got {output.shaped_reward}"
        )

    for code, gt in silent_cases:
        output = run_all_verifiers(code, gt)
        assert output.shaped_reward == 0.0, (
            f"Expected reward == 0 for {gt.bug_type}, got {output.shaped_reward}"
        )
        assert output.decisive_error_category == "silent_failure", (
            f"Expected 'silent_failure' for {gt.bug_type}, "
            f"got {output.decisive_error_category}"
        )
