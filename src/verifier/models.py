from __future__ import annotations

from dataclasses import dataclass, field

SEVERITY_WEIGHTS: dict[str, float] = {
    "E": 1.0,
    "W": 0.6,
    "C": 0.2,
    "R": 0.1,
}

REWARD_WEIGHTS: dict[str, float] = {
    "pylint": 0.5,
    "ast": 0.3,
    "execution": 0.2,
}

BUG_TYPE_TO_PYLINT_SYMBOL: dict[str, str] = {
    "undefined_variable": "undefined-variable",
    "mutable_default_argument": "dangerous-default-value",
    "bare_except": "bare-except",
    "equality_none_check": "singleton-comparison",
    "shadowed_builtin": "redefined-builtin",
    "missing_return": "inconsistent-return-statements",
}

PYTHON_BUILTINS: frozenset[str] = frozenset(
    [
        "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
        "bytes", "callable", "chr", "classmethod", "compile", "complex",
        "copyright", "credits", "delattr", "dict", "dir", "divmod", "enumerate",
        "eval", "exec", "exit", "filter", "float", "format", "frozenset",
        "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input",
        "int", "isinstance", "issubclass", "iter", "len", "license", "list",
        "locals", "map", "max", "memoryview", "min", "next", "object", "oct",
        "open", "ord", "pow", "print", "property", "quit", "range", "repr",
        "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod",
        "str", "sum", "super", "tuple", "type", "vars", "zip",
    ]
)


@dataclass
class PylintIssue:
    """A single issue reported by pylint."""

    code: str
    line: int
    column: int
    message: str
    severity: str
    symbol: str


@dataclass
class PylintResult:
    """Aggregated result from a pylint run."""

    issues: list[PylintIssue]
    passed: bool
    raw_output: str
    exit_code: int


@dataclass
class ASTPattern:
    """A structural code pattern detected by AST analysis."""

    name: str
    line: int
    description: str


@dataclass
class ASTAnalysis:
    """Full AST-based analysis of a code snippet."""

    complexity_score: float
    function_count: int
    avg_function_length: float
    max_nesting_depth: int
    patterns_found: list[ASTPattern]
    difficulty_score: float


@dataclass
class ExecutionResult:
    """Result of running code in an isolated subprocess."""

    ran_successfully: bool
    error_type: str | None
    error_message: str | None
    traceback: str | None
    timed_out: bool


@dataclass
class GroundTruth:
    """Known bug in a file, used for precision/recall computation."""

    file_path: str
    bug_type: str
    bug_line: int
    bug_description: str
    detectable_by_pylint: bool
    detectable_by_ast: bool
    detectable_by_execution: bool


@dataclass
class VerifierOutput:
    """Combined output from all three verifiers."""

    pylint: PylintResult
    ast: ASTAnalysis
    execution: ExecutionResult
    shaped_reward: float
    detected_issues: list[str]
    decisive_error_category: str | None
