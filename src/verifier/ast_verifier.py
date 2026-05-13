# AST-based static analysis — no LLM, no external tools.
# Doubles as difficulty scorer for curriculum learning (Bengio et al. 2009).
# Silent failure detection inspired by IBM Research (arXiv:2511.04032):
# structural code patterns map to "path features" in their trajectory analysis.
from __future__ import annotations

import ast
from typing import NamedTuple

import structlog

from src.verifier.models import ASTAnalysis, ASTPattern, PYTHON_BUILTINS

logger = structlog.get_logger(__name__)

COMPLEXITY_NODE_TYPES = (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)
MAX_NESTING_CAP = 5.0
MAX_FUNCTION_COUNT_CAP = 10.0


class _FunctionStats(NamedTuple):
    complexity: float
    length: int
    max_depth: int
    patterns: list[ASTPattern]


def analyze(code: str) -> ASTAnalysis:
    """Parse code with AST and return structural analysis."""
    if not code.strip():
        return _empty_analysis()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logger.warning("ast_verifier.parse_failed", error=str(exc))
        return _empty_analysis()

    function_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    function_count = len(function_nodes)

    per_function_stats: list[_FunctionStats] = [
        _analyze_function(fn) for fn in function_nodes
    ]

    if per_function_stats:
        complexity_score = max(s.complexity for s in per_function_stats)
        avg_function_length = sum(s.length for s in per_function_stats) / function_count
        max_nesting_depth = max(s.max_depth for s in per_function_stats)
        patterns_found: list[ASTPattern] = []
        for s in per_function_stats:
            patterns_found.extend(s.patterns)
    else:
        complexity_score = 0.0
        avg_function_length = 0.0
        max_nesting_depth = 0
        patterns_found = []

    module_patterns = _detect_module_patterns(tree)
    patterns_found.extend(module_patterns)

    difficulty_score = _compute_difficulty(
        complexity_score, function_count, max_nesting_depth
    )

    logger.info(
        "ast_verifier.done",
        function_count=function_count,
        complexity_score=complexity_score,
        pattern_count=len(patterns_found),
    )

    return ASTAnalysis(
        complexity_score=complexity_score,
        function_count=function_count,
        avg_function_length=avg_function_length,
        max_nesting_depth=max_nesting_depth,
        patterns_found=patterns_found,
        difficulty_score=difficulty_score,
    )


def _analyze_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _FunctionStats:
    """Compute per-function complexity, length, depth, and patterns."""
    all_nodes = list(ast.walk(node))
    total_nodes = len(all_nodes)

    complexity_count = sum(
        1 for n in all_nodes if isinstance(n, COMPLEXITY_NODE_TYPES)
    )
    raw_complexity = complexity_count / max(total_nodes / 10.0, 1.0)
    complexity = min(raw_complexity, 1.0)

    lines = set()
    for n in all_nodes:
        if hasattr(n, "lineno"):
            lines.add(n.lineno)
    length = len(lines)

    max_depth = _max_nesting_depth(node)
    patterns = _detect_function_patterns(node)

    return _FunctionStats(
        complexity=complexity,
        length=length,
        max_depth=max_depth,
        patterns=patterns,
    )


def _detect_function_patterns(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ASTPattern]:
    """Detect patterns that are scoped to a single function definition."""
    patterns: list[ASTPattern] = []
    patterns.extend(_check_mutable_default(node))
    patterns.extend(_check_missing_return(node))
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in PYTHON_BUILTINS:
            if isinstance(child.ctx, ast.Store):
                patterns.append(
                    ASTPattern(
                        name="shadowed_builtin",
                        line=child.lineno,
                        description=f"Variable '{child.id}' shadows a Python builtin",
                    )
                )
    return patterns


def _detect_module_patterns(tree: ast.Module) -> list[ASTPattern]:
    """Detect patterns that require walking the full module tree."""
    patterns: list[ASTPattern] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            patterns.append(
                ASTPattern(
                    name="bare_except",
                    line=node.lineno,
                    description="Bare except clause catches all exceptions including KeyboardInterrupt",
                )
            )
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant) and comparator.value is None:
                    patterns.append(
                        ASTPattern(
                            name="equality_none_check",
                            line=node.lineno,
                            description="Use 'is None' instead of '== None'",
                        )
                    )
    return patterns


def _check_mutable_default(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ASTPattern]:
    """Detect mutable default arguments (list or dict literals)."""
    patterns: list[ASTPattern] = []
    for default in node.args.defaults + node.args.kw_defaults:
        if default is None:
            continue
        if isinstance(default, ast.List | ast.Dict):
            patterns.append(
                ASTPattern(
                    name="mutable_default_argument",
                    line=node.lineno,
                    description=f"Function '{node.name}' has a mutable default argument",
                )
            )
    return patterns


def _check_missing_return(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ASTPattern]:
    """Detect functions with conditional returns but no guaranteed return."""
    has_conditional_return = False
    has_guaranteed_return = False

    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            has_conditional_return = True
            break

    if has_conditional_return:
        body = node.body
        last_stmt = body[-1] if body else None
        if not isinstance(last_stmt, ast.Return):
            has_guaranteed_return = False
        else:
            has_guaranteed_return = True

        if not has_guaranteed_return:
            return [
                ASTPattern(
                    name="missing_return",
                    line=node.lineno,
                    description=f"Function '{node.name}' has conditional returns but no guaranteed return",
                )
            ]
    return []


def _max_nesting_depth(node: ast.AST) -> int:
    """Compute maximum nesting depth of control flow within a node."""

    def _depth(n: ast.AST, current: int) -> int:
        max_d = current
        for child in ast.iter_child_nodes(n):
            if isinstance(child, ast.If | ast.For | ast.While | ast.Try | ast.With):
                max_d = max(max_d, _depth(child, current + 1))
            else:
                max_d = max(max_d, _depth(child, current))
        return max_d

    return _depth(node, 0)


def _compute_difficulty(
    complexity_score: float,
    function_count: int,
    max_nesting_depth: int,
) -> float:
    """Compute curriculum difficulty score from analysis components."""
    fn_component = min(function_count / MAX_FUNCTION_COUNT_CAP, 1.0)
    depth_component = min(max_nesting_depth / MAX_NESTING_CAP, 1.0)
    return 0.3 * complexity_score + 0.4 * fn_component + 0.3 * depth_component


def _empty_analysis() -> ASTAnalysis:
    """Return a zero-value ASTAnalysis for unparseable code."""
    return ASTAnalysis(
        complexity_score=0.0,
        function_count=0,
        avg_function_length=0.0,
        max_nesting_depth=0,
        patterns_found=[],
        difficulty_score=0.0,
    )
