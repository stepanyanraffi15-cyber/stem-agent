---
name: stem-agent-context
description: >-
  Project context and locked decisions for the stem-agent research codebase.
  Use when working on any file inside stem-agent/: writing new components,
  adding tests, modifying the verifier pipeline, discussing architecture,
  or onboarding to the project. Also use when the user says "update the skill",
  "record this decision", or changes an architecture or design choice.
---

# stem-agent Project Context

## SELF-UPDATE RULE

**When any decision in this file changes, update this file immediately.**
Do not wait for the user to ask. If a component is added, a formula changes,
a constant is renamed, or a design choice is reversed — edit this SKILL.md
in the same response where the change is made. Keep it the single source of
truth for the project.

---

## Research Question

Does outcome-based (RL-style) specialization generalize OOD better than
demonstration-based (SFT-style) specialization for Python code review?

Grounded in: Chu et al. 2025 (arXiv:2501.17161), TextGrad (arXiv:2406.07496),
Reflexion (arXiv:2303.11366), Voyager (arXiv:2305.16291),
AgenTracer (arXiv:2509.03312), IBM Silent Failures (arXiv:2511.04032),
KAMI benchmark (arXiv:2512.07497).

---

## What Is Built (as of last update)

```
src/verifier/
  models.py            — all dataclasses + mapping constants
  pylint_verifier.py   — subprocess + --output-format=json
  ast_verifier.py      — 5 patterns, per-function complexity
  execution_verifier.py — subprocess isolation, timeout=5s
  shaped_reward.py     — reward math only, injectable weights
  pipeline.py          — run_all_verifiers orchestration entry point

tests/
  conftest.py          — 7 fixtures (code strings + GroundTruth objects)
  unit/test_verifier.py — 26 tests, all green, 89% coverage
```

---

## Locked Architecture Decisions

### Pylint
- Always subprocess + `--output-format=json`. Never pylint Python API.
- Timeout: `PYLINT_TIMEOUT_SECONDS = 10`.
- Temp files: `tempfile.gettempdir()`.
- Parse `message-id` field for severity letter (first char: E/W/C/R).

### AST Verifier
- stdlib `ast` only — no external deps.
- Complexity: per-function, `count(If+For+While+Try+ExceptHandler) / max(total_nodes/10, 1)`, clamped to [0.0, 1.0], take max across functions.
- Difficulty: `0.3 * complexity + 0.4 * min(fn_count/10, 1.0) + 0.3 * min(max_depth/5, 1.0)`.
- Patterns detected: `mutable_default_argument`, `bare_except`, `equality_none_check`, `shadowed_builtin`, `missing_return`.

### Execution Verifier
- `subprocess.Popen` only. Never `eval()` or `exec()`.
- Timeout: `EXECUTION_TIMEOUT_SECONDS = 5`.
- Resource limits: `try/except ImportError` around `import resource`; no-op on macOS.
- Error classification: scan traceback, take **last** matching error type.

### Reward
- `compute_reward` signature: `(verifier_output, ground_truth, weights=REWARD_WEIGHTS) -> float`.
- Default weights: `{"pylint": 0.5, "ast": 0.3, "execution": 0.2}`.
- `execution_component` is 0 if pylint already caught the bug (no double-counting).
- Each component gated on `GroundTruth.detectable_by_*` flag.
- Result clamped to [0.0, 1.0].

### Pipeline
- `pipeline.run_all_verifiers(code, ground_truth) -> VerifierOutput` is the single entry point.
- Constructs `VerifierOutput` exactly once — no mutation after construction.
- `shaped_reward.py` owns math only. `pipeline.py` owns orchestration.

### Error Category Taxonomy (AgenTracer arXiv:2509.03312)
- `"static_error"` — pylint E-severity matched
- `"warning_level"` — only pylint W-severity matched
- `"runtime_error"` — execution caught error, detectable_by_execution=True
- `"silent_failure"` — bug exists, nothing caught it ← key OOD test signal
- `"clean"` — no known bug

---

## Key Data Model

```python
GroundTruth:
  file_path: str
  bug_type: str          # key into BUG_TYPE_TO_PYLINT_SYMBOL
  bug_line: int
  bug_description: str
  detectable_by_pylint: bool
  detectable_by_ast: bool
  detectable_by_execution: bool

BUG_TYPE_TO_PYLINT_SYMBOL = {
    "undefined_variable": "undefined-variable",
    "mutable_default_argument": "dangerous-default-value",
    "bare_except": "bare-except",
    "equality_none_check": "singleton-comparison",
    "shadowed_builtin": "redefined-builtin",
    "missing_return": "inconsistent-return-statements",
}
```

Ground truth matching predicate: `pylint_issue.symbol == BUG_TYPE_TO_PYLINT_SYMBOL[bug_type]`
with `abs(issue.line - bug_line) <= 2` as secondary filter.

---

## Code Conventions (Non-Negotiable)

- No comments inside code blocks.
- No `print()` anywhere — `structlog.get_logger(__name__)` at module level.
- Strict typing: all params and return types explicit.
- No `Optional` without defaults; no bare `dict` returns.
- Constants at top of each file — no magic numbers in function bodies.
- All subprocess calls wrapped in `try/except` with timeout.
- All `ast.parse()` calls wrapped in `try/except SyntaxError`.
- Inputs validated fail-fast at function entry.
- No mutation of inputs or globals.

---

## Curriculum Split (for downstream RL training)

- **Training set**: bugs detectable by pylint/AST (`detectable_by_pylint=True` or `detectable_by_ast=True`)
- **OOD test set**: logic-only bugs (`detectable_by_pylint=False`, `detectable_by_ast=False`, `detectable_by_execution=False`)
- Silent failures must produce `shaped_reward == 0.0` — agent must learn to detect them, not rely on the verifier.

---

## Open Questions (unresolved)

- Should `silent_failure` emit a small negative reward (-0.1) in the RL loop instead of 0.0?
- Should reward weights be per-bug-type or global?
- Should execution verifier use a Docker container or process pool for higher-throughput training?
