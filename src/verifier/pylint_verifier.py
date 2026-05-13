# Deterministic verifier — no LLM involved.
# Maps to "outcome-based reward signal" in Chu et al. 2025 (arXiv:2501.17161)
# and the shaped reward concept from reward shaping literature.
# Also maps to the "decisive error" annotation in AgenTracer (arXiv:2509.03312)
# where verifier output identifies what kind of failure occurred.
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import structlog

from src.verifier.models import PylintIssue, PylintResult, SEVERITY_WEIGHTS

logger = structlog.get_logger(__name__)

PYLINT_TIMEOUT_SECONDS = 10


def verify(code: str) -> PylintResult:
    """Run pylint on code string and return structured result."""
    if not code.strip():
        logger.info("pylint_verifier.empty_code")
        return PylintResult(issues=[], passed=True, raw_output="", exit_code=0)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=tempfile.gettempdir(),
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylint",
                str(tmp_path),
                "--output-format=json",
                "--score=no",
            ],
            capture_output=True,
            text=True,
            timeout=PYLINT_TIMEOUT_SECONDS,
        )
        raw_output = result.stdout + result.stderr
        issues = _parse_json_output(result.stdout, tmp_path)
        passed = not any(i.severity in ("E", "W") for i in issues)
        logger.info(
            "pylint_verifier.done",
            issue_count=len(issues),
            passed=passed,
            exit_code=result.returncode,
        )
        return PylintResult(
            issues=issues,
            passed=passed,
            raw_output=raw_output,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        logger.warning("pylint_verifier.timeout", timeout=PYLINT_TIMEOUT_SECONDS)
        return PylintResult(issues=[], passed=False, raw_output="timeout", exit_code=-1)
    except Exception as exc:
        logger.error("pylint_verifier.error", error=str(exc))
        return PylintResult(issues=[], passed=False, raw_output=str(exc), exit_code=-1)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _parse_json_output(stdout: str, tmp_path: Path) -> list[PylintIssue]:
    """Parse pylint JSON output into PylintIssue dataclasses."""
    if not stdout.strip():
        return []
    try:
        records = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("pylint_verifier.json_parse_failed", raw=stdout[:200])
        return []

    issues: list[PylintIssue] = []
    for record in records:
        msg_id: str = record.get("message-id", "")
        severity = _msg_id_to_severity(msg_id)
        issues.append(
            PylintIssue(
                code=msg_id,
                line=record.get("line", 0),
                column=record.get("column", 0),
                message=record.get("message", ""),
                severity=severity,
                symbol=record.get("symbol", ""),
            )
        )
    return issues


def _msg_id_to_severity(msg_id: str) -> str:
    """Extract severity letter from pylint message ID like 'E0602'."""
    if not msg_id:
        return "I"
    first = msg_id[0].upper()
    return first if first in SEVERITY_WEIGHTS else "I"
