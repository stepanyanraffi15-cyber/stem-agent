# Execution-based verification — catches runtime errors static analysis misses.
# Relevant to "tool failures" silent failure category in IBM Research (arXiv:2511.04032).
# Runtime errors in code review map to what AgenTracer calls "decisive errors"
# (arXiv:2509.03312) — the earliest point where the failure became inevitable.
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import structlog

from src.verifier.models import ExecutionResult

logger = structlog.get_logger(__name__)

EXECUTION_TIMEOUT_SECONDS = 5

KNOWN_ERROR_TYPES: tuple[str, ...] = (
    "SyntaxError",
    "NameError",
    "TypeError",
    "AttributeError",
    "ImportError",
    "IndexError",
    "KeyError",
    "ValueError",
    "ZeroDivisionError",
    "RuntimeError",
)

try:
    import resource as _resource_module

    _RESOURCE_AVAILABLE = True
except ImportError:
    _RESOURCE_AVAILABLE = False


def execute(code: str) -> ExecutionResult:
    """Run code in an isolated subprocess and return execution result."""
    if not code.strip():
        return ExecutionResult(
            ran_successfully=True,
            error_type=None,
            error_message=None,
            traceback=None,
            timed_out=False,
        )

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
        proc = subprocess.Popen(
            [sys.executable, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_apply_resource_limits if _RESOURCE_AVAILABLE else None,
        )
        try:
            _, stderr_bytes = proc.communicate(timeout=EXECUTION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.warning("execution_verifier.timeout", timeout=EXECUTION_TIMEOUT_SECONDS)
            return ExecutionResult(
                ran_successfully=False,
                error_type=None,
                error_message="Execution timed out",
                traceback=None,
                timed_out=True,
            )

        stderr = stderr_bytes.decode("utf-8", errors="replace")
        ran_successfully = proc.returncode == 0

        if ran_successfully:
            logger.info("execution_verifier.success")
            return ExecutionResult(
                ran_successfully=True,
                error_type=None,
                error_message=None,
                traceback=None,
                timed_out=False,
            )

        error_type = _classify_error(stderr)
        error_message = _extract_last_error_message(stderr)
        logger.info(
            "execution_verifier.error",
            error_type=error_type,
            exit_code=proc.returncode,
        )
        return ExecutionResult(
            ran_successfully=False,
            error_type=error_type,
            error_message=error_message,
            traceback=stderr,
            timed_out=False,
        )
    except Exception as exc:
        logger.error("execution_verifier.unexpected_error", error=str(exc))
        return ExecutionResult(
            ran_successfully=False,
            error_type=None,
            error_message=str(exc),
            traceback=None,
            timed_out=False,
        )
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _apply_resource_limits() -> None:
    """Apply CPU and memory limits on Linux; no-op on other platforms."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    except (ValueError, resource.error):
        pass


def _classify_error(stderr: str) -> str | None:
    """Extract the last error type from a Python traceback string."""
    last_error: str | None = None
    for line in stderr.splitlines():
        stripped = line.strip()
        for error_name in KNOWN_ERROR_TYPES:
            if stripped.startswith(error_name + ":") or stripped == error_name:
                last_error = error_name
    return last_error


def _extract_last_error_message(stderr: str) -> str | None:
    """Extract the last error line (type + message) from stderr."""
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    for line in reversed(lines):
        for error_name in KNOWN_ERROR_TYPES:
            if line.startswith(error_name):
                return line
    return lines[-1] if lines else None
