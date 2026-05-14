from __future__ import annotations

import re
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def load_prompt(relative_path: str, **kwargs: str) -> str:
    """Load a prompt file and format with provided kwargs.

    Extracts only the PROMPT section (before EXPECTED_OUTPUT_SCHEMA).
    Only substitutes {identifier} patterns that match kwargs keys —
    leaving literal JSON braces untouched.
    Raises FileNotFoundError if the file does not exist.
    Raises KeyError if a required placeholder is missing from kwargs.
    """
    full_path = PROMPTS_DIR / relative_path
    content = full_path.read_text(encoding="utf-8")
    prompt_section = content.split("EXPECTED_OUTPUT_SCHEMA:")[0]
    prompt_section = prompt_section.replace("PROMPT:\n", "").strip()

    for key, value in kwargs.items():
        prompt_section = prompt_section.replace(f"{{{key}}}", str(value))

    remaining = _PLACEHOLDER_RE.findall(prompt_section)
    if remaining:
        raise KeyError(remaining[0])

    log.debug("prompt_loaded", path=relative_path, kwargs_keys=list(kwargs.keys()))
    return prompt_section
