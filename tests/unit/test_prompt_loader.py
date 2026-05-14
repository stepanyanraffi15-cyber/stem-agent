from __future__ import annotations

import pytest

from src.prompts.loader import PROMPTS_DIR, load_prompt


def test_load_prompt_returns_prompt_section_only() -> None:
    """Loaded text must not contain EXPECTED_OUTPUT_SCHEMA or PAPER_CITATION."""
    text = load_prompt("stem_phase/theorize.md", task_description="test task")
    assert "EXPECTED_OUTPUT_SCHEMA" not in text
    assert "PAPER_CITATION" not in text


def test_load_prompt_formats_placeholders() -> None:
    """load_prompt fills {task_description} placeholder correctly."""
    text = load_prompt("stem_phase/theorize.md", task_description="my_task")
    assert "my_task" in text
    assert "{task_description}" not in text


def test_load_prompt_missing_file_raises() -> None:
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent/does_not_exist.md")


def test_load_prompt_missing_placeholder_raises() -> None:
    """Format with missing kwarg must raise KeyError, not silently skip."""
    with pytest.raises(KeyError):
        load_prompt("stem_phase/theorize.md")


def test_all_prompt_files_parseable() -> None:
    """Every .md file in src/prompts/ must load without error (no placeholder substitution)."""
    md_files = list(PROMPTS_DIR.rglob("*.md"))
    assert len(md_files) > 0, "No .md files found in src/prompts/"
    for md_file in md_files:
        relative = str(md_file.relative_to(PROMPTS_DIR))
        content = (PROMPTS_DIR / relative).read_text(encoding="utf-8")
        assert "PROMPT:" in content, f"{relative} missing PROMPT: section"
        assert "EXPECTED_OUTPUT_SCHEMA:" in content, f"{relative} missing EXPECTED_OUTPUT_SCHEMA: section"
        assert "PAPER_CITATION:" in content, f"{relative} missing PAPER_CITATION: section"
