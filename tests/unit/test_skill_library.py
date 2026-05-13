from __future__ import annotations

from pathlib import Path

from src.stem.models import Skill
from src.stem.skill_library import MIN_EVIDENCE_TO_LOCK, SkillLibrary


def _make_skill(
    name: str = "detect_undef",
    confidence: float = 0.5,
    locked: bool = False,
    evidence_count: int = 0,
    bug_types: list[str] | None = None,
) -> Skill:
    return Skill(
        name=name,
        description="Detect undefined variables",
        detection_pattern="check NameError",
        bug_types_covered=bug_types or ["undefined_variable"],
        confidence=confidence,
        locked=locked,
        evidence_count=evidence_count,
    )


def test_add_skill_stores_correctly() -> None:
    lib = SkillLibrary()
    skill = _make_skill("s1", confidence=0.7)
    lib.add_skill(skill)
    assert "s1" in lib.skills
    assert lib.skills["s1"].confidence == 0.7


def test_update_confidence_bayesian_increase() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("s1", confidence=0.5, evidence_count=2))
    lib.update_confidence("s1", confirmed=True)
    expected = (0.5 * 2 + 1) / (2 + 1)
    assert abs(lib.skills["s1"].confidence - expected) < 1e-9
    assert lib.skills["s1"].evidence_count == 3


def test_update_confidence_bayesian_decrease() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("s1", confidence=0.6, evidence_count=4))
    lib.update_confidence("s1", confirmed=False)
    expected = (0.6 * 4) / (4 + 1)
    assert abs(lib.skills["s1"].confidence - expected) < 1e-9
    assert lib.skills["s1"].evidence_count == 5


def test_skill_locks_at_threshold() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("s1", confidence=0.95, evidence_count=MIN_EVIDENCE_TO_LOCK - 1))
    lib.update_confidence("s1", confirmed=True)
    assert lib.skills["s1"].locked is True


def test_skill_does_not_lock_below_evidence_threshold() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("s1", confidence=0.95, evidence_count=0))
    lib.update_confidence("s1", confirmed=True)
    assert lib.skills["s1"].evidence_count == 1
    assert lib.skills["s1"].locked is False


def test_get_locked_skills_returns_only_locked() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("locked_one", locked=True))
    lib.add_skill(_make_skill("unlocked_one", locked=False))
    locked = lib.get_locked_skills()
    assert len(locked) == 1
    assert locked[0].name == "locked_one"


def test_get_skills_for_prompt_includes_all_skills() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("skill_a"))
    lib.add_skill(_make_skill("skill_b"))
    output = lib.get_skills_for_prompt()
    assert "skill_a" in output
    assert "skill_b" in output
    assert "DETECTION SKILLS:" in output


def test_get_skills_for_prompt_marks_locked() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("locked_skill", locked=True))
    lib.add_skill(_make_skill("free_skill", locked=False))
    output = lib.get_skills_for_prompt()
    assert "[LOCKED]" in output
    assert "locked_skill" in output
    assert "[LOCKED] free_skill" not in output


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("s1", confidence=0.7, bug_types=["undefined_variable"]))
    lib.add_skill(_make_skill("s2", confidence=0.4, locked=True, bug_types=["bare_except"]))

    save_path = str(tmp_path / "skills.json")
    lib.save(save_path)

    lib2 = SkillLibrary()
    lib2.load(save_path)

    assert set(lib2.skills.keys()) == {"s1", "s2"}
    assert abs(lib2.skills["s1"].confidence - 0.7) < 1e-9
    assert lib2.skills["s2"].locked is True
    assert lib2.skills["s2"].bug_types_covered == ["bare_except"]


def test_summary_counts_are_correct() -> None:
    lib = SkillLibrary()
    lib.add_skill(_make_skill("s1", locked=True, bug_types=["undefined_variable"]))
    lib.add_skill(_make_skill("s2", locked=False, bug_types=["undefined_variable", "bare_except"]))
    lib.add_skill(_make_skill("s3", locked=True, bug_types=["bare_except"]))

    summary = lib.summary()
    assert summary["total"] == 3
    assert summary["locked"] == 2
    assert summary["by_bug_type"]["undefined_variable"] == 2
    assert summary["by_bug_type"]["bare_except"] == 2
