# Skill library — discrete reusable detection strategies.
# Grounds: Voyager (arXiv:2305.16291) which builds a skill library
# of verified programs. We adapt this: skills are detection strategies,
# not executable programs.
#
# EWC connection (Kirkpatrick et al. 2017): locked skills are
# "important weights" that must be preserved during prompt updates.
# This prevents catastrophic forgetting of mastered bug types.
#
# Silent failure prevention: a skill for each failure mode from
# IBM Research (arXiv:2511.04032): drift detection, cycle detection,
# context propagation failures each get a corresponding skill.
from __future__ import annotations

import dataclasses
import json
from datetime import datetime

import structlog

from src.stem.models import Skill

logger = structlog.get_logger(__name__)

LOCK_THRESHOLD: float = 0.8
MIN_EVIDENCE_TO_LOCK: int = 3


class SkillLibrary:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def add_skill(self, skill: Skill) -> None:
        if not skill.name:
            raise ValueError("Skill name must not be empty")

        if skill.name in self.skills:
            existing = self.skills[skill.name]
            merged = Skill(
                name=existing.name,
                description=skill.description,
                detection_pattern=skill.detection_pattern,
                bug_types_covered=skill.bug_types_covered,
                confidence=max(existing.confidence, skill.confidence),
                locked=existing.locked or skill.locked,
                evidence_count=existing.evidence_count + skill.evidence_count,
                created_at=existing.created_at,
            )
            self.skills[skill.name] = merged
            logger.info("skill_library.add_skill.merged", name=skill.name)
        else:
            self.skills[skill.name] = skill
            logger.info("skill_library.add_skill.new", name=skill.name)

    def update_confidence(self, skill_name: str, confirmed: bool) -> None:
        if skill_name not in self.skills:
            raise ValueError(f"Unknown skill: {skill_name!r}")

        skill = self.skills[skill_name]
        count = skill.evidence_count
        conf = skill.confidence

        if confirmed:
            new_conf = (conf * count + 1) / (count + 1)
        else:
            new_conf = (conf * count) / (count + 1)

        new_count = count + 1
        should_lock = (
            new_conf > LOCK_THRESHOLD
            and new_count >= MIN_EVIDENCE_TO_LOCK
            and not skill.locked
        )

        self.skills[skill_name] = Skill(
            name=skill.name,
            description=skill.description,
            detection_pattern=skill.detection_pattern,
            bug_types_covered=skill.bug_types_covered,
            confidence=new_conf,
            locked=skill.locked or should_lock,
            evidence_count=new_count,
            created_at=skill.created_at,
        )

        if should_lock:
            logger.info(
                "skill_library.skill_locked",
                name=skill_name,
                confidence=f"{new_conf:.2f}",
            )

    def lock_skill(self, skill_name: str) -> None:
        if skill_name not in self.skills:
            raise ValueError(f"Unknown skill: {skill_name!r}")
        skill = self.skills[skill_name]
        self.skills[skill_name] = Skill(
            name=skill.name,
            description=skill.description,
            detection_pattern=skill.detection_pattern,
            bug_types_covered=skill.bug_types_covered,
            confidence=skill.confidence,
            locked=True,
            evidence_count=skill.evidence_count,
            created_at=skill.created_at,
        )
        logger.info("skill_library.lock_skill.forced", name=skill_name)

    def get_locked_skills(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.locked]

    def get_skills_for_prompt(self) -> str:
        if not self.skills:
            return "DETECTION SKILLS:\n(none)"

        lines = ["DETECTION SKILLS:"]
        for skill in self.skills.values():
            label = f"[LOCKED] {skill.name}" if skill.locked else skill.name
            lines.append(
                f"- {label}: {skill.detection_pattern} (confidence: {skill.confidence:.2f})"
            )
        return "\n".join(lines)

    def save(self, path: str) -> None:
        if not path:
            raise ValueError("path must not be empty")
        try:
            records = [
                {**dataclasses.asdict(s), "created_at": s.created_at.isoformat()}
                for s in self.skills.values()
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            logger.info("skill_library.saved", path=path, count=len(records))
        except OSError as exc:
            raise ValueError(f"Failed to save skill library to {path!r}: {exc}") from exc

    def load(self, path: str) -> None:
        if not path:
            raise ValueError("path must not be empty")
        try:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
        except OSError as exc:
            raise ValueError(f"Failed to read skill library from {path!r}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in skill library file {path!r}: {exc}") from exc

        self.skills = {}
        for record in records:
            created_at_raw = record.pop("created_at", None)
            created_at = (
                datetime.fromisoformat(created_at_raw)
                if created_at_raw
                else datetime.utcnow()
            )
            self.skills[record["name"]] = Skill(**record, created_at=created_at)
        logger.info("skill_library.loaded", path=path, count=len(self.skills))

    def summary(self) -> dict:
        by_bug_type: dict[str, int] = {}
        for skill in self.skills.values():
            for bt in skill.bug_types_covered:
                by_bug_type[bt] = by_bug_type.get(bt, 0) + 1
        return {
            "total": len(self.skills),
            "locked": len(self.get_locked_skills()),
            "by_bug_type": by_bug_type,
        }
