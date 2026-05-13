from __future__ import annotations

import dataclasses
from datetime import datetime

from src.stem.models import PromptVersion, Skill
from src.stem.prompt_manager import PromptManager
from src.stem.skill_library import SkillLibrary


def prompt_manager_to_dict(pm: PromptManager) -> dict:
    versions = []
    for v in pm.versions:
        d = dataclasses.asdict(v)
        d["created_at"] = v.created_at.isoformat()
        versions.append(d)
    return {
        "versions": versions,
        "current_index": pm.current_index,
        "rollback_events": list(pm._rollback_events),
    }


def prompt_manager_from_dict(d: dict) -> PromptManager:
    pm = PromptManager()
    pm._rollback_events = list(d.get("rollback_events", []))
    pm.current_index = d.get("current_index", -1)
    versions: list[PromptVersion] = []
    for v in d.get("versions", []):
        raw_ts = v.pop("created_at", None)
        created_at = datetime.fromisoformat(raw_ts) if raw_ts else datetime.utcnow()
        versions.append(PromptVersion(**v, created_at=created_at))
    pm.versions = versions
    return pm


def skill_library_to_dict(sl: SkillLibrary) -> dict:
    result: dict[str, dict] = {}
    for name, skill in sl.skills.items():
        d = dataclasses.asdict(skill)
        d["created_at"] = skill.created_at.isoformat()
        result[name] = d
    return result


def skill_library_from_dict(d: dict) -> SkillLibrary:
    sl = SkillLibrary()
    for name, record in d.items():
        rec = dict(record)
        raw_ts = rec.pop("created_at", None)
        created_at = datetime.fromisoformat(raw_ts) if raw_ts else datetime.utcnow()
        sl.skills[name] = Skill(**rec, created_at=created_at)
    return sl
