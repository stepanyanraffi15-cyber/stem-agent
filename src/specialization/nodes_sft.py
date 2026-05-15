from __future__ import annotations

import json
import os
from pathlib import Path

import structlog

from src.specialization.llm_factory import get_llm_client
from src.specialization.models import skill_library_from_dict, skill_library_to_dict
from src.specialization.state import SFTState
from src.stem.models import LLMMessage, Skill

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_BUGS_DIR = str(_PROJECT_ROOT / "data" / "training_bugs")
GROUND_TRUTH_PATH = str(_PROJECT_ROOT / "data" / "ground_truth.json")
DEMONSTRATIONS_COUNT = 20

_EXTRACT_SYSTEM = (
    "You are an expert AI researcher designing a code review agent. "
    "Analyze Python files paired with their known bugs and extract concrete, "
    "specific detection strategies. Each strategy must be actionable — not generic advice."
)

_CRITIQUE_SYSTEM = (
    "You are a constitutional AI critic. For each detection strategy, evaluate: "
    "1. Is it specific enough to be actionable? "
    "2. Does it generalize beyond training examples? "
    "3. Could it cause false positives? "
    "Revise each strategy to be maximally generalizable while remaining specific."
)

_REWRITE_SYSTEM = (
    "You are an AI agent rewriting your own system prompt. "
    "Incorporate all provided detection strategies as explicit, numbered instructions. "
    "Keep the original intent but make every instruction actionable. "
    "Return JSON: {\"system_prompt\": str, \"reasoning\": str}"
)

_SKILL_DEFAULT_CONFIDENCE = 0.5


def load_demonstrations(state: SFTState) -> dict:
    if not os.path.isdir(TRAINING_BUGS_DIR):
        raise ValueError(f"Training bugs directory not found: {TRAINING_BUGS_DIR!r}")

    files = sorted(
        os.path.join(TRAINING_BUGS_DIR, f)
        for f in os.listdir(TRAINING_BUGS_DIR)
        if f.endswith(".py")
    )[:DEMONSTRATIONS_COUNT]

    if not files:
        raise ValueError(f"No .py files found in {TRAINING_BUGS_DIR!r}")

    logger.info("sft.load_demonstrations", count=len(files))
    return {"demonstrations": files}


# Demonstration-based learning: showing the model (code, known_bug) pairs gives it
# a supervised signal it cannot derive from code alone. The key is pairing each file
# with its GT label — without the label the model might extract style observations
# instead of detection strategies. This step runs before critique because you cannot
# critique a strategy you haven't extracted yet.
def extract_patterns(state: SFTState) -> dict:
    if not state.get("demonstrations"):
        raise ValueError("demonstrations must not be empty")

    ground_truth = _load_ground_truth()
    pairs: list[str] = []
    for path in state["demonstrations"]:
        try:
            code = _read_file(path)
        except OSError:
            continue
        gt = ground_truth.get(path, {})
        pairs.append(
            f"FILE: {os.path.basename(path)}\n"
            f"BUG TYPE: {gt.get('bug_type', 'unknown')}\n"
            f"BUG LINE: {gt.get('bug_line', '?')}\n"
            f"CODE:\n{code}\n"
        )

    user_content = (
        "Here are Python files paired with their known bugs. "
        "Extract concrete, specific detection strategies you observe. "
        "Each strategy must be actionable.\n\n"
        + "\n---\n".join(pairs)
        + "\n\nReturn JSON: {\"patterns\": [{\"name\": str, \"strategy\": str, \"applies_to\": [str]}]}"
    )

    client = get_llm_client("gradient")
    messages = [
        LLMMessage(role="system", content=_EXTRACT_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]
    response = client.complete_json(messages)
    patterns = response if isinstance(response, list) else response.get("patterns", [])

    logger.info("sft.extract_patterns", count=len(patterns))
    return {"extracted_patterns": patterns}


# Constitutional AI (Bai et al. 2022, arXiv:2212.08073): self-critique separates
# memorization from generalization. The extracted patterns are shaped by the specific
# training examples — some will only work on those exact files. The critique forces
# the LLM to ask: "does this strategy survive outside the training set, and could it
# produce false positives on clean code?" Patterns that don't survive revision are
# precisely the ones that would hurt OOD performance.
def constitutional_critique(state: SFTState) -> dict:
    if not state.get("extracted_patterns"):
        raise ValueError("extracted_patterns must not be empty")

    patterns_json = json.dumps(state["extracted_patterns"], indent=2)
    user_content = (
        f"Critique and revise each detection strategy below. "
        f"Return the same JSON schema with revised strategies.\n\n{patterns_json}"
    )

    client = get_llm_client("gradient")
    messages = [
        LLMMessage(role="system", content=_CRITIQUE_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]
    response = client.complete_json(messages)
    revised = response if isinstance(response, list) else response.get("patterns", state["extracted_patterns"])

    logger.info("sft.constitutional_critique", count=len(revised))
    return {"critiqued_patterns": revised}


# Incorporate critiqued patterns into the base prompt, not replace it.
# The stem-phase base prompt encodes the agent's theory of the task (detection scope,
# false-positive control, output format). The critiqued patterns add concrete detection
# strategies on top of that theory. Replacing the base prompt would discard the theory
# and risk reverting to a generic reviewer — incorporating preserves both.
def rewrite_prompt(state: SFTState) -> dict:
    if not state.get("critiqued_patterns"):
        raise ValueError("critiqued_patterns must not be empty")

    base_prompt = state["stem_config"].get("system_prompt", "")
    patterns_text = "\n".join(
        f"{i + 1}. {p.get('name', '')}: {p.get('strategy', '')}"
        for i, p in enumerate(state["critiqued_patterns"])
    )
    user_content = (
        f"BASE SYSTEM PROMPT:\n{base_prompt}\n\n"
        f"DETECTION STRATEGIES TO INCORPORATE:\n{patterns_text}"
    )

    client = get_llm_client("gradient")
    messages = [
        LLMMessage(role="system", content=_REWRITE_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]
    response = client.complete_json(messages)

    final_prompt = response.get("system_prompt", base_prompt)
    reasoning = response.get("reasoning", "")

    logger.info("sft.rewrite_prompt", prompt_length=len(final_prompt))
    return {"final_prompt": final_prompt, "rewrite_reasoning": reasoning}


def update_skills_sft(state: SFTState) -> dict:
    if not state.get("critiqued_patterns"):
        raise ValueError("critiqued_patterns must not be empty")

    raw = state.get("skill_library") or {}
    library = skill_library_from_dict(raw)

    for pattern in state["critiqued_patterns"]:
        name = pattern.get("name", "").strip()
        if not name:
            continue
        strategy = pattern.get("strategy", "")
        applies_to: list[str] = pattern.get("applies_to", [])
        skill = Skill(
            name=name,
            description=strategy,
            detection_pattern=strategy,
            bug_types_covered=applies_to,
            confidence=_SKILL_DEFAULT_CONFIDENCE,
            locked=False,
            evidence_count=0,
        )
        library.add_skill(skill)

    logger.info("sft.update_skills", total=len(library.skills))
    return {"skill_library": skill_library_to_dict(library)}


def _load_ground_truth() -> dict[str, dict]:
    if not os.path.isfile(GROUND_TRUTH_PATH):
        return {}
    try:
        with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
            records: list[dict] = json.load(f)
        return {r["file_path"]: r for r in records if "file_path" in r}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()
