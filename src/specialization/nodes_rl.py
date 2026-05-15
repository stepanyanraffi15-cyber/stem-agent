from __future__ import annotations

import json
import os
from pathlib import Path

import structlog

from src.specialization.llm_factory import get_llm_client, get_max_tokens
from src.specialization.models import (
    prompt_manager_from_dict,
    prompt_manager_to_dict,
    skill_library_from_dict,
    skill_library_to_dict,
)
from src.specialization.state import AgentReview, RLState, VariantResult
from src.stem.models import LLMMessage, Skill

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_PATH = str(_PROJECT_ROOT / "data" / "ground_truth.json")
VALIDATION_DIR = str(_PROJECT_ROOT / "data" / "validation")
VALIDATION_BATCH_SIZE = 3
RECENT_FAILURES_MAX = 3
EARLY_EXIT_DELTA = 0.10
PLATEAU_THRESHOLD = 2
SKILL_DEFAULT_CONFIDENCE = 0.5

_GRADIENT_SYSTEM = (
    "You are a verbal gradient optimizer for LLM system prompts. "
    "Analyze prompt weaknesses from observed failures and propose precise improvements. "
    "Return JSON with keys: weakness, proposed_change, preserve, confidence."
)

_VARIANT_PROMPTS: dict[str, str] = {
    "focused": (
        "Apply this gradient precisely to the system prompt. Make only the specific change indicated.\n\n"
        "GRADIENT:\n{gradient}\n\nCURRENT PROMPT:\n{current_prompt}\n\n"
        'Return JSON: {{"system_prompt": str}}'
    ),
    "broad": (
        "Apply this gradient and generalize it to address related failure modes beyond the specific example.\n\n"
        "GRADIENT:\n{gradient}\n\nCURRENT PROMPT:\n{current_prompt}\n\n"
        'Return JSON: {{"system_prompt": str}}'
    ),
    "alternative": (
        "Take a completely different approach to the problem identified. "
        "Do not apply the gradient literally — invent an alternative solution.\n\n"
        "PROBLEM IDENTIFIED:\n{gradient}\n\nCURRENT PROMPT:\n{current_prompt}\n\n"
        'Return JSON: {{"system_prompt": str}}'
    ),
}

_REVIEW_SYSTEM_TEMPLATE = (
    "{system_prompt}\n\n"
    "Review this Python file for bugs. "
    "Return ONLY valid JSON: "
    "{{\"issues\": [{{\"line\": int, \"bug_type\": str, \"description\": str, \"confidence\": float}}], "
    "\"overall_confidence\": float, \"summary\": str}}"
)

_BATCH_REVIEW_SYSTEM_TEMPLATE = (
    "{system_prompt}\n\n"
    "Review these {count} Python files COMPLETELY INDEPENDENTLY. "
    "Do not let your review of one file influence another. "
    "Return JSON: {{\"reviews\": [/* one AgentReview per file */]}}"
)


def get_temperatures(iteration: int) -> list[float]:
    if iteration <= 5:
        return [0.6, 0.9, 1.2]
    if iteration <= 10:
        return [0.4, 0.6, 0.9]
    return [0.2, 0.4, 0.6]


def curriculum_step(state: RLState) -> dict:
    idx = state["curriculum_index"]
    order = state["curriculum_order"]
    if idx >= len(order):
        idx = 0
    temps = get_temperatures(state["iteration"])
    logger.info(
        "rl.curriculum_step",
        iteration=state["iteration"],
        curriculum_index=idx,
        file=order[idx] if order else None,
    )
    return {
        "curriculum_index": idx + 1,
        "temperatures": temps,
        "iteration": state["iteration"] + 1,
    }


def review_code_batch(state: RLState) -> dict:
    order = state["curriculum_order"]
    idx = max(0, state["curriculum_index"] - 1)
    file_path = order[idx] if order else ""

    try:
        code = _read_file(file_path)
    except OSError as exc:
        logger.warning("rl.review_code_batch.file_error", path=file_path, error=str(exc))
        code = ""

    system = _REVIEW_SYSTEM_TEMPLATE.format(system_prompt=state["current_prompt"])
    client = get_llm_client("agent_review")
    messages = [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=f"FILE: {os.path.basename(file_path)}\n\n{code}"),
    ]
    raw = client.complete_json(messages, max_tokens=get_max_tokens("agent_review"))

    review: AgentReview = {
        "file_path": file_path,
        "issues": raw.get("issues", []),
        "overall_confidence": float(raw.get("overall_confidence", 0.0)),
        "summary": raw.get("summary", ""),
    }
    logger.info("rl.review_code_batch.done", file=file_path, issues=len(review["issues"]))
    return {"agent_reviews": [review]}


from src.evaluation.metrics import bug_type_matches as _bug_type_matches


def _detection_f1(issues: list[dict], gt_bug_type: str) -> float:
    """Compute F1 of agent detection using normalized + alias-based matching."""
    tp = 1 if _bug_type_matches(gt_bug_type, issues) else 0
    fp = max(0, len(issues) - tp)
    fn = 1 - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_reward(state: RLState) -> dict:
    order = state["curriculum_order"]
    idx = max(0, state["curriculum_index"] - 1)
    file_path = order[idx] if order else ""

    ground_truth_map = _load_ground_truth()
    gt_dict = ground_truth_map.get(file_path)

    if gt_dict is None:
        logger.warning("rl.compute_reward.no_ground_truth", path=file_path)
        reward = 0.0
        bug_type = "unknown"
    else:
        bug_type = gt_dict["bug_type"]
        reviews = state.get("agent_reviews") or []
        last_review = reviews[-1] if reviews else {}
        issues: list[dict] = last_review.get("issues", [])
        reward = _detection_f1(issues, bug_type)

    history = list(state["performance_history"]) + [reward]

    failure_memory = dict(state["failure_memory"])
    recent_failures = list(state["recent_failures"])

    if reward == 0.0:
        failure_memory[bug_type] = failure_memory.get(bug_type, 0) + 1
        recent_failures.append({
            "bug_type": bug_type,
            "file_path": file_path,
            "iteration": state["iteration"],
        })
        recent_failures = recent_failures[-RECENT_FAILURES_MAX:]

    prev_best = max(history[:-1]) if len(history) > 1 else -1.0
    no_improvement = state["consecutive_no_improvement"]
    if reward <= prev_best:
        no_improvement += 1
    else:
        no_improvement = 0

    logger.info(
        "rl.compute_reward",
        reward=reward,
        bug_type=bug_type,
        no_improvement=no_improvement,
    )
    return {
        "performance_history": history,
        "failure_memory": failure_memory,
        "consecutive_no_improvement": no_improvement,
        "recent_failures": recent_failures,
    }


def check_forgetting(state: RLState) -> str:
    history = state["performance_history"]
    if not history:
        return "lazy_gradient"
    pm = prompt_manager_from_dict(state["prompt_manager"])
    latest = history[-1]
    if pm.detect_catastrophic_forgetting(latest):
        logger.info("rl.check_forgetting.ewc_rollback", score=latest)
        return "ewc_rollback"
    return "lazy_gradient"


def ewc_rollback(state: RLState) -> dict:
    pm = prompt_manager_from_dict(state["prompt_manager"])
    sl = skill_library_from_dict(state["skill_library"])
    locked = sl.get_locked_skills()
    client = get_llm_client("ewc_merge")
    merged = pm.ewc_constrained_rollback(
        new_prompt=state["current_prompt"],
        locked_skills=locked,
        llm_client=client,
    )
    events = list(state["rollback_events"]) + [{
        "iteration": state["iteration"],
        "reason": "catastrophic_forgetting",
        "merged_prompt_length": len(merged),
    }]
    logger.info("rl.ewc_rollback.done", iteration=state["iteration"])
    return {
        "current_prompt": merged,
        "rollback_events": events,
        "prompt_manager": prompt_manager_to_dict(pm),
    }


def lazy_gradient(state: RLState) -> dict:
    iteration = state["iteration"]
    no_improvement = state["consecutive_no_improvement"]
    rollback_events = state["rollback_events"]

    last_rollback_iter = rollback_events[-1]["iteration"] if rollback_events else -1
    just_rolled_back = last_rollback_iter == iteration

    should_recompute = (
        iteration <= 1
        or no_improvement >= PLATEAU_THRESHOLD
        or just_rolled_back
    )

    if not should_recompute and state.get("last_verbal_gradient") is not None:
        logger.info("rl.lazy_gradient.reuse", iteration=iteration)
        return {"last_verbal_gradient": state["last_verbal_gradient"]}

    recent = state.get("recent_failures", [])
    failure_memory = state["failure_memory"]

    user_content = (
        f"Current prompt:\n{state['current_prompt']}\n\n"
        f"Failure memory (persistent failures, weighted higher):\n{json.dumps(failure_memory)}\n\n"
        f"Recent failed reviews:\n{json.dumps(recent)}\n\n"
        "Compute a verbal gradient:\n"
        "1. What specific prompt weakness caused these failures?\n"
        "2. What exact change addresses it? Be specific, quote the prompt.\n"
        "3. What must NOT change (proven to work on these bug types)?\n"
        "Return JSON: {\"weakness\": str, \"proposed_change\": str, \"preserve\": str, \"confidence\": float}"
    )

    client = get_llm_client("gradient")
    messages = [
        LLMMessage(role="system", content=_GRADIENT_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]
    gradient = client.complete_json(messages, max_tokens=get_max_tokens("gradient"))
    logger.info("rl.lazy_gradient.recomputed", iteration=iteration, confidence=gradient.get("confidence"))
    return {"last_verbal_gradient": gradient}


def _generate_single_variant(
    state: RLState,
    variant_id: int,
    temperature: float,
    strategy: str,
) -> VariantResult:
    """Generate one prompt variant. Designed to run inside a ThreadPoolExecutor."""
    gradient = state.get("last_verbal_gradient") or {}
    template = _VARIANT_PROMPTS[strategy]
    user_content = template.format(
        gradient=json.dumps(gradient),
        current_prompt=state["current_prompt"],
    )

    client = get_llm_client("variant_gen")
    messages = [LLMMessage(role="user", content=user_content)]
    raw = client.complete(messages, temperature=temperature, max_tokens=get_max_tokens("variant_gen"))

    try:
        parsed = json.loads(raw.content)
        new_prompt = parsed.get("system_prompt", state["current_prompt"])
        if not isinstance(new_prompt, str) or not new_prompt.strip():
            logger.warning(
                "rl.generate_variant.invalid_prompt_type",
                variant_id=variant_id,
                got=type(new_prompt).__name__,
            )
            new_prompt = state["current_prompt"]
    except json.JSONDecodeError:
        new_prompt = state["current_prompt"]

    logger.info("rl.generate_variant", variant_id=variant_id, strategy=strategy)
    return VariantResult(
        variant_id=variant_id,
        prompt=new_prompt,
        temperature=temperature,
        score=0.0,
    )


def generate_all_variants(state: RLState) -> dict:
    """Generate all 3 prompt variants in parallel and return them as a single write."""
    temps = state["temperatures"]
    strategies = ["focused", "broad", "alternative"]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[VariantResult] = [None] * 3  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_idx = {
            pool.submit(_generate_single_variant, state, i, temps[i], strategies[i]): i
            for i in range(3)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.warning(
                    "rl.generate_all_variants.worker_error",
                    variant_id=idx,
                    error=str(exc),
                )
                results[idx] = VariantResult(
                    variant_id=idx,
                    prompt=state["current_prompt"],
                    temperature=temps[idx],
                    score=0.0,
                )

    logger.info("rl.generate_all_variants.done", temperatures=temps)
    return {"variant_results": [r for r in results if r is not None]}


def _score_variant(
    variant: VariantResult,
    batches: list[list[str]],
    ground_truth_map: dict[str, dict],
    current_score: float,
    best_score_ref: list[float],
) -> VariantResult:
    """Score a single variant against all validation batches.

    best_score_ref is a mutable single-element list shared across concurrent calls
    so that early-exit logic can be applied without locks (reads are safe; the
    worst case is a missed skip, not a crash).
    """
    if best_score_ref[0] >= current_score + EARLY_EXIT_DELTA:
        logger.info(
            "rl.score_variant.early_exit",
            variant_id=variant["variant_id"],
            best=best_score_ref[0],
        )
        return variant

    all_rewards: list[float] = []
    client = get_llm_client("agent_review")

    for batch in batches:
        codes_block = "\n\n---\n\n".join(
            f"FILE {j + 1}: {os.path.basename(p)}\n{_safe_read(p)}"
            for j, p in enumerate(batch)
        )
        system = _BATCH_REVIEW_SYSTEM_TEMPLATE.format(
            system_prompt=variant["prompt"],
            count=len(batch),
        )
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=codes_block),
        ]
        raw = client.complete_json(messages, max_tokens=get_max_tokens("agent_review"))
        reviews = raw.get("reviews", [])

        for k, review_dict in enumerate(reviews):
            if k >= len(batch):
                break
            file_path = batch[k]
            gt_dict = ground_truth_map.get(file_path)
            if gt_dict is None:
                continue
            issues = review_dict.get("issues", []) if isinstance(review_dict, dict) else []
            all_rewards.append(_detection_f1(issues, gt_dict["bug_type"]))

    score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    if score > best_score_ref[0]:
        best_score_ref[0] = score

    return VariantResult(
        variant_id=variant["variant_id"],
        prompt=variant["prompt"],
        temperature=variant["temperature"],
        score=score,
    )


def evaluate_batch(state: RLState) -> dict:
    """Score all 3 variants in parallel against the validation set."""
    validation_files = _load_validation_files()
    if not validation_files:
        logger.warning("rl.evaluate_batch.no_validation_files")
        return {"variant_results": state["variant_results"]}

    batches = [
        validation_files[i:i + VALIDATION_BATCH_SIZE]
        for i in range(0, len(validation_files), VALIDATION_BATCH_SIZE)
    ]
    ground_truth_map = _load_ground_truth()
    current_score = state["performance_history"][-1] if state["performance_history"] else 0.0

    best_score_ref: list[float] = [current_score]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import functools

    variants = list(state["variant_results"])
    score_fn = functools.partial(
        _score_variant,
        batches=batches,
        ground_truth_map=ground_truth_map,
        current_score=current_score,
        best_score_ref=best_score_ref,
    )

    scored_variants: list[VariantResult] = [None] * len(variants)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        future_to_idx = {pool.submit(score_fn, v): i for i, v in enumerate(variants)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                scored_variants[idx] = future.result()
            except Exception as exc:
                logger.warning(
                    "rl.evaluate_batch.worker_error",
                    variant_id=variants[idx]["variant_id"],
                    error=str(exc),
                )
                scored_variants[idx] = variants[idx]

    logger.info(
        "rl.evaluate_batch.done",
        variants=len(scored_variants),
        best_score=best_score_ref[0],
    )
    return {"variant_results": scored_variants}


def select_best_and_update(state: RLState) -> dict:
    variants = state["variant_results"]
    current_score = state["performance_history"][-1] if state["performance_history"] else 0.0

    best = max(variants, key=lambda v: v["score"]) if variants else None
    best_score = best["score"] if best else 0.0

    no_improvement = state["consecutive_no_improvement"]
    new_prompt = state["current_prompt"]

    pm = prompt_manager_from_dict(state["prompt_manager"])
    sl = skill_library_from_dict(state["skill_library"])

    if best and best_score > current_score:
        new_prompt = best["prompt"]
        no_improvement = 0
        locked_names = [s.name for s in sl.get_locked_skills()]
        pm.save_version(
            prompt=new_prompt,
            score=best_score,
            iteration=state["iteration"],
            skills_locked=locked_names,
        )
    else:
        no_improvement += 1

    max_iterations: int = state.get("max_iterations") or 15
    if state["iteration"] >= max_iterations or no_improvement >= 5:
        should_stop = True
        reason = "max_iterations_reached" if state["iteration"] >= max_iterations else "plateau"
    else:
        should_stop, reason = pm.should_stop(max_iterations=max_iterations)
    stopping_reason: str | None = reason if should_stop else None

    logger.info(
        "rl.select_best",
        best_score=best_score,
        current_score=current_score,
        stopping_reason=stopping_reason,
    )
    return {
        "current_prompt": new_prompt,
        "prompt_manager": prompt_manager_to_dict(pm),
        "skill_library": skill_library_to_dict(sl),
        "consecutive_no_improvement": no_improvement,
        "stopping_reason": stopping_reason,
    }


def check_stop(state: RLState) -> str:
    if state.get("stopping_reason") is not None:
        return "end"
    return "continue"


def finalize_rl(state: RLState) -> dict:
    prompt = state["current_prompt"]
    if not isinstance(prompt, str):
        raise TypeError(f"current_prompt must be str, got {type(prompt).__name__}")
    logger.info("rl.finalize", prompt_length=len(prompt))
    return {"final_prompt": prompt}


def _load_ground_truth() -> dict[str, dict]:
    if not os.path.isfile(GROUND_TRUTH_PATH):
        return {}
    try:
        with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
            records: list[dict] = json.load(f)
        return {r["file_path"]: r for r in records if "file_path" in r}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_validation_files() -> list[str]:
    if not os.path.isdir(VALIDATION_DIR):
        return []
    return sorted(
        os.path.join(VALIDATION_DIR, f)
        for f in os.listdir(VALIDATION_DIR)
        if f.endswith(".py")
    )


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _safe_read(path: str) -> str:
    try:
        return _read_file(path)
    except OSError:
        return ""
