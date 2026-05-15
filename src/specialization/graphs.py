from __future__ import annotations

import functools
import json
import os
import sqlite3
from pathlib import Path

import structlog
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from src.evaluation.metrics import bug_type_matches as _bug_type_matches
from src.specialization.llm_factory import get_llm_client, get_max_tokens
from src.specialization.models import prompt_manager_to_dict, skill_library_to_dict
from src.specialization.nodes_rl import (
    DEFAULT_MAX_ITERATIONS,
    TEMPERATURES_EARLY,
    check_forgetting,
    check_stop,
    compute_reward,
    curriculum_step,
    evaluate_batch,
    ewc_rollback,
    finalize_rl,
    generate_all_variants,
    lazy_gradient,
    review_code_batch,
    select_best_and_update,
)
from src.specialization.nodes_sft import (
    constitutional_critique,
    extract_patterns,
    load_demonstrations,
    rewrite_prompt,
    update_skills_sft,
)
from src.specialization.parallel import parallel_score_files
from src.specialization.state import OuterState, RLState, SFTState
from src.stem.models import LLMMessage
from src.stem.prompt_manager import PromptManager
from src.stem.skill_library import SkillLibrary
from src.verifier import pipeline
from src.verifier.models import GroundTruth

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_BUGS_DIR = str(_PROJECT_ROOT / "data" / "training_bugs")
HELD_OUT_DIR = str(_PROJECT_ROOT / "data" / "held_out_bugs")
GROUND_TRUTH_PATH = str(_PROJECT_ROOT / "data" / "ground_truth.json")

_eval_client = None


def _mean_f1(results: list[dict]) -> float:
    """Compute mean F1 across result dicts that contain an 'f1' key."""
    scores = [r["f1"] for r in results if "f1" in r]
    return sum(scores) / len(scores) if scores else 0.0


def _silent_count(results: list[dict]) -> int:
    """Count results where the reward is exactly 0.0 (silent failures)."""
    return sum(1 for r in results if r.get("reward", 0.0) == 0.0)


def _route_condition(state: OuterState) -> str:
    """Route outer graph to SFT or RL subgraph based on condition field."""
    return state["condition"]


def _prepare_rl(state: OuterState) -> dict:
    """Initialize all RL-specific state fields before entering the RL subgraph."""
    curriculum_order = _list_py_files(TRAINING_BUGS_DIR)
    return {
        "current_prompt": state["stem_config"].get("system_prompt", ""),
        "iteration": 0,
        "curriculum_index": 0,
        "curriculum_order": curriculum_order,
        "performance_history": [],
        "failure_memory": {},
        "consecutive_no_improvement": 0,
        "last_verbal_gradient": None,
        "temperatures": TEMPERATURES_EARLY,
        "variant_results": [],
        "skill_library": skill_library_to_dict(SkillLibrary()),
        "prompt_manager": prompt_manager_to_dict(PromptManager()),
        "stopping_reason": None,
        "max_iterations": state.get("max_iterations") or DEFAULT_MAX_ITERATIONS,
        "rollback_events": [],
        "recent_failures": [],
        "agent_reviews": [],
        "final_prompt": None,
    }


def _get_eval_client():
    global _eval_client
    if _eval_client is None:
        _eval_client = get_llm_client("agent_review")
    return _eval_client


def build_sft_subgraph() -> CompiledGraph:
    """Compile the SFT (demonstration-based) specialization subgraph."""
    graph: StateGraph = StateGraph(SFTState)
    graph.add_node("load_demonstrations", load_demonstrations)
    graph.add_node("extract_patterns", extract_patterns)
    graph.add_node("constitutional_critique", constitutional_critique)
    graph.add_node("rewrite_prompt", rewrite_prompt)
    graph.add_node("update_skills_sft", update_skills_sft)

    graph.add_edge(START, "load_demonstrations")
    graph.add_edge("load_demonstrations", "extract_patterns")
    graph.add_edge("extract_patterns", "constitutional_critique")
    graph.add_edge("constitutional_critique", "rewrite_prompt")
    graph.add_edge("rewrite_prompt", "update_skills_sft")
    graph.add_edge("update_skills_sft", END)

    return graph.compile()


def build_rl_subgraph() -> CompiledGraph:
    """Compile the RL (outcome-based) specialization subgraph."""
    graph: StateGraph = StateGraph(RLState)
    graph.add_node("curriculum_step", curriculum_step)
    graph.add_node("review_code_batch", review_code_batch)
    graph.add_node("compute_reward", compute_reward)
    graph.add_node("ewc_rollback", ewc_rollback)
    graph.add_node("lazy_gradient", lazy_gradient)
    graph.add_node("generate_all_variants", generate_all_variants)
    graph.add_node("evaluate_batch", evaluate_batch)
    graph.add_node("select_best_and_update", select_best_and_update)
    graph.add_node("finalize_rl", finalize_rl)

    graph.add_edge(START, "curriculum_step")
    graph.add_edge("curriculum_step", "review_code_batch")
    graph.add_edge("review_code_batch", "compute_reward")
    graph.add_conditional_edges(
        "compute_reward",
        check_forgetting,
        {"ewc_rollback": "ewc_rollback", "lazy_gradient": "lazy_gradient"},
    )
    graph.add_edge("ewc_rollback", "lazy_gradient")
    graph.add_edge("lazy_gradient", "generate_all_variants")
    graph.add_edge("generate_all_variants", "evaluate_batch")
    graph.add_edge("evaluate_batch", "select_best_and_update")
    graph.add_conditional_edges(
        "select_best_and_update",
        check_stop,
        {"continue": "curriculum_step", "end": "finalize_rl"},
    )
    graph.add_edge("finalize_rl", END)

    return graph.compile()


def build_outer_graph(
    checkpoint_path: str = "experiments/checkpoints.db",
    human_review: bool = False,
) -> CompiledGraph:
    """Compile the outer graph that routes to SFT or RL subgraph."""
    parent = os.path.dirname(checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    sft_graph = build_sft_subgraph()
    rl_graph = build_rl_subgraph()
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    graph: StateGraph = StateGraph(OuterState)
    graph.add_node("sft_subgraph", sft_graph)
    graph.add_node("prepare_rl", _prepare_rl)
    graph.add_node("rl_subgraph", rl_graph)
    graph.add_node("evaluate_final", evaluate_final_node)

    graph.add_conditional_edges(
        START,
        _route_condition,
        {"sft": "sft_subgraph", "rl": "prepare_rl"},
    )
    graph.add_edge("prepare_rl", "rl_subgraph")
    graph.add_edge("sft_subgraph", "evaluate_final")
    graph.add_edge("rl_subgraph", "evaluate_final")
    graph.add_edge("evaluate_final", END)

    interrupt: list[str] = ["sft_subgraph", "rl_subgraph"] if human_review else []
    return graph.compile(checkpointer=memory, interrupt_before=interrupt)


def evaluate_final_node(state: OuterState) -> dict:
    """Evaluate baseline vs specialised prompt on all files.

    Uses cached_baseline_eval from state when available to avoid re-running the
    same 50 baseline LLM calls that were already executed in run_baseline().
    """
    baseline_prompt = state["stem_config"].get("system_prompt", "")
    final_prompt = state.get("final_prompt") or baseline_prompt

    gt_map = _load_ground_truth()
    training_files = _list_py_files(TRAINING_BUGS_DIR)
    held_out_files = _list_py_files(HELD_OUT_DIR)
    all_files = training_files + held_out_files

    cached = state.get("cached_baseline_eval")
    if cached is not None:
        baseline_results = cached
        logger.info("evaluate_final.using_cached_baseline", count=len(baseline_results))
    else:
        baseline_results = _run_evaluation(baseline_prompt, all_files, gt_map)

    specialized_results = _run_evaluation(final_prompt, all_files, gt_map)

    in_dist_baseline = [r for r in baseline_results if r["file_path"] in set(training_files)]
    ood_baseline = [r for r in baseline_results if r["file_path"] in set(held_out_files)]
    in_dist_spec = [r for r in specialized_results if r["file_path"] in set(training_files)]
    ood_spec = [r for r in specialized_results if r["file_path"] in set(held_out_files)]

    baseline_in_f1 = _mean_f1(in_dist_baseline)
    baseline_ood_f1 = _mean_f1(ood_baseline)
    spec_in_f1 = _mean_f1(in_dist_spec)
    spec_ood_f1 = _mean_f1(ood_spec)

    perf_history: list[float] = state.get("performance_history") or []
    condition = state.get("condition", "unknown")

    if condition == "sft":
        pareto_data = [(0, baseline_in_f1), (1, spec_in_f1)]
    else:
        pareto_data = list(enumerate(perf_history))

    generalization_gap = spec_in_f1 - spec_ood_f1
    improvement = spec_ood_f1 - baseline_ood_f1

    logger.info(
        "evaluate_final",
        generalization_gap=generalization_gap,
        improvement=improvement,
        spec_ood_f1=spec_ood_f1,
        baseline_ood_f1=baseline_ood_f1,
    )
    return {
        "evaluation_results": {
            "baseline": {
                "in_dist_f1": baseline_in_f1,
                "ood_f1": baseline_ood_f1,
                "silent_failures": _silent_count(ood_baseline),
                "in_dist_file_scores": in_dist_baseline,
                "ood_file_scores": ood_baseline,
            },
            "specialized": {
                "in_dist_f1": spec_in_f1,
                "ood_f1": spec_ood_f1,
                "silent_failures": _silent_count(ood_spec),
                "in_dist_file_scores": in_dist_spec,
                "ood_file_scores": ood_spec,
            },
            "generalization_gap": generalization_gap,
            "improvement": improvement,
            "pareto_data": pareto_data,
        }
    }


def _score_one_file(
    file_path: str,
    system: str,
    gt_map: dict[str, dict],
    client: object,
) -> dict | None:
    """Score a single file against its ground truth. Returns None if file should be skipped."""
    gt_dict = gt_map.get(file_path)
    if gt_dict is None:
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            code = f.read()
    except OSError:
        return None

    gt = GroundTruth(
        file_path=gt_dict["file_path"],
        bug_type=gt_dict["bug_type"],
        bug_line=int(gt_dict["bug_line"]),
        bug_description=gt_dict.get("bug_description", ""),
        detectable_by_pylint=bool(gt_dict.get("detectable_by_pylint", False)),
        detectable_by_ast=bool(gt_dict.get("detectable_by_ast", False)),
        detectable_by_execution=bool(gt_dict.get("detectable_by_execution", False)),
    )

    messages = [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=code),
    ]
    try:
        raw = client.complete_json(
            messages,
            max_tokens=get_max_tokens("agent_review"),
        )
    except Exception as exc:
        logger.warning("evaluate_final.llm_error", path=file_path, error=str(exc))
        return None

    verifier_output = pipeline.run_all_verifiers(code, gt)
    reward = verifier_output.shaped_reward

    issues = raw.get("issues") or raw.get("findings", [])
    tp = 1 if _bug_type_matches(gt.bug_type, issues) else 0
    fp = max(0, len(issues) - tp)
    fn = 1 - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "file_path": file_path,
        "reward": reward,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _run_evaluation(
    prompt: str,
    files: list[str],
    gt_map: dict[str, dict],
) -> list[dict]:
    """Evaluate prompt on all files in parallel. Returns scored result dicts."""
    system = (
        f"{prompt}\n\n"
        "Review this Python file for bugs. "
        "Return JSON: {\"issues\": [{\"line\": int, \"bug_type\": str, \"description\": str, \"confidence\": float}], "
        "\"overall_confidence\": float, \"summary\": str}"
    )
    client = _get_eval_client()
    score_fn = functools.partial(
        _score_one_file,
        system=system,
        gt_map=gt_map,
        client=client,
    )
    return parallel_score_files(score_fn, files)


def _load_ground_truth() -> dict[str, dict]:
    """Load ground truth JSON into a file_path -> record map."""
    if not os.path.isfile(GROUND_TRUTH_PATH):
        return {}
    try:
        with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
            records: list[dict] = json.load(f)
        return {r["file_path"]: r for r in records if "file_path" in r}
    except (OSError, json.JSONDecodeError):
        return {}


def _list_py_files(directory: str) -> list[str]:
    """Return sorted list of .py file paths from a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".py")
    )
