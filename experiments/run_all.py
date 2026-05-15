"""Main experiment entry point.

Usage:
    python -m experiments.run_all --condition [all|sft|rl|baseline]
                                  [--skip-stem]
                                  [--experiment-id ID]
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

import structlog
from rich.console import Console
from rich.panel import Panel

from src.evaluation.benchmark import (
    print_comparison_table,
    run_full_benchmark,
    save_results,
)
from src.evaluation.metrics import ConditionResult, ExperimentResult, FileScore
from src.specialization.llm_factory import get_llm_client, get_max_tokens
from src.specialization.parallel import parallel_score_files
from src.specialization.runner import ExperimentRunner
from src.stem.agent import StemAgent
from src.stem.llm_client import LLMClient

log = structlog.get_logger(__name__)

CONSOLE = Console()
RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"
GROUND_TRUTH_PATH = _PROJECT_ROOT / "data" / "ground_truth.json"
TRAINING_BUGS_DIR = _PROJECT_ROOT / "data" / "training_bugs"
HELD_OUT_DIR = _PROJECT_ROOT / "data" / "held_out_bugs"


def _load_stem_cache() -> tuple[dict, dict, list[dict]]:
    """Load cached stem phase output from disk."""
    path = RESULTS_DIR / "stem_phase_output.json"
    with open(path, encoding="utf-8") as f:
        data: dict = json.load(f)
    return data["task_theory"], data["stem_config"], data["uncertainty_priors"]


def _save_stem_output(
    task_theory: dict,
    stem_config: dict,
    uncertainty_priors: list[dict],
) -> None:
    """Persist stem phase output to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "task_theory": task_theory,
        "stem_config": stem_config,
        "uncertainty_priors": uncertainty_priors,
    }
    path = RESULTS_DIR / "stem_phase_output.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("stem.saved", path=str(path))


def _list_py_files(directory: Path) -> list[str]:
    """Return sorted list of .py file paths from directory."""
    if not directory.is_dir():
        return []
    return sorted(str(directory / f) for f in os.listdir(directory) if f.endswith(".py"))


def _load_ground_truth() -> dict[str, dict]:
    """Load ground truth JSON into a file_path -> record map."""
    if not GROUND_TRUTH_PATH.is_file():
        return {}
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    return {r["file_path"]: r for r in records if "file_path" in r}


def _score_baseline_file(
    file_path: str,
    review_system: str,
    llm_client: LLMClient,
    gt_map: dict[str, dict],
) -> dict | None:
    """Score one file for the baseline condition. Returns None if file should be skipped."""
    from src.specialization.state import AgentReview
    from src.stem.models import LLMMessage
    from src.verifier import pipeline
    from src.verifier.models import GroundTruth

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
        LLMMessage(role="system", content=review_system),
        LLMMessage(role="user", content=code),
    ]
    try:
        raw = llm_client.complete_json(messages, max_tokens=get_max_tokens("agent_review"))
    except Exception as exc:
        log.warning("baseline.llm_error", path=file_path, error=str(exc))
        return None

    verifier_out = pipeline.run_all_verifiers(code, gt)
    return {
        "file_path": file_path,
        "issues": raw.get("issues", []),
        "overall_confidence": float(raw.get("overall_confidence", 0.0)),
        "summary": raw.get("summary", ""),
        "reward": verifier_out.shaped_reward,
        "_gt": gt,
    }


def run_baseline(
    stem_config: dict,
    llm_client: LLMClient,
) -> tuple[list[FileScore], list[dict]]:
    """Run agent with baseline stem prompt on all files in parallel.

    Returns both FileScore list (for benchmark) and raw eval dicts
    (for caching in evaluate_final_node).
    """
    from src.evaluation.metrics import precision_recall_f1
    from src.verifier.models import GroundTruth

    system_prompt: str = stem_config.get("system_prompt", "")
    review_system = (
        f"{system_prompt}\n\n"
        "Review this Python file for bugs. "
        'Return ONLY valid JSON: {"issues": [{"line": int, "bug_type": str, '
        '"description": str, "confidence": float}], '
        '"overall_confidence": float, "summary": str}'
    )

    training_files = _list_py_files(TRAINING_BUGS_DIR)
    held_out_files = _list_py_files(HELD_OUT_DIR)
    all_files = training_files + held_out_files
    gt_map = _load_ground_truth()

    score_fn = functools.partial(
        _score_baseline_file,
        review_system=review_system,
        llm_client=llm_client,
        gt_map=gt_map,
    )
    raw_results = parallel_score_files(score_fn, all_files)

    agent_reviews: list[dict] = []
    ground_truths: list[GroundTruth] = []
    for r in raw_results:
        gt = r.pop("_gt")
        agent_reviews.append(r)
        ground_truths.append(gt)

    file_scores = precision_recall_f1(agent_reviews, ground_truths)

    eval_dicts = [
        {
            "file_path": r["file_path"],
            "reward": r["reward"],
            "precision": s.precision,
            "recall": s.recall,
            "f1": s.f1,
        }
        for r, s in zip(agent_reviews, file_scores)
    ]

    log.info("baseline.done", n_files=len(file_scores))
    return file_scores, eval_dicts


def _evaluate_prompt(
    prompt: str,
    label: str,
    llm_client: LLMClient,
    gt_map: dict[str, dict],
) -> list[FileScore]:
    """Evaluate a prompt against all training + held-out files.

    Returns FileScore list ordered training-first then held-out (matches
    _build_condition_result's 30/20 split assumption).  Uses the same
    bug_type_matches normalization as run_baseline.
    """
    from src.evaluation.metrics import FileScore as _FS, bug_type_matches
    from src.specialization.parallel import parallel_score_files
    from src.stem.models import LLMMessage
    from src.verifier import pipeline
    from src.verifier.models import GroundTruth

    training_files = _list_py_files(TRAINING_BUGS_DIR)
    held_out_files = _list_py_files(HELD_OUT_DIR)
    all_files = training_files + held_out_files

    review_system = (
        f"{prompt}\n\n"
        "Review this Python file for bugs. "
        'Return ONLY valid JSON: {"issues": [{"line": int, "bug_type": str, '
        '"description": str, "confidence": float}], '
        '"overall_confidence": float, "summary": str}'
    )

    def _score_file(file_path: str) -> _FS | None:
        gt_dict = gt_map.get(file_path)
        if gt_dict is None:
            return None
        try:
            with open(file_path, encoding="utf-8") as fh:
                code = fh.read()
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
            LLMMessage(role="system", content=review_system),
            LLMMessage(role="user", content=code),
        ]
        try:
            raw = llm_client.complete_json(
                messages, max_tokens=get_max_tokens("agent_review")
            )
        except Exception as exc:
            log.warning(f"{label}.llm_error", path=file_path, error=str(exc))
            return None

        verifier_out = pipeline.run_all_verifiers(code, gt)
        issues: list[dict] = raw.get("issues", [])
        tp = 1 if bug_type_matches(gt.bug_type, issues) else 0
        fp = max(0, len(issues) - tp)
        fn = 1 - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return _FS(
            file_path=file_path,
            precision=precision,
            recall=recall,
            f1=f1,
            issues_found=len(issues),
            issues_expected=1,
            false_positives=fp,
            reward=verifier_out.shaped_reward,
            is_silent_failure=(verifier_out.shaped_reward == 0.0),
        )

    results: list[_FS] = parallel_score_files(_score_file, all_files)
    log.info(f"{label}.evaluate_prompt.done", n_files=len(results))
    return results


def _extract_skill_snapshots(
    runner: ExperimentRunner,
    experiment_id: str,
) -> list[dict]:
    """Collect skill_library dicts from each checkpoint in state history."""
    try:
        config = {"configurable": {"thread_id": experiment_id}}
        history = list(runner.graph.get_state_history(config))
        seen: set[int] = set()
        snapshots: list[dict] = []
        for h in reversed(history):
            if not isinstance(h.values, dict):
                continue
            sl = h.values.get("skill_library")
            if sl is None:
                continue
            skill_count = len(sl) if isinstance(sl, dict) else 0
            if skill_count not in seen:
                seen.add(skill_count)
                snapshots.append({"total": skill_count})
        return snapshots
    except Exception as exc:
        log.warning("skill_snapshots.error", error=str(exc))
        return []


def _panel(msg: str, style: str = "bold white") -> None:
    CONSOLE.print(Panel(msg, style=style))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stem-agent specialization experiments.",
        prog="python -m experiments.run_all",
    )
    parser.add_argument(
        "--condition",
        choices=["all", "sft", "rl", "baseline"],
        default="all",
        help="Which condition(s) to run.",
    )
    parser.add_argument(
        "--skip-stem",
        action="store_true",
        help="Load cached stem phase output instead of running it fresh.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment ID (default: auto-generated timestamp+condition).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run 3 iterations only. For testing pipeline correctness.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Max RL iterations. Default 15 for full run, use 3 for testing.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Skip specialization. Load sft_final_prompt.txt and rl_final_prompt.txt "
            "and re-run evaluation only. Requires --skip-stem."
        ),
    )
    return parser.parse_args()


def _build_initial_state(
    condition: str,
    task_theory: dict,
    stem_config: dict,
    uncertainty_priors: list[dict],
    experiment_id: str,
    max_iterations: int,
    cached_baseline_eval: list[dict] | None,
) -> dict:
    """Build the OuterState dict for a single condition run."""
    return {
        "task_theory": task_theory,
        "stem_config": stem_config,
        "uncertainty_priors": uncertainty_priors,
        "experiment_id": experiment_id,
        "condition": condition,
        "final_prompt": None,
        "evaluation_results": None,
        "cached_baseline_eval": cached_baseline_eval,
        "demonstrations": None,
        "extracted_patterns": None,
        "critiqued_patterns": None,
        "rewrite_reasoning": None,
        "skill_library": None,
        "performance_history": None,
        "max_iterations": max_iterations,
    }


def _raw_dict_to_file_score(r: dict) -> FileScore:
    """Convert a per-file result dict from evaluate_final_node into a FileScore."""
    precision = float(r.get("precision", 0.0))
    recall = float(r.get("recall", 0.0))
    f1 = float(r.get("f1", 0.0))
    reward = float(r.get("reward", 0.0))
    return FileScore(
        file_path=r["file_path"],
        precision=precision,
        recall=recall,
        f1=f1,
        issues_found=1,
        issues_expected=1,
        false_positives=max(0, round((1 - precision) * (1 if precision > 0 else 0))),
        reward=reward,
        is_silent_failure=(reward == 0.0),
    )


def _scores_from_eval_results(eval_results: dict, condition: str) -> list[FileScore]:
    """Convert evaluate_final_node result dict into a flat FileScore list.

    Returns in-dist scores first, then OOD scores — preserving the split order
    that _build_condition_result relies on.
    """
    spec = eval_results.get("specialized", {})
    if not spec:
        return []
    in_dist_raw: list[dict] = spec.get("in_dist_file_scores") or []
    ood_raw: list[dict] = spec.get("ood_file_scores") or []
    return [_raw_dict_to_file_score(r) for r in in_dist_raw] + [
        _raw_dict_to_file_score(r) for r in ood_raw
    ]


def main() -> None:
    """Orchestrate the full experiment pipeline."""
    args = _parse_args()
    condition: str = args.condition
    max_iterations: int = 3 if args.fast else args.iterations
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id: str = args.experiment_id or f"{timestamp_prefix}_{condition}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _panel(f"stem-agent experiment | id={experiment_id} | condition={condition}", "bold magenta")

    if args.skip_stem:
        _panel("Loading cached stem phase output...", "white")
        task_theory, stem_config, uncertainty_priors = _load_stem_cache()
        log.info("stem.loaded_from_cache")
    else:
        _panel("Stem phase: agent self-configuration...", "bright_magenta")
        stem_llm = get_llm_client("stem_phase")
        agent = StemAgent(llm_client=stem_llm)
        theory_obj, config_obj, priors_list = agent.run_stem_phase()
        task_theory = dataclasses.asdict(theory_obj, dict_factory=lambda x: {
            k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in x
        })
        stem_config = dataclasses.asdict(config_obj, dict_factory=lambda x: {
            k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in x
        })
        uncertainty_priors = [dataclasses.asdict(p) for p in priors_list]
        _save_stem_output(task_theory, stem_config, uncertainty_priors)

    baseline_prompt: str = stem_config.get("system_prompt", "")
    review_client = get_llm_client("agent_review")

    sft_final_prompt = baseline_prompt
    rl_final_prompt = baseline_prompt
    rl_performance_history: list[float] = []
    rl_skill_snapshots: list[dict] = []
    rl_rollback_events: list[dict] = []

    baseline_file_scores: list[FileScore] = []
    sft_file_scores: list[FileScore] = []
    rl_file_scores: list[FileScore] = []

    cached_baseline_eval: list[dict] | None = None
    gt_map = _load_ground_truth()

    if args.eval_only:
        _panel("Eval-only mode: loading saved prompts, skipping specialization...", "yellow")
        sft_prompt_path = RESULTS_DIR / "sft_final_prompt.txt"
        rl_prompt_path = RESULTS_DIR / "rl_final_prompt.txt"
        if sft_prompt_path.is_file():
            sft_final_prompt = sft_prompt_path.read_text(encoding="utf-8") or baseline_prompt
        if rl_prompt_path.is_file():
            rl_rl_content = rl_prompt_path.read_text(encoding="utf-8")
            rl_final_prompt = rl_rl_content if rl_rl_content.strip() else baseline_prompt
        perf_path = RESULTS_DIR / "rl_performance_history.json"
        if perf_path.is_file():
            with open(perf_path, encoding="utf-8") as f:
                rl_performance_history = json.load(f)

        _panel("Evaluating baseline prompt...", "white")
        baseline_file_scores = _evaluate_prompt(
            baseline_prompt, "baseline", review_client, gt_map
        )
        if condition in ("sft", "all"):
            _panel("Evaluating SFT final prompt...", "cyan")
            sft_file_scores = _evaluate_prompt(
                sft_final_prompt, "sft", review_client, gt_map
            )
        if condition in ("rl", "all"):
            _panel("Evaluating RL final prompt...", "bright_green")
            rl_file_scores = _evaluate_prompt(
                rl_final_prompt, "rl", review_client, gt_map
            )
    else:
        if condition in ("baseline", "all"):
            _panel("Running baseline (no specialization)...", "white")
            baseline_file_scores, cached_baseline_eval = run_baseline(stem_config, review_client)

        runner: ExperimentRunner | None = None

        if condition in ("sft", "all"):
            _panel("Condition A (SFT): demonstration-based specialization...", "cyan")
            sft_id = f"{experiment_id}_sft"
            runner = ExperimentRunner()
            sft_initial = _build_initial_state(
                condition="sft",
                task_theory=task_theory,
                stem_config=stem_config,
                uncertainty_priors=uncertainty_priors,
                experiment_id=sft_id,
                max_iterations=max_iterations,
                cached_baseline_eval=cached_baseline_eval,
            )
            sft_state = runner.run(
                condition="sft",
                task_theory=task_theory,
                stem_config=stem_config,
                uncertainty_priors=uncertainty_priors,
                experiment_id=sft_id,
                max_iterations=max_iterations,
                initial_state=sft_initial,
            )
            sft_final_prompt = sft_state.get("final_prompt") or baseline_prompt
            (RESULTS_DIR / "sft_final_prompt.txt").write_text(sft_final_prompt, encoding="utf-8")

            eval_results = sft_state.get("evaluation_results") or {}
            sft_file_scores = _scores_from_eval_results(eval_results, "sft")
            log.info("sft.done", prompt_length=len(sft_final_prompt))

        if condition in ("rl", "all"):
            _panel("Condition B (RL): outcome-based specialization...", "bright_green")
            rl_id = f"{experiment_id}_rl"
            if runner is None:
                runner = ExperimentRunner()
            rl_initial = _build_initial_state(
                condition="rl",
                task_theory=task_theory,
                stem_config=stem_config,
                uncertainty_priors=uncertainty_priors,
                experiment_id=rl_id,
                max_iterations=max_iterations,
                cached_baseline_eval=cached_baseline_eval,
            )
            rl_state = runner.run(
                condition="rl",
                task_theory=task_theory,
                stem_config=stem_config,
                uncertainty_priors=uncertainty_priors,
                experiment_id=rl_id,
                max_iterations=max_iterations,
                initial_state=rl_initial,
            )
            rl_final_prompt = rl_state.get("final_prompt") or baseline_prompt
            rl_performance_history = rl_state.get("performance_history") or []
            rl_rollback_events = rl_state.get("rollback_events") or []
            rl_skill_snapshots = _extract_skill_snapshots(runner, rl_id)

            (RESULTS_DIR / "rl_final_prompt.txt").write_text(rl_final_prompt, encoding="utf-8")
            with open(RESULTS_DIR / "rl_performance_history.json", "w", encoding="utf-8") as f:
                json.dump(rl_performance_history, f, indent=2)
            with open(RESULTS_DIR / "rl_rollback_events.json", "w", encoding="utf-8") as f:
                json.dump(rl_rollback_events, f, indent=2)

            eval_results = rl_state.get("evaluation_results") or {}
            rl_file_scores = _scores_from_eval_results(eval_results, "rl")
            log.info("rl.done", prompt_length=len(rl_final_prompt))

    _panel("Evaluation: running benchmark...", "bright_blue")
    result = run_full_benchmark(
        baseline_prompt=baseline_prompt,
        sft_final_prompt=sft_final_prompt,
        rl_final_prompt=rl_final_prompt,
        rl_performance_history=rl_performance_history,
        rl_skill_snapshots=rl_skill_snapshots,
        llm_client=review_client,
        output_path=str(RESULTS_DIR / "benchmark_results.json"),
        baseline_file_scores=baseline_file_scores,
        sft_file_scores=sft_file_scores,
        rl_file_scores=rl_file_scores,
    )

    print_comparison_table(result)

    table_path = RESULTS_DIR / "comparison_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        from rich.console import Console as _C
        capture = _C(file=f, width=120)
        from src.evaluation.benchmark import print_comparison_table as _pct
        _pct.__globals__["CONSOLE"] = capture
        _pct(result)

    _panel(f"Complete. Results saved to {RESULTS_DIR}/", "bright_green")


if __name__ == "__main__":
    main()
