"""Main experiment entry point.

Usage:
    python -m experiments.run_all --condition [all|sft|rl|baseline]
                                  [--skip-stem]
                                  [--experiment-id ID]
"""
from __future__ import annotations

import argparse
import dataclasses
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
from src.specialization.llm_factory import get_llm_client
from src.specialization.runner import ExperimentRunner
from src.stem.agent import StemAgent
from src.stem.llm_client import LLMClient

log = structlog.get_logger(__name__)

CONSOLE = Console()
RESULTS_DIR = Path("experiments/results")
GROUND_TRUTH_PATH = Path("data/ground_truth.json")
TRAINING_BUGS_DIR = Path("data/training_bugs")
HELD_OUT_DIR = Path("data/held_out_bugs")


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


def run_baseline(
    stem_config: dict,
    llm_client: LLMClient,
) -> list[FileScore]:
    """Run agent with baseline stem prompt on all files. No LangGraph."""
    from src.specialization.state import AgentReview
    from src.stem.models import LLMMessage
    from src.verifier import pipeline
    from src.verifier.models import GroundTruth
    from src.evaluation.metrics import precision_recall_f1

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

    agent_reviews: list[dict] = []
    ground_truths: list[GroundTruth] = []

    for file_path in all_files:
        gt_dict = gt_map.get(file_path)
        if gt_dict is None:
            continue
        try:
            with open(file_path, encoding="utf-8") as f:
                code = f.read()
        except OSError:
            continue

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
            raw = llm_client.complete_json(messages)
        except Exception as exc:
            log.warning("baseline.llm_error", path=file_path, error=str(exc))
            continue

        verifier_out = pipeline.run_all_verifiers(code, gt)
        agent_reviews.append({
            "file_path": file_path,
            "issues": raw.get("issues", []),
            "overall_confidence": float(raw.get("overall_confidence", 0.0)),
            "summary": raw.get("summary", ""),
            "reward": verifier_out.shaped_reward,
        })
        ground_truths.append(gt)

    return precision_recall_f1(agent_reviews, ground_truths)


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
    return parser.parse_args()


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
        _panel("🌱 Stem phase: agent self-configuration...", "bright_magenta")
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

    baseline_scores: list[FileScore] = []
    sft_file_scores: list[FileScore] = []
    rl_file_scores: list[FileScore] = []

    if condition in ("baseline", "all"):
        _panel("Running baseline (no specialization)...", "white")
        baseline_scores = run_baseline(stem_config, review_client)
        log.info("baseline.done", n_files=len(baseline_scores))

    runner: ExperimentRunner | None = None

    if condition in ("sft", "all"):
        _panel("📚 Condition A (SFT): demonstration-based specialization...", "cyan")
        sft_id = f"{experiment_id}_sft"
        runner = ExperimentRunner()
        sft_state = runner.run(
            condition="sft",
            task_theory=task_theory,
            stem_config=stem_config,
            uncertainty_priors=uncertainty_priors,
            experiment_id=sft_id,
            max_iterations=max_iterations,
        )
        sft_final_prompt = sft_state.get("final_prompt") or baseline_prompt
        (RESULTS_DIR / "sft_final_prompt.txt").write_text(sft_final_prompt, encoding="utf-8")

        eval_results = sft_state.get("evaluation_results") or {}
        sft_file_scores = _scores_from_eval_results(eval_results, "sft")
        log.info("sft.done", prompt_length=len(sft_final_prompt))

    if condition in ("rl", "all"):
        _panel("🔄 Condition B (RL): outcome-based specialization...", "bright_green")
        rl_id = f"{experiment_id}_rl"
        if runner is None:
            runner = ExperimentRunner()
        rl_state = runner.run(
            condition="rl",
            task_theory=task_theory,
            stem_config=stem_config,
            uncertainty_priors=uncertainty_priors,
            experiment_id=rl_id,
            max_iterations=max_iterations,
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

    _panel("📊 Evaluation: running benchmark...", "bright_blue")
    result = run_full_benchmark(
        baseline_prompt=baseline_prompt,
        sft_final_prompt=sft_final_prompt,
        rl_final_prompt=rl_final_prompt,
        rl_performance_history=rl_performance_history,
        rl_skill_snapshots=rl_skill_snapshots,
        llm_client=review_client,
        output_path=str(RESULTS_DIR / "benchmark_results.json"),
        baseline_file_scores=baseline_scores,
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

    _panel(f"✅ Complete. Results saved to {RESULTS_DIR}/", "bright_green")


def _scores_from_eval_results(eval_results: dict, condition: str) -> list[FileScore]:
    """Convert evaluate_final_node result dict into FileScore list."""
    from src.evaluation.metrics import FileScore
    scores: list[FileScore] = []
    spec = eval_results.get("specialized", {})
    if not spec:
        return scores
    in_f1 = float(spec.get("in_dist_f1", 0.0))
    ood_f1 = float(spec.get("ood_f1", 0.0))
    silent = int(spec.get("silent_failures", 0))
    scores.append(FileScore(
        file_path="__in_dist__",
        precision=in_f1,
        recall=in_f1,
        f1=in_f1,
        issues_found=1,
        issues_expected=1,
        false_positives=0,
        reward=in_f1,
        is_silent_failure=False,
    ))
    scores.append(FileScore(
        file_path="__ood__",
        precision=ood_f1,
        recall=ood_f1,
        f1=ood_f1,
        issues_found=1,
        issues_expected=1,
        false_positives=0,
        reward=ood_f1,
        is_silent_failure=(silent > 0),
    ))
    return scores


if __name__ == "__main__":
    main()
