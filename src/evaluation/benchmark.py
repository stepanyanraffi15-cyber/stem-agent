from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path

import structlog
from rich.console import Console
from rich.table import Table

from src.evaluation.metrics import (
    ConditionResult,
    ExperimentResult,
    FileScore,
    calibration_ece,
    generalization_gap,
    pareto_curve,
    precision_recall_f1,
    skill_growth_curve,
)

log = structlog.get_logger(__name__)

CONSOLE = Console()

GEN_GAP_WARN_THRESHOLD = 0.1
N_TRAINING_FILES = 30
N_HELD_OUT_FILES = 20


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_condition_result(
    condition: str,
    file_scores: list[FileScore],
) -> ConditionResult:
    """Aggregate FileScore list into ConditionResult.

    Expects file_scores ordered as training files first (N_TRAINING_FILES),
    held-out files second (N_HELD_OUT_FILES).  The split is used to populate
    the in_dist_* and ood_* sub-metrics so the display table shows them
    independently rather than averaged together.
    """
    if not file_scores:
        return ConditionResult(
            condition=condition,
            file_scores=[],
            mean_precision=0.0,
            mean_recall=0.0,
            mean_f1=0.0,
            mean_reward=0.0,
            silent_failure_count=0,
            silent_failure_rate=0.0,
        )
    n = len(file_scores)
    silent = [s for s in file_scores if s.is_silent_failure]

    in_dist = file_scores[:N_TRAINING_FILES]
    ood = file_scores[N_TRAINING_FILES:N_TRAINING_FILES + N_HELD_OUT_FILES]

    return ConditionResult(
        condition=condition,
        file_scores=file_scores,
        mean_precision=_mean([s.precision for s in file_scores]),
        mean_recall=_mean([s.recall for s in file_scores]),
        mean_f1=_mean([s.f1 for s in file_scores]),
        mean_reward=_mean([s.reward for s in file_scores]),
        silent_failure_count=len(silent),
        silent_failure_rate=len(silent) / n,
        in_dist_mean_precision=_mean([s.precision for s in in_dist]),
        in_dist_mean_recall=_mean([s.recall for s in in_dist]),
        in_dist_mean_f1=_mean([s.f1 for s in in_dist]),
        ood_mean_precision=_mean([s.precision for s in ood]),
        ood_mean_recall=_mean([s.recall for s in ood]),
        ood_mean_f1=_mean([s.f1 for s in ood]),
    )


def run_full_benchmark(
    baseline_prompt: str,
    sft_final_prompt: str,
    rl_final_prompt: str,
    rl_performance_history: list[float],
    rl_skill_snapshots: list[dict],
    llm_client: object,
    output_path: str = "experiments/results/benchmark_results.json",
    baseline_file_scores: list[FileScore] | None = None,
    sft_file_scores: list[FileScore] | None = None,
    rl_file_scores: list[FileScore] | None = None,
) -> ExperimentResult:
    """Compute all metrics from pre-evaluated file scores and save to JSON.

    Accepts pre-computed FileScore lists (from evaluate_final_node) — zero LLM calls.
    Builds ExperimentResult with all comparison metrics.
    """
    baseline_scores: list[FileScore] = baseline_file_scores or []
    sft_scores: list[FileScore] = sft_file_scores or []
    rl_scores: list[FileScore] = rl_file_scores or []

    baseline_result = _build_condition_result("baseline", baseline_scores)
    sft_result = _build_condition_result("sft", sft_scores)
    rl_result = _build_condition_result("rl", rl_scores)

    in_b = baseline_scores[:N_TRAINING_FILES]
    ood_b = baseline_scores[N_TRAINING_FILES:N_TRAINING_FILES + N_HELD_OUT_FILES]
    in_s = sft_scores[:N_TRAINING_FILES]
    ood_s = sft_scores[N_TRAINING_FILES:N_TRAINING_FILES + N_HELD_OUT_FILES]
    in_r = rl_scores[:N_TRAINING_FILES]
    ood_r = rl_scores[N_TRAINING_FILES:N_TRAINING_FILES + N_HELD_OUT_FILES]

    gap_baseline = generalization_gap(in_b, ood_b)
    gap_sft = generalization_gap(in_s, ood_s)
    gap_rl = generalization_gap(in_r, ood_r)

    rl_vs_sft = rl_result.ood_mean_f1 - sft_result.ood_mean_f1

    sft_confidences = [s.reward for s in sft_scores]
    sft_correct = [s.recall > 0.0 for s in sft_scores]
    rl_confidences = [s.reward for s in rl_scores]
    rl_correct = [s.recall > 0.0 for s in rl_scores]

    ece_sft = calibration_ece(sft_confidences, sft_correct)
    ece_rl = calibration_ece(rl_confidences, rl_correct)

    pareto_rl = pareto_curve(rl_performance_history)
    pareto_sft: list[tuple[int, float]] = (
        [(0, sft_result.in_dist_mean_f1)] if sft_result.in_dist_mean_f1 else []
    )

    skill_growth = skill_growth_curve(rl_skill_snapshots)

    result = ExperimentResult(
        baseline=baseline_result,
        sft=sft_result,
        rl=rl_result,
        generalization_gap_baseline=gap_baseline,
        generalization_gap_sft=gap_sft,
        generalization_gap_rl=gap_rl,
        rl_improvement_over_sft=rl_vs_sft,
        pareto_data_rl=pareto_rl,
        pareto_data_sft=pareto_sft,
        calibration_ece_sft=ece_sft,
        calibration_ece_rl=ece_rl,
        skill_growth_rl=skill_growth,
        timestamp=datetime.utcnow().isoformat(),
    )

    save_results(result, output_path)
    log.info("benchmark.complete", output_path=output_path)
    return result


def print_comparison_table(result: ExperimentResult) -> None:
    """Print a Rich comparison table for all three conditions."""

    def _fmt(value: float) -> str:
        return f"{value:.3f}"

    def _delta(rl_val: float, sft_val: float) -> str:
        diff = rl_val - sft_val
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.3f}"

    def _rl_color(rl_val: float, sft_val: float, fmt: str) -> str:
        if rl_val > sft_val:
            return f"[bright_green]{fmt}[/bright_green]"
        return fmt

    def _sft_color(sft_val: float, base_val: float, fmt: str) -> str:
        if sft_val > base_val:
            return f"[yellow]{fmt}[/yellow]"
        return fmt

    def _gap_color(gap: float, fmt: str) -> str:
        if gap > GEN_GAP_WARN_THRESHOLD:
            return f"[bright_red]{fmt}[/bright_red]"
        return fmt

    b = result.baseline
    s = result.sft
    r = result.rl

    table = Table(title="stem-agent Experiment Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="center")
    table.add_column("SFT", justify="center")
    table.add_column("RL", justify="center")
    table.add_column("RL Δ SFT", justify="center")

    rows: list[tuple[str, float, float, float]] = [
        ("Train Precision", b.in_dist_mean_precision, s.in_dist_mean_precision, r.in_dist_mean_precision),
        ("Train Recall", b.in_dist_mean_recall, s.in_dist_mean_recall, r.in_dist_mean_recall),
        ("Train F1", b.in_dist_mean_f1, s.in_dist_mean_f1, r.in_dist_mean_f1),
        ("OOD Precision", b.ood_mean_precision, s.ood_mean_precision, r.ood_mean_precision),
        ("OOD Recall", b.ood_mean_recall, s.ood_mean_recall, r.ood_mean_recall),
        ("OOD F1", b.ood_mean_f1, s.ood_mean_f1, r.ood_mean_f1),
    ]

    for label, bv, sv, rv in rows:
        table.add_row(
            label,
            _fmt(bv),
            _sft_color(sv, bv, _fmt(sv)),
            _rl_color(rv, sv, _fmt(rv)),
            _delta(rv, sv),
        )

    gap_rows: list[tuple[str, float, float, float]] = [
        ("Generalization Gap", result.generalization_gap_baseline,
         result.generalization_gap_sft, result.generalization_gap_rl),
    ]
    for label, bv, sv, rv in gap_rows:
        table.add_row(
            label,
            _gap_color(bv, _fmt(bv)),
            _gap_color(sv, _fmt(sv)),
            _gap_color(rv, _fmt(rv)),
            _delta(rv, sv),
        )

    sfr_b = b.silent_failure_rate
    sfr_s = s.silent_failure_rate
    sfr_r = r.silent_failure_rate
    table.add_row(
        "Silent Failure Rate",
        _fmt(sfr_b),
        _sft_color(sfr_b, sfr_s, _fmt(sfr_s)),
        _rl_color(sfr_s, sfr_r, _fmt(sfr_r)),
        _delta(sfr_r, sfr_s),
    )

    table.add_row(
        "Calibration ECE",
        "—",
        _fmt(result.calibration_ece_sft),
        _rl_color(result.calibration_ece_sft, result.calibration_ece_rl,
                  _fmt(result.calibration_ece_rl)),
        _delta(result.calibration_ece_rl, result.calibration_ece_sft),
    )

    CONSOLE.print(table)


def save_results(result: ExperimentResult, path: str) -> None:
    """Serialize ExperimentResult to JSON. Creates directory if needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj: object) -> object:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return obj

    data = dataclasses.asdict(result)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("benchmark.saved", path=path)


def load_results(path: str) -> ExperimentResult:
    """Deserialize ExperimentResult from JSON."""
    with open(path, encoding="utf-8") as f:
        data: dict = json.load(f)

    def _load_file_scores(records: list[dict]) -> list[FileScore]:
        return [FileScore(**r) for r in records]

    def _load_condition(d: dict) -> ConditionResult:
        return ConditionResult(
            condition=d["condition"],
            file_scores=_load_file_scores(d["file_scores"]),
            mean_precision=d["mean_precision"],
            mean_recall=d["mean_recall"],
            mean_f1=d["mean_f1"],
            mean_reward=d["mean_reward"],
            silent_failure_count=d["silent_failure_count"],
            silent_failure_rate=d["silent_failure_rate"],
        )

    return ExperimentResult(
        baseline=_load_condition(data["baseline"]),
        sft=_load_condition(data["sft"]),
        rl=_load_condition(data["rl"]),
        generalization_gap_baseline=data["generalization_gap_baseline"],
        generalization_gap_sft=data["generalization_gap_sft"],
        generalization_gap_rl=data["generalization_gap_rl"],
        rl_improvement_over_sft=data["rl_improvement_over_sft"],
        pareto_data_rl=[tuple(p) for p in data["pareto_data_rl"]],
        pareto_data_sft=[tuple(p) for p in data["pareto_data_sft"]],
        calibration_ece_sft=data["calibration_ece_sft"],
        calibration_ece_rl=data["calibration_ece_rl"],
        skill_growth_rl=[tuple(p) for p in data["skill_growth_rl"]],
        timestamp=data["timestamp"],
    )
