from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.verifier.models import GroundTruth

MATCH_LINE_TOLERANCE = 2
CALIBRATION_BINS = 10
BOOTSTRAP_SAMPLES = 1000


@dataclass
class FileScore:
    """Per-file precision/recall/F1 and metadata."""

    file_path: str
    precision: float
    recall: float
    f1: float
    issues_found: int
    issues_expected: int
    false_positives: int
    reward: float
    is_silent_failure: bool


@dataclass
class ConditionResult:
    """Aggregated metrics for one experimental condition."""

    condition: str
    file_scores: list[FileScore]
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_reward: float
    silent_failure_count: int
    silent_failure_rate: float


@dataclass
class ExperimentResult:
    """Full cross-condition comparison with all research metrics."""

    baseline: ConditionResult
    sft: ConditionResult
    rl: ConditionResult
    generalization_gap_baseline: float
    generalization_gap_sft: float
    generalization_gap_rl: float
    rl_improvement_over_sft: float
    pareto_data_rl: list[tuple[int, float]]
    pareto_data_sft: list[tuple[int, float]]
    calibration_ece_sft: float
    calibration_ece_rl: float
    skill_growth_rl: list[tuple[int, int]]
    timestamp: str


def precision_recall_f1(
    agent_reviews: list[dict],
    ground_truths: list[GroundTruth],
) -> list[FileScore]:
    """Match agent issues to ground truth by bug_type and line proximity.

    TP: bug_type match AND abs(issue_line - gt_line) <= MATCH_LINE_TOLERANCE.
    FP: agent issue with no matching ground truth.
    FN: ground truth bug matched by no agent issue.
    """
    gt_by_path: dict[str, GroundTruth] = {gt.file_path: gt for gt in ground_truths}
    scores: list[FileScore] = []

    for review in agent_reviews:
        file_path: str = review["file_path"]
        issues: list[dict] = review["issues"]
        reward: float = float(review.get("reward", 0.0))

        gt = gt_by_path.get(file_path)
        if gt is None:
            continue

        tp = 0
        for issue in issues:
            issue_line: int = int(issue["line"])
            issue_bug_type: str = issue["bug_type"]
            if (
                issue_bug_type == gt.bug_type
                and abs(issue_line - gt.bug_line) <= MATCH_LINE_TOLERANCE
            ):
                tp = 1
                break

        fp = max(0, len(issues) - tp)
        fn = 1 - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall + 1e-9)
        )

        scores.append(
            FileScore(
                file_path=file_path,
                precision=precision,
                recall=recall,
                f1=f1,
                issues_found=len(issues),
                issues_expected=1,
                false_positives=fp,
                reward=reward,
                is_silent_failure=(reward == 0.0),
            )
        )

    return scores


def generalization_gap(
    in_dist_scores: list[FileScore],
    ood_scores: list[FileScore],
) -> float:
    """Compute mean(in_dist F1) - mean(OOD F1). Positive = degradation on OOD."""
    if not in_dist_scores or not ood_scores:
        return 0.0
    in_mean = sum(s.f1 for s in in_dist_scores) / len(in_dist_scores)
    ood_mean = sum(s.f1 for s in ood_scores) / len(ood_scores)
    return in_mean - ood_mean


def calibration_ece(
    confidence_scores: list[float],
    correct_flags: list[bool],
) -> float:
    """Expected Calibration Error using CALIBRATION_BINS decile buckets.

    ECE = sum(|bucket_accuracy - bucket_mean_confidence| * bucket_weight).
    Lower is better; 0.0 = perfectly calibrated.
    """
    if not confidence_scores or not correct_flags:
        return 0.0
    if len(confidence_scores) != len(correct_flags):
        raise ValueError("confidence_scores and correct_flags must have equal length")

    n = len(confidence_scores)
    bin_edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    ece = 0.0

    for i in range(CALIBRATION_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        indices = [
            j for j, c in enumerate(confidence_scores)
            if lo <= c <= hi
        ]
        if not indices:
            continue
        bucket_conf = sum(confidence_scores[j] for j in indices) / len(indices)
        bucket_acc = sum(1 for j in indices if correct_flags[j]) / len(indices)
        weight = len(indices) / n
        ece += abs(bucket_acc - bucket_conf) * weight

    return float(ece)


def pareto_curve(
    performance_history: list[float],
) -> list[tuple[int, float]]:
    """Convert performance_history to (iteration, score) Pareto points."""
    return list(enumerate(performance_history))


def skill_growth_curve(
    skill_snapshots: list[dict],
) -> list[tuple[int, int]]:
    """Return (iteration, total_skill_count) from SkillLibrary.summary() snapshots."""
    return [(i, int(snap["total"])) for i, snap in enumerate(skill_snapshots)]


def bootstrap_ci(
    values: list[float],
    n_samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval. Returns (lower, upper)."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    means = np.array([
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_samples)
    ])
    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lower, upper)
