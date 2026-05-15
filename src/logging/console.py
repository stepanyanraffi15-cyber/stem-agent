"""Rich-based structured console logging for stem-agent.

Pure presentation layer — zero imports from src/evaluation/.
All domain objects are accepted as plain dicts.
"""
from __future__ import annotations

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONSOLE = Console()

PANEL_WIDTH = 100

COLORS: dict[str, str] = {
    "stem": "bright_magenta",
    "sft": "cyan",
    "rl": "bright_green",
    "verifier": "yellow",
    "evaluation": "bright_blue",
    "error": "bright_red",
    "success": "bright_green",
    "warning": "yellow",
    "info": "white",
}


def log_node_start(
    node_name: str,
    condition: str,
    iteration: int | None = None,
) -> None:
    """Rich panel: node name, condition color, optional iteration badge."""
    color = COLORS.get(condition, COLORS["info"])
    iter_badge = f" [bold white]iter {iteration:02d}[/bold white]" if iteration is not None else ""
    label = f"[bold {color}]{node_name}[/bold {color}]{iter_badge}"
    CONSOLE.print(Panel(label, width=PANEL_WIDTH, style=color))


def log_iteration_result(
    iteration: int,
    reward: float,
    best_reward: float,
    temperatures: list[float],
    rollback_fired: bool,
    gradient_recomputed: bool,
) -> None:
    """One-line Rich output per RL iteration.

    Format: [iter 05] reward=0.721 best=0.744 temps=[0.4,0.6,0.9] 🔄grad 🔙rollback
    Color reward green if > best_reward, red if rollback_fired, yellow otherwise.
    """
    if reward > best_reward:
        reward_color = "bright_green"
    elif rollback_fired:
        reward_color = "bright_red"
    else:
        reward_color = "yellow"

    temps_str = "[" + ",".join(f"{t}" for t in temperatures) + "]"
    grad_badge = " 🔄grad" if gradient_recomputed else ""
    rollback_badge = " 🔙" if rollback_fired else ""

    line = (
        f"[bold white][iter {iteration:02d}][/bold white] "
        f"reward=[{reward_color}]{reward:.3f}[/{reward_color}] "
        f"best={best_reward:.3f} "
        f"temps={temps_str}"
        f"{grad_badge}{rollback_badge}"
    )
    CONSOLE.print(line)


def log_skill_update(
    skill_name: str,
    confidence: float,
    locked: bool,
) -> None:
    """[SKILL] skill_name confidence=0.87 [LOCKED] in skill color."""
    color = COLORS["rl"]
    lock_badge = " [bold][[LOCKED]][/bold]" if locked else ""
    CONSOLE.print(
        f"[{color}][SKILL][/{color}] {skill_name} "
        f"confidence={confidence:.2f}{lock_badge}"
    )


def log_experiment_header(
    condition: str,
    experiment_id: str,
) -> None:
    """Rich Panel with big condition name, experiment ID, timestamp."""
    from datetime import datetime

    color = COLORS.get(condition, COLORS["info"])
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    content = (
        f"[bold {color}]{condition.upper()}[/bold {color}]\n"
        f"[white]experiment_id:[/white] {experiment_id}\n"
        f"[white]timestamp:[/white]     {ts}"
    )
    CONSOLE.print(Panel(content, width=PANEL_WIDTH, style=color))


def log_test_result(
    test_name: str,
    passed: bool,
    duration_ms: float,
    detail: str | None = None,
) -> None:
    """Colorful test output for pytest.

    ✅ PASS test_name (12ms)
    ❌ FAIL test_name (8ms): detail
    """
    if passed:
        CONSOLE.print(
            f"[bright_green]✅ PASS[/bright_green] {test_name} "
            f"([white]{duration_ms:.0f}ms[/white])"
        )
    else:
        detail_str = f": {detail}" if detail else ""
        CONSOLE.print(
            f"[bright_red]❌ FAIL[/bright_red] {test_name} "
            f"([white]{duration_ms:.0f}ms[/white]){detail_str}"
        )


def log_final_summary(result: dict) -> None:
    """Full Rich layout: two panels side by side.

    Left: key metrics (F1, gap, ECE).
    Right: Pareto curve as ASCII art.
    Bottom: conclusion line with RL vs SFT improvement.

    Accepts plain dict (dataclasses.asdict output). Zero domain imports.
    """
    rl = result.get("rl", {})
    sft = result.get("sft", {})
    gap_rl = result.get("generalization_gap_rl", 0.0)
    gap_sft = result.get("generalization_gap_sft", 0.0)
    ece_rl = result.get("calibration_ece_rl", 0.0)
    ece_sft = result.get("calibration_ece_sft", 0.0)
    rl_vs_sft = result.get("rl_improvement_over_sft", 0.0)

    metrics_lines = [
        f"[bold]RL mean F1:[/bold]      {rl.get('mean_f1', 0.0):.3f}",
        f"[bold]SFT mean F1:[/bold]     {sft.get('mean_f1', 0.0):.3f}",
        f"[bold]RL gen gap:[/bold]      {gap_rl:.3f}",
        f"[bold]SFT gen gap:[/bold]     {gap_sft:.3f}",
        f"[bold]RL ECE:[/bold]          {ece_rl:.3f}",
        f"[bold]SFT ECE:[/bold]         {ece_sft:.3f}",
        f"[bold]RL Δ SFT OOD:[/bold]    {rl_vs_sft:+.3f}",
    ]
    left_content = "\n".join(metrics_lines)

    pareto_data = result.get("pareto_data_rl", [])
    right_content = render_ascii_pareto(pareto_data)

    CONSOLE.print(
        Panel(left_content, title="Key Metrics", width=PANEL_WIDTH // 2, style="bright_blue")
    )
    CONSOLE.print(
        Panel(right_content, title="RL Pareto Curve", width=PANEL_WIDTH // 2, style="bright_green")
    )
    CONSOLE.print(
        "[bold]Conclusion:[/bold] "
        f"RL improvement over SFT OOD F1: [bright_green]{rl_vs_sft:+.3f}[/bright_green]"
    )


def render_ascii_pareto(
    pareto_data: list[tuple[int, float]],
    width: int = 40,
) -> str:
    """ASCII art Pareto curve for terminal display.

    Y axis: score 0.0-1.0 (10 rows).
    X axis: iterations.
    Points marked with * connected by dashes.
    """
    if not pareto_data:
        return "(no data)"

    height = 10
    iterations = [p[0] for p in pareto_data]
    scores = [p[1] for p in pareto_data]
    n = len(pareto_data)

    x_scale = max(1, (width - 2) / max(1, max(iterations) if iterations else 1))

    grid: list[list[str]] = [[" "] * width for _ in range(height)]

    def _row(score: float) -> int:
        return max(0, min(height - 1, int((1.0 - score) * (height - 1))))

    for i, (iteration, score) in enumerate(pareto_data):
        col = min(width - 1, int(iteration * x_scale))
        row = _row(score)
        grid[row][col] = "*"

        if i + 1 < n:
            next_col = min(width - 1, int(pareto_data[i + 1][0] * x_scale))
            for c in range(col + 1, next_col):
                r_interp = _row(
                    score + (scores[i + 1] - score) * (c - col) / max(1, next_col - col)
                )
                if grid[r_interp][c] == " ":
                    grid[r_interp][c] = "-"

    y_labels = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"]
    lines = [f"{y_labels[r]}|{''.join(grid[r])}" for r in range(height)]
    lines.append("    " + "-" * width)
    lines.append("    iter →")
    return "\n".join(lines)
