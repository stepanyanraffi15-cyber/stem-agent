from __future__ import annotations

from src.specialization.nodes_rl import (
    check_forgetting,
    check_stop,
    compute_reward,
    finalize_rl,
    get_temperatures,
    lazy_gradient,
    route_to_variants,
)
from src.specialization.state import RLState, VariantResult


def _base_rl_state(**overrides) -> RLState:
    pm_dict = {
        "versions": [],
        "current_index": -1,
        "rollback_events": [],
    }
    sl_dict: dict = {}
    base: RLState = {
        "task_theory": {},
        "stem_config": {"system_prompt": "Review code for bugs."},
        "uncertainty_priors": [],
        "experiment_id": "test-exp",
        "condition": "rl",
        "current_prompt": "Review code for bugs.",
        "iteration": 1,
        "curriculum_index": 0,
        "curriculum_order": ["data/training_bugs/bug_001.py"],
        "performance_history": [],
        "failure_memory": {},
        "consecutive_no_improvement": 0,
        "last_verbal_gradient": None,
        "temperatures": [0.6, 0.9, 1.2],
        "variant_results": [],
        "skill_library": sl_dict,
        "prompt_manager": pm_dict,
        "stopping_reason": None,
        "rollback_events": [],
        "recent_failures": [],
        "agent_reviews": [],
        "final_prompt": None,
    }
    base.update(overrides)
    return base


def test_sft_graph_compiles() -> None:
    from src.specialization.graphs import build_sft_subgraph
    graph = build_sft_subgraph()
    assert graph is not None


def test_rl_graph_compiles() -> None:
    from src.specialization.graphs import build_rl_subgraph
    graph = build_rl_subgraph()
    assert graph is not None


def test_outer_graph_compiles(tmp_path) -> None:
    from src.specialization.graphs import build_outer_graph
    graph = build_outer_graph(checkpoint_path=str(tmp_path / "test.db"))
    assert graph is not None


def test_route_condition_routes_to_sft(tmp_path) -> None:
    from src.specialization.graphs import build_outer_graph
    graph = build_outer_graph(checkpoint_path=str(tmp_path / "test.db"))
    next_nodes = graph.get_graph().nodes
    assert "sft_subgraph" in next_nodes


def test_route_condition_routes_to_rl(tmp_path) -> None:
    from src.specialization.graphs import build_outer_graph
    graph = build_outer_graph(checkpoint_path=str(tmp_path / "test.db"))
    nodes = graph.get_graph().nodes
    assert "rl_subgraph" in nodes


def test_check_forgetting_returns_ewc_rollback(mocker) -> None:
    mocker.patch(
        "src.specialization.nodes_rl.prompt_manager_from_dict",
        return_value=mocker.MagicMock(
            detect_catastrophic_forgetting=mocker.MagicMock(return_value=True)
        ),
    )
    state = _base_rl_state(performance_history=[0.3, 0.5, 0.2])
    result = check_forgetting(state)
    assert result == "ewc_rollback"


def test_check_forgetting_returns_lazy_gradient(mocker) -> None:
    mocker.patch(
        "src.specialization.nodes_rl.prompt_manager_from_dict",
        return_value=mocker.MagicMock(
            detect_catastrophic_forgetting=mocker.MagicMock(return_value=False)
        ),
    )
    state = _base_rl_state(performance_history=[0.5, 0.6, 0.7])
    result = check_forgetting(state)
    assert result == "lazy_gradient"


def test_check_stop_returns_end() -> None:
    state = _base_rl_state(stopping_reason="max_iterations_reached")
    assert check_stop(state) == "end"


def test_check_stop_returns_continue() -> None:
    state = _base_rl_state(stopping_reason=None)
    assert check_stop(state) == "continue"


def test_send_api_returns_three_sends() -> None:
    from langgraph.types import Send  # type: ignore[import-untyped]
    state = _base_rl_state(temperatures=[0.6, 0.9, 1.2])
    sends = route_to_variants(state)
    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)


def test_send_api_sends_have_different_strategies() -> None:
    state = _base_rl_state(temperatures=[0.6, 0.9, 1.2])
    sends = route_to_variants(state)
    strategies = [s.arg["variant_strategy"] for s in sends]
    assert len(set(strategies)) == 3


def test_simulated_annealing_early() -> None:
    temps = get_temperatures(3)
    assert len(temps) == 3
    assert all(0.5 <= t <= 1.3 for t in temps)


def test_simulated_annealing_late() -> None:
    temps = get_temperatures(12)
    assert len(temps) == 3
    assert all(0.1 <= t <= 0.7 for t in temps)


def test_temperatures_decrease_over_iterations() -> None:
    early = get_temperatures(1)
    mid = get_temperatures(7)
    late = get_temperatures(12)
    assert max(early) > max(mid) > max(late)


def test_agent_review_output_is_structured_json(mocker) -> None:
    mock_response = {
        "issues": [{"line": 5, "bug_type": "off_by_one", "description": "off by one", "confidence": 0.9}],
        "overall_confidence": 0.9,
        "summary": "Found one issue",
    }
    mock_client = mocker.MagicMock()
    mock_client.complete_json.return_value = mock_response
    mocker.patch("src.specialization.nodes_rl.get_llm_client", return_value=mock_client)
    mocker.patch("src.specialization.nodes_rl._read_file", return_value="def foo(): pass")

    state = _base_rl_state(
        curriculum_order=["data/training_bugs/bug_001.py"],
        curriculum_index=1,
    )
    from src.specialization.nodes_rl import review_code_batch
    result = review_code_batch(state)
    reviews = result["agent_reviews"]
    assert isinstance(reviews, list)
    assert len(reviews) == 1
    review = reviews[0]
    assert "issues" in review
    assert "overall_confidence" in review
    assert "summary" in review
    assert "file_path" in review


def test_lazy_gradient_reuses_on_no_change(mocker) -> None:
    existing_gradient = {"weakness": "missed nulls", "proposed_change": "add null check", "preserve": "syntax checks", "confidence": 0.8}
    mock_client = mocker.MagicMock()
    mocker.patch("src.specialization.nodes_rl.get_llm_client", return_value=mock_client)

    state = _base_rl_state(
        iteration=3,
        consecutive_no_improvement=0,
        rollback_events=[],
        last_verbal_gradient=existing_gradient,
        recent_failures=[],
    )
    result = lazy_gradient(state)
    mock_client.complete_json.assert_not_called()
    assert result["last_verbal_gradient"] == existing_gradient


def test_lazy_gradient_recomputes_on_plateau(mocker) -> None:
    new_gradient = {"weakness": "new", "proposed_change": "change", "preserve": "keep", "confidence": 0.7}
    mock_client = mocker.MagicMock()
    mock_client.complete_json.return_value = new_gradient
    mocker.patch("src.specialization.nodes_rl.get_llm_client", return_value=mock_client)

    state = _base_rl_state(
        iteration=5,
        consecutive_no_improvement=2,
        rollback_events=[],
        last_verbal_gradient={"weakness": "old"},
        recent_failures=[],
    )
    result = lazy_gradient(state)
    mock_client.complete_json.assert_called_once()
    assert result["last_verbal_gradient"] == new_gradient


def test_evaluate_batch_uses_two_calls_per_variant(mocker, tmp_path) -> None:
    validation_dir = tmp_path / "data" / "validation"
    validation_dir.mkdir(parents=True)
    for i in range(6):
        (validation_dir / f"file_{i}.py").write_text(f"def fn_{i}(): pass")

    mocker.patch(
        "src.specialization.nodes_rl.VALIDATION_DIR",
        str(validation_dir),
    )
    mocker.patch(
        "src.specialization.nodes_rl._load_ground_truth",
        return_value={},
    )

    mock_client = mocker.MagicMock()
    mock_client.complete_json.return_value = {"reviews": []}
    mocker.patch("src.specialization.nodes_rl.get_llm_client", return_value=mock_client)

    variants: list[VariantResult] = [
        {"variant_id": 0, "prompt": "p0", "temperature": 0.6, "score": 0.0},
        {"variant_id": 1, "prompt": "p1", "temperature": 0.9, "score": 0.0},
        {"variant_id": 2, "prompt": "p2", "temperature": 1.2, "score": 0.0},
    ]
    state = _base_rl_state(
        variant_results=variants,
        performance_history=[0.5],
    )
    from src.specialization.nodes_rl import evaluate_batch
    evaluate_batch(state)
    assert mock_client.complete_json.call_count == 3 * 2


def test_evaluate_batch_early_exit(mocker, tmp_path) -> None:
    validation_dir = tmp_path / "data" / "validation"
    validation_dir.mkdir(parents=True)
    for i in range(6):
        (validation_dir / f"file_{i}.py").write_text(f"def fn_{i}(): pass")

    mocker.patch("src.specialization.nodes_rl.VALIDATION_DIR", str(validation_dir))

    gt_map = {}
    mocker.patch("src.specialization.nodes_rl._load_ground_truth", return_value=gt_map)
    mocker.patch(
        "src.specialization.nodes_rl.pipeline.run_all_verifiers",
        return_value=mocker.MagicMock(shaped_reward=0.9),
    )

    call_count = 0

    def side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return {"reviews": []}

    mock_client = mocker.MagicMock()
    mock_client.complete_json.side_effect = side_effect
    mocker.patch("src.specialization.nodes_rl.get_llm_client", return_value=mock_client)

    variants: list[VariantResult] = [
        {"variant_id": 0, "prompt": "p0", "temperature": 0.6, "score": 0.0},
        {"variant_id": 1, "prompt": "p1", "temperature": 0.9, "score": 0.0},
        {"variant_id": 2, "prompt": "p2", "temperature": 1.2, "score": 0.0},
    ]
    state = _base_rl_state(
        variant_results=variants,
        performance_history=[0.5],
    )
    from src.specialization.nodes_rl import evaluate_batch
    result = evaluate_batch(state)
    assert result is not None


def test_failure_memory_accumulates(mocker) -> None:
    gt_dict = {
        "file_path": "data/training_bugs/bug_001.py",
        "bug_type": "off_by_one",
        "bug_line": 5,
        "bug_description": "off by one",
        "detectable_by_pylint": False,
        "detectable_by_ast": False,
        "detectable_by_execution": False,
    }
    mocker.patch(
        "src.specialization.nodes_rl._load_ground_truth",
        return_value={"data/training_bugs/bug_001.py": gt_dict},
    )
    mocker.patch(
        "src.specialization.nodes_rl._read_file",
        return_value="def f(): pass",
    )
    mocker.patch(
        "src.specialization.nodes_rl.pipeline.run_all_verifiers",
        return_value=mocker.MagicMock(shaped_reward=0.0),
    )

    state = _base_rl_state(
        curriculum_order=["data/training_bugs/bug_001.py"],
        curriculum_index=1,
        failure_memory={},
        recent_failures=[],
    )
    for _ in range(3):
        result = compute_reward(state)
        state = {**state, **result}

    assert state["failure_memory"].get("off_by_one", 0) == 3


def test_recent_failures_truncates_to_three(mocker) -> None:
    gt_dict = {
        "file_path": "data/training_bugs/bug_001.py",
        "bug_type": "off_by_one",
        "bug_line": 5,
        "bug_description": "off by one",
        "detectable_by_pylint": False,
        "detectable_by_ast": False,
        "detectable_by_execution": False,
    }
    mocker.patch(
        "src.specialization.nodes_rl._load_ground_truth",
        return_value={"data/training_bugs/bug_001.py": gt_dict},
    )
    mocker.patch("src.specialization.nodes_rl._read_file", return_value="def f(): pass")
    mocker.patch(
        "src.specialization.nodes_rl.pipeline.run_all_verifiers",
        return_value=mocker.MagicMock(shaped_reward=0.0),
    )

    state = _base_rl_state(
        curriculum_order=["data/training_bugs/bug_001.py"],
        curriculum_index=1,
        failure_memory={},
        recent_failures=[],
    )
    for i in range(5):
        state = {**state, "iteration": i + 1}
        result = compute_reward(state)
        state = {**state, **result}

    assert len(state["recent_failures"]) <= 3


def test_finalize_rl_maps_current_to_final_prompt() -> None:
    state = _base_rl_state(current_prompt="Optimized system prompt.")
    result = finalize_rl(state)
    assert result["final_prompt"] == "Optimized system prompt."


def test_evaluate_batch_parallel_equivalent_to_sequential(mocker, tmp_path) -> None:
    validation_dir = tmp_path / "data" / "validation"
    validation_dir.mkdir(parents=True)
    for i in range(6):
        (validation_dir / f"file_{i}.py").write_text(f"def fn_{i}(): pass")

    mocker.patch("src.specialization.nodes_rl.VALIDATION_DIR", str(validation_dir))
    mocker.patch("src.specialization.nodes_rl._load_ground_truth", return_value={})

    call_counts: list[int] = []

    def side_effect(*_args, **_kwargs):
        call_counts.append(1)
        return {"reviews": []}

    mock_client = mocker.MagicMock()
    mock_client.complete_json.side_effect = side_effect
    mocker.patch("src.specialization.nodes_rl.get_llm_client", return_value=mock_client)

    variants: list[VariantResult] = [
        {"variant_id": 0, "prompt": "p0", "temperature": 0.6, "score": 0.0},
        {"variant_id": 1, "prompt": "p1", "temperature": 0.9, "score": 0.0},
        {"variant_id": 2, "prompt": "p2", "temperature": 1.2, "score": 0.0},
    ]
    state = _base_rl_state(variant_results=variants, performance_history=[0.5])

    from src.specialization.nodes_rl import evaluate_batch
    result = evaluate_batch(state)

    scored = result["variant_results"]
    assert len(scored) == 3
    assert all(v["variant_id"] in (0, 1, 2) for v in scored)
    assert sum(call_counts) == 3 * 2
