from __future__ import annotations

import pytest

from src.specialization.state import OuterState, RLState, SFTState


def _mock_llm_client(mocker, json_return: dict | None = None, text_return: str = ""):
    mock_response = mocker.MagicMock()
    mock_response.content = text_return or "{}"
    mock_client = mocker.MagicMock()
    mock_client.complete.return_value = mock_response
    mock_client.complete_json.return_value = json_return or {}
    return mock_client


def _sft_initial_state() -> SFTState:
    return {
        "task_theory": {"difficulty_factors": [], "common_patterns": []},
        "stem_config": {"system_prompt": "Base prompt."},
        "experiment_id": "sft-integration-test",
        "demonstrations": [],
        "extracted_patterns": [],
        "critiqued_patterns": [],
        "rewrite_reasoning": "",
        "final_prompt": "",
        "skill_library": {},
        "agent_reviews": [],
    }


def _rl_initial_state(curriculum_order: list[str]) -> RLState:
    from src.specialization.models import prompt_manager_to_dict, skill_library_to_dict
    from src.stem.prompt_manager import PromptManager
    from src.stem.skill_library import SkillLibrary
    return {
        "task_theory": {},
        "stem_config": {"system_prompt": "Base prompt."},
        "uncertainty_priors": [],
        "experiment_id": "rl-integration-test",
        "condition": "rl",
        "current_prompt": "Base prompt.",
        "iteration": 0,
        "curriculum_index": 0,
        "curriculum_order": curriculum_order,
        "performance_history": [],
        "failure_memory": {},
        "consecutive_no_improvement": 0,
        "last_verbal_gradient": None,
        "temperatures": [0.6, 0.9, 1.2],
        "variant_results": [],
        "skill_library": skill_library_to_dict(SkillLibrary()),
        "prompt_manager": prompt_manager_to_dict(PromptManager()),
        "stopping_reason": None,
        "rollback_events": [],
        "recent_failures": [],
        "agent_reviews": [],
        "final_prompt": None,
    }


def _outer_state(condition: str) -> OuterState:
    from src.specialization.models import prompt_manager_to_dict, skill_library_to_dict
    from src.stem.prompt_manager import PromptManager
    from src.stem.skill_library import SkillLibrary
    return {
        "task_theory": {},
        "stem_config": {"system_prompt": "Base prompt."},
        "uncertainty_priors": [],
        "experiment_id": f"{condition}-outer-test",
        "condition": condition,
        "final_prompt": None,
        "evaluation_results": None,
        "demonstrations": None,
        "extracted_patterns": None,
        "critiqued_patterns": None,
        "rewrite_reasoning": None,
        "skill_library": None,
        "performance_history": None,
    }


def test_sft_full_run_with_mocked_llm(mocker, tmp_path) -> None:
    mocker.patch("src.specialization.nodes_sft.get_llm_client",
                 return_value=_sft_mock_client(mocker))
    mocker.patch("src.specialization.nodes_sft.os.path.isdir", return_value=True)
    mocker.patch("src.specialization.nodes_sft.os.listdir",
                 return_value=["bug_001.py", "bug_002.py"])
    mocker.patch("src.specialization.nodes_sft._read_file", return_value="def foo(): pass")
    mocker.patch("src.specialization.nodes_sft._load_ground_truth", return_value={})

    from src.specialization.graphs import build_sft_subgraph
    graph = build_sft_subgraph()
    initial = _sft_initial_state()
    config = {"configurable": {"thread_id": "sft-full-run"}}
    result = graph.invoke(initial, config=config)
    assert isinstance(result.get("final_prompt", ""), str)
    assert result["final_prompt"] != ""


def test_rl_three_iterations_with_mocked_llm(mocker, tmp_path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    for i in range(3):
        (training_dir / f"bug_{i:03d}.py").write_text(f"def f_{i}(): pass")

    curriculum_order = [str(training_dir / f"bug_{i:03d}.py") for i in range(3)]

    gt_map = {
        path: {
            "file_path": path,
            "bug_type": "off_by_one",
            "bug_line": 1,
            "bug_description": "off by one",
            "detectable_by_pylint": False,
            "detectable_by_ast": False,
            "detectable_by_execution": False,
        }
        for path in curriculum_order
    }
    mocker.patch("src.specialization.nodes_rl._load_ground_truth", return_value=gt_map)
    mocker.patch(
        "src.specialization.nodes_rl.pipeline.run_all_verifiers",
        return_value=mocker.MagicMock(shaped_reward=0.5),
    )
    mocker.patch("src.specialization.nodes_rl._load_validation_files", return_value=[])

    gradient_resp = {
        "weakness": "missed null",
        "proposed_change": "add check",
        "preserve": "syntax",
        "confidence": 0.7,
    }
    variant_resp = mocker.MagicMock()
    variant_resp.content = '{"system_prompt": "Improved."}'
    mock_client = mocker.MagicMock()
    mock_client.complete_json.return_value = gradient_resp
    mock_client.complete.return_value = variant_resp
    mocker.patch("src.specialization.nodes_rl.get_llm_client", return_value=mock_client)

    call_count = {"n": 0}

    def mock_pm_from_dict(d: dict):
        pm = _make_mock_pm(d)
        call_count["n"] += 1
        if call_count["n"] > 6:
            pm.should_stop = lambda: (True, "max_iterations_reached")
        return pm

    mocker.patch(
        "src.specialization.nodes_rl.prompt_manager_from_dict",
        side_effect=mock_pm_from_dict,
    )

    from src.specialization.graphs import build_rl_subgraph
    graph = build_rl_subgraph()
    initial = _rl_initial_state(curriculum_order)
    config = {"configurable": {"thread_id": "rl-3-iter"}}
    result = graph.invoke(initial, config=config)
    history = result.get("performance_history", [])
    assert len(history) >= 1


def _sft_mock_client(mocker, prompt_override: str = "Improved prompt."):
    _PATTERN = [{"name": "null_check", "strategy": "Check for None", "applies_to": ["null_deref"]}]

    def _json_side_effect(messages):
        content = messages[-1].content if messages else ""
        if "Extract" in content or "Critique" in content or "strategies" in content.lower():
            return {"patterns": _PATTERN}
        return {"system_prompt": prompt_override, "reasoning": "Added patterns."}

    mock_client = mocker.MagicMock()
    mock_client.complete_json.side_effect = _json_side_effect
    return mock_client


def _patch_sft_io(mocker) -> None:
    mocker.patch("src.specialization.nodes_sft.os.path.isdir", return_value=True)
    mocker.patch("src.specialization.nodes_sft.os.listdir", return_value=["a.py"])
    mocker.patch("src.specialization.nodes_sft._read_file", return_value="def foo(): pass")
    mocker.patch("src.specialization.nodes_sft._load_ground_truth", return_value={})
    mocker.patch("src.specialization.graphs._run_evaluation", return_value=[])
    mocker.patch("src.specialization.graphs._list_py_files", return_value=[])


def test_checkpoint_saves_after_each_node(tmp_path, mocker) -> None:
    _patch_sft_io(mocker)
    mocker.patch("src.specialization.nodes_sft.get_llm_client",
                 return_value=_sft_mock_client(mocker))

    db_path = str(tmp_path / "checkpoints.db")
    from src.specialization.graphs import build_outer_graph
    graph = build_outer_graph(checkpoint_path=db_path)
    state = _outer_state("sft")
    config = {"configurable": {"thread_id": "cp-test"}}
    graph.invoke(state, config=config)
    import os
    assert os.path.isfile(db_path)


def test_resume_from_checkpoint(tmp_path, mocker) -> None:
    _patch_sft_io(mocker)
    mocker.patch("src.specialization.nodes_sft.get_llm_client",
                 return_value=_sft_mock_client(mocker, "Resumed prompt."))

    db_path = str(tmp_path / "resume.db")
    from src.specialization.graphs import build_outer_graph
    graph = build_outer_graph(checkpoint_path=db_path)
    state = _outer_state("sft")
    exp_id = "resume-test"
    state["experiment_id"] = exp_id
    config = {"configurable": {"thread_id": exp_id}}
    graph.invoke(state, config=config)

    existing = graph.get_state(config)
    assert existing is not None
    assert existing.values


def test_streaming_yields_node_updates(mocker, tmp_path) -> None:
    _patch_sft_io(mocker)
    mocker.patch("src.specialization.nodes_sft.get_llm_client",
                 return_value=_sft_mock_client(mocker, "Streamed prompt."))

    db_path = str(tmp_path / "stream.db")
    from src.specialization.graphs import build_outer_graph
    graph = build_outer_graph(checkpoint_path=db_path)
    state = _outer_state("sft")
    state["experiment_id"] = "stream-test"
    config = {"configurable": {"thread_id": "stream-test"}}

    chunks = list(graph.stream(state, config=config, stream_mode="values"))
    assert len(chunks) >= 1
    assert all(isinstance(c, dict) for c in chunks)


def _make_mock_pm(d: dict):
    from src.stem.prompt_manager import PromptManager
    pm = PromptManager()
    pm.versions = []
    pm.current_index = -1
    pm._rollback_events = []
    pm.detect_catastrophic_forgetting = lambda score: False
    pm.should_stop = lambda: (False, "continue")
    pm.save_version = lambda **kwargs: None
    pm.ewc_constrained_rollback = lambda **kwargs: kwargs.get("new_prompt", "")
    return pm
