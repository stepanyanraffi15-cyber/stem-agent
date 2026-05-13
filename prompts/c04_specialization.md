# Component 4: Specialization Conditions

Corrected spec incorporating all Q&A answers from pre-build review.

---

## Research Question

Does outcome-based (RL-style) specialization generalize OOD better than
demonstration-based (SFT-style) specialization for Python code review?

---

## New Files

```
src/specialization/__init__.py
src/specialization/models.py        ← serialization helpers (PromptManager, SkillLibrary)
src/specialization/state.py         ← TypedDict state definitions with correct reducers
src/specialization/llm_factory.py   ← MODEL_ROUTING + get_llm_client()
src/specialization/nodes_sft.py     ← 5 sequential SFT nodes
src/specialization/nodes_rl.py      ← RL nodes + pure routing functions
src/specialization/graphs.py        ← graph compilation + evaluate_final_node
src/specialization/runner.py        ← ExperimentRunner with streaming
tests/unit/test_specialization.py
tests/integration/__init__.py
tests/integration/test_graphs.py
```

Add to requirements.txt:
```
langgraph>=0.2.0
langsmith>=0.1.0
langchain-core>=0.2.0
langgraph-checkpoint-sqlite>=1.0.0
```

Add to .env.example:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=stem-agent
MODEL_NAME_CHEAP=gpt-4o-mini
MODEL_NAME_STRONG=gpt-4o
```

---

## LLM Factory

File: src/specialization/llm_factory.py

```python
MODEL_ROUTING = {
    "agent_review": os.getenv("MODEL_NAME_CHEAP", "gpt-4o-mini"),
    "gradient":     os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
    "variant_gen":  os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
    "ewc_merge":    os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
}
```

`get_llm_client(model_key: str) -> LLMClient`
  - Reads MODEL_ROUTING[model_key]
  - Sets MODEL_NAME and MODEL_PROVIDER env vars temporarily
  - Instantiates and returns a fresh LLMClient
  - Raises ValueError on unknown model_key
  - Never touches src/stem/llm_client.py

---

## State Definitions

File: src/specialization/state.py

Reducers:
- `agent_reviews: Annotated[list[AgentReview], operator.add]`  — NOT add_messages
- `variant_results: Annotated[list[VariantResult], operator.add]` — fan-in for Send API

```python
class AgentReview(TypedDict):
    file_path: str
    issues: list[dict]        # {line, bug_type, description, confidence}
    overall_confidence: float
    summary: str

class VariantResult(TypedDict):
    variant_id: int
    prompt: str
    temperature: float
    score: float

class SFTState(TypedDict):
    task_theory: dict
    stem_config: dict
    experiment_id: str
    demonstrations: list[str]
    extracted_patterns: list[str]
    critiqued_patterns: list[str]
    rewrite_reasoning: str
    final_prompt: str
    skill_library: dict
    agent_reviews: Annotated[list[AgentReview], operator.add]

class RLState(TypedDict):
    task_theory: dict
    stem_config: dict
    uncertainty_priors: list[dict]
    experiment_id: str
    condition: str
    current_prompt: str
    iteration: int
    curriculum_index: int
    curriculum_order: list[str]
    performance_history: list[float]
    failure_memory: dict
    consecutive_no_improvement: int
    last_verbal_gradient: dict | None
    temperatures: list[float]
    variant_results: Annotated[list[VariantResult], operator.add]
    skill_library: dict
    prompt_manager: dict
    stopping_reason: str | None
    rollback_events: list[dict]
    recent_failures: list[dict]     # NEW — last 3 failed reviews for lazy_gradient
    agent_reviews: Annotated[list[AgentReview], operator.add]
    final_prompt: str | None        # set by finalize_rl before END

class VariantState(TypedDict):
    # All RLState fields plus variant-specific fields sent via Send API
    task_theory: dict
    stem_config: dict
    uncertainty_priors: list[dict]
    experiment_id: str
    condition: str
    current_prompt: str
    iteration: int
    curriculum_index: int
    curriculum_order: list[str]
    performance_history: list[float]
    failure_memory: dict
    consecutive_no_improvement: int
    last_verbal_gradient: dict | None
    temperatures: list[float]
    variant_results: Annotated[list[VariantResult], operator.add]
    skill_library: dict
    prompt_manager: dict
    stopping_reason: str | None
    rollback_events: list[dict]
    recent_failures: list[dict]
    agent_reviews: Annotated[list[AgentReview], operator.add]
    final_prompt: str | None
    variant_id: int
    target_temperature: float
    variant_strategy: str

class OuterState(TypedDict):
    task_theory: dict
    stem_config: dict
    uncertainty_priors: list[dict]
    experiment_id: str
    condition: str
    final_prompt: str | None
    evaluation_results: dict | None
    # SFTState keys with None defaults for subgraph key-mapping
    demonstrations: list[str] | None
    extracted_patterns: list[str] | None
    critiqued_patterns: list[str] | None
    rewrite_reasoning: str | None
    skill_library: dict | None
```

---

## Serialization Helpers

File: src/specialization/models.py

```python
def prompt_manager_to_dict(pm: PromptManager) -> dict:
    # serialize versions (list[PromptVersion]), current_index, _rollback_events
    # PromptVersion.created_at → isoformat string

def prompt_manager_from_dict(d: dict) -> PromptManager:
    # reconstruct PromptVersion objects, parse created_at from isoformat
    # set pm.versions, pm.current_index, pm._rollback_events

def skill_library_to_dict(sl: SkillLibrary) -> dict:
    # {name: dataclasses.asdict(skill)} with created_at isoformat

def skill_library_from_dict(d: dict) -> SkillLibrary:
    # reconstruct Skill objects with created_at parsing
```

---

## SFT Subgraph Nodes

File: src/specialization/nodes_sft.py

Pattern→Skill mapping in update_skills_sft:
  strategy     → detection_pattern
  applies_to   → bug_types_covered
  name         → name
  description  = strategy (same text)
  confidence   = 0.5 (default, unlocked)

node: load_demonstrations(state: SFTState) -> dict
  - Load 20 files from data/training_bugs/ using CurriculumBuilder
  - Return {"demonstrations": [file_paths]}

node: extract_patterns(state: SFTState) -> dict
  - LLM call (get_llm_client("gradient")): analyze demonstration code + ground truth pairs
  - Prompt: see original spec
  - Return {"extracted_patterns": [pattern_dicts]}

node: constitutional_critique(state: SFTState) -> dict
  - LLM call (get_llm_client("gradient")): critique each pattern
  - Grounds: Constitutional AI (Bai et al. 2022, arXiv:2212.08073)
  - Return {"critiqued_patterns": [revised_dicts]}

node: rewrite_prompt(state: SFTState) -> dict
  - LLM call (get_llm_client("gradient")): rewrite system prompt
  - Return {"final_prompt": str, "rewrite_reasoning": str}

node: update_skills_sft(state: SFTState) -> dict
  - Load SkillLibrary via skill_library_from_dict(state["skill_library"])
  - For each critiqued pattern, build Skill and call add_skill()
  - Return {"skill_library": skill_library_to_dict(library)}

No @traceable decorators. LangGraph handles LangSmith tracing automatically.

---

## RL Subgraph Nodes

File: src/specialization/nodes_rl.py

PURE ROUTING FUNCTIONS (never add_node, only add_conditional_edges):
  check_forgetting(state: RLState) -> str
  check_stop(state: RLState) -> str
  route_to_variants(state: RLState) -> list[Send]

SIMULATED ANNEALING:
  def get_temperatures(iteration: int) -> list[float]:
      if iteration <= 5:  return [0.6, 0.9, 1.2]
      elif iteration <= 10: return [0.4, 0.6, 0.9]
      else: return [0.2, 0.4, 0.6]

node: curriculum_step(state: RLState) -> dict
  - Next file from curriculum_order[curriculum_index]
  - Update temperatures via get_temperatures(state["iteration"])
  - Return {"curriculum_index": idx+1, "temperatures": temps, "iteration": iter+1}

node: review_code_batch(state: RLState) -> dict
  - get_llm_client("agent_review"), current_prompt as system
  - Return {"agent_reviews": [AgentReview(...)]}  ← operator.add accumulates

node: compute_reward(state: RLState) -> dict
  - Load ground truth from data/ground_truth.json for current file
  - Load file code, call pipeline.run_all_verifiers(code, ground_truth)
  - Read .shaped_reward
  - Update failure_memory: {bug_type: count+1} if reward == 0.0
  - Update recent_failures: append if reward == 0.0, keep [-3:]
  - Update consecutive_no_improvement
  - Return {"performance_history": [..., reward], "failure_memory": updated,
            "consecutive_no_improvement": N, "recent_failures": updated}

check_forgetting(state: RLState) -> str  [ROUTING FUNCTION]
  - Load PromptManager via prompt_manager_from_dict(state["prompt_manager"])
  - Call detect_catastrophic_forgetting(state["performance_history"][-1])
  - Return "ewc_rollback" if True, "lazy_gradient" otherwise

node: ewc_rollback(state: RLState) -> dict
  - get_llm_client("ewc_merge")
  - Call ewc_constrained_rollback()
  - Append rollback event
  - Return {"current_prompt": merged, "rollback_events": updated,
            "prompt_manager": updated_dict}

node: lazy_gradient(state: RLState) -> dict
  - RECOMPUTE if: iteration==1 OR consecutive_no_improvement>=2 OR last rollback_event.iteration==current
  - REUSE otherwise
  - If RECOMPUTE: get_llm_client("gradient"), uses state["recent_failures"]
  - Return {"last_verbal_gradient": gradient_dict}

route_to_variants(state: RLState) -> list[Send]  [ROUTING FUNCTION]
  - temps = state["temperatures"]
  - return [
        Send("generate_variant", {**state, "variant_id": 0, "target_temperature": temps[0], "variant_strategy": "focused"}),
        Send("generate_variant", {**state, "variant_id": 1, "target_temperature": temps[1], "variant_strategy": "broad"}),
        Send("generate_variant", {**state, "variant_id": 2, "target_temperature": temps[2], "variant_strategy": "alternative"}),
    ]

node: generate_variant(state: VariantState) -> dict
  - get_llm_client("variant_gen") at state["target_temperature"]
  - Prompt varies by state["variant_strategy"]: focused/broad/alternative
  - Return {"variant_results": [VariantResult(variant_id, prompt, temperature, score=0.0)]}

node: evaluate_batch(state: RLState) -> dict
  - GUARD: if len(state["variant_results"]) < 3: return {}
  - Load 6 validation files from data/validation/, split into 2 batches of 3
  - For each variant: 2 batched LLM calls (3 files each) via get_llm_client("agent_review")
  - Score = mean shaped_reward across all 6 reviews
  - Early exit: if best_score > current_score + 0.10, skip remaining variants
  - Return {"variant_results": [VariantResult with scores filled]}

node: select_best_and_update(state: RLState) -> dict
  - Pick best variant by score
  - Update current_prompt if best_score > current_score
  - Update PromptManager and SkillLibrary
  - Call should_stop(), set stopping_reason
  - Return {"current_prompt": best, "prompt_manager": updated,
            "skill_library": updated, "consecutive_no_improvement": N,
            "stopping_reason": reason_or_none}

check_stop(state: RLState) -> str  [ROUTING FUNCTION]
  - Return "end" if state["stopping_reason"] is not None, "continue" otherwise

node: finalize_rl(state: RLState) -> dict
  - Return {"final_prompt": state["current_prompt"]}

---

## Graph Wiring

File: src/specialization/graphs.py

### SFT Subgraph
```
START → load_demonstrations → extract_patterns → constitutional_critique
      → rewrite_prompt → update_skills_sft → END
```
Compiled with no checkpointer.

### RL Subgraph
```
START → curriculum_step → review_code_batch → compute_reward
  ↓ add_conditional_edges(compute_reward, check_forgetting, {"ewc_rollback": ..., "lazy_gradient": ...})
ewc_rollback → lazy_gradient
  ↓ add_conditional_edges(lazy_gradient, route_to_variants)   ← Send API
generate_variant → evaluate_batch → select_best_and_update
  ↓ add_conditional_edges(select_best_and_update, check_stop, {"continue": "curriculum_step", "end": "finalize_rl"})
finalize_rl → END
```
Compiled with no checkpointer.
variant_results MUST have Annotated[list[VariantResult], operator.add] for fan-in.

### Outer Graph
```python
def build_outer_graph(
    checkpoint_path: str = "experiments/checkpoints.db",
    human_review: bool = False,
) -> CompiledGraph:
    os.makedirs(os.path.dirname(checkpoint_path) or "experiments", exist_ok=True)
    memory = SqliteSaver.from_conn_string(checkpoint_path)
    interrupt = ["sft_subgraph", "rl_subgraph"] if human_review else []
    ...
    return graph.compile(checkpointer=memory, interrupt_before=interrupt)
```

route_condition(state: OuterState) -> str: returns state["condition"]

Nodes: sft_subgraph (compiled subgraph), rl_subgraph (compiled subgraph), evaluate_final
Edges: START → route_condition → {sft/rl}_subgraph → evaluate_final → END

### evaluate_final_node(state: OuterState) -> dict
  - baseline: state["stem_config"]["system_prompt"]
  - final: state["final_prompt"]
  - in-dist F1: run on data/training_bugs/ (20 files)
  - OOD F1: run on data/held_out_bugs/
  - generalization_gap = mean(training_F1) - mean(held_out_F1)
  - SFT Pareto: [(0, baseline_score), (1, specialized_score)]
  - RL Pareto: enumerate state["performance_history"] if present
  - Module-level _eval_client = get_llm_client("agent_review") at import time

---

## Runner

File: src/specialization/runner.py

```python
class ExperimentRunner:
    def __init__(self, checkpoint_path: str = "experiments/checkpoints.db"):
        self.graph = build_outer_graph(checkpoint_path=checkpoint_path)
        self._checkpoint_path = checkpoint_path

    def run(self, condition, task_theory, stem_config,
            uncertainty_priors, experiment_id) -> dict:
        config = {
            "configurable": {"thread_id": experiment_id},
            "tags": [f"condition:{condition}", f"experiment:{experiment_id}"],
            "metadata": {"condition": condition, "experiment_id": experiment_id,
                         "project": "stem-agent"},
        }
        initial_state = {...}
        final_state = None
        for chunk in self.graph.stream(initial_state, config=config, stream_mode="values"):
            node_name = list(chunk.keys())[0] if isinstance(chunk, dict) else "update"
            log.info("node_complete", experiment=experiment_id, condition=condition)
            final_state = chunk
        return final_state or {}

    def resume(self, experiment_id: str) -> dict:
        config = {"configurable": {"thread_id": experiment_id}}
        existing = self.graph.get_state(config)
        if not existing or not existing.values:
            raise ValueError(f"No checkpoint found for experiment_id: {experiment_id}")
        return self.graph.invoke(None, config=config)

    def get_state_at_iteration(self, experiment_id: str, iteration: int) -> dict:
        config = {"configurable": {"thread_id": experiment_id}}
        history = list(self.graph.get_state_history(config))
        target = [h for h in history if h.values.get("iteration") == iteration]
        return target[0].values if target else {}
```

---

## Tests

### Unit (tests/unit/test_specialization.py) — 22 tests, zero real API calls

def test_sft_graph_compiles()
def test_rl_graph_compiles()
def test_outer_graph_compiles(tmp_path)
def test_route_condition_routes_to_sft(mocker)
def test_route_condition_routes_to_rl(mocker)
def test_check_forgetting_returns_ewc_rollback(mocker)
def test_check_forgetting_returns_lazy_gradient(mocker)
def test_check_stop_returns_end()
def test_check_stop_returns_continue()
def test_send_api_returns_three_sends()
def test_send_api_sends_have_different_strategies()
def test_simulated_annealing_early()   # get_temperatures(3) in [0.5, 1.3]
def test_simulated_annealing_late()    # get_temperatures(12) in [0.1, 0.7]
def test_temperatures_decrease_over_iterations()
def test_agent_review_output_is_structured_json(mocker)
def test_lazy_gradient_reuses_on_no_change(mocker)
def test_lazy_gradient_recomputes_on_plateau(mocker)
def test_evaluate_batch_uses_two_calls_per_variant(mocker)
def test_evaluate_batch_early_exit(mocker)
def test_failure_memory_accumulates(mocker)
def test_recent_failures_truncates_to_three(mocker)
def test_finalize_rl_maps_current_to_final_prompt()

### Integration (tests/integration/test_graphs.py) — 5 tests

def test_sft_full_run_with_mocked_llm(mocker, tmp_path)
def test_rl_three_iterations_with_mocked_llm(mocker, tmp_path)
def test_checkpoint_saves_after_each_node(tmp_path, mocker)
def test_resume_from_checkpoint(tmp_path, mocker)
def test_streaming_yields_node_updates(mocker, tmp_path)

All pass checkpoint_path=str(tmp_path / "test.db") to build_outer_graph().

---

## Wiring Corrections vs Original Spec

- check_forgetting, check_stop, route_to_variants → add_conditional_edges ONLY, never add_node
- variant_results → Annotated[list[VariantResult], operator.add] (required for Send fan-in)
- agent_reviews → Annotated[list[AgentReview], operator.add] (not add_messages)
- generate_variant takes VariantState, not RLState
- finalize_rl node bridges current_prompt → final_prompt in RL path
- evaluate_batch guards: if len(variant_results) < 3: return {}
- select_best_and_update sets stopping_reason; check_stop reads it
- No @traceable decorators; LangGraph handles LangSmith tracing
- stream_mode="values" in runner (full state per chunk, not partial updates)
- build_outer_graph(checkpoint_path, human_review) — both parameterized
- evaluate_batch: 2 batched calls per variant (3 files each), 6 files total

---

## Git Commit Message

```
git commit -m "Add specialization conditions as LangGraph subgraphs

SFT subgraph: demonstrations → extract → constitutional critique → rewrite.
RL subgraph: curriculum loop with Send API parallel variant generation.

LangGraph features: StateGraph, compiled subgraphs, Send API (fan-out/fan-in),
SqliteSaver checkpointing, streaming (values mode), conditional edges,
LangSmith tracing via env, operator.add reducers, time travel.

Model routing: gpt-4o-mini for reviews, gpt-4o for gradients/variants.
Simulated annealing on temperatures (Kirkpatrick 1983).
Lazy verbal gradient with failure memory (AgenTracer arXiv:2509.03312).
Diverse beam: focused/broad/alternative strategies (Vijayakumar 2018).
Batched validation 3-file calls, context contamination prevention (KAMI arXiv:2512.07497).

Tests: 22 unit + 5 integration, all green. Zero real API calls in tests."
```
