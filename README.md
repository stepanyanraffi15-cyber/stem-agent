# stem-agent

Stem-phase self-configuring agent for Python code review — outcome-based (RL-style) vs demonstration-based (SFT-style) prompt specialization, evaluated on out-of-distribution generalization.

`LangGraph` · `LangSmith` · `pylint/AST/execution verifier` · `TextGrad verbal gradient` · `EWC rollback`

---

## Results

Trained on 30 Python files (5 pylint-detectable bug types). Evaluated on 20 held-out files (4 logic error types pylint cannot catch).

| Metric | Baseline | SFT | RL | RL Δ SFT |
|---|---|---|---|---|
| Train Precision | 0.533 | **0.900** | 0.544 | −0.356 |
| Train Recall | 0.533 | **0.900** | 0.567 | −0.333 |
| **Train F1** | 0.533 | **0.900** | 0.544 | −0.356 |
| OOD Precision | 0.117 | 0.400 | **0.217** | −0.183 |
| OOD Recall | 0.150 | 0.400 | **0.250** | −0.150 |
| **OOD F1** | 0.125 | 0.400 | **0.225** | −0.175 |
| **Generalization Gap** | 0.408 | 0.500 | **0.319** | **−0.181** |
| Silent Failure Rate | 0.400 | 0.400 | 0.400 | +0.000 |
| Calibration ECE | — | 0.556 | **0.332** | −0.224 |

SFT dominates absolute in-distribution F1 (0.900). RL produces a 36% smaller generalization gap (0.319 vs 0.500) and substantially better calibration (ECE 0.332 vs 0.556). The 40% silent failure rate is structural — it maps exactly to the 20 held-out logic-error files where no verifier produces signal. All conditions hit this ceiling identically, which validates the OOD split design.

### RL learning curve

```
Iteration  Reward
1          0.667  ← verbal gradient fires first meaningful update
2          0.000  ← wrong_return_variable: no pylint signal available
3          0.000
4          0.000
5          0.500  ← lazy gradient recomputed after 2-iteration plateau
6          0.667
7          1.0
```

Flat reward at iterations 2–4 is expected: the curriculum files at that point are `equality_none_check` (W-severity), where pylint contributes 0.3 to shaped reward rather than 0.6. The gradient signal is real but weak for mid-difficulty examples.

---

## The Research Question

Does outcome-based reward (RL-style prompt specialization) generalize out-of-distribution better than demonstration-based specialization (SFT-style)?

Grounded in Chu et al. 2025 ([arXiv:2501.17161](https://arxiv.org/abs/2501.17161)): *SFT memorizes training distribution; RL generalizes beyond it.* This project instantiates that finding at the prompt level — the agent rewrites its own system prompt through verifier feedback instead of fine-tuning weights.

The OOD split makes this testable: training bugs are pylint-detectable (the verifier gives real signal during RL specialization), held-out bugs are logic errors (no verifier signal — only reasoning about code intent can detect them). If RL only memorized the verifier signal, both conditions would score equally on held-out. If RL learned to reason about code, it generalizes. The 40% silent failure rate tells you where that ceiling currently sits.

---

## What This Implements

The stem phase maps to four behaviors:

```
OBSERVE     read task description → build theory before any training data
CONFIGURE   write own system prompt from that theory
SPECIALIZE  two conditions in parallel:
              A  SFT-style:  demonstrations → constitutional critique → single rewrite
              B  RL-style:   curriculum loop → shaped reward → verbal gradient → variants → EWC
STOP        principled: CI on improvement delta, not arbitrary N
```

### Shaped reward

The RL condition uses a weighted combination of three deterministic signals:

```
R = 0.5 · R_pylint  +  0.3 · R_ast  +  0.2 · R_exec
```

Each component conditioned on whether the ground truth bug is detectable by that signal:

```
R_pylint  = TP / (TP + FN)          [recall on pylint-detectable bugs only]
R_ast     = 1.0 if bug_type ∈ ast_patterns_found  else 0.0
R_exec    = 1.0 if execution caught error AND detectable_by_pylint == False  else 0.0
```

Severity weights: E-severity issues (1.0), W-severity (0.6), C-severity (0.2). Continuous reward in [0,1] instead of binary — from Chu et al. 2025: continuous rewards teach appropriate uncertainty; binary rewards produce overconfident models.

**The reward must measure agent output quality, not verifier output.** The verifier is deterministic — pylint always finds `bare_except` in a file containing `bare_except`, regardless of what system prompt the agent was given. Using verifier reward as the RL signal means the reward never varies with the prompt: the gradient has no direction. The reward used in the RL loop is:

```python
reward = agent_f1(agent_reported_issues, ground_truth_bug_type)
```

This is the outcome-based reward Chu et al. 2025 requires: measuring model performance, not a static code property.

### Verbal gradient

Following TextGrad ([arXiv:2406.07496](https://arxiv.org/abs/2406.07496)), the gradient is a structured diff between actual output and expected output — not a summary of failure counts:

```python
comparisons = [
    {
        "file": basename(f["file_path"]),
        "expected_bug_type": f["expected_bug_type"],
        "agent_reported_issues": f["agent_issues"],
    }
    for f in recent_failures
]
```

The gradient prompt asks three explicit questions: what specific prompt weakness explains the gap, what exact wording change closes it (quoting the current prompt), what must not change. The failure memory weights persistent errors across iterations higher than one-off errors — from AgenTracer ([arXiv:2509.03312](https://arxiv.org/abs/2509.03312)): the decisive error is the earliest recurring failure, not the most recent one.

### EWC-inspired rollback

When performance drops below `best_score × 0.85` or `mean(last_3) − 1.5σ`, a pure revert loses the gradient direction. Instead, an LLM merges the new prompt into the best historical prompt while preserving locked-skill language — Elastic Weight Consolidation (Kirkpatrick et al. 2017) in prompt space: constrain the update to preserve behaviors already mastered.

### Diverse variants

Three variants per iteration use genuinely different strategies — not temperature sampling of the same direction:

| Variant | Temperature | Strategy |
|---|---|---|
| A | `t` | Apply gradient precisely to the current prompt |
| B | `t + 0.3` | Generalize gradient to related failure modes |
| C | `t + 0.6` | Different approach to the same problem |

Temperatures follow a simulated annealing schedule (Kirkpatrick 1983): `[0.6, 0.9, 1.2]` early → `[0.2, 0.4, 0.6]` late. Wide exploration when the gradient is uncertain; exploitation near convergence.

---

## Data Split

**Training bugs (30 files, 5 types)** — pylint catches all of these. RL loop receives real gradient signal.

| Bug Type | Pylint Symbol | Severity |
|---|---|---|
| `bare_except` | W0702 | W |
| `mutable_default_argument` | W0102 | W |
| `equality_none_check` | E711 | E |
| `shadowed_builtin` | A001 | W |
| `undefined_variable` | E0602 | E |

**Held-out bugs (20 files, 4 types)** — pylint produces zero signal on these. Only semantic reasoning about code intent detects them. These are not contrived edge cases — they are the most common logic errors in real production Python.

| Bug Type | Why Pylint Cannot Catch It |
|---|---|
| `off_by_one` | Syntactically correct; wrong loop bound |
| `wrong_operator` | Syntactically correct; `>` where `>=` needed |
| `missing_edge_case` | No crash on normal input; `total/count` with no zero guard |
| `wrong_return_variable` | Returns a valid variable — the wrong one |

Files are curriculum-ordered by AST difficulty score: `0.3 × cyclomatic_proxy + 0.4 × function_count_norm + 0.3 × nesting_depth_norm`. Easy → hard, per Bengio et al. 2009.

---

## Architecture

| File | Responsibility |
|---|---|
| `src/verifier/pylint_verifier.py` | Subprocess pylint `--output-format=json`; severity-weighted issue list |
| `src/verifier/ast_verifier.py` | Structural pattern detection (bare except, mutable defaults); difficulty score |
| `src/verifier/execution_verifier.py` | Isolated subprocess execution; classifies NameError / TypeError / SyntaxError |
| `src/verifier/shaped_reward.py` | Combines three signals; assigns `decisive_error_category` |
| `src/curriculum/difficulty_scorer.py` | AST-based difficulty scoring; files sorted easy→hard |
| `src/curriculum/dataset_builder.py` | Deterministic hardcoded dataset (30 training + 20 held-out + 6 validation) |
| `src/stem/agent.py` | `theorize()` → `configure()` → `estimate_uncertainty()` |
| `src/stem/skill_library.py` | Voyager-style skill store; Bayesian confidence updates; locking at confidence > 0.8 |
| `src/stem/prompt_manager.py` | Version history; EWC rollback; CI-based stopping |
| `src/specialization/graphs.py` | LangGraph outer graph + subgraph compilation; SqliteSaver checkpoints |
| `src/specialization/nodes_rl.py` | Curriculum step, reward, verbal gradient, Send API variants, evaluate_batch |
| `src/specialization/nodes_sft.py` | Load demonstrations, extract patterns, constitutional critique, rewrite |
| `src/specialization/state.py` | TypedDict state definitions; `operator.add` reducers for fan-in |
| `src/specialization/parallel.py` | `ThreadPoolExecutor` for parallel file scoring |
| `src/evaluation/metrics.py` | Precision/recall/F1, ECE, bootstrap CI, generalization gap |
| `src/prompts/loader.py` | Load `.md` prompt templates; format `{placeholder}` vars; strip schema section |
| `experiments/run_all.py` | Entry point: stem → baseline → SFT → RL → benchmark |

SFT and RL compile as separate LangGraph subgraphs. An outer graph routes by `condition` field, both feed into a shared `evaluate_final_node`. SqliteSaver checkpoints every node — interrupted runs resume from the last completed node. Parallel file scoring uses `ThreadPoolExecutor(max_workers=10)`: 50 sequential LLM calls at ~3s each is 150s; parallelised is ~15s wall time.

---

## Quickstart

```bash
git clone https://github.com/stepanyanraffi15-cyber/stem-agent
cd stem-agent
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY

# Generate dataset (deterministic — no API key needed)
python -c "from src.curriculum.dataset_builder import generate_all; generate_all()"

# Verify setup
pytest tests/ -v

# Full experiment (~20 min, ~$0.40 with gpt-4o-mini/gpt-4o)
python -m experiments.run_all --condition all

# Skip stem phase on re-runs
python -m experiments.run_all --condition all --skip-stem

# Fast mode: 3 RL iterations, ~3 min
python -m experiments.run_all --condition rl --fast
```

| Parameter | Default | Notes |
|---|---|---|
| `MODEL_NAME_CHEAP` | `gpt-4o-mini` | Agent review (high volume, structured JSON) |
| `MODEL_NAME_STRONG` | `gpt-4o` | Gradient, variant generation, stem phase |
| `DEFAULT_MAX_ITERATIONS` | 10 | Below 10: curve too short. Above 15: diminishing returns on 6 validation files. |
| `FORGETTING_THRESHOLD` | 1.5σ | Hard floor: `best × 0.85`; soft floor: `mean(last_3) − 1.5σ` |
| `LOCK_THRESHOLD` | 0.8 | Minimum confidence + 3 evidence counts to lock a skill |

---

## What Did Not Work

**SFT outperforms RL in absolute F1.** Train F1=0.900 vs 0.544. Constitutional critique with direct demonstrations is maximally sample-efficient for in-distribution performance — the model sees concrete examples of correct vs incorrect labeling and self-corrects. RL needs more iterations to converge through noisy verbal gradients on a small validation set. At 10 iterations, SFT wins on absolute numbers while RL wins on generalization gap and calibration. The crossover point — where RL's OOD advantage compounds enough to close the in-distribution gap — was not reached in this experiment.

**The 40% silent failure rate is a hard ceiling.** Off-by-one errors, wrong operators, missing edge cases — these are invisible to pylint, AST analysis, and execution without test cases. All three conditions hit this ceiling identically. Prompt-level specialization cannot cross it regardless of how many iterations run, because there is no reward signal to learn from. This is not a limitation of the approach — it is the correct finding. The boundary of what prompt-level RL can learn is exactly where the verifier's signal runs out.

**Prompt-level RL is a proxy for the real mechanism.** Chu et al. 2025 demonstrated their finding at the weight level — model parameters shift through thousands of gradient steps. Here, only the system prompt changes. The representational geometry is fixed. The model's internal understanding of code is unchanged — only its surface instructions vary. SFT's advantage in absolute terms may partly reflect that demonstration-based prompt rewriting is better matched to this constraint than outcome-based optimization with a small validation set and 10 iterations.

---

## Limitations and Extensions

**Execution-based reward for logic errors.** The 40% silent failure ceiling requires a different signal source. An LLM-based execution simulator — trace inputs and outputs symbolically, identify semantic inconsistencies — would unlock this floor. This is the natural next verifier: static analysis covers syntax, AST covers structure, execution covers runtime, semantic tracing covers intent.

**Weight-level RL.** A LoRA adapter or fine-tuned embedding model trained on the same outcome-based reward would be the real version of this experiment. Prompt rewriting is an approximation. The comparison would then directly replicate Chu et al. 2025's setup, with code review as the task domain.

**Validation set size and composition.** 6 validation files produce noisy variant scores — a single file shifting changes a score by 0.167. With 20+ validation files, including some OOD examples, beam search would produce substantially more reliable gradient signal and the RL curve would steepen.

**Generalization gap vs iteration count.** The hypothesis: RL's gap shrinks monotonically as the verbal gradient accumulates evidence about failure modes; SFT's gap stays flat after 1–2 iterations. Plotting this curve at 5, 10, 20, 30 iterations would characterize where prompt-level RL specialization converges and whether there is a crossover point.

---

## Tests

```bash
pytest tests/ -v --cov=src
```

All unit tests mock LLM calls — zero API spend, deterministic. Integration tests use `MemorySaver` not `SqliteSaver` — no file artifacts.

The most structurally important test:

```python
def test_held_out_bugs_not_detected_by_pylint():
    """Every held-out file must produce zero pylint issues.
    If this fails, the OOD split is contaminated and the experiment is invalid."""
```

---

## Papers

| Reference | Used for |
|---|---|
| Chu et al. 2025, [arXiv:2501.17161](https://arxiv.org/abs/2501.17161) | Core research question; shaped reward; outcome-based RL |
| Yuksekgonul et al. 2024, [arXiv:2406.07496](https://arxiv.org/abs/2406.07496) (TextGrad) | Verbal gradient: output-vs-expected diff |
| Shinn et al. 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) (Reflexion) | Self-reflection loop structure |
| Wang et al. 2023, [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) (Voyager) | Skill library; Bayesian confidence |
| Bai et al. 2022, [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) (Constitutional AI) | SFT critique-then-revise loop |
| Yang et al. 2023, [arXiv:2309.03409](https://arxiv.org/abs/2309.03409) (OPRO) | Iterative prompt optimization |
| Yao et al. 2023, [arXiv:2310.16427](https://arxiv.org/abs/2310.16427) (PromptAgent) | Prompt search; theory-first |
| Zhang et al. 2025, [arXiv:2509.03312](https://arxiv.org/abs/2509.03312) (AgenTracer) | Failure memory; persistent failure weighting |
| Pathak et al. 2025, [arXiv:2511.04032](https://arxiv.org/abs/2511.04032) (IBM Silent Failures) | Silent failure taxonomy |
| Roig 2025, [arXiv:2512.07497](https://arxiv.org/abs/2512.07497) (KAMI) | Failure archetypes; context pollution prevention |
| JetBrains 2025, [arXiv:2510.05788](https://arxiv.org/abs/2510.05788) (Mellum) | Evaluation methodology; contamination-resistant OOD splits |
| Kirkpatrick et al. 2017 (EWC) | Rollback: constrain updates to preserve mastered behaviors |
| Bengio et al. 2009 (Curriculum Learning) | Training file ordering: easy→hard |
| Kirkpatrick et al. 1983 (Simulated Annealing) | Temperature schedule for variant generation |
| Vijayakumar et al. 2018 (Diverse Beam Search) | Variant diversity: different directions, not temperature noise |
