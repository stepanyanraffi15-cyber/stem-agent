# Stem Agent — Architecture

## Research question
Does outcome-based (RL-style) specialization produce an agent that
generalizes out-of-distribution better than demonstration-based (SFT-style)
specialization? Grounded in Chu et al. 2025 (arXiv:2501.17161).

## Four behaviors (stem cell metaphor)
1. Observe — reads task description, builds theory before touching data
2. Configure — writes its own system prompt from that theory
3. Specialize — improves through verifier feedback (two conditions)
4. Stop — principled stopping via confidence intervals, not arbitrary N

## Component map
[Verifier] → ground truth signals
[Curriculum] → ordered training data, easy→hard
[Stem phase] → self-configuration, no training data seen
[Specialization] → two LangGraph subgraphs (SFT + RL)
[Evaluation] → precision/recall/F1, OOD generalization gap, Pareto

## Prompt engineering discipline
All prompts versioned in src/prompts/ as markdown files.
Each prompt connects to a specific paper. No prompt is arbitrary.

## Papers this architecture instantiates
See .cursorrules for full citation list.
