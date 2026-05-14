PROMPT:
You are a Python code review agent about to begin specialization training.

Your current system prompt:
{system_prompt}

For each bug type listed below, honestly assess your confidence that
your current strategy will detect it. Be calibrated — not overconfident.

Bug types to assess: {bug_types_list}

Return ONLY valid JSON as a list:
[
  {"bug_type": "name", "confidence": 0.0_to_1.0, "reasoning": "why"},
  ...
]

EXPECTED_OUTPUT_SCHEMA:
list of {"bug_type": str, "confidence": float, "reasoning": str}

PAPER_CITATION:
Kuhn et al. 2023 arXiv:2302.09664 — semantic entropy and uncertainty estimation.
