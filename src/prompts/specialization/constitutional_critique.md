PROMPT:
You are evaluating detection strategies for Python code review.

Strategies to critique:
{patterns_json}

For each strategy, evaluate:
1. Is it specific enough to be actionable?
2. Does it generalize beyond the training examples, or is it overfit?
3. Could it produce false positives on clean code?

Revise each strategy to maximize generalizability while preserving precision.

Return ONLY valid JSON with the same schema as input, revised strategies:
{
  "patterns": [
    {
      "name": "same_name",
      "strategy": "revised strategy text",
      "applies_to": ["bug_types"]
    }
  ]
}

EXPECTED_OUTPUT_SCHEMA:
{"patterns": list of {"name": str, "strategy": str, "applies_to": list[str]}}

PAPER_CITATION:
Constitutional AI arXiv:2212.08073 — self-critique to improve outputs.
