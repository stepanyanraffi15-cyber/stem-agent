PROMPT:
You are improving a system prompt for Python code review.

Current prompt:
{current_prompt}

Verbal gradient (the problem to solve):
{verbal_gradient}

Take a COMPLETELY DIFFERENT approach to solving the problem identified
in the gradient. Do not apply the gradient directly — instead, find
an alternative strategy that addresses the same root cause differently.

Return ONLY the improved system prompt as plain text.
No JSON wrapper. No explanation. Just the prompt.

EXPECTED_OUTPUT_SCHEMA:
plain text — no JSON schema required

PAPER_CITATION:
Diverse beam search Vijayakumar 2018 — alternative explores orthogonal direction.
