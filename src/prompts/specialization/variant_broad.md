PROMPT:
You are improving a system prompt for Python code review.

Current prompt:
{current_prompt}

Verbal gradient (what to fix):
{verbal_gradient}

Apply this gradient AND generalize it — if the gradient addresses one
failure mode, also address related failure modes that likely share
the same root cause.

Return ONLY the improved system prompt as plain text.
No JSON wrapper. No explanation. Just the prompt.

EXPECTED_OUTPUT_SCHEMA:
plain text — no JSON schema required

PAPER_CITATION:
Diverse beam search Vijayakumar 2018 — broad variant explores neighboring directions.
