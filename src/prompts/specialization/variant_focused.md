PROMPT:
You are improving a system prompt for Python code review.

Current prompt:
{current_prompt}

Verbal gradient (what to fix):
{verbal_gradient}

Apply this gradient PRECISELY. Make the minimum change that addresses
the identified weakness. Preserve everything else exactly.

Return ONLY the improved system prompt as plain text.
No JSON wrapper. No explanation. Just the prompt.

EXPECTED_OUTPUT_SCHEMA:
plain text — no JSON schema required

PAPER_CITATION:
OPRO arXiv:2309.03409 — focused prompt optimization from gradient.
