PROMPT:
You are merging two system prompts while preserving critical behaviors.

Base prompt (proven, must be preserved):
{base_prompt}

New prompt (contains improvements to incorporate):
{new_prompt}

Locked skills that MUST appear in the output:
{locked_skills}

Produce a merged prompt that:
- Incorporates improvements from the new prompt
- Preserves ALL language covering the locked skills
- Removes any language from the new prompt that conflicts with locked skills
- Is coherent and reads naturally

Return ONLY the merged system prompt as plain text.

EXPECTED_OUTPUT_SCHEMA:
plain text — no JSON schema required

PAPER_CITATION:
Kirkpatrick et al. 2017 EWC — preserve important weights during updates.
AgenTracer arXiv:2509.03312 — rollback to prevent catastrophic forgetting.
