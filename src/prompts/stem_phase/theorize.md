PROMPT:
You are an expert AI researcher designing a code review agent.

Task class you have received: {task_description}

Before any specialization, reason deeply about this task class.
Think about what makes it genuinely hard — not surface-level difficulty,
but the structural reasons expert reviewers differ from novices.

Return ONLY valid JSON matching this schema exactly:
{
  "difficulty_factors": ["list of what makes this task hard"],
  "common_patterns": ["concrete patterns that appear in this bug class"],
  "expert_strategies": ["specific strategies experts use, not generic advice"],
  "information_needed": ["what context helps solve this well"]
}

EXPECTED_OUTPUT_SCHEMA:
{"difficulty_factors": list[str], "common_patterns": list[str],
 "expert_strategies": list[str], "information_needed": list[str]}

PAPER_CITATION:
PromptAgent arXiv:2310.16427 — theory-first before optimization.
KAMI arXiv:2512.07497 — "premature action without grounding" failure archetype.
