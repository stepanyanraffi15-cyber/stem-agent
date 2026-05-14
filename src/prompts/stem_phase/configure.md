PROMPT:
You are an AI agent that has just developed a theory about a task class.

Your theory:
{theory_json}

Based on this theory, write your own system prompt that you will use
when performing code reviews. Your system prompt must:
- Be concrete and specific — encode your theory as explicit instructions
- Not be generic advice — every sentence must be actionable
- Be written in second person as if instructing yourself

Return ONLY valid JSON:
{
  "system_prompt": "the complete system prompt you will use",
  "tools_selected": ["list of tool names you will use"],
  "reasoning": "why these choices follow from your theory"
}

EXPECTED_OUTPUT_SCHEMA:
{"system_prompt": str, "tools_selected": list[str], "reasoning": str}

PAPER_CITATION:
PromptAgent arXiv:2310.16427 — strategic prompt construction from task analysis.
