PROMPT:
You are analyzing expert Python code reviews to extract detection strategies.

Below are {n_examples} Python files paired with their known bugs:

{demonstrations}

Extract the specific, concrete strategies that an expert would use to detect
these bugs. Each strategy must be:
- Actionable (not "look carefully" but "check if variable is defined before use")
- Specific to a bug category
- Generalizable beyond these exact examples

Return ONLY valid JSON:
{
  "patterns": [
    {
      "name": "snake_case_identifier",
      "strategy": "exact detection instruction",
      "applies_to": ["bug_type_1", "bug_type_2"]
    }
  ]
}

EXPECTED_OUTPUT_SCHEMA:
{"patterns": list of {"name": str, "strategy": str, "applies_to": list[str]}}

PAPER_CITATION:
Constitutional AI arXiv:2212.08073 — pattern extraction before critique.
Chu et al. 2025 arXiv:2501.17161 — SFT condition: demonstration-based.
