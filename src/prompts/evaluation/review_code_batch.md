PROMPT:
{system_prompt}

Review the following {n_files} Python files for bugs.
Review each file COMPLETELY INDEPENDENTLY.
Do not let your analysis of one file influence another.

{files_block}

Return ONLY valid JSON:
{
  "reviews": [
    {
      "file_path": "path/to/file.py",
      "issues": [
        {
          "line": 12,
          "bug_type": "undefined_variable",
          "description": "Variable 'total' used before assignment",
          "confidence": 0.95
        }
      ],
      "overall_confidence": 0.85,
      "summary": "one sentence"
    }
  ]
}

EXPECTED_OUTPUT_SCHEMA:
{"reviews": list of AgentReview dicts}

PAPER_CITATION:
KAMI arXiv:2512.07497 — independent review prevents context pollution
("Chekhov's gun" effect where all context is treated as signal).
