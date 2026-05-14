PROMPT:
You are computing a verbal gradient for prompt optimization.

Current system prompt:
{current_prompt}

Persistent failure memory (sorted by frequency, highest first):
{failure_memory}

Recent failed reviews (agent missed the bug):
{recent_failures}

Compute a precise verbal gradient — a direction for prompt improvement:
1. What specific weakness in the prompt caused these failures?
   Quote the exact phrase(s) that are insufficient.
2. What exact change addresses it? Be specific — rewrite the phrase.
3. What must NOT change? (list what is already working)
4. Confidence that this gradient addresses the root cause: 0.0-1.0

Return ONLY valid JSON:
{
  "weakness": "specific problem, quoting exact prompt phrase",
  "proposed_change": "exact new text to replace the weakness",
  "preserve": "what must not change",
  "confidence": 0.0_to_1.0
}

EXPECTED_OUTPUT_SCHEMA:
{"weakness": str, "proposed_change": str, "preserve": str, "confidence": float}

PAPER_CITATION:
TextGrad arXiv:2406.07496 — verbal gradients as text-space optimization.
AgenTracer arXiv:2509.03312 — failure memory weights persistent errors.
