"""QA generation prompts for all 5 modes defined in CLAUDE.md."""

SYSTEM_PROMPT = """You are a legal dataset builder for a U.S. immigration AI project.
Your job is to generate accurate, source-grounded Q&A pairs from provided text.
Rules:
- Every answer must be derivable from the provided source text. No hallucination.
- Questions must be clear, standalone, and answerable from the text.
- Answers must be factual, concise, and cite the relevant passage.
- Output ONLY valid JSON. No explanation outside the JSON.
- If the text does not support a good Q&A pair, return an empty list [].
"""

MODE1_FAQ_EXTRACTION = """Extract existing question-answer pairs from this FAQ or help page text.
Return a JSON array of objects with keys: question, answer, source_span (exact quote from text, max 200 chars).

Source text:
{text}
"""

MODE2_RULE_TO_QUESTION = """Convert the legal/policy statements in this text into plain-language Q&A pairs.
Focus on: eligibility rules, definitions, requirements, procedures, exceptions.
Return a JSON array of objects with keys: question, answer, source_span (exact quote, max 200 chars), answer_type.
answer_type must be one of: factual, eligibility, procedural, definition, exception.
Generate at most {max_pairs} pairs.

Source text:
{text}
"""

MODE3_FORM_INSTRUCTIONS = """From this form instruction text, generate practical filing Q&A pairs.
Focus on: who qualifies, required documents, where to file, fees, what happens after filing, common mistakes.
Return a JSON array of objects with keys: question, answer, source_span (exact quote, max 200 chars), answer_type.
answer_type must be one of: eligibility, required_docs, filing_procedure, timing, post_filing.
Generate at most {max_pairs} pairs.

Source text:
{text}
"""

MODE4_PRECEDENT = """From this immigration court or BIA/AG decision text, generate legal Q&A pairs.
Focus on: what rule was established, what facts led to the outcome, how a legal term was interpreted.
Return a JSON array of objects with keys: question, answer, source_span (exact quote, max 200 chars), answer_type.
answer_type must be one of: legal_rule, case_outcome, statutory_interpretation, procedural_rule.
Generate at most {max_pairs} pairs.

Source text:
{text}
"""

MODE5_STATISTICS = """From this statistics or data table text, generate factual Q&A pairs.
Each Q&A MUST include the specific year/date and dataset name.
Do NOT mix statistics into legal/doctrinal answers.
Return a JSON array of objects with keys: question, answer, source_span (exact quote, max 200 chars), answer_type.
answer_type must be "statistical_fact".
Generate at most {max_pairs} pairs.

Source text:
{text}
"""

MODE4_PARAPHRASE = """For each Q&A pair in the input JSON, generate 2-3 natural-language paraphrases of the question only.
Keep the answer identical. Return a JSON array where each object has:
  question (the new paraphrase), answer (unchanged), source_span (unchanged), answer_type (unchanged),
  paraphrase_of (the original question).

Input Q&A pairs:
{text}
"""


def get_prompt(mode: str, text: str, max_pairs: int = 10) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given mode."""
    templates = {
        "faq": MODE1_FAQ_EXTRACTION,
        "rule": MODE2_RULE_TO_QUESTION,
        "form": MODE3_FORM_INSTRUCTIONS,
        "precedent": MODE4_PRECEDENT,
        "statistics": MODE5_STATISTICS,
    }
    template = templates.get(mode, MODE2_RULE_TO_QUESTION)
    # Hard cap at 6000 chars to stay well within Bedrock token limits
    user = template.format(text=text[:6000], max_pairs=max_pairs)
    return SYSTEM_PROMPT, user


def infer_mode(record: dict) -> str:
    """Pick the best generation mode for a record based on source_type."""
    source_type = record.get("source_type", "")
    data_class = record.get("data_class", "")
    title = record.get("title", "").lower()
    if source_type in ("official_faq",) or "faq" in title or "help" in title:
        return "faq"
    if source_type in ("official_form",) or any(f in title for f in ["i-130", "i-485", "n-400", "i-765"]):
        return "form"
    if source_type in ("case_law",) or "bia" in title or "decision" in title:
        return "precedent"
    if data_class == "statistics" or source_type == "statistics":
        return "statistics"
    return "rule"
