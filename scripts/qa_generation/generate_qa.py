"""
Generate Q&A pairs from canonical corpus using Claude via AWS Bedrock (Account 2 IAM).
"""

import json
import os
import time
import uuid

import botocore.exceptions
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scripts.aws_config import BEDROCK_MODEL_ID, bedrock_client
from scripts.qa_generation.prompts import get_prompt, infer_mode
from scripts.s3_store import upload_jsonl

log = structlog.get_logger()

BATCH_SIZE = int(os.getenv("QA_BATCH_SIZE", "10"))
MAX_PAIRS_PER_DOC = int(os.getenv("QA_MAX_PAIRS_PER_DOC", "50"))
HIGH_RISK_TOPICS = {"asylum", "removal", "admissibility", "humanitarian"}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=20),
    retry=retry_if_exception_type(botocore.exceptions.ClientError),
    reraise=True,
)
def _invoke_claude(system_prompt: str, user_prompt: str) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    })
    resp = bedrock_client().invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]


def _parse_qa_response(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        import re
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return []


def _generate_paraphrases(qa_pairs: list[dict]) -> list[dict]:
    """Mode 4: generate 2-3 paraphrases per QA pair."""
    if not qa_pairs:
        return []
    import json as _json
    input_text = _json.dumps(qa_pairs[:10], indent=2)[:6000]
    system_p = "You are a legal dataset builder. Output only valid JSON."
    from scripts.qa_generation.prompts import MODE4_PARAPHRASE
    user_p = MODE4_PARAPHRASE.format(text=input_text)
    try:
        raw = _invoke_claude(system_p, user_p)
    except Exception as e:
        log.warning("paraphrase_error", error=str(e))
        return []
    paraphrases = _parse_qa_response(raw)
    out = []
    for p in paraphrases:
        if p.get("question") and p.get("answer"):
            p["extraction_type"] = "paraphrase"
            out.append(p)
    return out


def generate_qa_for_record(record: dict) -> list[dict]:
    mode = infer_mode(record)
    system_p, user_p = get_prompt(mode, record.get("text", ""), MAX_PAIRS_PER_DOC)

    try:
        raw = _invoke_claude(system_p, user_p)
    except Exception as e:
        log.warning("bedrock_error", doc_id=record.get("doc_id"), error=str(e))
        return []

    pairs = _parse_qa_response(raw)
    qa_records = []
    for pair in pairs:
        question = (pair.get("question") or "").strip()
        answer = (pair.get("answer") or "").strip()
        if not question or not answer:
            continue

        is_high_risk = any(t in HIGH_RISK_TOPICS for t in record.get("topic_tags", []))
        qa_records.append({
            "qa_id": f"qa-{uuid.uuid4().hex[:12]}",
            "question": question,
            "question_paraphrases": [],
            "answer": answer,
            "answer_type": pair.get("answer_type", "factual"),
            "extraction_type": "direct" if mode == "faq" else "rule_derived" if mode == "rule" else "case_derived" if mode == "precedent" else "rule_derived",
            "source_doc_id": record.get("doc_id", ""),
            "source_url": record.get("url", ""),
            "source_span": pair.get("source_span", ""),
            "authority_level": record.get("authority_level", ""),
            "topic_tags": record.get("topic_tags", []),
            "immigration_subtopic": next(iter(record.get("topic_tags", [])), "general"),
            "generation_mode": mode,
            "review_status": "manual_review_required" if is_high_risk else "draft",
            "time_sensitive": record.get("time_sensitive", False),
        })
    return qa_records


def generate_qa_corpus(
    corpus_records: list[dict],
    output_prefix: str = "data_processed/qa_pairs",
    first_scope_only: bool = True,
) -> list[dict]:
    if first_scope_only:
        records = [r for r in corpus_records if r.get("in_first_scope")]
        log.info("qa_scope_filter", total=len(corpus_records), in_scope=len(records))
    else:
        records = corpus_records

    primary = [r for r in records if r.get("authority_level") == "primary_official"]
    log.info("qa_generation_start", primary_records=len(primary))

    all_qa: list[dict] = []
    batch: list[dict] = []

    for i, record in enumerate(primary):
        pairs = generate_qa_for_record(record)
        all_qa.extend(pairs)
        batch.extend(pairs)

        if len(batch) >= BATCH_SIZE * 10:
            batch_num = i // (BATCH_SIZE * 10)
            upload_jsonl(batch, f"{output_prefix}/batch_{batch_num:04d}.jsonl")
            log.info("qa_batch_uploaded", batch=batch_num, pairs=len(batch), total_so_far=len(all_qa))
            batch = []

        # Mode 4: paraphrase augmentation every 10 records
        if pairs and i % 10 == 0:
            paraphrases = _generate_paraphrases(pairs)
            all_qa.extend(paraphrases)
            batch.extend(paraphrases)

        time.sleep(0.5)

    if batch:
        upload_jsonl(batch, f"{output_prefix}/batch_final.jsonl")

    log.info("qa_generation_complete", total_pairs=len(all_qa))
    return all_qa


def build_eval_set(qa_pairs: list[dict], per_category: int = 50) -> list[dict]:
    candidates = [
        q for q in qa_pairs
        if q.get("authority_level") == "primary_official"
        and q.get("source_span")
        and q.get("review_status") != "manual_review_required"
    ]

    eval_set = []
    by_type: dict[str, list] = {}
    for q in candidates:
        key = q.get("answer_type", "factual")
        by_type.setdefault(key, []).append(q)

    for answer_type, items in by_type.items():
        for item in items[:per_category]:
            item = dict(item)
            item["eval_category"] = answer_type
            eval_set.append(item)

    log.info("eval_set_built", size=len(eval_set))
    return eval_set
