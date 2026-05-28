"""
Benchmark three models on the USA Immigration Law Q&A eval set.

Models evaluated:
  1. llama3_8b_base   — meta.llama3-8b-instruct-v1:0, zero-shot via Bedrock
                        (closest available base model; same Llama 3 architecture family)
  2. claude_sonnet    — us.anthropic.claude-sonnet-4-6, zero-shot via Bedrock
                        (strong frontier model baseline)
  3. llama32_finetuned — nshportun/usa-immigration-llama-3.2-3b via SageMaker endpoint
                         (our fine-tuned model)

Evaluation:
  - Stratified sample of 8 per subtopic (~100 questions across 13 subdomains)
  - LLM-as-judge: Claude Sonnet 4.6 scores each answer 0–3
      0 = wrong / hallucinated
      1 = partially correct
      2 = mostly correct, minor gaps
      3 = fully correct and grounded
  - Reports mean score, % fully correct, per model and per subdomain

Output:
  - data_local/benchmark/results.jsonl  — per-question scores
  - data_local/benchmark/summary.json   — aggregate table

Run: python scripts/benchmark/run_benchmark.py
"""

import boto3
import json
import os
import pathlib
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

BASE      = pathlib.Path(__file__).resolve().parents[2]
EVAL_PATH = BASE / "data_local" / "splits" / "eval.jsonl"
OUT_DIR   = BASE / "data_local" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEDROCK_REGION       = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_JUDGE_MODEL  = "us.anthropic.claude-sonnet-4-6"
BEDROCK_CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-6"
BEDROCK_LLAMA_MODEL  = "meta.llama3-8b-instruct-v1:0"

SAGEMAKER_ENDPOINT   = os.getenv("BENCHMARK_SAGEMAKER_ENDPOINT", "")
SAGEMAKER_REGION     = os.getenv("SAGEMAKER_REGION", "us-east-1")

SAMPLE_PER_SUBTOPIC  = 8
JUDGE_RETRIES        = 3
REQUEST_DELAY        = 0.5

SYSTEM_PROMPT = (
    "You are an expert on U.S. immigration law and policy. "
    "Answer accurately and specifically based on official USCIS, 8 CFR, and BIA sources. "
    "Be concise and factual."
)


# ── Clients ───────────────────────────────────────────────────────────────────

def make_bedrock():
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=BEDROCK_REGION,
    )

def make_smr():
    return boto3.client(
        "sagemaker-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=SAGEMAKER_REGION,
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

def bedrock_converse(client, model_id: str, question: str,
                     system: str = SYSTEM_PROMPT, max_tokens: int = 400) -> str:
    resp = client.converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()


def sagemaker_infer(smr_client, endpoint_name: str, question: str) -> str:
    """Call a SageMaker real-time endpoint (DJL LMI container).

    DJL LMI uses the 'inputs' + 'parameters' schema, same as TGI.
    The chat template is applied manually so the model sees the full
    Llama 3.1 instruction format.
    """
    # Build the Llama 3.1 chat-formatted prompt manually
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 350,
            "do_sample": False,
            "return_full_text": False,
        },
    }
    for attempt in range(3):
        try:
            resp = smr_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            result = json.loads(resp["Body"].read())
            # DJL returns {"generated_text": "..."} or [{"generated_text": "..."}]
            if isinstance(result, list):
                return result[0].get("generated_text", "").strip()
            if isinstance(result, dict):
                return result.get("generated_text", str(result)).strip()
            return str(result).strip()
        except Exception as e:
            log.warning("sagemaker_error", attempt=attempt, error=str(e)[:120])
            time.sleep(5)
    return "[ENDPOINT_ERROR]"


# ── LLM Judge ─────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert evaluator assessing answers to U.S. immigration law questions.\n\n"
    "Score the model's answer against the reference answer on a 0–3 scale:\n"
    "  3 = Fully correct: covers all key facts, no hallucinations, matches reference\n"
    "  2 = Mostly correct: correct core answer but missing some details\n"
    "  1 = Partially correct: some relevant info but significant gaps or minor errors\n"
    "  0 = Wrong: incorrect, hallucinated facts, or completely off-topic\n\n"
    "Respond ONLY with valid JSON: {\"score\": <0-3>, \"reason\": \"<one sentence>\"}"
)

def judge(bedrock_client, question: str, reference: str, prediction: str) -> dict:
    prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Model answer: {prediction}\n\n"
        "Score the model answer."
    )
    for _ in range(JUDGE_RETRIES):
        try:
            text = bedrock_converse(
                bedrock_client, BEDROCK_JUDGE_MODEL, prompt,
                system=JUDGE_SYSTEM, max_tokens=150
            )
            # Strip markdown code fences if present
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            result = json.loads(text)
            if "score" in result:
                return {"score": int(result["score"]), "reason": result.get("reason", "")}
        except Exception as e:
            log.warning("judge_error", error=str(e)[:80])
            time.sleep(2)
    return {"score": -1, "reason": "judge_failed"}


# ── Sampling ──────────────────────────────────────────────────────────────────

def load_sample() -> list[dict]:
    rows = []
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    by_topic: dict[str, list] = defaultdict(list)
    for r in rows:
        by_topic[r["immigration_subtopic"]].append(r)
    random.seed(42)
    sample = []
    for topic, items in sorted(by_topic.items()):
        sample.extend(random.sample(items, min(SAMPLE_PER_SUBTOPIC, len(items))))
    log.info("sample_loaded", n=len(sample), subtopics=len(by_topic))
    return sample


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    bedrock = make_bedrock()
    smr     = make_smr() if SAGEMAKER_ENDPOINT else None
    sample  = load_sample()

    results_path = OUT_DIR / "results.jsonl"
    done_ids: set[str] = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["qa_id"])
        log.info("resuming", already_done=len(done_ids))

    models: dict[str, callable] = {
        "llama3_8b_base_zeroshot": lambda q: bedrock_converse(bedrock, BEDROCK_LLAMA_MODEL, q),
        "claude_sonnet_zeroshot":  lambda q: bedrock_converse(bedrock, BEDROCK_CLAUDE_MODEL, q),
    }
    if smr and SAGEMAKER_ENDPOINT:
        models["llama32_finetuned"] = lambda q: sagemaker_infer(smr, SAGEMAKER_ENDPOINT, q)
    else:
        log.warning("no_sagemaker_endpoint",
                    msg="Set BENCHMARK_SAGEMAKER_ENDPOINT env var to include fine-tuned model")

    with open(results_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(sample):
            if row["qa_id"] in done_ids:
                continue
            log.info("eval", i=i+1, n=len(sample),
                     subtopic=row["immigration_subtopic"], qa_id=row["qa_id"])

            record = {
                "qa_id":          row["qa_id"],
                "question":       row["question"],
                "reference":      row["answer"],
                "subtopic":       row["immigration_subtopic"],
                "answer_type":    row.get("answer_type", ""),
                "authority_level": row.get("authority_level", ""),
                "scores":         {},
                "predictions":    {},
                "reasons":        {},
            }

            for name, fn in models.items():
                pred = fn(row["question"])
                time.sleep(REQUEST_DELAY)
                j = judge(bedrock, row["question"], row["answer"], pred)
                time.sleep(REQUEST_DELAY)
                record["predictions"][name] = pred
                record["scores"][name]      = j["score"]
                record["reasons"][name]     = j["reason"]
                log.info("scored", model=name, score=j["score"])

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    summarize(results_path)


def summarize(results_path: pathlib.Path):
    rows = [json.loads(l) for l in open(results_path, encoding="utf-8")]
    all_models = sorted({k for r in rows for k in r["scores"]})

    def stats(scores):
        v = [s for s in scores if s >= 0]
        if not v:
            return {"n": 0, "mean": None, "pct_full": None}
        return {
            "n": len(v),
            "mean": round(sum(v) / len(v), 3),
            "pct_full": round(100 * sum(1 for s in v if s == 3) / len(v), 1),
        }

    overall   = {m: stats([r["scores"].get(m, -1) for r in rows]) for m in all_models}
    by_topic  = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in all_models:
            by_topic[r["subtopic"]][m].append(r["scores"].get(m, -1))
    by_topic_stats = {
        t: {m: stats(by_topic[t][m]) for m in all_models}
        for t in sorted(by_topic)
    }

    summary = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "eval_date":     datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sample_size":   len(rows),
        "judge_model":   BEDROCK_JUDGE_MODEL,
        "scoring_scale": "0-3 (0=wrong, 1=partial, 2=mostly correct, 3=fully correct)",
        "overall":       overall,
        "by_subtopic":   by_topic_stats,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print table
    labels = {
        "llama3_8b_base_zeroshot": "Llama 3 8B (zero-shot)",
        "claude_sonnet_zeroshot":  "Claude Sonnet 4.6 (zero-shot)",
        "llama32_finetuned":       "Llama 3.2 3B fine-tuned (ours)",
    }
    lines = ["\n=== BENCHMARK RESULTS ===\n",
             f"{'Model':<35} {'Mean (0-3)':<14} {'% Score=3':<14} {'N'}"]
    lines.append("-" * 70)
    for m in all_models:
        s = overall[m]
        label = labels.get(m, m)
        lines.append(f"{label:<35} {str(s['mean']):<14} {str(s['pct_full'])+'%':<14} {s['n']}")
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    return summary


if __name__ == "__main__":
    run()
