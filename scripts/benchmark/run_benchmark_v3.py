"""
Benchmark v3: evaluate the stable fine-tuned model alongside the same baselines.

Models evaluated:
  1. llama3_8b_base_zeroshot   — meta.llama3-8b-instruct-v1:0 via Bedrock
  2. claude_sonnet_zeroshot    — us.anthropic.claude-sonnet-4-6 via Bedrock
  3. llama32_finetuned_v3      — nshportun/usa-immigration-llama-3.2-3b-v3
                                 via SageMaker real-time endpoint

Output:
  data_local/benchmark_v3/results.jsonl  — per-question scores
  data_local/benchmark_v3/summary.json   — aggregate table

Run:
  BENCHMARK_SAGEMAKER_ENDPOINT=immigration-llama32-benchmark \\
    python scripts/benchmark/run_benchmark_v3.py

Reuses all inference/judge/sampling logic from run_benchmark.py.
"""

import json
import os
import pathlib
import sys
import time

import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

BASE     = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR  = BASE / "data_local" / "benchmark_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import shared logic from run_benchmark
sys.path.insert(0, str(BASE))
from scripts.benchmark.run_benchmark import (
    make_bedrock,
    make_smr,
    bedrock_converse,
    sagemaker_infer,
    judge,
    load_sample,
    BEDROCK_LLAMA_MODEL,
    BEDROCK_CLAUDE_MODEL,
    REQUEST_DELAY,
)

SAGEMAKER_ENDPOINT = os.getenv("BENCHMARK_SAGEMAKER_ENDPOINT", "")
SAGEMAKER_REGION   = os.getenv("SAGEMAKER_REGION", "us-west-2")

MODEL_KEY_V3 = "llama32_finetuned_v3"


def summarize_v3(results_path: pathlib.Path):
    """Summarize results to summary.json and print table."""
    from collections import defaultdict
    from datetime import datetime, timezone

    rows       = [json.loads(l) for l in open(results_path, encoding="utf-8")]
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

    overall      = {m: stats([r["scores"].get(m, -1) for r in rows]) for m in all_models}
    by_topic     = defaultdict(lambda: defaultdict(list))
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
        "judge_model":   "us.anthropic.claude-sonnet-4-6",
        "scoring_scale": "0-3 (0=wrong, 1=partial, 2=mostly correct, 3=fully correct)",
        "overall":       overall,
        "by_subtopic":   by_topic_stats,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    labels = {
        "llama3_8b_base_zeroshot":  "Llama 3 8B (zero-shot)",
        "claude_sonnet_zeroshot":   "Claude Sonnet 4.6 (zero-shot)",
        MODEL_KEY_V3:               "Llama 3.2 3B fine-tuned v3 (ours)",
    }
    lines = ["\n=== BENCHMARK V3 RESULTS ===\n",
             f"{'Model':<40} {'Mean (0-3)':<14} {'% Score=3':<14} {'N'}"]
    lines.append("-" * 75)
    for m in all_models:
        s     = overall[m]
        label = labels.get(m, m)
        lines.append(
            f"{label:<40} {str(s['mean']):<14} {str(s['pct_full'])+'%':<14} {s['n']}"
        )
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    return summary


def run():
    if not SAGEMAKER_ENDPOINT:
        print("ERROR: set BENCHMARK_SAGEMAKER_ENDPOINT to the deployed endpoint name.")
        print("Example: BENCHMARK_SAGEMAKER_ENDPOINT=immigration-llama32-benchmark python scripts/benchmark/run_benchmark_v3.py")
        sys.exit(1)

    bedrock = make_bedrock()
    smr     = make_smr()
    sample  = load_sample()

    results_path = OUT_DIR / "results.jsonl"
    done_ids: set[str] = set()
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["qa_id"])
        log.info("resuming", already_done=len(done_ids))

    models = {
        "llama3_8b_base_zeroshot": lambda q: bedrock_converse(bedrock, BEDROCK_LLAMA_MODEL, q),
        "claude_sonnet_zeroshot":  lambda q: bedrock_converse(bedrock, BEDROCK_CLAUDE_MODEL, q),
        MODEL_KEY_V3:              lambda q: sagemaker_infer(smr, SAGEMAKER_ENDPOINT, q),
    }

    log.info("benchmark_v3_start",
             endpoint=SAGEMAKER_ENDPOINT,
             models=list(models.keys()),
             n=len(sample))

    with open(results_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(sample):
            if row["qa_id"] in done_ids:
                continue
            log.info("eval", i=i+1, n=len(sample),
                     subtopic=row["immigration_subtopic"], qa_id=row["qa_id"])

            record = {
                "qa_id":           row["qa_id"],
                "question":        row["question"],
                "reference":       row["answer"],
                "subtopic":        row["immigration_subtopic"],
                "answer_type":     row.get("answer_type", ""),
                "authority_level": row.get("authority_level", ""),
                "scores":          {},
                "predictions":     {},
                "reasons":         {},
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

    summarize_v3(results_path)


if __name__ == "__main__":
    run()
