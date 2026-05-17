"""
USA Immigration Dataset Pipeline — Main Orchestrator

Usage:
  python main.py setup          # One-time AWS setup (bucket, budget)
  python main.py crawl          # Crawl all 8 source tiers (priority order)
  python main.py parse          # Parse raw S3 files → canonical corpus
  python main.py normalize      # Normalize + dedup + chunk
  python main.py qa             # Generate Q&A pairs via Bedrock (Account 2)
  python main.py validate       # Validate all outputs, write reports
  python main.py run            # Full pipeline end-to-end
  python main.py budget         # Check current Account 1 spend

Budget enforcement: stops automatically when Account 1 spend >= $130.
Switch AWS credentials to Account 2 before running `qa` step.
"""

import sys
from pathlib import Path

import structlog
import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from scripts.budget_monitor import BudgetExceeded, check_budget
from scripts.s3_store import upload_json, upload_jsonl

log = structlog.get_logger()
console = Console()
app = typer.Typer(help="USA Immigration Dataset Pipeline")


def _guard_budget(label: str):
    try:
        spend = check_budget(label=label)
        console.print(f"[green]Budget OK[/green]: ${spend:.2f} / $130.00 spent")
    except BudgetExceeded as e:
        console.print(f"[bold red]BUDGET LIMIT REACHED[/bold red]: {e}")
        console.print("Switch to Account 2 credentials and continue with: python main.py qa")
        raise typer.Exit(1)


@app.command()
def setup():
    """One-time AWS setup: create S3 bucket and AWS Budget alert."""
    from scripts.setup_aws import main as aws_setup
    aws_setup()


@app.command()
def budget():
    """Check current Account 1 AWS spend."""
    _guard_budget("manual_check")


@app.command()
def crawl(
    source: str = typer.Option("all", help="all | uscis | statutes | eoir | dhs | open | secondary | community"),
):
    """Crawl sources in priority order. Checks budget before each source tier."""
    _guard_budget("pre_crawl")

    log_path = Path("metadata/crawl_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    crawlers = {
        "uscis": ("scripts.crawl.crawl_uscis", "USCISCrawler"),
        "statutes": ("scripts.crawl.crawl_statutes", "StatutesCrawler"),
        "eoir": ("scripts.crawl.crawl_eoir", "EOIRCrawler"),
        "dhs": ("scripts.crawl.crawl_dhs_stats", "DHSStatsCrawler"),
        "open": ("scripts.crawl.crawl_open_datasets", "OpenDatasetsCrawler"),
        "secondary": ("scripts.crawl.crawl_secondary", "SecondaryCrawler"),
        "community": ("scripts.crawl.crawl_community", "CommunityCrawler"),
    }

    order = list(crawlers.keys()) if source == "all" else [source]

    for name in order:
        if name not in crawlers:
            console.print(f"[red]Unknown source: {name}[/red]")
            continue
        _guard_budget(f"pre_{name}_crawl")
        console.print(f"[cyan]Crawling: {name}[/cyan]")
        module_path, cls_name = crawlers[name]
        import importlib
        mod = importlib.import_module(module_path)
        crawler_cls = getattr(mod, cls_name)
        crawler_cls(log_path).run()
        console.print(f"[green]Done: {name}[/green]")

    console.print("[bold green]Crawl phase complete.[/bold green]")


@app.command()
def parse():
    """Parse raw S3 files into canonical corpus."""
    _guard_budget("pre_parse")
    from scripts.parse.run_parse import parse_all
    records = parse_all()
    console.print(f"[green]Parsed {len(records)} corpus records.[/green]")


@app.command()
def normalize():
    """Normalize metadata, deduplicate, and chunk corpus."""
    _guard_budget("pre_normalize")
    from scripts.s3_store import download_jsonl, list_keys, upload_jsonl
    from scripts.normalize.normalize_metadata import normalize_corpus
    from scripts.normalize.dedup import dedup
    from scripts.normalize.chunker import chunk_corpus

    # Load all canonical corpus batches
    raw_keys = list_keys("data_processed/canonical_corpus/")
    all_records = []
    for key in raw_keys:
        if key.endswith(".jsonl"):
            rel = key.replace("v1/", "", 1)
            try:
                all_records.extend(download_jsonl(rel))
            except Exception as e:
                log.warning("load_error", key=key, error=str(e))

    console.print(f"Loaded {len(all_records)} raw corpus records")

    normalized = normalize_corpus(all_records)
    deduped = dedup(normalized)
    upload_jsonl(deduped, "data_processed/canonical_corpus/normalized.jsonl")
    console.print(f"[green]Normalized: {len(deduped)} records[/green]")

    chunks = chunk_corpus(deduped)
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch_num = i // batch_size
        upload_jsonl(chunks[i:i+batch_size], f"data_processed/chunks/batch_{batch_num:04d}.jsonl")
    console.print(f"[green]Chunked: {len(chunks)} chunks[/green]")


@app.command()
def qa(first_scope: bool = typer.Option(True, help="Restrict to 5 first-scope topics")):
    """
    Generate Q&A pairs via AWS Bedrock (Account 2 credentials required).
    Set BEDROCK_* env vars to Account 2 before running.
    """
    console.print("[yellow]Using Bedrock (Account 2). Ensure BEDROCK_* env vars are set.[/yellow]")
    from scripts.s3_store import download_jsonl
    from scripts.qa_generation.generate_qa import generate_qa_corpus, build_eval_set
    from scripts.validation.validate import flag_contradictions

    records = download_jsonl("data_processed/canonical_corpus/normalized.jsonl")
    console.print(f"Loaded {len(records)} normalized records")

    qa_pairs = generate_qa_corpus(records, first_scope_only=first_scope)
    qa_pairs = flag_contradictions(qa_pairs)

    upload_jsonl(qa_pairs, "data_processed/qa_pairs/all.jsonl")
    console.print(f"[green]Generated {len(qa_pairs)} Q&A pairs[/green]")

    eval_set = build_eval_set(qa_pairs)
    upload_jsonl(eval_set, "data_processed/eval/eval_set.jsonl")
    console.print(f"[green]Eval set: {len(eval_set)} items[/green]")


@app.command()
def validate():
    """Validate all outputs and write coverage/quality reports to S3 and reports/."""
    from scripts.s3_store import download_jsonl
    from scripts.validation.validate import validate_corpus, validate_chunks, validate_qa_pairs
    from scripts.validation.coverage_report import generate_coverage_report, generate_data_quality_report

    corpus = download_jsonl("data_processed/canonical_corpus/normalized.jsonl")
    valid_corpus, rejected_corpus = validate_corpus(corpus)

    chunk_keys = list_keys("data_processed/chunks/")
    all_chunks = []
    for key in chunk_keys:
        if key.endswith(".jsonl"):
            rel = key.replace("v1/", "", 1)
            try:
                all_chunks.extend(download_jsonl(rel))
            except Exception:
                pass
    valid_chunks, rejected_chunks = validate_chunks(all_chunks)

    qa_pairs = download_jsonl("data_processed/qa_pairs/all.jsonl")
    valid_qa, rejected_qa = validate_qa_pairs(qa_pairs)

    # Upload cleaned outputs
    upload_jsonl(valid_corpus, "data_processed/canonical_corpus/validated.jsonl")
    upload_jsonl(valid_qa, "data_processed/qa_pairs/validated.jsonl")

    coverage = generate_coverage_report(valid_corpus, valid_qa)
    quality = generate_data_quality_report(rejected_corpus, rejected_chunks, rejected_qa)

    upload_json({"coverage": coverage}, "reports/coverage.json")
    import sys, json as _json
    sys.stdout.buffer.write((_json.dumps(coverage, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    console.print(f"\n[green]Validation complete.[/green]")
    console.print(f"  Valid corpus: {len(valid_corpus)} | Rejected: {len(rejected_corpus)}")
    console.print(f"  Valid Q&A:    {len(valid_qa)} | Rejected: {len(rejected_qa)}")


@app.command()
def run(first_scope: bool = typer.Option(True)):
    """Run full pipeline: crawl → parse → normalize → qa → validate."""
    console.print("[bold cyan]Starting full pipeline run[/bold cyan]")
    _guard_budget("pipeline_start")

    crawl(source="all")
    _guard_budget("post_crawl")
    parse()
    _guard_budget("post_parse")
    normalize()
    _guard_budget("post_normalize")

    console.print("[yellow]Crawl/parse/normalize complete.[/yellow]")
    console.print("[yellow]Switch to Account 2 credentials, then run: python main.py qa[/yellow]")
    console.print("[yellow]Then run: python main.py validate[/yellow]")


def list_keys(prefix: str):
    from scripts.s3_store import list_keys as _list_keys
    return _list_keys(prefix)


if __name__ == "__main__":
    app()
