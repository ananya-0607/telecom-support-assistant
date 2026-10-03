"""Tag prepared KB chunks using Groq. Supports sample selection and resume.

Sample: python scripts/classify_kb.py --chunks 1 8 15
Full:   python scripts/classify_kb.py
No source PDFs or prepared input records are modified.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", help="Chunk numbers to process; selects matches across PDFs")
    parser.add_argument("--limit", type=int, help="Maximum selected chunks to process")
    parser.add_argument("--delay", type=float, default=30.0, help="Seconds between API requests (default: 30)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1 or args.delay < 0:
        parser.error("Limit must be positive; delay must be nonnegative.")
    try:
        from dotenv import load_dotenv
        from groq import Groq, APIStatusError, APIConnectionError
        from telecom_support.ingestion.schemas import KBChunk
        from telecom_support.ingestion.metadata import cache_key, classify_chunk, enrich_chunk, load_cache, save_cache
    except ImportError:
        print("Install dependencies: python -m pip install -r requirements.txt")
        return 1
    folder = ROOT / "data/indexes"
    try:
        original = [KBChunk.model_validate_json(line) for line in
                    (folder / "kb_sections.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not original or len({c.source_id for c in original}) != len(original):
            raise ValueError("Prepared KB must contain nonempty, uniquely identified chunks.")
        selected = [c for c in original if not args.chunks or c.chunk_number in args.chunks]
        if args.chunks and set(args.chunks) - {c.chunk_number for c in selected}:
            raise ValueError("Some selected chunk numbers do not exist.")
        if args.limit:
            selected = selected[:args.limit]
        entries = load_cache(folder / "kb_metadata_cache.jsonl")
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1

    load_dotenv(ROOT / ".env", override=False)
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not model:
        print("FAIL: GROQ_MODEL is empty.")
        return 1
    requests = 0
    failure = None
    client = None
    try:
        for chunk in selected:
            key = cache_key(chunk, model)
            if key in entries:
                print(f"Chunk {chunk.chunk_number}: cached")
                continue
            if not api_key or api_key == "your_key_here":
                failure = "Set GROQ_API_KEY in .env. Never share your key."
                break
            if client is None:
                client = Groq(api_key=api_key, timeout=60.0, max_retries=0)
            if requests:
                time.sleep(args.delay)
            requests += 1
            try:
                result = classify_chunk(client, chunk, model)
                entries[key] = result
                save_cache(folder / "kb_metadata_cache.jsonl", entries)
                print(f"Chunk {chunk.chunk_number}: {result.title}; categories={[x.value for x in result.categories]}; products={[x.value for x in result.products]}")
            except APIStatusError as error:
                failure = f"Groq HTTP {error.status_code}. Check key/model access or quota; rerun later to resume."
                break
            except APIConnectionError:
                failure = "Groq connection failed or timed out; rerun later to resume."
                break
            except (OSError, ValueError):
                failure = "Invalid/incomplete metadata or cache write failed. Inspect configuration before retrying."
                break
    except KeyboardInterrupt:
        failure = "Interrupted; completed cached results retained."
    finally:
        if client is not None:
            client.close()

    # Rebuild output from current input plus compatible cache, never stale enriched files.
    enriched = []
    for chunk in original:
        key = cache_key(chunk, model)
        if key in entries:
            enriched.append(enrich_chunk(chunk, entries[key], model, key))
        else:
            enriched.append(chunk)
    classified = [c for c in enriched if c.metadata_status == "classified"]
    summary = {
        "total_chunks": len(original), "selected_chunks": len(selected), "api_requests_this_run": requests,
        "classified_chunks": len(classified), "pending_chunks": len(original) - len(classified),
        "model": model, "failure": failure,
        "note": "Predicted titles and labels are not human approval or procedure verification.",
    }
    try:
        atomic_text(folder / "kb_sections_enriched.jsonl", "".join(c.model_dump_json() + "\n" for c in enriched))
        atomic_text(folder / "kb_metadata_summary.json", json.dumps(summary, indent=2) + "\n")
    except OSError:
        print("FAIL: Cannot write enriched output. Cached successes are retained where saved.")
        return 1
    print(json.dumps(summary, indent=2))
    if failure:
        print(f"FAIL: {failure}")
        return 1
    print("PASS: Selected chunks processed; inspect labels before full ingestion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
