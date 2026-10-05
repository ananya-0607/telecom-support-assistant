"""Build a new SQL/vector snapshot; activate only after successful verification."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telecom_support.indexing.records import load_records, make_passages
from telecom_support.indexing.storage import write_database
from telecom_support.indexing.embeddings import LocalEmbedder
from telecom_support.indexing.vectors import write_vectors, COLLECTION
from telecom_support.ingestion.chunking import TOKENIZER_MODEL
from telecom_support.taxonomy import TAXONOMY

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    index_dir = ROOT / 'data' / 'indexes'
    try:
        records = load_records(index_dir)
        print('Loading local MiniLM embedding model (first run downloads weights)...', flush=True)
        embedder = LocalEmbedder(index_dir / 'embedding_model_cache')
        passages = make_passages(records, embedder.tokenizer)
        print(f'Embedding {len(passages)} passages from {len(records)} sources...', flush=True)
        vectors = embedder.encode([p['text'] for p in passages], args.batch_size)
        build_id = uuid.uuid4().hex
        snapshot = index_dir / 'search' / 'builds' / build_id
        snapshot.mkdir(parents=True)
        manifest = {'schema_version': 1, 'build_id': build_id, 'taxonomy':TAXONOMY,
            'created_at': datetime.now(timezone.utc).isoformat(), 'model': TOKENIZER_MODEL,
            'dimension': len(vectors[0]), 'collection': COLLECTION,
            'ticket_sources': sum(r.source_type == 'ticket' for r in records),
            'kb_sources': sum(r.source_type == 'kb' for r in records),
            'passages': len(passages), 'ticket_embedding_field': 'conversation',
            'kb_embedding_field': 'text', 'max_tokens': 240, 'overlap_tokens': 32,
            'evaluation_tickets_indexed': 0,
            'note': 'Full rebuild. Older snapshots retained; incremental updates are not implemented.'}
        write_database(snapshot / 'evidence.sqlite3', records, passages, manifest)
        write_vectors(snapshot / 'qdrant', passages, vectors)
        (snapshot / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        active = index_dir / 'search' / 'active.json'
        temporary = active.with_suffix('.tmp')
        temporary.write_text(json.dumps({'build_id': build_id, 'taxonomy':TAXONOMY}, indent=2), encoding='utf-8')
        temporary.replace(active)
        print(json.dumps(manifest, indent=2))
        print(f'PASS: Verified snapshot activated at {snapshot}')
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'FAIL: {exc}. Previous active snapshot remains unchanged.', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
