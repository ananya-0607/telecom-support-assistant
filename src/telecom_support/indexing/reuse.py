"""Reuse persisted normalized vectors by exact text, not document/chunk position."""
import hashlib
import json
import numpy as np
from telecom_support.indexing.vectors import COLLECTION
from telecom_support.ingestion.chunking import TOKENIZER_MODEL


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def encode_with_reuse(index, passages):
    cached = {}
    manifest = json.loads((index.path/'manifest.json').read_text(encoding='utf-8'))
    if getattr(index, 'client', None) is not None and manifest.get('model') == TOKENIZER_MODEL:
        by_id = {row['passage_id']:row['text'] for row in index.rows}
        offset = None
        while True:
            points, offset = index.client.scroll(COLLECTION, limit=128, offset=offset,
                with_payload=False, with_vectors=True)
            for point in points:
                text = by_id.get(str(point.id))
                vector = point.vector
                if text is not None and isinstance(vector, list) and vector and np.isfinite(vector).all():
                    cached[text_hash(text)] = vector
            if offset is None:
                break
    missing = {}
    for passage in passages:
        key = text_hash(passage['text'])
        if key not in cached:
            missing[key] = passage['text']
    if missing:
        encoded = index.embedder.encode(list(missing.values()))
        cached.update((key, [float(x) for x in vector]) for key, vector in zip(missing, encoded))
    vectors = np.asarray([cached[text_hash(p['text'])] for p in passages], dtype=np.float32)
    return vectors, {'embedded_unique_texts':len(missing),
        'reused_passages':sum(text_hash(p['text']) not in missing for p in passages)}
