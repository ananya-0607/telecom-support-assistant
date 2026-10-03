"""Local Qdrant persistence with filterable metadata and SQL IDs."""
import math

COLLECTION = "support_passages"

def write_vectors(path, passages, vectors):
    from qdrant_client import QdrantClient, models
    if len(vectors) != len(passages) or not passages:
        raise ValueError("Each passage needs exactly one vector.")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(v) != dimension or
        not all(math.isfinite(float(x)) for x in v) for v in vectors):
        raise ValueError("Invalid vector dimensions or non-finite values.")
    client = QdrantClient(path=str(path))
    try:
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE))
        for start in range(0, len(passages), 64):
            client.upsert(COLLECTION, points=[models.PointStruct(
                id=p['passage_id'], vector=[float(x) for x in vectors[i]],
                payload={key: p[key] for key in
                         ('passage_id', 'source_id', 'source_type', 'categories', 'products')})
                for i, p in enumerate(passages[start:start+64], start)], wait=True)
        # Compare every ID, not only counts, before activating the snapshot.
        stored, offset = set(), None
        while True:
            points, offset = client.scroll(COLLECTION, limit=128, offset=offset,
                                            with_payload=False, with_vectors=False)
            stored.update(str(p.id) for p in points)
            if offset is None:
                break
        if stored != {p['passage_id'] for p in passages}:
            raise ValueError("SQL and Qdrant passage IDs do not match.")
    finally:
        client.close()
    return dimension
