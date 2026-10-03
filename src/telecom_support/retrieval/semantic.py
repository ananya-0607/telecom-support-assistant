from qdrant_client import models
from telecom_support.indexing.vectors import COLLECTION

def semantic_search(client, vector, rows, threshold, limit=12):
    if not rows:
        return [], {}
    # The SQL scope has already applied category/product and general-KB rules.
    scope = models.Filter(must=[models.HasIdCondition(has_id=[r['passage_id'] for r in rows])])
    result = client.query_points(collection_name=COLLECTION, query=vector,
        query_filter=scope, limit=limit, score_threshold=threshold, with_payload=False)
    return [str(p.id) for p in result.points], {str(p.id):float(p.score) for p in result.points}
