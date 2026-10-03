import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def keyword_search(query, rows, limit=12):
    if not rows:
        return [], {}
    corpus = [tokenize(row['text']) for row in rows]
    if not any(corpus):
        return [], {}
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    pairs = sorted([(r['passage_id'], float(s)) for r,s in zip(rows,scores) if s > 0],
                   key=lambda pair: (-pair[1],pair[0]))[:limit]
    return [p[0] for p in pairs], dict(pairs)
