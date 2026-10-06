from telecom_support.indexing.storage import fetch_evidence
from .keyword import keyword_search
from .semantic import semantic_search
from .ranking import reciprocal_rank_fusion

def matches(row, categories, products, scope):
    if scope == 'global':
        return True
    if row['source_type'] == 'kb' and not row['categories'] and not row['products']:
        return True
    product_ok = not products or bool(set(products) & set(row['products']))
    category_ok = not categories or bool(set(categories) & set(row['categories']))
    return product_ok and (scope == 'product' or category_ok)

def retrieve(index, complaint, labels, threshold=0.30, per_type=3):
    vector = index.embedder.encode([complaint])[0].tolist()
    categories = [c.value for c in labels.categories]
    products = [p.value for p in labels.products]
    evidence, diagnostics = [], []
    for source_type in ['ticket','kb']:
        for scope in (['category_product','product','global'] if categories or products else ['global']):
            rows = [r for r in index.rows if r['source_type'] == source_type and matches(r,categories,products,scope)]
            semantic, sem_scores = semantic_search(index.client,vector,rows,threshold)
            lexical, lex_scores = keyword_search(complaint,rows)
            ranking, scores = reciprocal_rank_fusion([semantic,lexical])
            by_id = {r['passage_id']:r for r in rows}
            unique, used = [], set()
            for pid in ranking:
                # BM25 boosts ordering but cannot independently admit evidence.
                if pid not in sem_scores:
                    continue
                if by_id[pid]['source_id'] not in used:
                    unique.append(pid)
                    used.add(by_id[pid]['source_id'])
            # Keep the narrowest scope with qualifying evidence; three is a cap,
            # not a quota to fill with broader or keyword-only matches.
            if unique or scope == 'global':
                break
        diagnostics.append({'source_type':source_type, 'scope':scope,
            'semantic_hits':len(semantic), 'keyword_hits':len(lexical), 'threshold':threshold,
            'selected_sources':min(len(unique), per_type),
            'warning':None})
        for pid in unique[:per_type]:
            stored = fetch_evidence(index.database,pid)
            evidence.append({'citation_id':f'S{len(evidence)+1}', 'passage_id':pid,
                **stored, 'rrf_score':scores[pid], 'semantic_score':sem_scores.get(pid),
                'keyword_score':lex_scores.get(pid)})
    return evidence, diagnostics
