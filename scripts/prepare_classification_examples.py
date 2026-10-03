"""Select compact few-shot examples from training tickets only; no API calls."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from telecom_support.ingestion.schemas import TicketRecord
from telecom_support.taxonomy import TAXONOMY

def select_examples(records, limit=14):
    if any(r.split != 'train' for r in records):
        raise ValueError('Only training records may supply examples.')
    selected, covered = [], set()
    for category in TAXONOMY['categories']:
        candidates = [r for r in records if r.category.value == category]
        if candidates:
            chosen = min(candidates, key=lambda r: (len(r.customer_complaint), r.ticket_id))
            selected.append(chosen)
            covered.update([('severity',chosen.severity.value), ('sentiment',chosen.sentiment.value)])
    while len(selected) < limit:
        remaining = [r for r in records if r not in selected]
        if not remaining:
            break
        best = max(remaining, key=lambda r: sum(x not in covered for x in
            [('severity',r.severity.value), ('sentiment',r.sentiment.value)]))
        missing = {('severity',best.severity.value), ('sentiment',best.sentiment.value)} - covered
        if not missing:
            break
        selected.append(best)
        covered.update(missing)
    return [{'ticket_id':r.ticket_id, 'complaint':r.customer_complaint, 'labels':{
        'categories':[r.category.value], 'products':[r.product.value],
        'severity':r.severity.value, 'sentiment':r.sentiment.value}} for r in selected]

if __name__ == '__main__':
    path = ROOT/'data/indexes'
    records = [TicketRecord.model_validate_json(line) for line in
               (path/'historical_tickets.jsonl').read_text(encoding='utf-8').splitlines()]
    output = {'taxonomy_version':TAXONOMY['version'], 'examples':select_examples(records)}
    (path/'classification_examples.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f"Saved {len(output['examples'])} training-only examples. No API requests.")
