"""Classification metrics; no network calls and no additional dependencies."""
import hashlib
import json
import math
from statistics import mean

FIELDS = ('categories', 'products', 'severity', 'sentiment')

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()

def same_field(field, expected, predicted):
    if field in ('categories', 'products'):
        return set(expected[field]) == set(predicted[field])
    return expected[field] == predicted[field]

def category_f1(rows):
    # Include labels appearing in references OR predictions; undefined F1 is zero.
    labels = sorted({c for row in rows for side in ('expected', 'prediction')
                     for c in row[side]['categories']})
    scores = []
    for label in labels:
        tp = sum(label in r['expected']['categories'] and label in r['prediction']['categories'] for r in rows)
        fp = sum(label not in r['expected']['categories'] and label in r['prediction']['categories'] for r in rows)
        fn = sum(label in r['expected']['categories'] and label not in r['prediction']['categories'] for r in rows)
        denominator = 2*tp + fp + fn
        scores.append(2*tp/denominator if denominator else 0.0)
    return {'value':mean(scores) if scores else None, 'labels_included':labels}

def group_metrics(rows):
    if not rows:
        return {'valid_predictions':0, 'field_accuracy':{f:None for f in FIELDS},
                'all_fields_accuracy':None, 'category_macro_f1':category_f1(rows)}
    return {'valid_predictions':len(rows),
        'field_accuracy':{f:sum(same_field(f,r['expected'],r['prediction']) for r in rows)/len(rows) for f in FIELDS},
        'all_fields_accuracy':sum(all(same_field(f,r['expected'],r['prediction']) for f in FIELDS)
                                  for r in rows)/len(rows), 'category_macro_f1':category_f1(rows)}

def build_report(cases, attempts):
    """Quality uses latest successful predictions; reliability counts every attempt."""
    selected = {case['ticket_id']:case for case in cases}
    successes = {}
    for attempt in attempts:
        if attempt['ticket_id'] in selected and attempt['status'] == 'success':
            successes[attempt['ticket_id']] = attempt
    rows = [{'ticket_id':tid, 'complaint':selected[tid]['complaint'],
             'expected':selected[tid]['expected'], 'prediction':attempt['prediction']}
            for tid,attempt in successes.items()]
    relevant = [a for a in attempts if a['ticket_id'] in selected]
    successful = [a for a in relevant if a['status'] == 'success']
    failures = [a for a in relevant if a['status'] == 'error']
    correct = sum(all(same_field(f,r['expected'],r['prediction']) for f in FIELDS) for r in rows)
    wrong = [{**r,'incorrect_fields':[f for f in FIELDS if not same_field(f,r['expected'],r['prediction'])]}
             for r in rows if any(not same_field(f,r['expected'],r['prediction']) for f in FIELDS)]
    return {'selected_tickets':len(cases), 'completed_tickets':len(rows),
        'remaining_ticket_ids':[tid for tid in selected if tid not in successes],
        'quality_on_valid_predictions':{'overall':group_metrics(rows)},
        'request_health':{'attempts':len(relevant),'successes':len(successful),'failures':len(failures),
            'success_rate':len(successful)/len(relevant) if relevant else None,
            'average_success_latency_seconds':mean(a['latency_seconds'] for a in successful) if successful else None},
        'successful_and_all_fields_correct_fraction_of_selected':correct/len(cases) if cases else None,
        'incorrect_cases':wrong, 'failed_attempts':failures,
        'notes':['Accuracy and F1 are fractions from 0 to 1; null means no denominator.',
                 'Quality excludes failed calls; request health and selected-ticket coverage expose failures.',
                 'Unattempted selected tickets count against selected-ticket coverage until the run is complete.',
                 'Average latency excludes deliberate delay and measures successful classifier calls only.',
                 'Scores measure agreement with synthetic reference labels, not guaranteed correctness.',
                 'Small samples do not establish final quality; macro F1 uses labels present in references or predictions.']}

def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)

def open_run(base, manifest):
    """Changed inputs automatically select a different directory."""
    identity = fingerprint(manifest)
    import re
    model = re.sub(r'[^a-zA-Z0-9-]+', '-', manifest.get('model', 'evaluation')).strip('-')
    prefix = 'classification' if base.name == 'classification' else 'rag-evaluation'
    folder = base/f'{prefix}-{model}-{identity[:8]}'
    # Reuse compatible older reports without discarding their completed predictions.
    legacy = base/identity[:16]
    if legacy.exists() and not folder.exists():
        if json.loads((legacy/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Run manifest mismatch; refusing to mix evaluation results.')
        legacy.rename(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Run manifest mismatch; refusing to mix evaluation results.')
    else:
        write_json(path,manifest)
    return folder

def load_attempts(path, cases):
    if not path.exists():
        return []
    from telecom_support.classification.schemas import ComplaintLabels
    allowed = {c['ticket_id'] for c in cases}
    attempts = []
    for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        try:
            item = json.loads(line)
            if item['ticket_id'] not in allowed or item['status'] not in ('success','error'):
                raise ValueError('Invalid ticket/status')
            if (not isinstance(item['latency_seconds'], (float,int))
                    or not math.isfinite(item['latency_seconds']) or item['latency_seconds'] < 0):
                raise ValueError('Invalid latency')
            if item['status'] == 'success':
                ComplaintLabels.model_validate(item['prediction'])
            elif not isinstance(item['error'],str):
                raise ValueError('Invalid error')
        except (ValueError,KeyError,TypeError) as exc:
            raise ValueError(f'Invalid saved attempt at line {number}; restore or move this run aside.') from exc
        attempts.append(item)
    return attempts

def check_examples(examples, training, evaluation):
    from telecom_support.classification.schemas import ComplaintLabels
    by_id = {r.ticket_id:r for r in training}
    if len(by_id) != len(training) or any(r.split != 'train' for r in training):
        raise ValueError('Invalid training records.')
    test_ids = {r.ticket_id for r in evaluation}
    normalize = lambda text:' '.join(text.lower().split())
    test_texts = {normalize(r.customer_complaint) for r in evaluation}
    if test_ids & set(by_id) or test_texts & {normalize(r.customer_complaint) for r in training}:
        raise ValueError('Training/evaluation overlap detected.')
    if not examples:
        raise ValueError('Classification examples are empty.')
    seen = set()
    for example in examples:
        tid = example['ticket_id']
        if tid not in by_id or tid in seen:
            raise ValueError('Examples must be unique training tickets.')
        seen.add(tid)
        record = by_id[tid]
        if example['complaint'] != record.customer_complaint:
            raise ValueError('Example complaint differs from its training record; regenerate examples.')
        expected = {'categories':[record.category.value], 'products':[record.product.value],
                    'severity':record.severity.value,'sentiment':record.sentiment.value}
        parsed = ComplaintLabels.model_validate(example['labels']).model_dump(mode='json')
        if parsed != expected:
            raise ValueError('Example labels differ from training labels; regenerate examples.')

def ordered_cases(records):
    """Round-robin telecom categories for a useful small sample."""
    groups = {}
    for record in records:
        case = {'ticket_id':record.ticket_id, 'complaint':record.customer_complaint,
            'expected':{'categories':[record.category.value], 'products':[record.product.value],
                        'severity':record.severity.value, 'sentiment':record.sentiment.value}}
        groups.setdefault(record.category.value,[]).append(case)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    return ordered
