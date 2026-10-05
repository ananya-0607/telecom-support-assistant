"""RAG evaluation data, readable reports and exact generation-context extraction."""
import json
import math
from statistics import mean
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class ReferenceCase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ticket_id: str
    complaint: str = Field(min_length=1)
    reference_resolution: str = Field(min_length=1)
    ticket_version: str
    review_status: Literal['draft', 'source_checked', 'reviewed'] = 'draft'
    review_notes: str = ''
    supporting_training_ticket_ids: list[str] = Field(default_factory=list)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
    temporary.replace(path)


def answer_text(result):
    answer = result['resolution']
    parts = [answer['summary']] + [s['instruction'] for s in answer['steps']]
    parts += ['Missing information: ' + x for x in answer['missing_information']]
    if answer['escalation']:
        parts.append('Escalation: ' + answer['escalation'])
    return '\n'.join(parts)


def evidence_contexts(result):
    # Exactly the evidence fields the generation prompt sees, in the same order.
    from telecom_support.generation.service import generation_messages
    from telecom_support.classification.schemas import ComplaintLabels
    messages = generation_messages('', ComplaintLabels.model_validate(result['classification']), result['sources'])
    return [json.dumps(item, ensure_ascii=False) for item in json.loads(messages[-1]['content'])['evidence']]


def citation_validity(result):
    allowed = {s['citation_id'] for s in result['sources']}
    references = [c for s in result['resolution']['steps'] for c in s['citations']]
    return sum(c in allowed for c in references) / len(references) if references else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


METRIC_NAMES = ['Context precision', 'Context recall', 'Faithfulness', 'Answer relevance', 'Answer correctness', 'Citation support']


def report(cases, pipeline_rows, metric_rows):
    ids = {c.ticket_id for c in cases}
    successful = {r['ticket_id']: r for r in pipeline_rows if r['ticket_id'] in ids and r['status'] == 'success'}
    relevant = [r for r in pipeline_rows if r['ticket_id'] in ids]
    metrics = {(r['ticket_id'], r['metric']): r for r in metric_rows if r['ticket_id'] in ids and r['status'] == 'success'}
    scores = {}
    for name in METRIC_NAMES:
        values = [r['score'] for (_, metric), r in metrics.items() if metric == name and r['score'] is not None]
        scores[name] = {'average_score': mean(values) if values else None, 'scored_cases': len(values)}
    latencies = [r['latency_seconds'] for r in successful.values()]
    valid = [citation_validity(r['result']) for r in successful.values()]
    valid = [x for x in valid if x is not None]
    return {'Selected complaints': len(cases), 'Answers saved': len(successful),
        'Reference status': ('reviewed' if all(c.review_status == 'reviewed' for c in cases)
            else 'contains draft references — exploratory scores' if any(c.review_status == 'draft' for c in cases)
            else 'source-checked synthetic references; human review remains advisable'),
        'RAG quality': scores,
        'Citation validity': {'average_score': mean(valid) if valid else None, 'applicable_cases': len(valid)},
        'Request success rate': sum(r['status'] == 'success' for r in relevant) / len(relevant) if relevant else None,
        'Average end-to-end time in seconds': mean(latencies) if latencies else None,
        'End-to-end time for 95 percent of requests in seconds': percentile(latencies, .95),
        'Cases without usable evidence': sum(r['result']['resolution']['status'] == 'insufficient_evidence' for r in successful.values()),
        'Evaluation errors': [r for r in metric_rows if r['ticket_id'] in ids and r['status'] == 'error'],
        'Pipeline errors': [r for r in relevant if r['status'] == 'error'],
        'Notes': ['Ragas judges meaning rather than exact wording. Scores are estimates, not human approval.',
                  'Missing or undefined scores are null, never replaced with zero.',
                  'Quality averages exclude failed/undefined scores; scored_cases exposes coverage.',
                  'Citation support is a separate LLM judgment of each step against its cited sources.',
                  'Latency excludes startup, deliberate API spacing and judge calls.',
                  'Faithfulness and citation support do not verify provider policy or technical safety.',
                  'Evaluation embeddings split long text into bounded nonoverlapping parts and pool normalized vectors with token-count weights. The LLM receives full text.']}
