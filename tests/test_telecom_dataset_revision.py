"""The telecom-only replacement cases remain held out and internally consistent."""
import json
from collections import Counter
from pathlib import Path
from telecom_support.classification.schemas import ComplaintLabels
from telecom_support.taxonomy import TAXONOMY

# This fixed benchmark covers the original classes, not later uploaded categories.
BENCHMARK_CATEGORIES = {
    'Broadband intermittent drops', 'Slow speed', 'No connectivity / outage',
    'Router / modem hardware', 'Billing dispute', 'Plan change', 'Mobile signal',
    'SIM / activation / porting', 'Installation / technician visit', 'Account / login / KYC',
}

def test_replacements_cover_categories_without_duplicate_complaints():
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'dataset_creation/telecom_evaluation_replacements.json').read_text())
    cases=data['cases']
    assert len(cases)==20
    assert len({c['ticket_id'] for c in cases})==20
    assert BENCHMARK_CATEGORIES <= set(TAXONOMY['categories'])
    assert Counter(c['category'] for c in cases)=={category:2 for category in BENCHMARK_CATEGORIES}
    normalized={' '.join(c['customer_complaint'].lower().split()) for c in cases}
    assert len(normalized)==20
    for c in cases:
        ComplaintLabels(categories=[c['category']],products=[c['product']],
                        severity=c['severity'],sentiment=c['sentiment'])
        assert all(c[field].strip() for field in ('conversation','resolution_steps','resolution_summary'))


def test_covered_benchmark_has_explicit_source_links_and_three_label_values():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'dataset_creation/covered_evaluation_scenarios.json').read_text())
    cases = data['cases']
    assert len(cases) == len({c['ticket_id'] for c in cases}) == 40
    assert BENCHMARK_CATEGORIES <= set(TAXONOMY['categories'])
    assert Counter(c['category'] for c in cases) == {category: 4 for category in BENCHMARK_CATEGORIES}
    assert {c['severity'] for c in cases} == set(TAXONOMY['severities'])
    assert {c['sentiment'] for c in cases} == set(TAXONOMY['sentiments'])
    assert all(c['supporting_training_ticket_ids'] and c['ticket_id'] not in c['supporting_training_ticket_ids'] for c in cases)
    assert len({' '.join(c['customer_complaint'].lower().split()) for c in cases}) == 40
