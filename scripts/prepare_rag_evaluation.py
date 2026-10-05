"""Create draft reference resolutions from held-out tickets, without API calls."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from telecom_support.ingestion.schemas import TicketRecord
from telecom_support.evaluation.classification import ordered_cases, fingerprint
from telecom_support.evaluation.rag import ReferenceCase, save_rows, read_rows


def main():
    path = ROOT / 'data/evaluation/rag_reference_cases.jsonl'
    tickets = [TicketRecord.model_validate(r) for r in read_rows(ROOT / 'data/indexes/evaluation_tickets.jsonl')]
    if not tickets or any(r.split != 'test' for r in tickets):
        raise ValueError('Only held-out test tickets may supply references.')
    existing = {r.ticket_id: r for r in (ReferenceCase.model_validate(x) for x in read_rows(path))}
    by_id = {r.ticket_id: r for r in tickets}
    training = {r.ticket_id: r for r in (TicketRecord.model_validate(x) for x in read_rows(ROOT / 'data/indexes/historical_tickets.jsonl'))}
    scenario_path = ROOT / 'dataset_creation/covered_evaluation_scenarios.json'
    scenarios = {r['ticket_id']: r for r in json.loads(scenario_path.read_text(encoding='utf-8'))['cases']} if scenario_path.exists() else {}
    rows = []
    for item in ordered_cases(tickets):
        ticket = by_id[item['ticket_id']]
        version = fingerprint(ticket.model_dump(mode='json'))
        previous = existing.get(ticket.ticket_id)
        if previous and previous.ticket_version != version:
            raise ValueError(f'{ticket.ticket_id} changed; review its existing reference before replacing it.')
        scenario = scenarios.get(ticket.ticket_id)
        source_ids = scenario.get('supporting_training_ticket_ids', []) if scenario else []
        covered = bool(source_ids) and scenario['customer_complaint'] == ticket.customer_complaint
        if covered:
            for source_id in source_ids:
                source = training.get(source_id)
                if source is None or source.split != 'train' or source.category != ticket.category or source.product != ticket.product or source.resolution_steps not in ticket.resolution_steps:
                    raise ValueError(f'{ticket.ticket_id}: declared procedure source is missing or inconsistent.')
        case = previous or ReferenceCase(ticket_id=ticket.ticket_id, complaint=ticket.customer_complaint,
            reference_resolution=ticket.resolution_steps if covered else ticket.resolution_steps + '\n' + ticket.resolution_summary,
            ticket_version=version, review_status='source_checked' if covered else 'draft',
            supporting_training_ticket_ids=source_ids if covered else [],
            review_notes='Covered synthetic paraphrase; procedure checked against training sources. Corrective actions are conditional. Not an independent unseen-problem benchmark or human approval.' if covered else
                'Excel historical resolution: check later-discovered facts and accept valid alternatives.')
        rows.append(case.model_dump(mode='json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_rows(path, rows)
    print(f'Saved {len(rows)} references: {path}')
    print('No API calls. Declared procedure sources checked; draft references still require review.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
