"""Evaluate the existing classifier, preserving progress without changing prompts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from telecom_support.classification.service import classify, classification_messages
from telecom_support.classification.schemas import ComplaintLabels
from telecom_support.ingestion.schemas import TicketRecord
from telecom_support.taxonomy import TAXONOMY
from telecom_support.llm import StructuredLLM, ServiceError
from telecom_support.evaluation.classification import (
    fingerprint, open_run, load_attempts, check_examples, ordered_cases, build_report, write_json)

def read_tickets(path):
    records = [TicketRecord.model_validate_json(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if not records or len({r.ticket_id for r in records}) != len(records):
        raise ValueError('Ticket files must be nonempty with unique IDs.')
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,help='Select first N cases; telecom categories are round-robin ordered.')
    parser.add_argument('--delay',type=float,default=30,help='Minimum wait between requests, in seconds.')
    parser.add_argument('--report-only',action='store_true',help='Score saved results; make no API calls.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1 or args.delay < 0:
        parser.error('Limit must be positive; delay must be nonnegative.')
    llm, attempts, folder, selected, failure = None, [], None, [], None
    try:
        data = ROOT/'data/indexes'
        training = read_tickets(data/'historical_tickets.jsonl')
        evaluation = read_tickets(data/'evaluation_tickets.jsonl')
        if any(r.split != 'test' for r in evaluation):
            raise ValueError('Evaluation file must contain test tickets only.')
        examples_data = json.loads((data/'classification_examples.json').read_text(encoding='utf-8'))
        if examples_data['taxonomy_version'] != TAXONOMY['version']:
            raise ValueError('Taxonomy changed; regenerate classification examples.')
        examples = examples_data['examples']
        check_examples(examples,training,evaluation)
        load_dotenv(ROOT/'.env',override=False)
        model = os.getenv('GROQ_MODEL','openai/gpt-oss-20b').strip()
        if not model:
            raise ValueError('GROQ_MODEL cannot be empty.')
        cases = ordered_cases(evaluation)
        manifest = {'schema_version':1,'model':model,'taxonomy':TAXONOMY,
            'dataset_fingerprint':fingerprint(cases),
            'training_fingerprint':fingerprint([r.model_dump(mode='json') for r in training]),
            'messages_template':classification_messages('__EVALUATION_COMPLAINT__',examples),
            'output_schema':ComplaintLabels.model_json_schema(),
            'classifier_code_fingerprint':fingerprint([
                (ROOT/'src/telecom_support/classification/service.py').read_text(encoding='utf-8'),
                (ROOT/'src/telecom_support/llm.py').read_text(encoding='utf-8')]),
            'temperature':0,'max_completion_tokens':2048,
            'ordering':'telecom-category-round-robin-v2'}
        folder = open_run(data/'evaluations/classification',manifest)
        attempts = load_attempts(folder/'predictions.jsonl',cases)
        selected = cases[:args.limit] if args.limit else cases
        done = {a['ticket_id'] for a in attempts if a['status']=='success'}
        print(f'Run: {folder.name}; selected: {len(selected)}; model: {model}',flush=True)
        requested = False
        if not args.report_only:
            for case in selected:
                if case['ticket_id'] in done:
                    print(f"{case['ticket_id']}: saved prediction reused",flush=True)
                    continue
                if llm is None:
                    llm = StructuredLLM(ROOT)
                if requested:
                    time.sleep(args.delay)
                started = time.perf_counter()
                try:
                    prediction = classify(llm,case['complaint'],examples).model_dump(mode='json')
                    attempt = {'ticket_id':case['ticket_id'],'status':'success','prediction':prediction,
                        'expected':case['expected'],'complaint':case['complaint']}
                except ServiceError as exc:
                    failure = str(exc)
                    attempt = {'ticket_id':case['ticket_id'],'status':'error','error':failure,
                        'expected':case['expected'],'complaint':case['complaint']}
                attempt.update(latency_seconds=round(time.perf_counter()-started,4),
                               attempted_at=datetime.now(timezone.utc).isoformat())
                requested = True
                attempts.append(attempt)
                # Save after every attempt. Atomic rewrite avoids a partially appended last line.
                path = folder/'predictions.jsonl'
                temporary = path.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in attempts),encoding='utf-8')
                temporary.replace(path)
                print(f"{case['ticket_id']}: {attempt['status']}",flush=True)
                if failure:
                    break
    except KeyboardInterrupt:
        failure = 'Interrupted; previously saved predictions retained.'
    except (OSError,ValueError,KeyError,ServiceError) as exc:
        failure = str(exc)
    finally:
        if llm is not None:
            llm.close()
    if folder is not None and selected:
        report = build_report(selected,attempts)
        report.update(run_id=folder.name,report_scope='sample' if args.limit else 'full',
                      generated_at=datetime.now(timezone.utc).isoformat(),failure=failure)
        write_json(folder/'report.json',report)
        print(json.dumps({key:report[key] for key in ['selected_tickets','completed_tickets',
            'quality_on_valid_predictions','request_health',
            'successful_and_all_fields_correct_fraction_of_selected']},indent=2))
        print(f'Report: {folder / "report.json"}')
    if failure:
        print(f'FAIL: {failure}',file=sys.stderr)
        return 1
    print('PASS: Report generated. Inspect incorrect cases; scores are not a correctness guarantee.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
