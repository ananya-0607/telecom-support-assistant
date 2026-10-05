"""Single-process background ingestion with durable jobs and snapshot publication."""
import hashlib
import json
import sqlite3
import threading
import time
import uuid
import re
import copy
from contextlib import closing
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from telecom_support.classification.schemas import ComplaintLabels
from telecom_support.taxonomy import TAXONOMY, staged_taxonomy, activate_taxonomy, current_taxonomy
from telecom_support.classification.service import classify
from telecom_support.indexing.records import make_passages
from telecom_support.indexing.storage import write_database
from telecom_support.indexing.vectors import write_vectors
from telecom_support.indexing.reuse import encode_with_reuse, text_hash
from telecom_support.retrieval.index import SearchIndex
from .schemas import TicketRecord, KBChunk
from .chunking import chunk_blocks, document_identity, token_count
from .pdf import extract_pdf
from .metadata import cache_key, classify_chunk, enrich_chunk, load_cache, save_cache, MetadataResult, PROMPT_VERSION
from .pdf_labels import PDFLabels, expanded_taxonomy, suggest_missing


class ResolvedConversation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    customer_complaint: str = Field(min_length=1, max_length=6000)
    conversation: str = Field(min_length=1, max_length=100000)
    resolution_steps: str = Field(min_length=1, max_length=30000)
    resolution_summary: str = Field(min_length=1, max_length=6000)
    classification: ComplaintLabels | None = None
    reviewed: bool = False
    generated_response: dict | None = None

    @field_validator('customer_complaint', 'conversation', 'resolution_steps', 'resolution_summary')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Fields cannot be blank.')
        return value.strip()

    @model_validator(mode='after')
    def review_generated(self):
        if self.classification is not None and not self.reviewed:
            raise ValueError('Review the generated resolution before saving it as knowledge.')
        return self


def next_ticket_id(database, evaluation_path=None):
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
        ids = [json.loads(row[0])['ticket_id'] for row in connection.execute(
            "SELECT record_json FROM sources WHERE source_type='ticket'")]
    if evaluation_path is not None and evaluation_path.exists():
        ids.extend(json.loads(line)['ticket_id'] for line in evaluation_path.read_text(encoding='utf-8').splitlines() if line.strip())
    numbers = [int(match.group(1)) for value in ids if (match := re.fullmatch(r'T-(\d+)', value))]
    return f'T-{max(numbers, default=0)+1:04d}'


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class AdditionJobs:
    def __init__(self, root, state, delay=60):
        self.root, self.state, self.delay = root, state, delay
        self.folder = root/'data/indexes/additions'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.thread = None
        for path in self.folder.glob('*/status.json'):
            status = json.loads(path.read_text(encoding='utf-8'))
            if status['status'] in ('queued', 'processing'):
                status.update(status='interrupted', stage='Backend stopped; resubmit the source.')
                atomic_json(path, status)

    def status(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise FileNotFoundError('Unknown job.')
        return json.loads((self.folder/job_id/'status.json').read_text(encoding='utf-8'))

    def documents(self):
        grouped = {}
        with closing(sqlite3.connect(self.state.pipeline.index.database.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
            for row in connection.execute("SELECT record_json FROM sources WHERE source_type='kb'"):
                record = json.loads(row[0])
                doc_id = record['document_id']
                if doc_id not in grouped:
                    name = record.get('document_name') or Path(record['source_file']).name
                    if name == 'upload.pdf':
                        name = record.get('title') or 'Uploaded knowledge base'
                    grouped[doc_id] = dict(document_id=doc_id, name=name, chunks=0,
                        source_version=record['source_version'])
                grouped[doc_id]['chunks'] += 1
        return sorted(grouped.values(), key=lambda d:(d['name'], d['document_id']))

    def submit(self, kind, payload, options=None, replacement_id=None, filename=None):
        if not self.state.lock.acquire(blocking=False):
            raise RuntimeError('Another search or ingestion is running; retry later.')
        job_id = uuid.uuid4().hex
        folder = self.folder/job_id
        try:
            folder.mkdir()
            if kind == 'pdf':
                (folder/'upload.pdf').write_bytes(payload)
                atomic_json(folder/'document.json', {'replacement_id':replacement_id, 'filename':filename})
                if options:
                    atomic_json(folder/'labels.json', options.model_dump())
            else:
                atomic_json(folder/'input.json', payload.model_dump())
            atomic_json(folder/'status.json', dict(job_id=job_id, kind=kind,
                status='queued', stage='Waiting to process', created_at=datetime.now(timezone.utc).isoformat()))
            self.thread = threading.Thread(target=self.process, args=(job_id, kind, payload, options, replacement_id, filename), daemon=False)
            self.thread.start()
        except Exception:
            self.state.lock.release()
            raise
        return self.status(job_id)

    def update(self, job_id, **values):
        status = self.status(job_id)
        status.update(values)
        atomic_json(self.folder/job_id/'status.json', status)

    def confirm_labels(self, job_id):
        status = self.status(job_id)
        if status['status'] != 'awaiting_confirmation':
            raise ValueError('This upload is not waiting for label confirmation.')
        if not self.state.lock.acquire(blocking=False):
            raise RuntimeError('Another search or ingestion is running; retry later.')
        try:
            if status['base_taxonomy_version'] != TAXONOMY['version']:
                raise ValueError('Taxonomy changed while waiting. Upload again to obtain a current suggestion.')
            folder = self.folder/job_id
            options = PDFLabels.model_validate({**json.loads((folder/'labels.json').read_text()), 'confirmed':True})
            details = json.loads((folder/'document.json').read_text()) if (folder/'document.json').exists() else {}
            self.update(job_id, status='queued', stage='Preparing confirmed labels')
            self.thread = threading.Thread(target=self.process,
                args=(job_id, 'pdf', (folder/'upload.pdf').read_bytes(), options, None, details.get('filename')), daemon=False)
            self.thread.start()
        except Exception:
            self.state.lock.release()
            raise
        return self.status(job_id)

    def process(self, job_id, kind, payload, options=None, replacement_id=None, filename=None):
        proposed = copy.deepcopy(TAXONOMY)
        try:
            if options:
                if not options.category or not options.product:
                    self.update(job_id, status='processing', stage='Suggesting the missing label')
                    blocks, _ = extract_pdf(self.folder/job_id/'upload.pdf')
                    if not blocks:
                        raise ValueError('No extractable PDF text found.')
                    suggestion = suggest_missing(self.state.pipeline.llm, options, blocks)
                    values = options.model_dump()
                    for name in ('category', 'product'):
                        if not values[name]:
                            label = getattr(suggestion, name)
                            if label is None:
                                raise ValueError(f'No supported existing {name} found. Upload again and provide both labels.')
                            values[name] = label.value
                    options = PDFLabels.model_validate(values)
                    atomic_json(self.folder/job_id/'labels.json', options.model_dump())
                    self.update(job_id, status='awaiting_confirmation', stage='Confirm the suggested label',
                        proposed_labels={'category':options.category, 'product':options.product},
                        suggestion_reason=suggestion.reason, base_taxonomy_version=TAXONOMY['version'])
                    self.state.lock.release()
                    return
                proposed = expanded_taxonomy(options, proposed)
        except Exception as exc:
            code = getattr(exc, 'status_code', None)
            error = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else (
                f'Groq HTTP {code}; retry the upload later.' if code else 'Could not prepare label suggestions.')
            self.update(job_id, status='failed', stage='Label preparation failed', error=error)
            self.state.lock.release()
            return
        with staged_taxonomy(proposed):
            self._process(job_id, kind, payload, proposed, replacement_id, filename)

    def _process(self, job_id, kind, payload, proposed, replacement_id=None, filename=None):
        candidate = None
        try:
            self.update(job_id, status='processing', stage='Preparing source')
            index = self.state.pipeline.index
            embedder = index.embedder
            folder = self.folder/job_id
            source_file = (folder/('upload.pdf' if kind == 'pdf' else 'input.json')).relative_to(self.root).as_posix()
            with closing(sqlite3.connect(index.database.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
                existing = [TicketRecord.model_validate_json(r[0]) if r[1]=='ticket' else KBChunk.model_validate_json(r[0])
                    for r in connection.execute('SELECT record_json, source_type FROM sources')]
            old_chunks = [r for r in existing if isinstance(r, KBChunk) and r.document_id == replacement_id] if replacement_id else []
            if replacement_id and not old_chunks:
                raise ValueError('Selected document no longer exists. Refresh the document list.')
            if kind == 'pdf':
                version = hashlib.sha256(payload).hexdigest()
                if old_chunks and all(r.source_version == version for r in old_chunks):
                    atomic_json(folder/'records.json', [])
                    self.update(job_id, status='completed', stage='No changes found', no_changes=True,
                        added_sources=0, added_passages=0, embedded_unique_texts=0,
                        build_id=index.path.name)
                    return
                if any(isinstance(r, KBChunk) and r.source_version == version and r.document_id != replacement_id for r in existing):
                    raise ValueError('This PDF content is already indexed; no duplicate was added.')
            if kind == 'ticket':
                if token_count(embedder.tokenizer, payload.customer_complaint) > embedder.model.max_seq_length:
                    raise ValueError('Use a shorter complaint; the full conversation may be long.')
                self.update(job_id, stage='Using saved classification' if payload.classification else 'Classifying complaint')
                labels = payload.classification or classify(self.state.pipeline.llm, payload.customer_complaint, self.state.pipeline.examples)
                if len(labels.categories) != 1 or len(labels.products) != 1:
                    raise ValueError('Historical ticket records currently require one category and one product. Submit separate resolved issues.')
                ticket_id = next_ticket_id(index.database, self.root/'data/indexes/evaluation_tickets.jsonl')
                records = [TicketRecord(source_id=ticket_id, ticket_id=ticket_id,
                    source_file=source_file, source_version=hashlib.sha256(payload.model_dump_json().encode()).hexdigest(),
                    taxonomy_version=current_taxonomy()['version'], search_text=payload.conversation,
                    created_at=datetime.now(timezone.utc).isoformat(), category=labels.categories[0],
                    product=labels.products[0], severity=labels.severity, sentiment=labels.sentiment,
                    split='train', origin='agent_reviewed_generated' if payload.classification else 'historical',
                    **payload.model_dump(exclude={'classification', 'reviewed'}))]
                self.update(job_id, ticket_id=ticket_id)
            else:
                blocks, details = extract_pdf(folder/'upload.pdf')
                records = chunk_blocks(blocks, embedder.tokenizer, replacement_id or document_identity(source_file),
                    source_file, hashlib.sha256(payload).hexdigest(), current_taxonomy()['version'])
                document_name = (filename or (old_chunks[0].document_name if old_chunks else None))
                for record in records:
                    record.document_name = document_name
                if not records:
                    raise ValueError('No extractable text found. Scanned PDFs require OCR, which is not supported.')
                registry_key = hashlib.sha256(json.dumps(proposed, sort_keys=True).encode()).hexdigest()[:16]
                cache_path = self.folder/f'metadata_cache-{registry_key}.jsonl'
                cache = load_cache(cache_path)
                enriched, last_request = [], None
                llm = self.state.pipeline.llm
                reusable_metadata = {text_hash(r.text):r for r in existing if isinstance(r, KBChunk)
                    and r.title and r.metadata_status == 'classified' and r.taxonomy_version == proposed['version']
                    and r.metadata_model == llm.model and r.metadata_prompt_version == PROMPT_VERSION}
                for number, chunk in enumerate(records, 1):
                    self.update(job_id, stage='Classifying PDF chunks', chunks_total=len(records), chunks_completed=number-1)
                    key = cache_key(chunk, llm.model)
                    previous = reusable_metadata.get(text_hash(chunk.text))
                    if key not in cache and previous is not None:
                        cache[key] = MetadataResult(title=previous.title, categories=previous.categories, products=previous.products)
                    if key not in cache:
                        if last_request is not None:
                            time.sleep(max(0, self.delay-(time.monotonic()-last_request)))
                        cache[key] = classify_chunk(llm.client, chunk, llm.model)
                        last_request = time.monotonic()
                        save_cache(cache_path, cache)
                    enriched.append(enrich_chunk(chunk, cache[key], llm.model, key))
                records = enriched
                self.update(job_id, chunks_completed=len(records), pages=details['pages'])
                if replacement_id:
                    old_counts = Counter(text_hash(r.text) for r in old_chunks)
                    new_counts = Counter(text_hash(r.text) for r in records)
                    self.update(job_id, replaced_document_id=replacement_id,
                        unchanged_chunks=sum((old_counts & new_counts).values()),
                        changed_or_new_chunks=sum((new_counts-old_counts).values()),
                        removed_chunks=sum((old_counts-new_counts).values()))
                    existing = [r for r in existing if not isinstance(r, KBChunk) or r.document_id != replacement_id]
            if kind == 'ticket' and any(isinstance(r, TicketRecord) and
                    r.conversation.strip() == payload.conversation.strip() for r in existing):
                raise ValueError('This conversation is already indexed; no duplicate was added.')
            all_records = existing+records
            self.update(job_id, stage='Chunking and embedding sources')
            passages = make_passages(all_records, embedder.tokenizer)
            vectors, reuse_stats = encode_with_reuse(index, passages)
            self.update(job_id, **reuse_stats)
            build_id = uuid.uuid4().hex
            snapshot = self.root/'data/indexes/search/builds'/build_id
            snapshot.mkdir(parents=True)
            manifest = json.loads((index.path/'manifest.json').read_text())
            manifest.update(build_id=build_id, created_at=datetime.now(timezone.utc).isoformat(),
                taxonomy=proposed,
                ticket_sources=sum(r.source_type=='ticket' for r in all_records),
                kb_sources=sum(r.source_type=='kb' for r in all_records), passages=len(passages),
                note='Full snapshot rebuild including interface additions.')
            self.update(job_id, stage='Writing and verifying SQLite and Qdrant')
            write_database(snapshot/'evidence.sqlite3', all_records, passages, manifest)
            write_vectors(snapshot/'qdrant', passages, vectors)
            atomic_json(snapshot/'manifest.json', manifest)
            candidate = SearchIndex(self.root, embedder=embedder, build_id=build_id)
            # Persist records before activation; CLI rebuilds include only published jobs.
            atomic_json(folder/'records.json', [r.model_dump(mode='json') for r in records])
            taxonomy_path = self.root/'config/taxonomy.json'
            previous_taxonomy = copy.deepcopy(TAXONOMY)
            if proposed != previous_taxonomy:
                taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(taxonomy_path, proposed)
            try:
                atomic_json(self.root/'data/indexes/search/active.json', {'build_id': build_id, 'taxonomy':proposed})
            except Exception:
                if proposed != previous_taxonomy:
                    atomic_json(taxonomy_path, previous_taxonomy)
                raise
            activate_taxonomy(proposed)
            self.state.pipeline.taxonomy_version = proposed['version']
            self.state.pipeline.index = candidate
            self.state.index = candidate
            candidate = None
            index.close()
            self.update(job_id, status='completed', stage='Completed — searchable now', build_id=build_id,
                taxonomy_version=proposed['version'],
                added_sources=len(records), added_passages=sum(p['source_id'] in {r.source_id for r in records} for p in passages))
        except Exception as exc:
            code = getattr(exc, 'status_code', None)
            message = f'Groq HTTP {code}; source not activated. Check quota/model access and resubmit.' if code else (
                str(exc) if isinstance(exc, (ValueError, RuntimeError)) else 'Ingestion failed; check backend configuration and resubmit.')
            self.update(job_id, status='failed', stage='Processing failed', error=message)
        finally:
            if candidate is not None:
                candidate.close()
            self.state.lock.release()

    def close(self):
        if self.thread is not None:
            self.thread.join()
