"""Single-process prototype API; owns the local Qdrant connection."""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from telecom_support.llm import StructuredLLM, ServiceError
from telecom_support.retrieval.index import SearchIndex
from telecom_support.pipeline import ResolutionPipeline
from telecom_support.ingestion.additions import AdditionJobs, ResolvedConversation
from telecom_support.ingestion.pdf_labels import PDFLabels
from telecom_support.taxonomy import TAXONOMY

ROOT = Path(__file__).resolve().parents[2]

@asynccontextmanager
async def lifespan(app):
    index, llm = None, None
    try:
        threshold = float(os.getenv('SEMANTIC_THRESHOLD','0.30'))
        if not -1 <= threshold <= 1:
            raise ValueError('SEMANTIC_THRESHOLD must be between -1 and 1.')
        index = SearchIndex(ROOT)
        llm = StructuredLLM(ROOT)
        app.state.pipeline = ResolutionPipeline(ROOT,index,llm,threshold)
        app.state.lock = Lock()
        app.state.index = index
        app.state.jobs = AdditionJobs(ROOT, app.state)
        yield
    finally:
        if hasattr(app.state, 'jobs'):
            app.state.jobs.close()
        if index is not None:
            getattr(app.state, 'index', index).close()
        if llm is not None:
            llm.close()

app = FastAPI(title='Telecom Support Resolution Assistant', lifespan=lifespan)

class ComplaintRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    complaint: str = Field(min_length=1, max_length=6000)

@app.get('/health')
def health():
    return {'status':'ready'}

@app.post('/ingestion/conversations', status_code=202)
def add_conversation(record: ResolvedConversation):
    try:
        return app.state.jobs.submit('ticket', record)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None

@app.post('/ingestion/pdfs', status_code=202)
async def add_pdf(request: Request, new_labels: bool = False, category: str = '',
                  category_description: str = '', product: str = '', product_description: str = '',
                  replacement_id: str = '', filename: str = ''):
    if replacement_id and new_labels:
        raise HTTPException(422, 'Replace a PDF using the current taxonomy; register new labels with a separate upload.')
    if replacement_id and replacement_id not in {d['document_id'] for d in app.state.jobs.documents()}:
        raise HTTPException(404, 'Selected document not found.')
    filename = filename.replace('\\', '/').rsplit('/', 1)[-1].strip()
    if len(filename) > 200 or any(ord(c) < 32 for c in filename):
        raise HTTPException(422, 'Invalid document filename.')
    try:
        options = PDFLabels(category=category, category_description=category_description,
            product=product, product_description=product_description) if new_labels else None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if request.headers.get('content-type', '').split(';')[0] != 'application/pdf':
        raise HTTPException(415, 'Upload a PDF with application/pdf content type.')
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 10 * 1024 * 1024:
            raise HTTPException(413, 'PDF limit is 10 MB.')
        chunks.append(chunk)
    data = b''.join(chunks)
    if not data.startswith(b'%PDF-'):
        raise HTTPException(400, 'File is not a PDF.')
    try:
        return app.state.jobs.submit('pdf', data, options, replacement_id or None, filename or None)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None

@app.get('/taxonomy')
def taxonomy():
    return TAXONOMY

@app.get('/ingestion/documents')
def indexed_documents():
    return app.state.jobs.documents()

@app.post('/ingestion/jobs/{job_id}/confirm-labels', status_code=202)
def confirm_pdf_labels(job_id: str):
    try:
        return app.state.jobs.confirm_labels(job_id)
    except FileNotFoundError:
        raise HTTPException(404, 'Unknown ingestion job.') from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None

@app.get('/ingestion/jobs/{job_id}')
def ingestion_status(job_id: str):
    try:
        return app.state.jobs.status(job_id)
    except FileNotFoundError:
        raise HTTPException(404, 'Unknown ingestion job.') from None

@app.post('/resolve')
def resolve(request: ComplaintRequest):
    # Local Qdrant/CPU model are shared; reject concurrent requests rather than overload.
    if not app.state.lock.acquire(blocking=False):
        raise HTTPException(503,'Another request is running; retry shortly.')
    try:
        return app.state.pipeline.resolve(request.complaint)
    except ServiceError as exc:
        raise HTTPException(400,str(exc)) from None
    except Exception:
        raise HTTPException(500,'Resolution failed. Check the local API terminal.') from None
    finally:
        app.state.lock.release()
