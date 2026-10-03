"""Single-process prototype API; owns the local Qdrant connection."""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from telecom_support.llm import StructuredLLM, ServiceError
from telecom_support.retrieval.index import SearchIndex
from telecom_support.pipeline import ResolutionPipeline

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
        yield
    finally:
        if index is not None:
            index.close()
        if llm is not None:
            llm.close()

app = FastAPI(title='Telecom Support Resolution Assistant', lifespan=lifespan)

class ComplaintRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    complaint: str = Field(min_length=1, max_length=6000)

@app.get('/health')
def health():
    return {'status':'ready'}

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
