"""Optional Ragas 0.2.15 adapters: Groq judging and existing local embeddings."""
import asyncio
import time
import threading
import numpy as np
from langchain_core.outputs import Generation, LLMResult
from langchain_core.embeddings import Embeddings
from ragas.llms import BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy, AnswerCorrectness
from telecom_support.llm import ServiceError


class RequestPacer:
    """One shared gate for every generation/judging call, including internal calls."""
    def __init__(self, delay):
        self.delay, self.last, self.wait_seconds = delay, None, 0.0
        self.lock = threading.Lock()

    def run(self, function):
        with self.lock:
            if self.last is not None:
                wait = max(0, self.delay - (time.monotonic() - self.last))
                started = time.monotonic()
                time.sleep(wait)
                self.wait_seconds += time.monotonic() - started
            try:
                return function()
            finally:
                self.last = time.monotonic()


class PacedPipelineLLM:
    def __init__(self, wrapped, pacer):
        self.wrapped, self.pacer = wrapped, pacer
        self.model = wrapped.model

    def call(self, *args, **kwargs):
        return self.pacer.run(lambda: self.wrapped.call(*args, **kwargs))


class GroqRagasJudge(BaseRagasLLM):
    def __init__(self, llm, pacer):
        super().__init__(run_config=RunConfig(max_retries=1, timeout=600, max_workers=1))
        self.llm, self.pacer = llm, pacer
        self.calls = 0
        self.judgments = []

    def is_finished(self, response):
        return all(g.generation_info['finish_reason'] == 'stop' for group in response.generations for g in group)

    def generate_text(self, prompt, n=1, temperature=0, stop=None, callbacks=None):
        generations = []
        for _ in range(n):
            def request():
                self.calls += 1
                try:
                    response = self.llm.client.chat.completions.create(model=self.llm.model,
                        messages=[{'role': 'user', 'content': prompt.to_string()}], temperature=0,
                        max_completion_tokens=4096, stop=stop,
                        response_format={'type': 'json_object'})
                except Exception as exc:
                    # Do not persist provider response bodies or secrets.
                    code = getattr(exc, 'status_code', None)
                    raise ServiceError(f'Groq judge request failed (HTTP {code or "connection error"}); rerun later.') from None
                if not response.choices or response.choices[0].finish_reason != 'stop':
                    raise ServiceError('Groq judge output was incomplete.')
                choice = response.choices[0]
                self.judgments.append({'prompt': prompt.to_string(), 'response': choice.message.content or ''})
                return Generation(text=choice.message.content or '', generation_info={'finish_reason': choice.finish_reason})
            generations.append(self.pacer.run(request))
        return LLMResult(generations=[generations])

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)


class LocalEmbeddings(Embeddings):
    """Evaluation-only token-weighted pooling for texts exceeding MiniLM's limit."""
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts):
        if not texts or any(not text.strip() for text in texts):
            raise ValueError('Evaluation embedding inputs must contain text.')
        results = []
        tokenizer = self.embedder.tokenizer
        limit = self.embedder.model.max_seq_length
        for text in texts:
            # Keep short-text embeddings identical to the existing implementation.
            if len(tokenizer.encode(text, add_special_tokens=True, verbose=False)) <= limit:
                results.append(self.embedder.encode([text])[0].tolist())
                continue
            offsets = tokenizer(text, add_special_tokens=False,
                return_offsets_mapping=True, verbose=False)['offset_mapping']
            budget = min(240, limit - tokenizer.num_special_tokens_to_add(pair=False))
            if budget < 1:
                raise ValueError('Embedding model has no usable token budget.')
            parts, weights, start = [], [], 0
            while start < len(offsets):
                end = min(start + budget, len(offsets))
                while end > start:
                    part = text[offsets[start][0]:offsets[end - 1][1]]
                    if len(tokenizer.encode(part, add_special_tokens=True, verbose=False)) <= limit:
                        break
                    end -= 1
                if end == start:
                    raise ValueError('Cannot fit evaluation text within embedding token limit.')
                parts.append(part)
                weights.append(end - start)
                start = end
            vectors = self.embedder.encode(parts)
            pooled = np.average(vectors, axis=0, weights=weights)
            norm = np.linalg.norm(pooled)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError('Invalid pooled evaluation embedding.')
            results.append((pooled / norm).tolist())
        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def create_metrics(judge, embedder):
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings(embedder))
    metrics = {'Context precision': ContextPrecision(llm=judge),
        'Context recall': ContextRecall(llm=judge), 'Faithfulness': Faithfulness(llm=judge),
        'Answer relevance': AnswerRelevancy(llm=judge, embeddings=embeddings),
        'Answer correctness': AnswerCorrectness(llm=judge, embeddings=embeddings)}
    for metric in metrics.values():
        metric.init(judge.run_config)
    return metrics
