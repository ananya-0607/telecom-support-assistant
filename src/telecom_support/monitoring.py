"""Process-local request metrics and bounded logs containing no customer content."""
import json
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
from datetime import datetime, timezone


class Monitoring:
    def __init__(self, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.counts = dict(resolution_requests=0, successful_requests=0,
                           failed_requests=0, requests_without_evidence=0)
        self.total_seconds = self.retrieval_seconds = 0.0
        self.retrieval_samples = 0
        self.status_counts = {}
        self.logger = logging.Logger('resolution_metrics', level=logging.INFO)
        self.handler = RotatingFileHandler(log_path, maxBytes=1_000_000,
                                          backupCount=3, encoding='utf-8')
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)

    def record(self, request_id, status, duration, retrieval=None, no_evidence=False,
               error_type=None):
        success = 200 <= status < 300
        with self.lock:
            self.counts['resolution_requests'] += 1
            self.counts['successful_requests' if success else 'failed_requests'] += 1
            self.counts['requests_without_evidence'] += int(no_evidence)
            self.total_seconds += duration
            self.status_counts[str(status)] = self.status_counts.get(str(status), 0) + 1
            if retrieval is not None:
                self.retrieval_seconds += retrieval
                self.retrieval_samples += 1
        # Fixed fields only: never serialize requests, exception messages or LLM output.
        self.logger.info(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id, event='resolution_request', status_code=status,
            outcome='success' if success else 'failure', duration_seconds=round(duration, 6),
            retrieval_seconds=retrieval, no_usable_evidence=no_evidence,
            error_type=error_type)))

    def snapshot(self):
        with self.lock:
            return {**self.counts, 'status_codes':dict(self.status_counts),
                'average_response_seconds':self.total_seconds / self.counts['resolution_requests']
                    if self.counts['resolution_requests'] else None,
                'average_retrieval_seconds':self.retrieval_seconds / self.retrieval_samples
                    if self.retrieval_samples else None,
                'retrieval_timing_samples':self.retrieval_samples,
                'scope':'POST /resolve; process-local counters reset on restart'}

    def close(self):
        self.handler.close()
        self.logger.removeHandler(self.handler)
