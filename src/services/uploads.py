"""Bounded upload work outside Streamlit's script thread; no UI calls in workers."""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from src.knowledge.parsing import DocumentError
from src.services.agents import AgentNotFoundError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadResult:
    ready: tuple[str, ...]
    errors: tuple[str, ...]


class UploadQueue:
    def __init__(self):
        # Serial indexing avoids competing model initialization and memory spikes.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-upload")
        self.jobs: dict[str, tuple[str, Future]] = {}
        self.lock = Lock()

    def start(self, agent_id, files, documents):
        with self.lock:
            if any(owner == agent_id and not future.done() for owner, future in self.jobs.values()):
                raise DocumentError("Masih ada dokumen agent ini yang sedang diproses.")
            # Retain only a small number of completed status receipts, never file bytes.
            completed = [key for key, (_, future) in self.jobs.items() if future.done()]
            for key in completed[:-32]:
                del self.jobs[key]
            job_id = uuid4().hex
            self.jobs[job_id] = (
                agent_id,
                self.executor.submit(self._process, agent_id, files, documents),
            )
            return job_id

    @staticmethod
    def _process(agent_id, files, documents):
        ready, errors = [], []
        for filename, content in files:
            try:
                doc = documents.upload(agent_id, filename, content)
                if doc.status == "ready":
                    ready.append(doc.filename)
                else:
                    errors.append(f"{doc.filename}: {doc.error_message}")
            except (DocumentError, AgentNotFoundError) as error:
                errors.append(f"{filename}: {error}")
            except Exception as error:
                logger.error("upload_job_failed error_type=%s", type(error).__name__)
                errors.append(
                    f"{filename}: Pemrosesan gagal. Periksa penyimpanan dan model embedding."
                )
        return UploadResult(tuple(ready), tuple(errors))

    def get(self, job_id, agent_id):
        with self.lock:
            item = self.jobs.get(job_id)
            if not item or item[0] != agent_id:
                return None
            future = item[1]
        return future.result() if future.done() else "processing"

    def busy(self):
        with self.lock:
            return any(not future.done() for _, future in self.jobs.values())
