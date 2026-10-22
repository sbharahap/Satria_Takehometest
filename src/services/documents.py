import json
import logging
from dataclasses import asdict
from pathlib import PurePosixPath
from uuid import uuid4

from src.config import PROJECT_ROOT, Settings
from src.database.connection import Database
from src.knowledge.ocr import OCRConfig
from src.knowledge.parsing import DocumentError, PageText, chunk_pages, parse_docx, parse_pdf
from src.models import Document
from src.services.agents import AgentService

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self, database: Database, knowledge, embeddings, agents: AgentService, settings: Settings
    ):
        self.database, self.knowledge, self.embeddings = database, knowledge, embeddings
        self.agents, self.settings = agents, settings

    def list_documents(self, agent_id: str) -> list[Document]:
        self.agents.get(agent_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE agent_id=? ORDER BY created_at,id", (agent_id,)
            ).fetchall()
        return [Document(**dict(row)) for row in rows]

    def get(self, agent_id: str, document_id: str) -> Document:
        for doc in self.list_documents(agent_id):
            if doc.id == document_id:
                return doc
        raise DocumentError("Dokumen tidak ditemukan pada agent ini atau sudah dihapus.")

    def extraction(self, agent_id: str, document_id: str, *, original=False) -> list[dict]:
        """Current working text, or the immutable parser snapshot when original=True."""
        self.agents.get(agent_id)
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT e.pages_json,x.pages_json FROM documents d "
                "LEFT JOIN document_extractions e ON e.document_id=d.id "
                "LEFT JOIN document_edits x ON x.document_id=d.id "
                "WHERE d.agent_id=? AND d.id=?",
                (agent_id, document_id),
            ).fetchone()
        if row is None:
            raise DocumentError("Dokumen tidak ditemukan pada agent ini.")
        text = row[0] if original else (row[1] or row[0])
        return json.loads(text) if text else []

    def original_file(self, agent_id: str, document_id: str) -> bytes | None:
        self.agents.get(agent_id)
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT o.file_bytes FROM documents d LEFT JOIN document_originals o "
                "ON o.document_id=d.id WHERE d.agent_id=? AND d.id=?",
                (agent_id, document_id),
            ).fetchone()
        if row is None:
            raise DocumentError("Dokumen tidak ditemukan pada agent ini.")
        return row[0]

    def ready_versions(self, agent_id: str) -> dict[str, int]:
        return {d.id: d.revision for d in self.list_documents(agent_id) if d.status == "ready"}

    def ready_ids(self, agent_id: str) -> list[str]:
        return list(self.ready_versions(agent_id))

    def _index(self, agent_id, document_id, filename, pages, revision):
        chunks = chunk_pages(
            pages,
            self.embeddings.model.tokenizer,
            self.settings.chunk_size_tokens,
            self.settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise DocumentError("Tidak ada potongan teks yang dapat diindeks.")
        vectors = self.embeddings.embed_passages([c.text for c in chunks])
        self.agents.get(agent_id)
        # Only this revision may write vectors; a deleted document cannot be resurrected.
        current = self.get(agent_id, document_id)
        if current.status != "processing" or current.revision != revision:
            raise DocumentError("Dokumen sudah dihapus atau pemrosesannya dibatalkan.")
        self.knowledge.delete_document(agent_id, document_id, max_revision=revision)
        self.knowledge.add_document(agent_id, document_id, filename, chunks, vectors, revision)
        with self.database.connect() as connection:
            active = connection.execute(
                "SELECT id FROM agents WHERE id=? AND status='active'", (agent_id,)
            ).fetchone()
            if not active:
                raise DocumentError("Agent sudah dihapus. Unggah ke agent yang aktif.")
            updated = connection.execute(
                "UPDATE documents SET status='ready',chunks_count=?,error_message=NULL "
                "WHERE id=? AND agent_id=? AND status='processing' AND revision=?",
                (len(chunks), document_id, agent_id, revision),
            ).rowcount
            if not updated:
                raise DocumentError("Dokumen sudah dihapus atau pemrosesannya dibatalkan.")
        logger.info(
            "document_ready document_id=%s revision=%s chunks=%s",
            document_id,
            revision,
            len(chunks),
        )

    def _failed(self, agent_id, document_id, error, revision):
        message = (
            str(error)
            if isinstance(error, DocumentError)
            else ("Pemrosesan gagal. Periksa model embedding dan penyimpanan, lalu coba kembali.")
        )
        try:
            # A late failure from an old worker must not erase a newer ready revision.
            self.knowledge.delete_document(agent_id, document_id, max_revision=revision)
        except Exception:
            logger.error("partial_index_cleanup_failed document_id=%s", document_id)
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE documents SET status='failed',error_message=? "
                "WHERE id=? AND agent_id=? AND revision=?",
                (message, document_id, agent_id, revision),
            )
        logger.warning(
            "document_failed document_id=%s error_type=%s", document_id, type(error).__name__
        )

    def upload(self, agent_id: str, filename: str, data: bytes) -> Document:
        self.agents.get(agent_id)
        filename = PurePosixPath(filename.replace("\\", "/")).name[:200]
        extension = PurePosixPath(filename).suffix.lower().lstrip(".")
        if extension not in {"pdf", "docx"}:
            raise DocumentError("Format tidak didukung. Unggah file PDF atau DOCX.")
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise DocumentError(f"Ukuran file melebihi {self.settings.max_upload_mb} MB.")
        document_id = uuid4().hex
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO documents(id,agent_id,filename,file_type,status) "
                "VALUES(?,?,?,?,'processing')",
                (document_id, agent_id, filename, extension),
            )
        try:
            tessdata = self.settings.ocr_tessdata_dir
            local_data = PROJECT_ROOT / "data/ocr/tessdata"
            if tessdata is None and local_data.is_dir():
                tessdata = local_data
            ocr = (
                OCRConfig(max_pages=self.settings.ocr_max_pages, tessdata_dir=tessdata)
                if self.settings.ocr_enabled
                else None
            )
            pages = parse_pdf(data, ocr=ocr) if extension == "pdf" else parse_docx(data)
            with self.database.connect() as connection:
                stored = connection.execute(
                    "INSERT INTO document_extractions(document_id,pages_json) "
                    "SELECT id,? FROM documents WHERE id=? AND agent_id=? AND status='processing'",
                    (
                        json.dumps([asdict(page) for page in pages], ensure_ascii=False),
                        document_id,
                        agent_id,
                    ),
                ).rowcount
                if not stored:
                    raise DocumentError("Dokumen sudah dihapus saat ekstraksi berlangsung.")
                connection.execute(
                    "INSERT INTO document_originals(document_id,file_bytes) VALUES(?,?)",
                    (document_id, data),
                )
            self._index(agent_id, document_id, filename, pages, 1)
        except Exception as error:
            self._failed(agent_id, document_id, error, 1)
        return self.get(agent_id, document_id)

    def edit_page(self, agent_id, document_id, page_index, text, expected_revision) -> Document:
        doc = self.get(agent_id, document_id)
        if doc.status == "processing":
            raise DocumentError("Dokumen sedang diproses. Tunggu hingga selesai.")
        pages = self.extraction(agent_id, document_id)
        if not isinstance(page_index, int) or not 0 <= page_index < len(pages):
            raise DocumentError("Halaman atau bagian tidak ditemukan.")
        if not isinstance(text, str) or "\x00" in text:
            raise DocumentError("Teks koreksi tidak valid.")
        pages[page_index]["text"] = text.strip()
        if sum(len(p["text"]) for p in pages) > 1_000_000:
            raise DocumentError("Teks terlalu besar; maksimal satu juta karakter per dokumen.")
        if sum(c.isalnum() for p in pages for c in p["text"]) < 20:
            raise DocumentError("Hasil koreksi terlalu pendek atau kosong untuk diindeks.")
        revision = expected_revision + 1
        with self.database.connect() as connection:
            updated = connection.execute(
                "UPDATE documents SET status='processing',revision=?,error_message=NULL "
                "WHERE id=? AND agent_id=? AND revision=? AND status IN ('ready','failed')",
                (revision, document_id, agent_id, expected_revision),
            ).rowcount
            if not updated:
                raise DocumentError("Dokumen telah berubah. Muat ulang halaman sebelum menyimpan.")
            connection.execute(
                "INSERT INTO document_edits(document_id,pages_json) VALUES(?,?) "
                "ON CONFLICT(document_id) DO UPDATE SET pages_json=excluded.pages_json",
                (document_id, json.dumps(pages, ensure_ascii=False)),
            )
        try:
            self._index(
                agent_id, document_id, doc.filename, [PageText(**p) for p in pages], revision
            )
        except Exception as error:
            self._failed(agent_id, document_id, error, revision)
        return self.get(agent_id, document_id)

    def delete(self, agent_id: str, document_id: str) -> None:
        self.agents.get(agent_id)
        with self.database.connect() as connection:
            found = connection.execute(
                "UPDATE documents SET status='failed',error_message='Penghapusan belum selesai; "
                "coba hapus kembali.' WHERE id=? AND agent_id=?",
                (document_id, agent_id),
            ).rowcount
            if not found:
                raise DocumentError("Dokumen tidak ditemukan pada agent ini.")
        try:
            self.knowledge.delete_document(agent_id, document_id)
        except Exception:
            raise DocumentError("Pembersihan indeks gagal. Coba hapus kembali.") from None
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE id=? AND agent_id=?", (document_id, agent_id)
            )
        logger.info("document_deleted agent_id=%s document_id=%s", agent_id, document_id)
