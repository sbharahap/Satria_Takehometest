import logging
from pathlib import PurePosixPath
from uuid import uuid4

from src.config import Settings
from src.database.connection import Database
from src.knowledge.parsing import DocumentError, chunk_pages, parse_pdf
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
                "SELECT * FROM documents WHERE agent_id=? ORDER BY created_at,id",
                (agent_id,),
            ).fetchall()
        return [Document(**dict(row)) for row in rows]

    def ready_ids(self, agent_id: str) -> list[str]:
        return [d.id for d in self.list_documents(agent_id) if d.status == "ready"]

    def upload(self, agent_id: str, filename: str, data: bytes) -> Document:
        self.agents.get(agent_id)
        filename = PurePosixPath(filename.replace("\\", "/")).name[:200]
        if not filename.lower().endswith(".pdf"):
            raise DocumentError(
                "Tahap ini menerima PDF. Dukungan DOCX akan ditambahkan berikutnya."
            )
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise DocumentError(f"Ukuran file melebihi {self.settings.max_upload_mb} MB.")
        document_id = uuid4().hex
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO documents(id,agent_id,filename,file_type,status) VALUES(?,?,?,'pdf',"
                "'processing')",
                (document_id, agent_id, filename),
            )
        try:
            pages = parse_pdf(data)
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
            self.knowledge.add_document(agent_id, document_id, filename, chunks, vectors)
            with self.database.connect() as connection:
                active = connection.execute(
                    "SELECT id FROM agents WHERE id=? AND status='active'",
                    (agent_id,),
                ).fetchone()
                if not active:
                    raise DocumentError("Agent sudah dihapus. Unggah ke agent yang aktif.")
                connection.execute(
                    "UPDATE documents SET status='ready',chunks_count=? WHERE id=?",
                    (len(chunks), document_id),
                )
            logger.info(
                "document_ready agent_id=%s document_id=%s chunks=%s",
                agent_id,
                document_id,
                len(chunks),
            )
        except Exception as error:
            message = (
                str(error)
                if isinstance(error, DocumentError)
                else (
                    "Pemrosesan gagal. Periksa model embedding dan penyimpanan, lalu unggah ulang."
                )
            )
            try:
                self.knowledge.delete_document(agent_id, document_id)
            except Exception:
                logger.error("partial_index_cleanup_failed document_id=%s", document_id)
            with self.database.connect() as connection:
                connection.execute(
                    "UPDATE documents SET status='failed',error_message=? WHERE id=?",
                    (message, document_id),
                )
            logger.warning(
                "document_failed document_id=%s error_type=%s", document_id, type(error).__name__
            )
        docs = self.list_documents(agent_id)
        return next(d for d in docs if d.id == document_id)

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
        # A failed cleanup must never leave the document eligible for retrieval.
        try:
            self.knowledge.delete_document(agent_id, document_id)
        except Exception:
            raise DocumentError("Pembersihan indeks gagal. Coba hapus kembali.") from None
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE id=? AND agent_id=?", (document_id, agent_id)
            )
        logger.info("document_deleted agent_id=%s document_id=%s", agent_id, document_id)
