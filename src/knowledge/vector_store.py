from src.config import Settings
from src.models import Source


class KnowledgeStore:
    """Persistent vectors; embedding is always supplied explicitly by the application."""

    def __init__(self, settings: Settings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension,
            },
        )
        metadata = self.collection.metadata or {}
        if (
            metadata.get("embedding_model") != settings.embedding_model
            or metadata.get("embedding_dimension") != settings.embedding_dimension
        ):
            raise ValueError(
                "Konfigurasi embedding berbeda dari indeks tersimpan. "
                "Gunakan collection baru dan indeks ulang dokumen."
            )

    def delete_for_agent(self, agent_id: str) -> None:
        self.collection.delete(where={"agent_id": agent_id})

    def count(self) -> int:
        return self.collection.count()

    def delete_document(self, agent_id: str, document_id: str) -> None:
        self.collection.delete(
            where={
                "$and": [
                    {"agent_id": agent_id},
                    {"document_id": document_id},
                ]
            }
        )

    def add_document(self, agent_id, document_id, filename, chunks, vectors) -> None:
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            self.collection.upsert(
                ids=[f"{document_id}:{start + i}" for i in range(len(batch))],
                embeddings=vectors[start : start + 64],
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {
                        "agent_id": agent_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page": chunk.page,
                    }
                    for chunk in batch
                ],
            )

    def search(self, agent_id, document_ids, vector, top_k) -> list[Source]:
        if not document_ids or self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.count()),
            where={"$and": [{"agent_id": agent_id}, {"document_id": {"$in": document_ids}}]},
            include=["documents", "metadatas", "distances"],
        )
        sources = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            sources.append(
                Source(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page=meta["page"],
                    text=text,
                    score=max(0.0, min(1.0, 1.0 - distance)),
                )
            )
        return sources
