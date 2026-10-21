from src.config import Settings


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
