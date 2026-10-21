from functools import cached_property

from src.config import Settings


class LocalEmbeddings:
    """Multilingual E5, normalized, with distinct query and passage prefixes."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            self.settings.embedding_model,
            cache_folder=str(self.settings.embedding_cache_dir),
            trust_remote_code=False,
            device="cpu",
        )
        if model.get_sentence_embedding_dimension() != self.settings.embedding_dimension:
            raise ValueError("Dimensi embedding tidak sesuai konfigurasi indeks.")
        model.max_seq_length = 512
        return model

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Pertanyaan tidak boleh kosong.")
        return self.model.encode(
            [f"query: {text.strip()}"], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Potongan dokumen tidak boleh kosong.")
        return self.model.encode(
            [f"passage: {text.strip()}" for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
