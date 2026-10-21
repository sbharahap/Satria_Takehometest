"""Validated configuration. Environment variables override the local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Ruang Dokumen"
    data_dir: Path = Path("data")
    sqlite_filename: str = "app.db"
    chroma_collection: str = "agent_documents"
    default_chat_provider: Literal["litellm", "openai", "openrouter"] = "litellm"
    litellm_model: str = "ollama_chat/qwen3:8b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_model: str = "gpt-4o-mini"
    openrouter_model: str = "openrouter/free"
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    litellm_api_key: SecretStr | None = None
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: int = Field(default=384, ge=1)
    chunk_size_tokens: int = Field(default=350, ge=32, le=480)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    retrieval_top_k: int = Field(default=4, ge=1, le=30)
    similarity_threshold: float = Field(default=0.83, ge=0, le=1)
    max_upload_mb: int = Field(default=20, ge=1, le=100)
    provider_timeout_seconds: float = Field(default=60, gt=0, le=300)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("openai_api_key", "openrouter_api_key", "litellm_api_key", mode="before")
    @classmethod
    def empty_key_is_unconfigured(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("data_dir")
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("sqlite_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("SQLITE_FILENAME harus berupa nama file, tanpa direktori.")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL harus berupa URL HTTP/HTTPS.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OLLAMA_BASE_URL tidak boleh memuat kredensial atau query.")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_chunk_overlap(self):
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS harus lebih kecil daripada CHUNK_SIZE_TOKENS.")
        return self

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.embedding_cache_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
