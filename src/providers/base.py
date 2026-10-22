import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 700
    temperature: float = 0
    json_mode: bool = False
    reasoning_mode: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    content: str
    fallback_model: str | None = None


class ProviderError(RuntimeError):
    pass


def mapped_error(provider: str, error: Exception) -> ProviderError:
    status = getattr(error, "status_code", None)
    if status is None and getattr(error, "response", None) is not None:
        status = error.response.status_code
    kind = type(error).__name__.lower()
    logger.warning(
        "provider_failed provider=%s status=%s error_type=%s",
        provider,
        status,
        type(error).__name__,
    )
    if status in (401, 403) or "authentication" in kind:
        message = "API key tidak valid atau akses ditolak. Periksa konfigurasi provider."
    elif status == 429 or "ratelimit" in kind:
        message = "Batas permintaan provider tercapai. Tunggu sebentar lalu coba kembali."
    elif status == 402:
        message = "Kredit provider tidak mencukupi. Periksa saldo atau pilih provider lain."
    elif "timeout" in kind:
        message = "Provider melewati batas waktu. Coba kembali atau pilih model lebih kecil."
    elif status == 404:
        message = "Model atau endpoint tidak ditemukan. Periksa ID model dan koneksi provider."
    elif status == 400:
        message = (
            "Konfigurasi permintaan tidak didukung model. "
            "Periksa format JSON, ID model, dan pengaturan reasoning."
        )
    else:
        message = "Provider gagal merespons. Periksa koneksi, model, dan konfigurasi layanan."
    return ProviderError(message)


class ChatProvider(ABC):
    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse: ...


def checked_response(content) -> ChatResponse:
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("Provider tidak menghasilkan teks jawaban. Coba model lain.")
    return ChatResponse(content.strip())
