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
    fallback_reason: str | None = None


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, reason: str = "unavailable"):
        super().__init__(message)
        self.reason = reason


def failure_summary(reason: str | None) -> str:
    """Only predefined labels may reach persisted chat; never raw provider errors."""
    return {
        "authentication": "API key tidak valid atau akses ditolak",
        "rate_limit": "batas permintaan tercapai; jenis batas belum diketahui",
        "daily_quota": "kuota harian model gratis habis",
        "capacity": "model upstream sedang penuh atau tidak tersedia",
        "credits": "kredit provider tidak mencukupi",
        "timeout": "provider melewati batas waktu",
        "connection": "koneksi ke provider gagal",
        "model_not_found": "model atau rute tidak ditemukan",
        "invalid_request": "konfigurasi permintaan tidak didukung",
        "token_limit": "batas token habis sebelum jawaban selesai",
        "empty_response": "provider tidak menghasilkan teks",
    }.get(reason, "provider gagal merespons")


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
        reason = "authentication"
        message = "API key tidak valid atau akses ditolak. Periksa konfigurasi provider."
    elif status == 429 or "ratelimit" in kind:
        reason = "rate_limit"
        message = "Batas permintaan provider tercapai. Tunggu sebentar lalu coba kembali."
    elif status == 402:
        reason = "credits"
        message = "Kredit provider tidak mencukupi. Periksa saldo atau pilih provider lain."
    elif status == 408 or "timeout" in kind:
        reason = "timeout"
        message = "Provider melewati batas waktu. Coba kembali atau pilih model lebih kecil."
    elif "connect" in kind:
        reason = "connection"
        message = "Koneksi ke provider gagal. Periksa jaringan dan ketersediaan layanan."
    elif status == 404:
        reason = "model_not_found"
        message = "Model atau endpoint tidak ditemukan. Periksa ID model dan koneksi provider."
    elif status == 400:
        reason = "invalid_request"
        message = (
            "Konfigurasi permintaan tidak didukung model. "
            "Periksa format JSON, ID model, dan pengaturan reasoning."
        )
    elif status in (502, 503):
        reason = "capacity"
        message = "Model upstream sedang penuh atau tidak tersedia. Coba kembali nanti."
    else:
        reason = "unavailable"
        message = "Provider gagal merespons. Periksa koneksi, model, dan konfigurasi layanan."
    return ProviderError(message, reason=reason)


class ChatProvider(ABC):
    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse: ...


def checked_response(content) -> ChatResponse:
    if not isinstance(content, str) or not content.strip():
        raise ProviderError(
            "Provider tidak menghasilkan teks jawaban. Coba model lain.", reason="empty_response"
        )
    return ChatResponse(content.strip())
