"""One bounded local retry, independent of retrieval and chat orchestration."""

import logging
from dataclasses import replace

from src.providers.base import ChatProvider, ChatRequest, ChatResponse, ProviderError

logger = logging.getLogger(__name__)


class LocalFallbackProvider(ChatProvider):
    def __init__(self, primary: ChatProvider, local: ChatProvider, model: str):
        self.primary, self.local, self.model = primary, local, model

    def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            return self.primary.complete(request)
        except ProviderError as error:
            reason = error.reason
            # Never log prompts, response contents, credentials, or raw exception text.
            logger.warning("provider_local_fallback")
        try:
            response = self.local.complete(replace(request, model=self.model))
        except ProviderError:
            raise ProviderError(
                "OpenRouter gagal dan fallback lokal juga gagal. "
                "Periksa koneksi Ollama dan ketersediaan model lokal, lalu coba kembali."
            ) from None
        return replace(response, fallback_model=self.model, fallback_reason=reason)
