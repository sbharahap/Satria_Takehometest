import httpx

from src.config import Settings
from src.providers.base import (
    ChatProvider,
    ChatRequest,
    ChatResponse,
    ProviderError,
    checked_response,
    mapped_error,
)


class OpenRouterProvider(ChatProvider):
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, request: ChatRequest) -> ChatResponse:
        if self.settings.openrouter_free_only and not (
            request.model == "openrouter/free" or request.model.endswith(":free")
        ):
            raise ProviderError(
                "Mode gratis aktif. Pilih openrouter/free atau ID model berakhiran :free."
            )
        if not self.settings.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY belum diisi pada environment/.env.")
        try:
            with httpx.Client(timeout=self.settings.provider_timeout_seconds) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer "
                        + self.settings.openrouter_api_key.get_secret_value()
                    },
                    json={
                        "model": request.model,
                        "messages": request.messages,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "reasoning": {
                            "enabled": request.reasoning_mode == "thinking"
                            if request.reasoning_mode is not None
                            else self.settings.openrouter_reasoning_enabled
                        },
                        **(
                            {"response_format": {"type": "json_object"}}
                            if request.json_mode
                            else {}
                        ),
                    },
                )
                response.raise_for_status()
                body = response.json()
                if "error" in body:
                    raise ProviderError("Provider mengembalikan error. Periksa model dan kuota.")
                content = body["choices"][0]["message"]["content"]
                if not content and body["choices"][0].get("finish_reason") == "length":
                    raise ProviderError(
                        "Model menghabiskan batas token sebelum menjawab. "
                        "Pilih model non-reasoning atau ubah konfigurasi reasoning."
                    )
        except ProviderError:
            raise
        except Exception as error:
            raise mapped_error("openrouter", error) from None
        return checked_response(content)
