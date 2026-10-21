from src.config import Settings
from src.providers.base import (
    ChatProvider,
    ChatRequest,
    ChatResponse,
    ProviderError,
    checked_response,
    mapped_error,
)


class OpenAIProvider(ChatProvider):
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, request: ChatRequest) -> ChatResponse:
        from openai import OpenAI

        if not self.settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY belum diisi pada environment/.env.")
        try:
            with OpenAI(
                api_key=self.settings.openai_api_key.get_secret_value(),
                timeout=self.settings.provider_timeout_seconds,
                max_retries=0,
            ) as client:
                response = client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    max_completion_tokens=request.max_tokens,
                    **({"response_format": {"type": "json_object"}} if request.json_mode else {}),
                )
        except Exception as error:
            raise mapped_error("openai", error) from None
        return checked_response(response.choices[0].message.content)
