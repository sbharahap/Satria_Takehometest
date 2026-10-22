from src.config import Settings
from src.providers.base import (
    ChatProvider,
    ChatRequest,
    ChatResponse,
    checked_response,
    mapped_error,
)


class LiteLLMProvider(ChatProvider):
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, request: ChatRequest) -> ChatResponse:
        import litellm

        kwargs = {}
        if request.model.startswith(("ollama/", "ollama_chat/")):
            kwargs["api_base"] = self.settings.ollama_base_url
            kwargs["num_ctx"] = 8192
            if (
                request.model.startswith("ollama_chat/qwen3")
                and request.reasoning_mode != "thinking"
            ):
                kwargs["reasoning_effort"] = "none"
        elif self.settings.litellm_api_key:
            kwargs["api_key"] = self.settings.litellm_api_key.get_secret_value()
        if request.json_mode:
            if request.model.startswith(("ollama/", "ollama_chat/")):
                kwargs["format"] = "json"
            else:
                kwargs["response_format"] = {"type": "json_object"}
        try:
            response = litellm.completion(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                timeout=self.settings.provider_timeout_seconds,
                num_retries=0,
                **kwargs,
            )
            content = response.choices[0].message.content
        except Exception as error:
            raise mapped_error("litellm", error) from None
        return checked_response(content)
