from collections.abc import Callable

from src.config import Settings
from src.providers.base import ChatProvider, ProviderError
from src.providers.fallback import LocalFallbackProvider
from src.providers.litellm_adapter import LiteLLMProvider
from src.providers.openai_adapter import OpenAIProvider
from src.providers.openrouter_adapter import OpenRouterProvider


class ProviderRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.factories: dict[str, Callable[[Settings], ChatProvider]] = {
            "litellm": LiteLLMProvider,
            "openai": OpenAIProvider,
            "openrouter": OpenRouterProvider,
        }

    def register(self, name: str, factory: Callable[[Settings], ChatProvider]) -> None:
        self.factories[name] = factory

    def get(self, name: str) -> ChatProvider:
        if name not in self.factories:
            raise ProviderError("Provider belum terdaftar.")
        primary = self.factories[name](self.settings)
        if name == "openrouter" and self.settings.local_fallback_enabled:
            return LocalFallbackProvider(
                primary,
                self.factories["litellm"](self.settings),
                self.settings.local_fallback_model,
            )
        return primary
