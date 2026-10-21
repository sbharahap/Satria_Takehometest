from types import SimpleNamespace

import httpx
import pytest

from src.config import Settings
from src.providers.base import ChatProvider, ChatRequest, ChatResponse, ProviderError, mapped_error
from src.providers.litellm_adapter import LiteLLMProvider
from src.providers.openai_adapter import OpenAIProvider
from src.providers.openrouter_adapter import OpenRouterProvider
from src.providers.registry import ProviderRegistry


def request():
    return ChatRequest(model="example-model", messages=[{"role": "user", "content": "Halo"}])


def response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Jawaban"))])


@pytest.mark.parametrize("model", ["ollama_chat/llama3.1:8b", "ollama_chat/qwen3:8b"])
def test_litellm_uses_own_adapter_and_local_endpoint(monkeypatch, model):
    import litellm

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return response()

    monkeypatch.setattr(litellm, "completion", completion)
    provider = LiteLLMProvider(Settings(_env_file=None))
    result = provider.complete(
        ChatRequest(model=model, messages=request().messages, json_mode=True)
    )
    assert result.content == "Jawaban"
    assert calls[0]["api_base"] == "http://127.0.0.1:11434"
    assert calls[0]["num_retries"] == 0
    assert "api_key" not in calls[0]
    assert calls[0]["format"] == "json"
    if "qwen3" in model:
        assert calls[0]["reasoning_effort"] == "none"


def test_openai_uses_sdk_directly(monkeypatch):
    import openai

    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))

        def complete(self, **kwargs):
            calls.append(kwargs)
            return response()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(openai, "OpenAI", Client)
    result = OpenAIProvider(Settings(_env_file=None, openai_api_key="test-placeholder")).complete(
        request()
    )
    assert result.content == "Jawaban" and calls[0]["model"] == "example-model"


@pytest.mark.parametrize("status, expected", [(200, "Jawaban"), (401, "API key"), (429, "Batas")])
def test_openrouter_direct_http_and_safe_errors(monkeypatch, status, expected):
    original = httpx.Client

    def handler(req):
        assert req.url == "https://openrouter.ai/api/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer test-placeholder"
        return httpx.Response(status, json={"choices": [{"message": {"content": "Jawaban"}}]})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    provider = OpenRouterProvider(Settings(_env_file=None, openrouter_api_key="test-placeholder"))
    if status == 200:
        assert provider.complete(request()).content == expected
    else:
        with pytest.raises(ProviderError, match=expected):
            provider.complete(request())


@pytest.mark.parametrize("provider", [OpenAIProvider, OpenRouterProvider])
def test_missing_cloud_key_is_clear(provider, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="belum diisi"):
        provider(Settings(_env_file=None)).complete(request())


def test_error_mapping_redacts_raw_error_and_fourth_provider_needs_no_chat_changes(caplog):
    error = mapped_error("test", httpx.ReadTimeout("SENSITIVE-PLACEHOLDER"))
    assert "batas waktu" in str(error)
    assert "SENSITIVE-PLACEHOLDER" not in caplog.text

    class FourthProvider(ChatProvider):
        def complete(self, request):
            return ChatResponse("Fourth")

    registry = ProviderRegistry(Settings(_env_file=None))
    registry.register("fourth", lambda settings: FourthProvider())
    assert registry.get("fourth").complete(request()).content == "Fourth"
