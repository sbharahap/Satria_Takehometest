from dataclasses import replace

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.providers.base import ChatRequest, ChatResponse, ProviderError
from src.providers.registry import ProviderRegistry
from tests.test_rag import agent, make_pdf, rag  # noqa: F401


class Stub:
    def __init__(self, error=None, response=None):
        self.error = error
        self.response = response or ChatResponse("local response")
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.response


def registry(primary, local, **config):
    result = ProviderRegistry(Settings(_env_file=None, **config))
    result.register("openrouter", lambda settings: primary)
    result.register("litellm", lambda settings: local)
    return result


def test_fallback_preserves_request_and_reports_model_without_recursion():
    primary, local = Stub(ProviderError("timeout", reason="timeout")), Stub()
    request = ChatRequest("cloud:free", [{"role": "user", "content": "question"}], 321, 0.2, True)
    result = registry(primary, local).get("openrouter").complete(request)
    assert result.fallback_model == "ollama_chat/qwen3:8b"
    assert result.fallback_reason == "timeout"
    assert local.calls == [replace(request, model=result.fallback_model)]
    assert primary.calls == [request]


def test_success_disabled_and_non_cloud_paths_do_not_fallback():
    primary, local = Stub(response=ChatResponse("cloud")), Stub()
    providers = registry(primary, local)
    request = ChatRequest("cloud:free", [])
    assert providers.get("openrouter").complete(request) == ChatResponse("cloud")
    assert not local.calls
    primary.error = ProviderError("failed")
    with pytest.raises(ProviderError, match="failed"):
        registry(primary, local, local_fallback_enabled=False).get("openrouter").complete(request)
    assert not local.calls
    local.error = ProviderError("local failure")
    with pytest.raises(ProviderError, match="local failure"):
        providers.get("litellm").complete(request)
    assert len(primary.calls) == 2


def test_both_fail_safely_and_only_once(caplog):
    primary = Stub(ProviderError("SENSITIVE-CLOUD"))
    local = Stub(ProviderError("SENSITIVE-LOCAL"))
    with pytest.raises(ProviderError, match="fallback lokal juga gagal") as error:
        registry(primary, local).get("openrouter").complete(ChatRequest("cloud:free", []))
    assert len(primary.calls) == len(local.calls) == 1
    assert "SENSITIVE" not in str(error.value) + caplog.text


def test_fallback_configuration_cannot_route_to_paid_api():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, local_fallback_model="openai/gpt-4o")
    assert Settings(_env_file=None, local_fallback_model="ollama_chat/llama3.1:8b")


def test_fallback_notice_persists_and_invalid_citations_still_fail_closed(rag):  # noqa: F811
    services, _, provider = rag
    hr = agent(services, "HR")
    services.documents.upload(hr.id, "people.pdf", make_pdf("Pembelajaran karyawan berkelanjutan."))
    original = provider.complete

    def complete(request):
        return replace(
            original(request), fallback_model="ollama_chat/qwen3:8b", fallback_reason="timeout"
        )

    provider.complete = complete
    conversation = services.chat.new_conversation(hr.id)
    result = services.chat.ask(hr.id, conversation, "Apa kebijakan pengembangan karyawan?")
    assert result["citations"] and "fallback lokal digunakan" in result["content"]
    assert "batas waktu" in result["content"]
    assert services.chat.messages(hr.id, conversation)[-1]["content"] == result["content"]
    provider.content = "Klaim palsu [S99]"
    result = services.chat.ask(
        hr.id, services.chat.new_conversation(hr.id), "Apa kebijakan karyawan?"
    )
    assert not result["citations"] and "sumber terverifikasi" in result["content"]
