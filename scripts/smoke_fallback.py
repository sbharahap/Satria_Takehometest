"""Opt-in: simulated OpenRouter outage, real E5/retrieval/Ollama, no cloud call."""

import json
from uuid import uuid4

from src.bootstrap import initialize
from src.config import Settings
from src.knowledge.embeddings import LocalEmbeddings
from src.models import AgentInput
from src.providers.base import ChatProvider, ProviderError
from src.services.chat import UNKNOWN


class SimulatedOutage(ChatProvider):
    calls = 0

    def __init__(self, settings):
        pass

    def complete(self, request):
        self.calls += 1
        raise ProviderError("Simulated cloud outage for local fallback smoke test.")


def main():
    settings = Settings()
    assert settings.local_fallback_enabled, "Enable LOCAL_FALLBACK_ENABLED first."
    scratch = settings.data_dir / "smoke-fallback" / uuid4().hex
    services = initialize(settings.model_copy(update={"data_dir": scratch}))
    embeddings = LocalEmbeddings(settings)
    services.documents.embeddings = services.chat.embeddings = embeddings
    outage = SimulatedOutage(settings)
    services.chat.providers.register("openrouter", lambda settings: outage)
    agent = services.agents.create(
        AgentInput(
            name="HR local fallback smoke",
            instructions="Jawab dari dokumen aktif.",
            provider="openrouter",
            model=settings.openrouter_model,
        )
    )
    source = settings.data_dir / "demo/hr/people_policy_ptvi_pol_0014_id.pdf"
    doc = services.documents.upload(agent.id, source.name, source.read_bytes())
    assert doc.status == "ready", doc.error_message
    conversation = services.chat.new_conversation(agent.id)
    question = "Seberapa sering siklus karyawan terintegrasi di perusahaan harus dilaksanakan?"
    answer = services.chat.ask(agent.id, conversation, question)
    assert answer["citations"] and "tahun" in answer["content"].lower()
    assert "fallback lokal digunakan" in answer["content"] and outage.calls == 1
    assert services.chat.messages(agent.id, conversation)[-1]["content"] == answer["content"]
    services.documents.delete(agent.id, doc.id)
    after_delete = services.chat.ask(agent.id, conversation, question)
    assert after_delete["content"] == UNKNOWN and not after_delete["citations"]
    assert outage.calls == 1  # Empty knowledge must not invoke either model.
    report = {
        "cloud_outage": "simulated; no OpenRouter network request",
        "local_model": settings.local_fallback_model,
        "answer": answer,
        "after_delete": after_delete,
    }
    (scratch / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"answer": answer["content"], "report": str(scratch / "results.json")}))
    print(
        "PASS: simulated cloud outage -> real local RAG, persisted notice, deleted source rejected."
    )


if __name__ == "__main__":
    main()
