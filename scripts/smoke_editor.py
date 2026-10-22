"""Opt-in local RAG after a whitespace-only correction to a public policy PDF."""

import json
from uuid import uuid4

from src.bootstrap import initialize
from src.config import Settings
from src.knowledge.embeddings import LocalEmbeddings
from src.models import AgentInput


def main():
    settings = Settings()
    if not settings.litellm_model.startswith(("ollama/", "ollama_chat/")):
        raise ValueError("This test only permits local Ollama models.")
    scratch = settings.data_dir / "smoke-editor" / uuid4().hex
    services = initialize(settings.model_copy(update={"data_dir": scratch}))
    embeddings = LocalEmbeddings(settings)
    services.documents.embeddings = services.chat.embeddings = embeddings
    agent = services.agents.create(
        AgentInput(
            name="HR editor smoke",
            instructions="Jawab dari sumber aktif.",
            provider="litellm",
            model=settings.litellm_model,
        )
    )
    path = settings.data_dir / "demo/hr/people_policy_ptvi_pol_0014_id.pdf"
    original = path.read_bytes()
    doc = services.documents.upload(agent.id, path.name, original)
    assert doc.status == "ready"
    pages = services.documents.extraction(agent.id, doc.id)
    index = next(i for i, page in enumerate(pages) if page["page"] == 3)
    corrected = " ".join(pages[index]["text"].split())
    updated = services.documents.edit_page(agent.id, doc.id, index, corrected, doc.revision)
    assert updated.status == "ready" and updated.revision == 2
    assert services.documents.original_file(agent.id, doc.id) == original
    assert services.documents.extraction(agent.id, doc.id, original=True) == pages
    answer = services.chat.ask(
        agent.id,
        services.chat.new_conversation(agent.id),
        "Seberapa sering siklus karyawan terintegrasi di perusahaan harus dilaksanakan?",
    )
    assert answer["citations"] and "tahun" in answer["content"].lower()
    assert all(c["revision"] == 2 for c in answer["citations"])
    report = {
        "answer_mode": settings.answer_mode,
        "correction": "whitespace only, no company policy facts changed",
        "revision": updated.revision,
        "original_file_unchanged": True,
        "answer": answer,
    }
    (scratch / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"answer": answer["content"], "revision": updated.revision}))
    print(f"PASS: edited text -> reindex -> cited RAG ({settings.answer_mode}). {scratch}")


if __name__ == "__main__":
    main()
