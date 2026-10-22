import json

import pytest
from test_docx import make_docx
from test_rag import agent, make_pdf
from test_rag import rag as rag

from src.knowledge.parsing import DocumentError
from src.providers.base import ChatResponse
from src.services.chat import UNKNOWN


def test_doctor_does_not_cancel_upload_running_in_ui(rag, monkeypatch, capsys):
    import src.cli

    services, _, _ = rag
    owner = agent(services, "HR")
    with services.database.connect() as connection:
        connection.execute(
            "INSERT INTO documents(id,agent_id,filename,file_type,status) "
            "VALUES('in-flight',?,'working.pdf','pdf','processing')",
            (owner.id,),
        )
    monkeypatch.setattr(src.cli, "get_settings", lambda: services.documents.settings)
    monkeypatch.setattr(
        src.cli,
        "environment_report",
        lambda settings: {
            "python_supported": True,
            "packages": {"example": "installed"},
            "embedding_downloaded": True,
            "ollama": {"selected_model_available": True},
            "ocr": {"indonesian_and_english_ready": True},
        },
    )
    monkeypatch.setattr("sys.argv", ["cli", "doctor"])
    assert src.cli.main() == 0
    assert services.documents.get(owner.id, "in-flight").status == "processing"
    capsys.readouterr()


def test_docx_upload_cites_section_and_deletes_without_harming_pdf(rag):
    services, _, provider = rag
    hr = agent(services, "HR")
    pdf = services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    docx = services.documents.upload(hr.id, "mentoring.docx", make_docx())
    assert docx.status == "ready" and docx.file_type == "docx"
    provider.content = "Program mentoring diselenggarakan setiap bulan. [S1]"
    chat = services.chat.new_conversation(hr.id)
    result = services.chat.ask(hr.id, chat, "Bagaimana program mentoring karyawan?")
    citation = result["citations"][0]
    assert citation["filename"] == "mentoring.docx" and citation["page"] is None
    assert "Program mentoring" in citation["locator"]
    assert "halaman PDF None" not in provider.calls[-1].messages[-1]["content"]
    services.documents.delete(hr.id, docx.id)
    assert services.documents.ready_ids(hr.id) == [pdf.id]
    assert services.knowledge.collection.get(where={"document_id": docx.id})["ids"] == []


def test_corrupt_docx_status_failed_and_processing_visible_before_ready(rag, monkeypatch):
    services, embeddings, _ = rag
    hr = agent(services, "HR")
    failed = services.documents.upload(hr.id, "bad.docx", b"PKbroken")
    assert failed.status == "failed" and "DOCX" in failed.error_message
    original = embeddings.embed_passages

    def observe_processing(texts):
        assert any(d.status == "processing" for d in services.documents.list_documents(hr.id))
        assert services.documents.ready_ids(hr.id) == []
        return original(texts)

    monkeypatch.setattr(embeddings, "embed_passages", observe_processing)
    assert services.documents.upload(hr.id, "ok.docx", make_docx()).status == "ready"


def test_document_deleted_during_indexing_cannot_be_resurrected(rag, monkeypatch):
    services, _, _ = rag
    hr = agent(services, "HR")
    original = services.knowledge.add_document

    def delete_during_add(agent_id, document_id, *args):
        original(agent_id, document_id, *args)
        services.documents.delete(agent_id, document_id)

    monkeypatch.setattr(services.knowledge, "add_document", delete_during_add)
    with pytest.raises(DocumentError, match="sudah dihapus"):
        services.documents.upload(hr.id, "race.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    assert services.documents.list_documents(hr.id) == [] and services.knowledge.count() == 0


def test_multiturn_rewrites_numbered_reference_and_retrieves_fresh_sources(rag, monkeypatch):
    services, _, provider = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    chat = services.chat.new_conversation(hr.id)
    provider.content = "1. Pelatihan karyawan. 2. Mentoring karyawan. [S1]"
    services.chat.ask(hr.id, chat, "Apa dua program pengembangan karyawan?")
    requests = []

    def complete(request):
        requests.append(request)
        if "Tulis ulang" in request.messages[0]["content"]:
            assert "Mentoring karyawan" in request.messages[1]["content"]
            return ChatResponse(
                json.dumps({"query": "Jelaskan mentoring karyawan", "resolved": True})
            )
        assert "Jelaskan mentoring karyawan" in request.messages[1]["content"]
        return ChatResponse(
            json.dumps(
                {"status": "answered", "answer": "Mentoring karyawan.", "source_ids": ["S1"]}
            )
        )

    monkeypatch.setattr(provider, "complete", complete)
    result = services.chat.ask(hr.id, chat, "Jelaskan poin kedua itu.")
    assert result["citations"] and len(requests) == 2
    services.documents.delete(hr.id, doc.id)
    count = len(requests)
    assert services.chat.ask(hr.id, chat, "Jelaskan lagi poin kedua.")["content"] == UNKNOWN
    assert len(requests) == count


def test_deleted_answer_not_forwarded_to_rewriter_with_other_documents_present(rag):
    services, _, provider = rag
    hr = agent(services, "HR")
    first = services.documents.upload(
        hr.id, "first.pdf", make_pdf("Pengembangan karyawan perusahaan.")
    )
    chat = services.chat.new_conversation(hr.id)
    services.chat.ask(hr.id, chat, "Apa kebijakan pengembangan karyawan?")
    services.documents.upload(
        hr.id, "second.pdf", make_pdf("Karyawan memperoleh pendidikan keselamatan.")
    )
    services.documents.delete(hr.id, first.id)
    provider.fail = True
    assert services.chat.ask(hr.id, chat, "Jelaskan lagi hal tersebut.")["content"] == UNKNOWN


def test_agent_provider_model_updates_are_used_without_rebuilding_chat(rag):
    from src.models import AgentInput

    services, _, _ = rag
    hr = agent(services, "HR")
    services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    calls = []

    class ContractProvider:
        def __init__(self, name):
            self.name = name

        def complete(self, request):
            calls.append((self.name, request.model))
            return ChatResponse(
                json.dumps(
                    {"status": "answered", "answer": "Karyawan belajar.", "source_ids": ["S1"]}
                )
            )

    for name in ("litellm", "openai", "openrouter"):
        services.chat.providers.register(name, lambda settings, name=name: ContractProvider(name))
        services.agents.update(
            hr.id,
            AgentInput(
                name="HR",
                instructions="Gunakan sumber.",
                provider=name,
                model=f"test-{name}",
            ),
        )
        chat = services.chat.new_conversation(hr.id)
        assert services.chat.ask(hr.id, chat, "Apa kebijakan karyawan?")["citations"]
        assert calls[-1] == (name, f"test-{name}")


def test_docx_table_answer_uses_row_locator(rag):
    services, _, provider = rag
    hr = agent(services, "HR")
    services.documents.upload(hr.id, "mentoring.docx", make_docx())
    provider.content = "Tim Pembelajaran bertanggung jawab atas mentoring. [S1]"
    chat = services.chat.new_conversation(hr.id)
    result = services.chat.ask(hr.id, chat, "Siapa penanggung jawab program mentoring?")
    assert result["citations"]
    assert "tabel 1, baris 2" in result["citations"][0]["locator"]


def test_conversation_deletion_is_scoped_and_generation_cannot_revive_it(rag, monkeypatch):  # noqa: F811
    from tests.test_rag import agent

    services, _, _ = rag
    first, second = agent(services, "HR scoped chat"), agent(services, "Legal scoped chat")
    conversation = services.chat.new_conversation(first.id)
    services.chat.delete_conversation(second.id, conversation)
    assert services.chat.messages(first.id, conversation) == []

    def delete_while_generating(state):
        services.chat.delete_conversation(first.id, conversation)
        return {"answer": "Tidak ada bukti.", "citations": []}

    monkeypatch.setattr(services.chat.graph, "invoke", delete_while_generating)
    with pytest.raises(ValueError, match="Percakapan sudah dihapus"):
        services.chat.ask(first.id, conversation, "Jelaskan kebijakannya.")
    with services.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
