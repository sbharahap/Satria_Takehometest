import io
import json
import re
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.bootstrap import initialize
from src.config import Settings
from src.knowledge.parsing import DocumentError, parse_pdf
from src.models import AgentInput
from src.providers.base import ChatProvider, ChatResponse, ProviderError
from src.services.chat import UNKNOWN, UNVERIFIED


def make_pdf(text=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font)}
                ),
            }
        )
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class FakeEmbeddings:
    fail = False

    @property
    def model(self):
        return SimpleNamespace(
            tokenizer=lambda text, **kw: {
                "offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)],
            }
        )

    def embed_query(self, text):
        return ([1.0, 0.0] if "karyawan" in text.lower() else [0.0, 1.0]) + [0.0] * 382

    def embed_passages(self, texts):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [self.embed_query(text) for text in texts]


class FakeProvider(ChatProvider):
    content = "Pengembangan karyawan mengutamakan pembelajaran berkelanjutan. [S1]"
    fail = False

    def __init__(self):
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        if self.fail:
            raise ProviderError("Provider timeout")
        ids = re.findall(r"\[S(\d+)\]", self.content)
        return ChatResponse(
            json.dumps(
                {"status": "answered", "answer": self.content, "source_ids": [f"S{i}" for i in ids]}
            )
        )


@pytest.fixture
def rag(tmp_path):
    services = initialize(
        Settings(_env_file=None, data_dir=tmp_path, citation_verifier_enabled=False)
    )
    embeddings, provider = FakeEmbeddings(), FakeProvider()
    services.documents.embeddings = embeddings
    services.chat.embeddings = embeddings
    services.chat.providers.register("litellm", lambda settings: provider)
    return services, embeddings, provider


def agent(services, name):
    return services.agents.create(
        AgentInput(
            name=name, instructions="Gunakan sumber aktif.", provider="litellm", model="test-model"
        )
    )


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "kosong"),
        (b"not a pdf", "rusak"),
        (b"%PDF-broken", "rusak"),
        (make_pdf(), "scan"),
    ],
)
def test_bad_pdf_has_clear_error(data, message):
    with pytest.raises(DocumentError, match=message):
        parse_pdf(data)


def test_upload_retrieve_cite_isolate_and_delete_in_same_conversation(rag):
    services, _, provider = rag
    hr, legal = agent(services, "HR"), agent(services, "Legal")
    hr_doc = services.documents.upload(
        hr.id, "hr.pdf", make_pdf("Pengembangan karyawan mengutamakan pembelajaran berkelanjutan.")
    )
    legal_doc = services.documents.upload(
        legal.id,
        "legal.pdf",
        make_pdf("Larangan suap berlaku kepada pemasok serta mitra usaha perusahaan."),
    )
    assert hr_doc.status == legal_doc.status == "ready"
    hr_chat = services.chat.new_conversation(hr.id)
    result = services.chat.ask(hr.id, hr_chat, "Apa kebijakan pengembangan karyawan?")
    assert result["citations"][0]["filename"] == "hr.pdf"
    assert result["citations"][0]["page"] == 1
    assert "legal.pdf" not in provider.calls[-1].messages[-1]["content"]
    with pytest.raises(ValueError):
        services.chat.messages(legal.id, hr_chat)
    for owner, other_doc in ((hr, legal_doc), (legal, hr_doc)):
        vector = services.chat.embeddings.embed_query("karyawan")
        hits = services.knowledge.search(owner.id, [hr_doc.id, legal_doc.id], vector, 4)
        assert hits and all(hit.document_id != other_doc.id for hit in hits)
    services.documents.delete(hr.id, hr_doc.id)
    calls = len(provider.calls)
    repeated = services.chat.ask(hr.id, hr_chat, "Jelaskan lagi kebijakan karyawan itu.")
    assert repeated["content"] == UNKNOWN and repeated["citations"] == []
    assert len(provider.calls) == calls
    assert services.documents.list_documents(legal.id)[0].status == "ready"
    assert len(services.chat.messages(hr.id, hr_chat)) == 4


def test_failed_ingestion_and_failed_delete_are_not_retrievable(rag, monkeypatch):
    services, embeddings, _ = rag
    hr = agent(services, "HR")
    embeddings.fail = True
    doc = services.documents.upload(
        hr.id, "fail.pdf", make_pdf("Pengembangan karyawan perusahaan.")
    )
    assert doc.status == "failed" and services.knowledge.count() == 0
    embeddings.fail = False
    doc = services.documents.upload(hr.id, "ok.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    monkeypatch.setattr(
        services.knowledge,
        "delete_document",
        lambda *args: (_ for _ in ()).throw(RuntimeError("store unavailable")),
    )
    with pytest.raises(DocumentError):
        services.documents.delete(hr.id, doc.id)
    assert services.documents.ready_ids(hr.id) == []
    chat = services.chat.new_conversation(hr.id)
    assert services.chat.ask(hr.id, chat, "Apa kebijakan karyawan?")["content"] == UNKNOWN


def test_invalid_citation_and_provider_failure_do_not_become_grounded_answers(rag):
    services, _, provider = rag
    hr = agent(services, "HR")
    services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    chat = services.chat.new_conversation(hr.id)
    provider.content = "Jawaban dengan sumber palsu. [S99]"
    assert services.chat.ask(hr.id, chat, "Apa kebijakan karyawan?")["content"] == UNVERIFIED
    provider.fail = True
    with pytest.raises(ProviderError):
        services.chat.ask(hr.id, chat, "Apa kebijakan karyawan?")
    assert len(services.chat.messages(hr.id, chat)) == 2


def test_multiple_documents_and_interrupted_ingestion_recovery(rag):
    services, _, _ = rag
    hr = agent(services, "HR")
    for name in ("first.pdf", "second.pdf"):
        assert (
            services.documents.upload(
                hr.id, name, make_pdf("Pengembangan karyawan perusahaan.")
            ).status
            == "ready"
        )
    with services.database.connect() as connection:
        connection.execute(
            "INSERT INTO documents(id,agent_id,filename,file_type,status) "
            "VALUES('interrupted',?,'third.pdf','pdf','processing')",
            (hr.id,),
        )
    restarted = initialize(services.documents.settings)
    docs = restarted.documents.list_documents(hr.id)
    assert len(docs) == 3
    assert next(d for d in docs if d.id == "interrupted").status == "failed"
    assert len(restarted.documents.ready_ids(hr.id)) == 2


def test_chat_ui_renders_citation_and_delete_affects_next_turn(rag, monkeypatch):
    from pathlib import Path

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import src.bootstrap
    from src.config import get_settings

    services, _, _ = rag
    hr = agent(services, "HR UI")
    doc = services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    monkeypatch.setenv("DATA_DIR", str(services.database.path.parent))
    monkeypatch.setattr(src.bootstrap, "initialize", lambda settings: services)
    get_settings.cache_clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
        app.run()
        app.chat_input(key=f"question_{hr.id}").set_value("Apa kebijakan karyawan?").run()
        assert not app.exception
        assert any("[S1]" in element.value for element in app.markdown)
        assert any("halaman PDF 1" in expander.label for expander in app.expander)
        app.button(key=f"delete_doc_{doc.id}").click().run()
        app.chat_input(key=f"question_{hr.id}").set_value("Jelaskan lagi kebijakan itu.").run()
        assert not app.exception
        assert any(UNKNOWN == element.value for element in app.markdown)
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"status":"unknown","answer":"","source_ids":[]}', UNKNOWN),
        ('{"status":"answered","answer":"Unsupported","source_ids":[]}', UNVERIFIED),
        ("not-json", UNVERIFIED),
        ("[]", UNVERIFIED),
    ],
)
def test_structured_output_abstains_or_fails_closed(rag, monkeypatch, raw, expected):
    services, _, provider = rag
    hr = agent(services, "HR")
    services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    monkeypatch.setattr(provider, "complete", lambda request: ChatResponse(raw))
    chat = services.chat.new_conversation(hr.id)
    result = services.chat.ask(hr.id, chat, "Apa kebijakan karyawan?")
    assert result["content"] == expected and not result["citations"]


def test_reranking_prefers_explicit_attachment_clause():
    from src.knowledge.reranking import rerank
    from src.models import Source

    general = Source(
        chunk_id="general",
        document_id="doc",
        filename="policy.pdf",
        page=4,
        text="Perusahaan menggunakan klausul kontrak khusus tentang hak asasi manusia "
        "dan mengharapkan mitra menghormati nilai perusahaan.",
        score=0.90,
    )
    explicit = Source(
        chunk_id="explicit",
        document_id="doc",
        filename="policy.pdf",
        page=1,
        text="Kebijakan ini harus dilampirkan sebagai lampiran pada setiap kontrak "
        "yang dimiliki perusahaan dengan pihak ketiga manapun.",
        score=0.84,
    )
    hits = rerank(
        "Kebijakan apa yang harus menjadi lampiran kontrak pihak ketiga?", [general, explicit], 1
    )
    assert hits[0].chunk_id == "explicit"
    assert hits[0].score == 0.84


def test_agent_delete_removes_ingested_vectors_documents_and_chat(rag):
    services, _, _ = rag
    hr, legal = agent(services, "HR"), agent(services, "Legal")
    services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan perusahaan."))
    services.documents.upload(
        legal.id, "legal.pdf", make_pdf("Larangan suap pada mitra perusahaan.")
    )
    chat = services.chat.new_conversation(hr.id)
    services.chat.ask(hr.id, chat, "Apa kebijakan karyawan?")
    services.agents.delete(hr.id)
    assert services.knowledge.collection.get(where={"agent_id": hr.id})["ids"] == []
    assert services.knowledge.collection.get(where={"agent_id": legal.id})["ids"]
    with services.database.connect() as connection:
        for table in ("documents", "conversations", "messages"):
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE agent_id=?", (hr.id,)
                ).fetchone()[0]
                == 0
            )
