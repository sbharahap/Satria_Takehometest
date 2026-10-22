import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.database.connection import Database
from src.knowledge.parsing import DocumentError, normalize_pdf_text
from src.models import Citation
from src.providers.base import ChatResponse, ProviderError
from src.services.chat import UNVERIFIED
from src.services.verification import CitationVerifier
from tests.test_docx import make_docx
from tests.test_rag import agent, make_pdf, rag  # noqa: F401


def test_extraction_is_original_prechunk_text_isolated_and_cascades(rag):  # noqa: F811
    services, _, _ = rag
    hr, legal = agent(services, "HR"), agent(services, "Legal")
    text = "Pengembangan karyawan berkelanjutan. " * 200
    doc = services.documents.upload(hr.id, "people.pdf", make_pdf(text))
    pages = services.documents.extraction(hr.id, doc.id)
    assert len(pages) == 1 and text.strip() in pages[0]["text"]
    assert pages[0]["page"] == 1 and pages[0]["locator"] is None
    assert doc.chunks_count > 1
    services.database.initialize()
    assert services.documents.extraction(hr.id, doc.id) == pages
    with pytest.raises(DocumentError, match="agent ini"):
        services.documents.extraction(legal.id, doc.id)
    services.documents.delete(hr.id, doc.id)
    with services.database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM document_extractions").fetchone()[0] == 0
    doc = services.documents.upload(hr.id, "people.docx", make_docx())
    pages = services.documents.extraction(hr.id, doc.id)
    assert all(p["page"] is None for p in pages)
    assert any("tabel" in p["locator"].lower() for p in pages)
    services.agents.delete(hr.id)
    with services.database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM document_extractions").fetchone()[0] == 0


def test_pdf_normalizer_joins_soft_wraps_and_keeps_paragraph_list_and_hyphenation():
    text = (
        "Kebijakan ini berlaku untuk seluruh tenaga\n"
        "kerja dan pihak terkait.\n\n"
        "1. Ketentuan utama\nRincian terpisah.\n"
        "anti-\nkorupsi"
    )
    assert normalize_pdf_text(text) == (
        "Kebijakan ini berlaku untuk seluruh tenaga kerja dan pihak terkait.\n\n"
        "1. Ketentuan utama Rincian terpisah.\nantikorupsi"
    )


def test_parse_success_embedding_failure_remains_reviewable_but_corrupt_has_no_text(rag):  # noqa: F811
    services, embeddings, _ = rag
    hr = agent(services, "HR")
    embeddings.fail = True
    doc = services.documents.upload(
        hr.id, "people.pdf", make_pdf("Karyawan belajar bersama pembimbing setiap tahun.")
    )
    assert doc.status == "failed" and services.documents.extraction(hr.id, doc.id)
    corrupt = services.documents.upload(hr.id, "broken.pdf", b"broken")
    assert corrupt.status == "failed" and not services.documents.extraction(hr.id, corrupt.id)


def test_additive_v1_migration_preserves_existing_data(rag):  # noqa: F811
    services, _, _ = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(
        hr.id, "old.pdf", make_pdf("Karyawan belajar bersama pembimbing setiap tahun.")
    )
    with services.database.connect() as db:
        db.execute("DROP TABLE document_extractions")
        db.execute("PRAGMA user_version=1")
    Database(services.database.path).initialize()
    assert services.agents.get(hr.id).name == "HR"
    assert services.documents.extraction(hr.id, doc.id) == []
    with services.database.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def test_extraction_ui_displays_text_and_location(rag, monkeypatch):  # noqa: F811
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import src.bootstrap
    from src.config import get_settings

    services, _, _ = rag
    hr = agent(services, "HR preview")
    doc = services.documents.upload(hr.id, "review.pdf", make_pdf("Karyawan belajar setiap tahun."))
    monkeypatch.setattr(src.bootstrap, "initialize", lambda settings: services)
    monkeypatch.setenv("DATA_DIR", str(services.database.path.parent))
    get_settings.cache_clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        assert not app.exception
        assert app.selectbox(key=f"extraction_page_{doc.id}").options == ["Halaman PDF 1"]
        text = app.text_area(key=f"extraction_text_{doc.id}_1_0")
        assert "Karyawan belajar setiap tahun." in text.value and not text.disabled
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()


class Verdict:
    def __init__(self, body):
        self.body, self.calls = body, []

    def complete(self, request):
        self.calls.append(request)
        if isinstance(self.body, Exception):
            raise self.body
        return ChatResponse(json.dumps(self.body))


def citation():
    return Citation(
        tag="S1",
        chunk_id="c1",
        document_id="d1",
        filename="policy.pdf",
        page=1,
        text="Siklus manajemen karyawan dilakukan setiap tahun.",
        score=0.9,
    )


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"supported": True, "evidence": [{"tag": "S1", "quote": "dilakukan setiap tahun"}]}, True),
        ({"supported": False, "evidence": []}, False),
        (
            {"supported": True, "evidence": [{"tag": "S1", "quote": "dilakukan setiap bulan"}]},
            False,
        ),
        (
            {"supported": True, "evidence": [{"tag": "S99", "quote": "dilakukan setiap tahun"}]},
            False,
        ),
        ({"supported": True, "evidence": []}, False),
        ({"supported": True, "evidence": [None]}, False),
        ([], False),
        (ProviderError("provider failed"), False),
    ],
)
def test_verifier_fails_closed_and_uses_local_model(body, expected):
    from src.config import Settings

    provider = Verdict(body)
    names = []

    def get(name):
        names.append(name)
        return provider

    settings = Settings(_env_file=None)
    verifier = CitationVerifier(SimpleNamespace(get=get), settings)
    assert verifier.verify("Seberapa sering?", "Setiap tahun. [S1]", [citation()]) is expected
    assert names == ["litellm"]
    assert provider.calls[0].model.startswith("ollama_chat/")


def test_semantic_rejection_removes_citations_in_full_graph(rag):  # noqa: F811
    services, _, _ = rag
    hr = agent(services, "HR")
    services.documents.upload(hr.id, "hr.pdf", make_pdf("Pengembangan karyawan setiap tahun."))
    services.chat.settings.citation_verifier_enabled = True
    rejecting = Verdict({"supported": False, "evidence": []})
    services.chat.verifier.providers = SimpleNamespace(get=lambda name: rejecting)
    answer = services.chat.ask(
        hr.id, services.chat.new_conversation(hr.id), "Apa kebijakan karyawan?"
    )
    assert answer["content"] == UNVERIFIED and not answer["citations"]
    assert rejecting.calls
