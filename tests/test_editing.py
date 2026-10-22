from pathlib import Path

import pytest

from src.knowledge.parsing import DocumentError
from src.services.chat import UNKNOWN
from tests.test_docx import make_docx
from tests.test_rag import agent, make_pdf, rag  # noqa: F401


def test_original_file_is_isolated_immutable_and_cascades(rag):  # noqa: F811
    services, _, _ = rag
    hr, legal = agent(services, "HR"), agent(services, "Legal")
    original = make_pdf("Karyawan mengikuti pengembangan setiap tahun.")
    doc = services.documents.upload(hr.id, "people.pdf", original)
    with pytest.raises(DocumentError):
        services.documents.original_file(legal.id, doc.id)
    services.documents.edit_page(hr.id, doc.id, 0, "Karyawan mengikuti mentoring setiap bulan.", 1)
    assert services.documents.original_file(hr.id, doc.id) == original
    assert "tahun" in services.documents.extraction(hr.id, doc.id, original=True)[0]["text"]
    assert "bulan" in services.documents.extraction(hr.id, doc.id)[0]["text"]
    services.agents.delete(hr.id)
    with services.database.connect() as db:
        for table in ("document_originals", "document_edits", "document_extractions"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_edits_replace_all_vectors_and_citations_record_revision(rag):  # noqa: F811
    services, _, provider = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(
        hr.id, "people.pdf", make_pdf("Karyawan mengikuti program lama. " * 200)
    )
    assert doc.chunks_count > 1
    updated = services.documents.edit_page(
        hr.id, doc.id, 0, "Karyawan mengikuti program mentoring setiap bulan.", doc.revision
    )
    assert updated.status == "ready" and updated.revision == 2 and updated.chunks_count == 1
    assert services.knowledge.count() == 1
    found = services.knowledge.search(
        hr.id, [doc.id], services.chat.embeddings.embed_query("karyawan"), 4
    )
    assert found[0].revision == 2 and "program lama" not in found[0].text
    provider.content = "Karyawan mengikuti mentoring setiap bulan. [S1]"
    answer = services.chat.ask(
        hr.id, services.chat.new_conversation(hr.id), "Apa program karyawan?"
    )
    assert answer["citations"][0]["revision"] == 2
    assert "teks dikoreksi pengguna" in provider.calls[-1].messages[-1]["content"]


def test_failed_reindex_excludes_document_and_preserves_draft_for_retry(rag, monkeypatch):  # noqa: F811
    services, embeddings, _ = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(hr.id, "people.pdf", make_pdf("Karyawan belajar setiap tahun."))
    original = embeddings.embed_passages

    def fail(texts):
        assert services.documents.ready_ids(hr.id) == []
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(embeddings, "embed_passages", fail)
    draft = "Karyawan mengikuti pembelajaran setiap bulan."
    result = services.documents.edit_page(hr.id, doc.id, 0, draft, 1)
    assert result.status == "failed" and result.revision == 2
    assert services.documents.extraction(hr.id, doc.id)[0]["text"] == draft
    assert services.documents.ready_ids(hr.id) == [] and services.knowledge.count() == 0
    monkeypatch.setattr(embeddings, "embed_passages", original)
    assert services.documents.edit_page(hr.id, doc.id, 0, draft, 2).status == "ready"


def test_empty_and_stale_edits_cannot_overwrite_current_knowledge(rag):  # noqa: F811
    services, _, _ = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(hr.id, "hr.docx", make_docx())
    current = services.documents.extraction(hr.id, doc.id)
    result = services.documents.edit_page(hr.id, doc.id, 0, "Karyawan belajar bersama mentor.", 1)
    assert result.revision == 2
    with pytest.raises(DocumentError, match="telah berubah"):
        services.documents.edit_page(hr.id, doc.id, 0, "Karyawan memakai versi usang.", 1)
    pages = services.documents.extraction(hr.id, doc.id)
    assert pages[0]["text"] == "Karyawan belajar bersama mentor."
    assert all(p["page"] is None for p in pages)
    assert pages[0]["locator"] == current[0]["locator"]
    one = services.documents.upload(hr.id, "single.pdf", make_pdf("Karyawan belajar setiap tahun."))
    with pytest.raises(DocumentError, match="pendek atau kosong"):
        services.documents.edit_page(hr.id, one.id, 0, "", 1)
    assert services.documents.get(hr.id, one.id).revision == 1


def test_deleted_document_cannot_be_resurrected_by_editor(rag, monkeypatch):  # noqa: F811
    services, _, _ = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(hr.id, "single.pdf", make_pdf("Karyawan belajar setiap tahun."))
    original = services.knowledge.add_document

    def delete_during_add(agent_id, document_id, *args):
        original(agent_id, document_id, *args)
        services.documents.delete(agent_id, document_id)

    monkeypatch.setattr(services.knowledge, "add_document", delete_during_add)
    with pytest.raises(DocumentError, match="sudah dihapus"):
        services.documents.edit_page(hr.id, doc.id, 0, "Karyawan belajar setiap bulan.", 1)
    assert services.knowledge.count() == 0 and services.documents.list_documents(hr.id) == []


def test_old_revision_is_not_sent_to_followup_rewriter(rag):  # noqa: F811
    services, _, provider = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(hr.id, "single.pdf", make_pdf("Karyawan belajar setiap tahun."))
    chat = services.chat.new_conversation(hr.id)
    services.chat.ask(hr.id, chat, "Apa kebijakan karyawan?")
    services.documents.edit_page(hr.id, doc.id, 0, "Karyawan belajar setiap bulan.", 1)
    provider.fail = True
    assert services.chat.ask(hr.id, chat, "Jelaskan lagi poin tersebut.")["content"] == UNKNOWN


def test_revision_changed_during_verification_cannot_be_saved_as_current_answer(rag):  # noqa: F811
    services, _, _ = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(hr.id, "single.pdf", make_pdf("Karyawan belajar setiap tahun."))
    services.chat.settings.citation_verifier_enabled = True

    def edit_before_commit(*args):
        services.documents.edit_page(hr.id, doc.id, 0, "Karyawan belajar setiap bulan.", 1)
        return True

    services.chat.verifier.verify = edit_before_commit
    result = services.chat.ask(
        hr.id, services.chat.new_conversation(hr.id), "Apa kebijakan karyawan?"
    )
    assert result["content"] == UNKNOWN and not result["citations"]


def test_editor_ui_saves_and_reindexes(rag, monkeypatch):  # noqa: F811
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import src.bootstrap
    from src.config import get_settings

    services, _, _ = rag
    hr = agent(services, "HR editor")
    doc = services.documents.upload(hr.id, "review.pdf", make_pdf("Karyawan belajar setiap tahun."))
    monkeypatch.setattr(src.bootstrap, "initialize", lambda settings: services)
    monkeypatch.setenv("DATA_DIR", str(services.database.path.parent))
    get_settings.cache_clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
        assert not app.exception
        app.text_area(key=f"extraction_text_{doc.id}_1_0").input("Karyawan belajar setiap bulan.")
        next(b for b in app.button if b.label == "Simpan koreksi dan indeks ulang").click().run()
        assert not app.exception
        assert (
            app.text_area(key=f"extraction_text_{doc.id}_2_0").value
            == "Karyawan belajar setiap bulan."
        )
        assert services.documents.get(hr.id, doc.id).revision == 2
        assert any("Knowledge memakai teks koreksi" in w.value for w in app.warning)
    finally:
        st.cache_resource.clear()
        get_settings.cache_clear()


def test_pdf_viewer_page_matches_editor_selection_without_changing_original():
    import io

    from pypdf import PdfReader, PdfWriter

    from src.knowledge.parsing import pdf_page_preview

    writer = PdfWriter()
    for text in (
        "Halaman pertama untuk kebijakan karyawan.",
        "Halaman kedua untuk evaluasi karyawan.",
    ):
        writer.add_page(PdfReader(io.BytesIO(make_pdf(text))).pages[0])
    output = io.BytesIO()
    writer.write(output)
    original = output.getvalue()
    preview = PdfReader(io.BytesIO(pdf_page_preview(original, 2)))
    assert len(preview.pages) == 1
    assert "Halaman kedua" in preview.pages[0].extract_text()
    assert len(PdfReader(io.BytesIO(original)).pages) == 2
    with pytest.raises(DocumentError, match="tidak dapat ditampilkan"):
        pdf_page_preview(original, 3)


def test_v2_migration_keeps_parser_snapshot_and_adds_revision_and_original_storage(rag):  # noqa: F811
    services, _, _ = rag
    hr = agent(services, "HR")
    doc = services.documents.upload(hr.id, "old.pdf", make_pdf("Karyawan belajar setiap tahun."))
    snapshot = services.documents.extraction(hr.id, doc.id)
    with services.database.connect() as db:
        db.execute("DROP TABLE document_originals")
        db.execute("DROP TABLE document_edits")
        db.execute("ALTER TABLE documents DROP COLUMN revision")
        db.execute("PRAGMA user_version=2")
    services.database.initialize()
    assert services.documents.get(hr.id, doc.id).revision == 1
    assert services.documents.extraction(hr.id, doc.id) == snapshot
    assert services.documents.original_file(hr.id, doc.id) is None
