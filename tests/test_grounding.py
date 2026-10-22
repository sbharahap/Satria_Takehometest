"""Closed-set provenance and conservative abstention, independent of policy benchmark."""

import json
from types import SimpleNamespace

import pytest

from src.knowledge.grounding import (
    evidence_units,
    has_requested_value,
    mentioned_documents,
    render_evidence,
    select_evidence,
)
from src.models import Source
from src.providers.base import ChatResponse
from src.services.chat import UNKNOWN, UNVERIFIED
from tests.test_rag import agent, make_pdf, rag  # noqa: F401


def source(text, **values):
    return Source(
        **dict(
            chunk_id="chunk",
            document_id="doc",
            filename="policy.pdf",
            page=2,
            text=text,
            score=0.95,
            **values,
        )
    )


def test_explicit_document_scope_and_ambiguous_shared_alias():
    docs = [
        SimpleNamespace(id="a", filename="benefit_policy_2025.pdf"),
        SimpleNamespace(id="b", filename="benefit_policy_2026.pdf"),
        SimpleNamespace(id="c", filename="security_manual.pdf"),
    ]
    assert mentioned_documents("Jelaskan benefit_policy_2026.pdf", docs) == {"b"}
    assert mentioned_documents("Jelaskan benefit policy", docs) == {"a", "b"}
    assert mentioned_documents("Apa kebijakan perusahaan?", docs) is None


@pytest.mark.parametrize(
    "question,text,expected",
    [
        ("Berapa nominal hadiah?", "Hadiah tunai dilarang menurut kebijakan 2025.", False),
        ("Tanggal berapa gaji dibayar?", "Upah dibayarkan setiap tanggal 5.", True),
        ("Berapa jam kerja?", "Pekerjaan dilakukan empat puluh jam per minggu.", True),
        ("Berapa nominal hadiah?", "Batas hadiah adalah Rp 250.000 per transaksi.", True),
        ("Berapa nominal hadiah?", "Batas hadiah adalah dua puluh ribu rupiah.", True),
        ("Berapa hari prosesnya?", "Proses dilaksanakan sesuai prosedur perusahaan.", False),
        ("Berapa hari prosesnya?", "Proses berlangsung lima hari kerja.", True),
        ("Berapa nomor hotline?", "Hubungi pusat layanan melalui formulir resmi.", False),
        ("Berapa nomor hotline?", "Hubungi hotline 0800-123-4567 untuk bantuan.", True),
    ],
)
def test_requested_value_type(question, text, expected):
    assert has_requested_value(question, text) is expected


def test_selector_ignores_model_authored_facts_and_preserves_literal_source():
    units = evidence_units(
        [source("Pengajuan diproses setiap tahun sesuai syarat yang berlaku.")], "Kapan?"
    )
    chosen = select_evidence(
        {"status": "answered", "evidence_ids": ["E1"], "answer": "Diproses setiap bulan."}, units
    )
    assert render_evidence(chosen) == units[0].text + " [S1]"
    assert "bulan" not in render_evidence(chosen)
    for ids in (["E99"], ["E1", "E1"], [], [1], "E1"):
        assert select_evidence({"status": "answered", "evidence_ids": ids}, units) is None


def test_units_preserve_conditions_and_source_mapping():
    text = "Hadiah boleh diterima hanya jika memenuhi prosedur dan hukum yang berlaku."
    units = evidence_units(
        [source(text), source("Aturan kedua berlaku kepada seluruh mitra.")], "Apa aturan?"
    )
    assert render_evidence(
        select_evidence({"status": "answered", "evidence_ids": ["E2", "E1"]}, units)
    ) == (f"1. {text} [S1]\n2. Aturan kedua berlaku kepada seluruh mitra. [S2]")


def test_extractive_pipeline_scopes_document_and_rejects_free_text(rag, monkeypatch):  # noqa: F811
    services, _, provider = rag
    services.chat.settings.answer_mode = "extractive"
    services.chat.settings.citation_verifier_enabled = True
    owner = agent(services, "HR")
    wanted = services.documents.upload(
        owner.id,
        "training_policy.pdf",
        make_pdf("Pengembangan karyawan dilaksanakan setiap tahun sesuai ketentuan perusahaan."),
    )
    services.documents.upload(
        owner.id,
        "benefit_policy.pdf",
        make_pdf("Pengembangan karyawan dilaksanakan setiap bulan sesuai ketentuan perusahaan."),
    )

    def complete(request):
        provider.calls.append(request)
        payload = json.loads(request.messages[-1]["content"])
        assert {u["filename"] for u in payload["evidence"]} == {"training_policy.pdf"}
        return ChatResponse(
            json.dumps({"status": "answered", "evidence_ids": ["E1"], "answer": "Setiap bulan."})
        )

    monkeypatch.setattr(provider, "complete", complete)
    monkeypatch.setattr(
        services.chat.verifier,
        "verify",
        lambda *a: pytest.fail("No LLM verdict for literal answers"),
    )
    result = services.chat.ask(
        owner.id, services.chat.new_conversation(owner.id), "Jelaskan training policy karyawan."
    )
    assert "setiap tahun" in result["content"] and "bulan" not in result["content"]
    assert result["citations"][0]["document_id"] == wanted.id
    services.documents.delete(owner.id, wanted.id)
    remaining = services.chat._retrieve(
        {"agent": owner, "question": "Jelaskan karyawan.", "query": "Jelaskan karyawan."}
    )["sources"]
    assert remaining and all(hit.document_id != wanted.id for hit in remaining)


def test_extractive_missing_amount_abstains_without_provider(rag):  # noqa: F811
    services, _, provider = rag
    services.chat.settings.answer_mode = "extractive"
    owner = agent(services, "Legal")
    services.documents.upload(
        owner.id,
        "gift_policy.pdf",
        make_pdf("Hadiah tunai dilarang bagi seluruh pemasok dan mitra perusahaan."),
    )
    result = services.chat.ask(
        owner.id, services.chat.new_conversation(owner.id), "Berapa nominal maksimal hadiah?"
    )
    assert result["content"] == UNKNOWN and not result["citations"]
    assert not provider.calls


def test_extractive_invalid_selector_and_tampered_answer_fail_closed(rag, monkeypatch):  # noqa: F811
    services, _, provider = rag
    services.chat.settings.answer_mode = "extractive"
    owner = agent(services, "HR")
    services.documents.upload(
        owner.id,
        "training.pdf",
        make_pdf("Pengembangan karyawan dilakukan setiap tahun sesuai prosedur perusahaan."),
    )
    monkeypatch.setattr(
        provider,
        "complete",
        lambda request: ChatResponse(
            '{"status":"answered","evidence_ids":["E999"],"answer":"Klaim buatan."}'
        ),
    )
    result = services.chat.ask(
        owner.id, services.chat.new_conversation(owner.id), "Jelaskan karyawan."
    )
    assert result["content"] == UNVERIFIED and not result["citations"]
    assert services.chat._verify_claims(
        {"answer": "Teks diganti", "evidence_answer": "Teks asli", "citations": [object()]}
    ) == {"answer": UNVERIFIED, "citations": []}


def test_numbered_followup_refinds_displayed_item_in_fresh_evidence():
    from src.knowledge.grounding import focus_numbered_followup

    sources = [
        source(
            "(i) Evaluasi kinerja dilakukan sesuai prosedur; "
            "(ii) Pengembangan dibahas bersama manajer; "
            "(iii) Risiko kehilangan talenta dievaluasi."
        )
    ]
    units = evidence_units(sources, "Apa poinnya?")
    history = [
        {
            "role": "assistant",
            "content": render_evidence(units),
            "citations": [{"document_id": "doc", "page": 2, "revision": 1}],
        }
    ]
    focused = focus_numbered_followup(
        "Jelaskan poin kedua itu", history, units, sources, {"doc": 1}
    )
    assert len(focused) == 1 and focused[0].text.startswith("(ii)")
    assert not focus_numbered_followup("Jelaskan poin keempat", history, units, sources, {"doc": 1})
    # History must not supply facts missing from retrieval or survive revision change.
    assert not focus_numbered_followup(
        "Jelaskan poin kedua", history, units[:1], sources, {"doc": 1}
    )
    assert (
        focus_numbered_followup("Jelaskan poin kedua", history, units, sources, {"doc": 2}) == units
    )


def test_numbered_followup_tracks_display_order_instead_of_source_item_number():
    from src.knowledge.grounding import focus_numbered_followup

    sources = [
        source(
            "(i) Evaluasi kinerja dilakukan sesuai prosedur; "
            "(ii) Pengembangan dibahas bersama manajer; "
            "(iii) Risiko kehilangan talenta dievaluasi."
        )
    ]
    units = evidence_units(sources, "Apa poinnya?")
    history = [
        {
            "role": "assistant",
            "content": render_evidence(units[1:]),
            "citations": [{"document_id": "doc", "page": 2, "revision": 1}],
        }
    ]
    focused = focus_numbered_followup("Jelaskan poin kedua", history, units, sources, {"doc": 1})
    assert len(focused) == 1 and focused[0].text.startswith("(iii)")


def test_multi_part_answer_accepts_five_literal_units_but_rejects_six():
    units = evidence_units(
        [
            source(
                "\n\n".join(f"Ketentuan nomor {i} berlaku bagi seluruh karyawan." for i in range(6))
            )
        ],
        "Jelaskan semua ketentuan.",
    )
    assert len(units) == 6
    assert (
        len(
            select_evidence(
                {"status": "answered", "evidence_ids": [u.id for u in units[:5]]}, units
            )
        )
        == 5
    )
    assert (
        select_evidence({"status": "answered", "evidence_ids": [u.id for u in units]}, units)
        is None
    )
