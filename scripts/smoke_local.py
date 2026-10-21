"""Opt-in live PDF/RAG smoke test. Uses local Ollama; no paid/cloud calls.

Run from repository root: uv run python -m scripts.smoke_local
Prepare demo PDFs first with scripts.prepare_demo. Test records are isolated under data/.
"""

import argparse
import json
from uuid import uuid4

from src.bootstrap import initialize
from src.config import Settings
from src.knowledge.embeddings import LocalEmbeddings
from src.models import AgentInput
from src.services.chat import UNKNOWN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    settings = Settings()
    model = args.model or settings.litellm_model
    if not model.startswith(("ollama/", "ollama_chat/")):
        raise ValueError("This smoke test only allows local Ollama models.")
    scratch = settings.data_dir / "smoke" / uuid4().hex
    services = initialize(settings.model_copy(update={"data_dir": scratch}))
    embeddings = LocalEmbeddings(settings)
    services.documents.embeddings = services.chat.embeddings = embeddings
    agents = {}
    for group in ("hr", "legal"):
        agent = services.agents.create(
            AgentInput(
                name=f"{group.upper()} PT Vale Indonesia",
                provider="litellm",
                model=model,
                instructions="Jawab berdasarkan sumber aktif. Jangan mengarang kebijakan.",
            )
        )
        agents[group] = agent
        files = sorted((settings.data_dir / "demo" / group).glob("*_id.pdf"))
        assert len(files) == 2, "Prepare the four demo PDFs first."
        for path in files:
            doc = services.documents.upload(agent.id, path.name, path.read_bytes())
            assert doc.status == "ready", doc.error_message
            print(f"Indexed {group}: {doc.filename}, {doc.chunks_count} chunks", flush=True)
    results = []

    def ask(group, conversation, question, expected_file=None, expect_unknown=False):
        result = services.chat.ask(agents[group].id, conversation, question)
        record = {
            "agent": group,
            "question": question,
            "answer": result["content"],
            "citations": [
                {key: citation[key] for key in ("filename", "page", "score")}
                for citation in result["citations"]
            ],
        }
        results.append(record)
        (scratch / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if expect_unknown:
            assert result["content"] == UNKNOWN and not result["citations"], "Expected abstention"
        else:
            assert result["citations"], "Expected a verified citation"
            assert any(expected_file in c["filename"] for c in result["citations"])
        return result

    hr_chat = services.chat.new_conversation(agents["hr"].id)
    first = ask(
        "hr",
        hr_chat,
        "Apa tujuan siklus Manajemen Karyawan yang Terintegrasi dan siapa yang bertanggung jawab?",
        "people_policy",
    )
    assert "human capital" in first["content"].lower() and "kinerja" in first["content"].lower()
    followup = ask("hr", hr_chat, "Seberapa sering siklus tersebut dilaksanakan?", "people_policy")
    assert "setiap tahun" in followup["content"].lower()
    fresh = services.chat.new_conversation(agents["hr"].id)
    ask("hr", fresh, "Berapa jumlah hari cuti tahunan untuk karyawan PT Vale?", expect_unknown=True)
    fresh = services.chat.new_conversation(agents["hr"].id)
    ask("hr", fresh, "Apa definisi Red Flags dalam kebijakan antikorupsi?", expect_unknown=True)
    legal_chat = services.chat.new_conversation(agents["legal"].id)
    ask(
        "legal",
        legal_chat,
        "Apa yang dimaksud Red Flags dalam kebijakan antikorupsi?",
        "anti_corruption",
    )
    doc = next(
        d
        for d in services.documents.list_documents(agents["legal"].id)
        if "anti_corruption" in d.filename
    )
    services.documents.delete(agents["legal"].id, doc.id)
    ask(
        "legal",
        legal_chat,
        "Apa yang dimaksud Red Flags dalam kebijakan antikorupsi?",
        expect_unknown=True,
    )
    assert len(services.documents.ready_ids(agents["legal"].id)) == 1
    fresh = services.chat.new_conversation(agents["legal"].id)
    remaining = ask(
        "legal",
        fresh,
        "Apa kewajiban melampirkan kebijakan HAM pada kontrak pihak ketiga?",
        "human_rights",
    )
    assert any(c["page"] == 1 for c in remaining["citations"])
    print(f"PASS: seven live cases. Report: {scratch / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
