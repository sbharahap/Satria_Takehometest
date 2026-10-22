"""Create the selected PT XYZ simulation agents from prepared local PDFs. No LLM calls."""

import argparse
from pathlib import Path

from src.bootstrap import initialize
from src.config import get_settings
from src.models import AgentInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/demo-current"))
    parser.add_argument("--include-ocr", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    groups = [("hr", "HR PT XYZ (simulasi)"), ("legal", "Legal PT XYZ (simulasi)")]
    if args.include_ocr:
        groups.append(("ocr-example", "OCR Jamkrida (contoh terpisah)"))
    for group, _ in groups:
        if not list((args.source_dir / group).glob("*.pdf")):
            raise ValueError("Prepare the selected local PDFs with scripts.prepare_demo.")
    services = initialize(settings)
    for group, name in groups:
        owner = next((a for a in services.agents.list_agents() if a.name == name), None)
        if owner is None:
            owner = services.agents.create(
                AgentInput(
                    name=name,
                    provider="litellm",
                    model=settings.litellm_model,
                    instructions=(
                        "Gunakan dokumen aktif. Bedakan template simulasi dan regulasi umum."
                    ),
                )
            )
        ready = {
            d.filename for d in services.documents.list_documents(owner.id) if d.status == "ready"
        }
        for path in sorted((args.source_dir / group).glob("*.pdf")):
            if path.name in ready:
                print(f"{name}: {path.name} already ready", flush=True)
                continue
            doc = services.documents.upload(owner.id, path.name, path.read_bytes())
            if doc.status != "ready":
                raise ValueError(doc.error_message or "Ingestion failed")
            print(f"{name}: {path.name} ready ({doc.chunks_count} chunks)", flush=True)


if __name__ == "__main__":
    main()
