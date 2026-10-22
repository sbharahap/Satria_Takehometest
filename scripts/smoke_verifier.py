"""Opt-in local semantic-check smoke: supported claim versus fabricated claims."""

import json
from uuid import uuid4

from pypdf import PdfReader

from src.config import Settings
from src.models import Citation
from src.providers.registry import ProviderRegistry
from src.services.verification import CitationVerifier


def main():
    settings = Settings()
    path = settings.data_dir / "demo/hr/people_policy_ptvi_pol_0014_id.pdf"
    citation = Citation(
        tag="S1",
        chunk_id="smoke",
        document_id="smoke",
        filename=path.name,
        page=3,
        text=PdfReader(path).pages[2].extract_text(),
        score=1,
    )
    verifier = CitationVerifier(ProviderRegistry(settings), settings)
    records = []
    for answer, expected in [
        ("Siklus Manajemen Karyawan yang Terintegrasi dilaksanakan setiap tahun. [S1]", True),
        ("Siklus Manajemen Karyawan yang Terintegrasi dilaksanakan setiap bulan. [S1]", False),
        ("Karyawan berhak atas cuti tahunan sebanyak 60 hari. [S1]", False),
    ]:
        result = verifier.verify("Apa ketentuan dalam kebijakan ini?", answer, [citation])
        records.append({"answer": answer, "accepted": result, "expected": expected})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        assert result is expected
    scratch = settings.data_dir / "smoke-verifier" / uuid4().hex
    scratch.mkdir(parents=True)
    (scratch / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(
        "PASS: local verifier accepted annual cycle, rejected monthly cycle and invented leave. "
        f"{scratch}"
    )


if __name__ == "__main__":
    main()
