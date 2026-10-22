"""Import the selected Indonesian demo PDFs without changing company identities.

User-supplied binaries remain in ignored data/. See docs/DEMO_DATASET.md for provenance.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pypdf import PdfReader

REFERENCES = {
    "Contoh-PP-Sederhana-2023.pdf": (
        "https://disnakertrans.bantulkab.go.id/storage/disnakertrans/menu/294/"
        "Contoh-PP-Sederhana-2023.pdf"
    ),
    "Permenaker 28:2014.pdf": "https://bphn.go.id/data/documents/14pm028.pdf",
    "uu40-2007.pdf": "https://peraturan.go.id/id/uu-no-40-tahun-2007",
    "Scan-Surat24-07-2023-094710.pdf": (
        "https://jamkrida-jateng.co.id/wp-content/uploads/Scan-Surat24-07-2023-094710.pdf"
    ),
}

DOCUMENTS = [
    ("hr", "Contoh-PP-Sederhana-2023.pdf", "template"),
    ("hr", "Permenaker 28:2014.pdf", "regulation"),
    ("legal", "Permenaker 28:2014.pdf", "regulation"),
    ("legal", "uu40-2007.pdf", "regulation"),
    ("ocr-example", "Scan-Surat24-07-2023-094710.pdf", "external-company-example"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo-current"))
    args = parser.parse_args()
    records = []
    for group, name, kind in DOCUMENTS:
        source = args.source_dir / name
        if not source.exists() and ":" in name:
            source = args.source_dir / name.replace(":", "-")
        raw = source.read_bytes()
        target = args.output_dir / group / name.replace(":", "-")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append(
            {
                "file": str(target.relative_to(args.output_dir)),
                "kind": kind,
                "reference_url": REFERENCES[name],
                "pages": len(PdfReader(target).pages),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        print(f"{group}/{target.name}: {records[-1]['pages']} pages")
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
